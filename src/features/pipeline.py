"""Per-checkpoint feature engineering.

Turns the simulated orders and their lifecycle event stream into a supervised
learning table whose target is the delivery time remaining at each checkpoint.

The central concern is temporal leakage. A feature may enter a checkpoint's row
only if it is observable at that checkpoint. Placement-time attributes are known
from the first checkpoint onward; lifecycle measurements such as the realised
assignment delay or preparation time become available only once their stage has
passed. Signals that are not yet observable are left missing, which the gradient
boosted models consume natively.

:class:`FeatureBuilder` follows the scikit-learn ``fit``/``transform`` contract.
Historical aggregates are learned during ``fit`` on training orders only, so the
same object can transform a held-out set, or a single live order in the serving
layer, without leaking information across the split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Placement-time attributes carried unchanged onto every checkpoint row.
_PLACEMENT_NUMERIC = [
    "distance_km",
    "restaurant_prep_estimate",
    "hour_of_day",
    "day_of_week",
    "rider_available_count_nearby",
]
_PLACEMENT_BOOL = ["is_weekend", "is_peak", "weather_rain", "restaurant_high_volume"]

# Lifecycle measurements and the first checkpoint index at which each is observed.
_OBSERVED_AT = {
    "assignment_delay_min": 1,  # known once a rider is assigned
    "actual_prep_min": 2,  # known once the order is picked up
    "to_restaurant_min": 2,  # known once the rider reaches the restaurant
}

_ZONE_COLUMNS = [f"zone_{name}" for name in config.ZONE_NAMES]


def _zone_congestion(zones: pd.Series) -> np.ndarray:
    lookup = {z.name: z.congestion for z in config.ZONES}
    return zones.map(lookup).to_numpy()


def _expected_to_customer_min(orders: pd.DataFrame) -> np.ndarray:
    """Distance-based travel-time prior, computable from placement-time fields."""
    congestion = _zone_congestion(orders["traffic_zone"])
    rain_extra = np.where(orders["weather_rain"].to_numpy(), config.DELAY.rain_congestion_extra, 0.0)
    speed = config.DELAY.base_speed_kmpm / (congestion + rain_extra)
    return orders["distance_km"].to_numpy() / speed


class FeatureBuilder:
    """Learns historical aggregates and builds per-checkpoint feature tables."""

    def __init__(self) -> None:
        self._restaurant_prep: dict[str, float] = {}
        self._zone_hour_delivery: dict[tuple[str, int], float] = {}
        self._global_prep = 0.0
        self._global_delivery = 0.0
        self._fitted = False

    # -- fit ----------------------------------------------------------------
    def fit(self, orders: pd.DataFrame) -> FeatureBuilder:
        """Learn restaurant- and zone-level history from training orders only."""
        self._global_prep = float(orders["actual_prep_min"].mean())
        self._global_delivery = float(orders["actual_delivery_min"].mean())
        self._restaurant_prep = (
            orders.groupby("restaurant_id")["actual_prep_min"].mean().to_dict()
        )
        self._zone_hour_delivery = (
            orders.groupby(["traffic_zone", "hour_of_day"])["actual_delivery_min"]
            .mean()
            .to_dict()
        )
        self._fitted = True
        return self

    # -- transform ----------------------------------------------------------
    def transform(self, orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        """Build the checkpoint feature table for ``orders`` and their ``events``.

        Returns a frame with identifier columns (``order_id``, ``checkpoint``,
        ``checkpoint_index``, ``event_offset_min``), the target ``remaining_min``,
        and one column per feature. One row is produced per order and checkpoint.
        """
        if not self._fitted:
            raise RuntimeError("FeatureBuilder.transform called before fit.")

        base = self._order_level_features(orders)
        merged = events.merge(base, on="order_id", how="left", validate="many_to_one")

        idx = merged["checkpoint_index"].to_numpy()
        features = pd.DataFrame(index=merged.index)

        # Identifiers and target.
        for col in ("order_id", "checkpoint", "checkpoint_index"):
            features[col] = merged[col].to_numpy()
        features["event_offset_min"] = merged["event_offset_min"].to_numpy()
        features["remaining_min"] = merged["remaining_min"].to_numpy()

        # Placement-time features, always observable.
        for col in _PLACEMENT_NUMERIC:
            features[col] = merged[col].to_numpy()
        for col in _PLACEMENT_BOOL:
            features[col] = merged[col].astype(int).to_numpy()
        for col in _ZONE_COLUMNS:
            features[col] = merged[col].to_numpy()
        features["zone_congestion"] = merged["zone_congestion"].to_numpy()
        features["restaurant_hist_prep"] = merged["restaurant_hist_prep"].to_numpy()
        features["zone_hour_hist_delivery"] = merged["zone_hour_hist_delivery"].to_numpy()
        features["expected_to_customer_min"] = merged["expected_to_customer_min"].to_numpy()

        # Progress flags derived from how far the order has advanced.
        features["picked_up"] = (idx >= 2).astype(int)
        features["enroute"] = (idx >= 3).astype(int)

        # Lifecycle measurements, revealed only once their stage has passed.
        for col, first_index in _OBSERVED_AT.items():
            observable = idx >= first_index
            features[f"obs_{col}"] = np.where(observable, merged[col].to_numpy(), np.nan)

        # Observed preparation overrun, meaningful only after pickup.
        prep_overrun = merged["actual_prep_min"].to_numpy() - merged["restaurant_prep_estimate"].to_numpy()
        features["obs_prep_overrun_min"] = np.where(idx >= 2, prep_overrun, np.nan)

        return features

    def fit_transform(self, orders: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
        return self.fit(orders).transform(orders, events)

    # -- helpers ------------------------------------------------------------
    def _order_level_features(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Assemble the per-order attributes reused across all of an order's rows."""
        out = orders[
            ["order_id", "restaurant_id", "traffic_zone", *_PLACEMENT_NUMERIC, *_PLACEMENT_BOOL]
        ].copy()

        # Carry the raw lifecycle measurements; the transform masks them per checkpoint.
        for col in (*_OBSERVED_AT, "actual_prep_min"):
            out[col] = orders[col].to_numpy()

        zones = orders["traffic_zone"]
        for name, column in zip(config.ZONE_NAMES, _ZONE_COLUMNS):
            out[column] = (zones == name).astype(int).to_numpy()
        out["zone_congestion"] = _zone_congestion(zones)

        out["restaurant_hist_prep"] = (
            orders["restaurant_id"].map(self._restaurant_prep).fillna(self._global_prep).to_numpy()
        )
        keys = list(zip(orders["traffic_zone"], orders["hour_of_day"]))
        out["zone_hour_hist_delivery"] = [
            self._zone_hour_delivery.get(k, self._global_delivery) for k in keys
        ]
        out["expected_to_customer_min"] = _expected_to_customer_min(orders)
        return out


# Columns that are identifiers or the target, not model inputs.
NON_FEATURE_COLUMNS = ["order_id", "checkpoint", "checkpoint_index", "remaining_min"]


def feature_columns(table: pd.DataFrame) -> list[str]:
    """Return the model-input columns of a feature table built by :class:`FeatureBuilder`."""
    return [c for c in table.columns if c not in NON_FEATURE_COLUMNS]


# Columns that carry no information at order placement: they are either observed
# lifecycle measurements (missing at checkpoint 0) or progress markers that are
# constant there. The static model is restricted to the remaining placement-time
# columns so it depends only on what is known when the order is created.
def placement_feature_columns(table: pd.DataFrame) -> list[str]:
    """Return the feature columns observable at order placement (checkpoint 0)."""
    excluded = {"event_offset_min", "picked_up", "enroute"}
    return [
        c
        for c in feature_columns(table)
        if c not in excluded and not c.startswith("obs_")
    ]
