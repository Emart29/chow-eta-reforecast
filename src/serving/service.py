"""Single-order inference and order state.

Wraps the serving bundle so a single order can be scored the same way the batch
models are trained: the raw order is turned into a one-row order table and a
one-row event stream, then passed through the fitted feature pipeline. Reusing
the pipeline keeps serving-time features identical to training-time features.

An in-memory :class:`OrderStore` records each order's placement ETA and latest
risk assessment, so the three lifecycle endpoints act on shared state rather than
recomputing from scratch on every call.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src import config
from src.risk.detector import suggested_message
from src.serving.bundle import ServingBundle, load_bundle

_OBSERVED_FIELDS = ("assignment_delay_min", "actual_prep_min", "to_restaurant_min")


@dataclass
class OrderContext:
    """Placement-time attributes of an order, shared by every prediction call."""

    order_id: str
    restaurant_id: str
    traffic_zone: str
    distance_km: float
    restaurant_prep_estimate: float
    rider_available_count_nearby: int
    restaurant_high_volume: bool
    weather_rain: bool
    order_placed_at: datetime


@dataclass
class RiskAssessment:
    order_id: str
    checkpoint: str
    risk_score: float
    flagged: bool
    placement_eta_minutes: float
    current_eta_minutes: float
    suggested_message: str | None


@dataclass
class OrderRecord:
    order_id: str
    placement_eta_minutes: float
    latest_eta_minutes: float
    latest_checkpoint: str
    risk: RiskAssessment | None = None


class OrderStore:
    """Minimal in-memory store of order predictions and risk state."""

    def __init__(self) -> None:
        self._records: dict[str, OrderRecord] = {}

    def upsert_placement(self, order_id: str, eta_minutes: float) -> None:
        self._records[order_id] = OrderRecord(
            order_id=order_id,
            placement_eta_minutes=eta_minutes,
            latest_eta_minutes=eta_minutes,
            latest_checkpoint=config.CHECKPOINTS[0],
        )

    def update_checkpoint(
        self, order_id: str, checkpoint: str, eta_minutes: float, risk: RiskAssessment | None
    ) -> None:
        record = self._records.get(order_id)
        if record is None:
            record = OrderRecord(order_id, eta_minutes, eta_minutes, checkpoint)
            self._records[order_id] = record
        record.latest_eta_minutes = eta_minutes
        record.latest_checkpoint = checkpoint
        if risk is not None:
            record.risk = risk

    def get(self, order_id: str) -> OrderRecord | None:
        return self._records.get(order_id)


def _temporal(ts: datetime) -> dict:
    hour = ts.hour
    dow = ts.weekday()
    is_peak = any(start <= hour < end for start, end in config.PEAK_HOURS)
    return {
        "hour_of_day": hour,
        "day_of_week": dow,
        "is_weekend": dow in config.WEEKEND_DAYS,
        "is_peak": is_peak,
    }


class ModelService:
    """Turns raw orders into ETA and risk predictions using the serving bundle."""

    def __init__(self, bundle: ServingBundle | None = None) -> None:
        self.bundle = bundle or load_bundle()

    # -- feature assembly ---------------------------------------------------
    def _feature_row(
        self, ctx: OrderContext, checkpoint: str, event_offset_min: float, observed: dict
    ) -> pd.DataFrame:
        temporal = _temporal(ctx.order_placed_at)
        order_row = pd.DataFrame(
            [
                {
                    "order_id": ctx.order_id,
                    "restaurant_id": ctx.restaurant_id,
                    "traffic_zone": ctx.traffic_zone,
                    "distance_km": ctx.distance_km,
                    "restaurant_prep_estimate": ctx.restaurant_prep_estimate,
                    "rider_available_count_nearby": ctx.rider_available_count_nearby,
                    "restaurant_high_volume": ctx.restaurant_high_volume,
                    "weather_rain": ctx.weather_rain,
                    **temporal,
                    **{f: observed.get(f, np.nan) for f in _OBSERVED_FIELDS},
                }
            ]
        )
        event_row = pd.DataFrame(
            [
                {
                    "order_id": ctx.order_id,
                    "checkpoint": checkpoint,
                    "checkpoint_index": config.CHECKPOINTS.index(checkpoint),
                    "event_offset_min": event_offset_min,
                    "remaining_min": np.nan,  # the prediction target; unknown at serving
                }
            ]
        )
        return self.bundle.builder.transform(order_row, event_row)

    # -- predictions --------------------------------------------------------
    def predict_eta(self, ctx: OrderContext) -> float:
        """Return the initial ETA in minutes from placement-time information."""
        features = self._feature_row(ctx, config.CHECKPOINTS[0], 0.0, {})
        remaining = float(self.bundle.eta_model.predict(features[self.bundle.eta_feature_columns])[0])
        return remaining  # elapsed is zero at placement

    def reforecast(
        self, ctx: OrderContext, checkpoint: str, elapsed_min: float, observed: dict
    ) -> float:
        """Return the revised total ETA in minutes at a lifecycle checkpoint."""
        features = self._feature_row(ctx, checkpoint, elapsed_min, observed)
        remaining = float(self.bundle.eta_model.predict(features[self.bundle.eta_feature_columns])[0])
        return elapsed_min + remaining

    def assess_risk(
        self, ctx: OrderContext, elapsed_min: float, observed: dict, placement_eta: float
    ) -> RiskAssessment:
        """Score silent-overrun risk at rider assignment and build the nudge."""
        checkpoint = "rider_assigned"
        features = self._feature_row(ctx, checkpoint, elapsed_min, observed)
        current_eta = elapsed_min + float(
            self.bundle.eta_model.predict(features[self.bundle.eta_feature_columns])[0]
        )
        features = features.assign(reforecast_growth=current_eta - placement_eta)
        score = float(self.bundle.risk_model.predict_proba(features[self.bundle.risk_feature_columns])[:, 1][0])
        flagged = score >= self.bundle.risk_threshold
        return RiskAssessment(
            order_id=ctx.order_id,
            checkpoint=checkpoint,
            risk_score=score,
            flagged=flagged,
            placement_eta_minutes=placement_eta,
            current_eta_minutes=current_eta,
            suggested_message=suggested_message(current_eta) if flagged else None,
        )

    @staticmethod
    def delivery_time(ctx: OrderContext, eta_minutes: float) -> datetime:
        return ctx.order_placed_at + timedelta(minutes=eta_minutes)
