"""Shared dataset preparation for the ETA models.

Loads the simulated orders and event stream, splits orders into train and test
sets, and fits the feature pipeline on the training orders only. Both the static
baseline and the dynamic re-forecaster consume the tables produced here, so the
split and the fitted :class:`~src.features.pipeline.FeatureBuilder` are identical
across models and the comparison between them is fair.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.features.pipeline import FeatureBuilder


@dataclass
class Dataset:
    """Train/test feature tables plus the artefacts needed to evaluate them."""

    features_train: pd.DataFrame
    features_test: pd.DataFrame
    orders_train: pd.DataFrame
    orders_test: pd.DataFrame
    builder: FeatureBuilder


def _split_order_ids(order_ids: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = order_ids.copy()
    rng.shuffle(shuffled)
    cut = int(len(shuffled) * (1.0 - test_size))
    return shuffled[:cut], shuffled[cut:]


def load_simulated() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the enriched orders and event stream written by the simulator."""
    if not config.ORDERS_PATH.exists() or not config.EVENTS_PATH.exists():
        raise FileNotFoundError(
            "Simulated data not found. Run `python -m src.simulate.lifecycle` first."
        )
    orders = pd.read_parquet(config.ORDERS_PATH)
    events = pd.read_parquet(config.EVENTS_PATH)
    return orders, events


def build_dataset(
    test_size: float = 0.2,
    seed: int = config.SEED,
    orders: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
) -> Dataset:
    """Split orders, fit the feature pipeline on the training split, and transform both.

    ``orders`` and ``events`` may be supplied directly (for tests or ad-hoc runs);
    when omitted they are read from the simulated parquet files.
    """
    if orders is None or events is None:
        orders, events = load_simulated()

    train_ids, test_ids = _split_order_ids(orders["order_id"].to_numpy(), test_size, seed)
    train_mask = orders["order_id"].isin(train_ids)

    orders_train = orders[train_mask].reset_index(drop=True)
    orders_test = orders[~train_mask].reset_index(drop=True)

    events_train = events[events["order_id"].isin(train_ids)].reset_index(drop=True)
    events_test = events[events["order_id"].isin(test_ids)].reset_index(drop=True)

    builder = FeatureBuilder().fit(orders_train)
    features_train = builder.transform(orders_train, events_train)
    features_test = builder.transform(orders_test, events_test)

    return Dataset(
        features_train=features_train,
        features_test=features_test,
        orders_train=orders_train,
        orders_test=orders_test,
        builder=builder,
    )
