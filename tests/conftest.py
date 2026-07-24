"""Shared pytest fixtures.

Every fixture builds its data in memory, so the suite never depends on the
generated parquet files (which are not checked in). Models are trained on a
small sample and shared across the session to keep the run fast.
"""
from __future__ import annotations

import pytest

from src.features.pipeline import feature_columns
from src.models import dynamic
from src.models.dataset import Dataset, build_dataset
from src.risk import detector
from src.serving.bundle import ServingBundle
from src.serving.service import ModelService
from src.simulate.lifecycle import simulate_lifecycle
from src.simulate.orders import generate_orders

N_ORDERS = 3000


@pytest.fixture(scope="session")
def simulated():
    orders = generate_orders(N_ORDERS, days=14, seed=123)
    orders, events = simulate_lifecycle(orders, seed=123)
    return orders, events


@pytest.fixture(scope="session")
def dataset(simulated) -> Dataset:
    orders, events = simulated
    return build_dataset(orders=orders, events=events, seed=123)


@pytest.fixture(scope="session")
def serving_bundle(dataset) -> ServingBundle:
    eta_model, _ = dynamic.train(dataset)
    risk_model, risk_metrics, _ = detector.train(dataset)
    eta_cols = feature_columns(dataset.features_test)
    return ServingBundle(
        builder=dataset.builder,
        eta_model=eta_model,
        eta_feature_columns=eta_cols,
        risk_model=risk_model,
        risk_feature_columns=[*eta_cols, "reforecast_growth"],
        risk_threshold=float(risk_metrics["threshold"]),
    )


@pytest.fixture(scope="session")
def service(serving_bundle) -> ModelService:
    return ModelService(bundle=serving_bundle)
