"""Tests for the model training contracts and metrics."""
from __future__ import annotations

import numpy as np

from src.models import baseline, dynamic
from src.models.metrics import error_metrics, silent_overrun_rate


def test_error_metrics_keys():
    predicted = np.array([10.0, 20.0, 30.0])
    actual = np.array([12.0, 18.0, 35.0])
    metrics = error_metrics(predicted, actual)
    assert set(metrics) >= {"mae", "rmse", "p50_abs_error", "p90_abs_error", "mean_bias"}
    assert metrics["mae"] > 0


def test_silent_overrun_rate_bounds():
    shown = np.array([30.0, 30.0, 30.0, 30.0])
    actual = np.array([25.0, 45.0, 50.0, 31.0])
    rate = silent_overrun_rate(shown, actual, tolerance_min=10.0)
    assert rate == 0.5  # two of four exceed by more than 10 minutes


def test_baseline_trains_and_reports(dataset):
    _, metrics = baseline.train(dataset)
    assert metrics["mae"] > 0
    assert 0.0 <= metrics["silent_overrun_rate"] <= 1.0
    assert metrics["n_test"] > 0


def test_dynamic_improves_at_later_checkpoints(dataset):
    _, metrics = dynamic.train(dataset)
    cps = metrics["per_checkpoint"]
    # Accuracy should be at least as good once the order is picked up.
    assert cps["pickup_confirmed"]["mae"] <= cps["order_placed"]["mae"]
    assert cps["pickup_confirmed"]["silent_overrun_rate"] <= cps["order_placed"]["silent_overrun_rate"]


def test_dynamic_matches_baseline_at_placement(dataset):
    _, base_metrics = baseline.train(dataset)
    _, dyn_metrics = dynamic.train(dataset)
    placement_mae = dyn_metrics["per_checkpoint"]["order_placed"]["mae"]
    # Same information at placement, so within a small tolerance of the baseline.
    assert abs(placement_mae - base_metrics["mae"]) < 1.0
