"""Evaluation metrics for delivery-time predictions.

Accuracy is reported as absolute error in minutes. The headline product metric
is the silent-overrun rate: the share of orders whose actual delivery time runs
past the ETA the customer was shown by more than a tolerance, without that ETA
having been revised. It captures the failure the whole project targets, where a
customer is left waiting well beyond a promise that is never updated.
"""
from __future__ import annotations

import numpy as np


def error_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    """Return absolute-error summary statistics for a set of predictions."""
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    abs_err = np.abs(predicted - actual)
    signed = predicted - actual
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(signed**2))),
        "p50_abs_error": float(np.quantile(abs_err, 0.50)),
        "p90_abs_error": float(np.quantile(abs_err, 0.90)),
        "mean_bias": float(signed.mean()),  # positive => over-promising speed
    }


def silent_overrun_rate(
    shown_eta: np.ndarray,
    actual: np.ndarray,
    tolerance_min: float,
) -> float:
    """Share of orders that finish more than ``tolerance_min`` past the shown ETA."""
    shown_eta = np.asarray(shown_eta, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.mean(actual - shown_eta > tolerance_min))
