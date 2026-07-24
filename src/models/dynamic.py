"""Dynamic ETA re-forecasting.

A single gradient-boosted model predicts the delivery time remaining at any
checkpoint, conditioned on the signals observable there. Because the feature set
records how far the order has progressed (elapsed time, pickup and en-route
flags, and the lifecycle measurements revealed so far), one model re-forecasts at
every stage rather than a separate model per checkpoint: each checkpoint is a
fresh prediction on newly available information.

The ETA quoted at a checkpoint is the exactly known elapsed time plus the
predicted remaining time. Gradient boosting is a deliberate choice here: it is
fast, handles the mixed numeric and flag features with missing lifecycle values
natively, and stays explainable through feature attributions.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src import config
from src.features.pipeline import feature_columns
from src.models.dataset import Dataset, build_dataset
from src.models.metrics import error_metrics, silent_overrun_rate

MODEL_PATH = config.MODELS_DIR / "dynamic_reforecast.joblib"
METRICS_PATH = config.REPORTS_DIR / "dynamic_metrics.json"


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        random_state=config.SEED,
        n_jobs=-1,
    )


def _predicted_eta(rows: pd.DataFrame, predicted_remaining: np.ndarray) -> np.ndarray:
    """Reconstruct the total delivery ETA from known elapsed time plus prediction."""
    return rows["event_offset_min"].to_numpy() + predicted_remaining


def _actual_total(rows: pd.DataFrame) -> np.ndarray:
    """Recover the realised total delivery time from any checkpoint row."""
    return rows["event_offset_min"].to_numpy() + rows["remaining_min"].to_numpy()


def train(dataset: Dataset | None = None) -> tuple[XGBRegressor, dict]:
    """Train the re-forecaster and evaluate it at each lifecycle checkpoint."""
    dataset = dataset or build_dataset()

    train_rows = dataset.features_train
    test_rows = dataset.features_test
    columns = feature_columns(train_rows)

    model = _new_model()
    model.fit(train_rows[columns], train_rows["remaining_min"])

    predicted_remaining = model.predict(test_rows[columns])
    predicted_eta = _predicted_eta(test_rows, predicted_remaining)
    actual_total = _actual_total(test_rows)

    per_checkpoint: dict[str, dict] = {}
    for index, name in enumerate(config.CHECKPOINTS):
        mask = (test_rows["checkpoint_index"] == index).to_numpy()
        cp_metrics = error_metrics(predicted_eta[mask], actual_total[mask])
        cp_metrics["silent_overrun_rate"] = silent_overrun_rate(
            shown_eta=predicted_eta[mask],
            actual=actual_total[mask],
            tolerance_min=config.SILENT_OVERRUN_THRESHOLD_MIN,
        )
        cp_metrics["n"] = int(mask.sum())
        per_checkpoint[name] = cp_metrics

    metrics = {
        "per_checkpoint": per_checkpoint,
        "overall": error_metrics(predicted_eta, actual_total),
        "feature_columns": columns,
        "n_train_rows": int(len(train_rows)),
        "n_test_rows": int(len(test_rows)),
    }
    return model, metrics


def _print_metrics(metrics: dict) -> None:
    print("Dynamic re-forecaster (ETA revised at each checkpoint)")
    header = f"  {'checkpoint':<18}{'MAE':>8}{'P90 err':>10}{'silent overrun':>17}"
    print(header)
    for name, cp in metrics["per_checkpoint"].items():
        print(
            f"  {name:<18}{cp['mae']:>7.2f}{cp['p90_abs_error']:>10.2f}"
            f"{cp['silent_overrun_rate']:>16.1%}"
        )


def main() -> None:
    config.ensure_dirs()
    model, metrics = train()

    joblib.dump(model, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    _print_metrics(metrics)
    print(f"\nsaved model   -> {MODEL_PATH}")
    print(f"saved metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
