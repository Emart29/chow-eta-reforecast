"""Static ETA baseline.

Predicts a single delivery-time estimate at order placement from placement-time
features and never revises it. This mirrors the common production shortcut of
quoting one ETA up front, and it is the reference the dynamic re-forecaster is
measured against. Alongside accuracy, it reports the silent-overrun rate: how
often that never-updated ETA is left far behind by the actual delivery.
"""
from __future__ import annotations

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor

from src import config
from src.features.pipeline import placement_feature_columns
from src.models.dataset import Dataset, build_dataset
from src.models.metrics import error_metrics, silent_overrun_rate

MODEL_PATH = config.MODELS_DIR / "baseline_static.joblib"
METRICS_PATH = config.REPORTS_DIR / "baseline_metrics.json"


def _checkpoint0(features: pd.DataFrame) -> pd.DataFrame:
    """Select the order-placement rows; the static model uses these only."""
    return features[features["checkpoint_index"] == 0].reset_index(drop=True)


def _new_model() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        random_state=config.SEED,
        n_jobs=-1,
    )


def train(dataset: Dataset | None = None) -> tuple[XGBRegressor, dict]:
    """Train the static baseline and return the fitted model with its test metrics."""
    dataset = dataset or build_dataset()

    train_rows = _checkpoint0(dataset.features_train)
    test_rows = _checkpoint0(dataset.features_test)
    columns = placement_feature_columns(train_rows)

    model = _new_model()
    model.fit(train_rows[columns], train_rows["remaining_min"])

    # At placement the shown ETA equals the prediction and the target is the
    # full delivery time, so the prediction is both the estimate and the promise.
    predicted = model.predict(test_rows[columns])
    actual = test_rows["remaining_min"].to_numpy()

    metrics = error_metrics(predicted, actual)
    metrics["silent_overrun_rate"] = silent_overrun_rate(
        shown_eta=predicted,
        actual=actual,
        tolerance_min=config.SILENT_OVERRUN_THRESHOLD_MIN,
    )
    metrics["n_train"] = len(train_rows)
    metrics["n_test"] = len(test_rows)
    metrics["feature_columns"] = columns
    return model, metrics


def _print_metrics(metrics: dict) -> None:
    print("Static baseline (single ETA at order placement)")
    print(f"  test orders:          {metrics['n_test']:,}")
    print(f"  MAE:                  {metrics['mae']:.2f} min")
    print(f"  RMSE:                 {metrics['rmse']:.2f} min")
    print(f"  median abs error:     {metrics['p50_abs_error']:.2f} min")
    print(f"  P90 abs error:        {metrics['p90_abs_error']:.2f} min")
    print(f"  mean bias:            {metrics['mean_bias']:+.2f} min")
    print(
        f"  silent-overrun rate:  {metrics['silent_overrun_rate']:.1%} "
        f"(> {config.SILENT_OVERRUN_THRESHOLD_MIN:.0f} min past the shown ETA)"
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
