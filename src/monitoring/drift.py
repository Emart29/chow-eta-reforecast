"""Data-drift monitoring scenario.

A model trained on one environment silently loses accuracy when the world moves
underneath it. This scenario makes that visible. A reference period is drawn from
the normal regime the serving model was trained on; a current period is drawn
from a wetter, more congested "rainy season". The already-trained model scores
both, and the two things a monitoring system must catch are reported side by
side: input drift, via Evidently, and the accuracy degradation that follows.

Outputs an Evidently HTML report and a JSON summary in ``reports/``.
"""
from __future__ import annotations

import json

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src import config
from src.features.pipeline import placement_feature_columns
from src.models.metrics import error_metrics, silent_overrun_rate
from src.serving.bundle import load_bundle
from src.simulate.lifecycle import simulate_lifecycle
from src.simulate.orders import generate_orders

DRIFT_HTML_PATH = config.REPORTS_DIR / "drift_report.html"
DRIFT_METRICS_PATH = config.REPORTS_DIR / "drift_metrics.json"

# The "rainy season" regime: rain is far more common and traffic is heavier.
RAINY_SEASON_RAIN_PROB = 0.55
RAINY_SEASON_CONGESTION_SCALE = 1.35


def _placement_frame(orders: pd.DataFrame, events: pd.DataFrame, bundle) -> pd.DataFrame:
    """Build the placement-checkpoint feature rows with predictions and actuals."""
    features = bundle.builder.transform(orders, events)
    cp0 = features[features["checkpoint_index"] == 0].reset_index(drop=True)
    cp0 = cp0.assign(
        predicted_eta=bundle.eta_model.predict(cp0[bundle.eta_feature_columns]),
        actual_delivery_min=cp0["remaining_min"],  # equals total delivery at placement
    )
    return cp0


def _performance(frame: pd.DataFrame) -> dict:
    predicted = frame["predicted_eta"].to_numpy()
    actual = frame["actual_delivery_min"].to_numpy()
    metrics = error_metrics(predicted, actual)
    metrics["silent_overrun_rate"] = silent_overrun_rate(
        predicted, actual, config.SILENT_OVERRUN_THRESHOLD_MIN
    )
    return metrics


def _parse_drift(snapshot_dict: dict) -> dict:
    """Extract the drift summary and per-column p-values from an Evidently snapshot."""
    overall = {"drifted_columns": None, "drifted_share": None}
    columns = {}
    for metric in snapshot_dict["metrics"]:
        mtype = metric.get("config", {}).get("type", "")
        if mtype.endswith("DriftedColumnsCount"):
            overall["drifted_columns"] = int(metric["value"]["count"])
            overall["drifted_share"] = float(metric["value"]["share"])
        elif mtype.endswith("ValueDrift"):
            column = metric["config"]["column"]
            p_value = float(metric["value"])
            columns[column] = {"p_value": p_value, "drifted": p_value < 0.05}
    return {"overall": overall, "columns": columns}


def run(n_orders: int = 12_000) -> dict:
    """Generate the two regimes, run drift detection, and measure degradation."""
    config.ensure_dirs()
    bundle = load_bundle()

    reference_orders = generate_orders(n_orders, seed=101)
    reference_orders, reference_events = simulate_lifecycle(reference_orders, seed=101)

    current_orders = generate_orders(n_orders, seed=202, rain_prob=RAINY_SEASON_RAIN_PROB)
    current_orders, current_events = simulate_lifecycle(
        current_orders, seed=202, congestion_scale=RAINY_SEASON_CONGESTION_SCALE
    )

    reference = _placement_frame(reference_orders, reference_events, bundle)
    current = _placement_frame(current_orders, current_events, bundle)

    # Compare the placement-time inputs the model consumes, plus its prediction
    # and the realised delivery time. The prediction and actual are appended
    # explicitly, so they are excluded from the feature list to avoid duplicates.
    derived = ["predicted_eta", "actual_delivery_min"]
    feature_cols = [c for c in placement_feature_columns(reference) if c not in derived]
    monitored = feature_cols + derived
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current[monitored], reference_data=reference[monitored])
    snapshot.save_html(str(DRIFT_HTML_PATH))

    drift = _parse_drift(snapshot.dict())
    summary = {
        "regime": {
            "reference": {"rain_prob": config.RAIN_PROB, "congestion_scale": 1.0},
            "current": {
                "rain_prob": RAINY_SEASON_RAIN_PROB,
                "congestion_scale": RAINY_SEASON_CONGESTION_SCALE,
            },
        },
        "drift": drift,
        "performance": {
            "reference": _performance(reference),
            "current": _performance(current),
        },
        "n_orders": n_orders,
    }
    with DRIFT_METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _print_summary(summary: dict) -> None:
    d = summary["drift"]["overall"]
    ref = summary["performance"]["reference"]
    cur = summary["performance"]["current"]
    print("Drift scenario: normal regime -> rainy season")
    print(f"  drifted columns:        {d['drifted_columns']} ({d['drifted_share']:.0%} of monitored)")
    print(f"  placement MAE:          {ref['mae']:.2f} -> {cur['mae']:.2f} min")
    print(f"  silent-overrun rate:    {ref['silent_overrun_rate']:.1%} -> {cur['silent_overrun_rate']:.1%}")
    print(f"  mean bias:              {ref['mean_bias']:+.2f} -> {cur['mean_bias']:+.2f} min")
    print(f"\nsaved report  -> {DRIFT_HTML_PATH}")
    print(f"saved metrics -> {DRIFT_METRICS_PATH}")


def main() -> None:
    _print_summary(run())


if __name__ == "__main__":
    main()
