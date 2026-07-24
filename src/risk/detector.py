"""Silent-overrun risk detector.

The re-forecaster keeps the ETA honest, but a customer only benefits if someone
acts on a slipping estimate. This model closes that loop. At the rider-assignment
checkpoint, the earliest point where the realised assignment delay is known, it
predicts whether an order is heading for a silent overrun, defined as finishing
more than the tolerance past the ETA shown at order placement.

A flagged order is a cue to notify the customer proactively with the re-forecast
ETA, before they are left wondering. Because every nudge has a cost in attention
and trust, the detector is evaluated on precision and recall, and compared with a
simple rule that flags on how much the ETA has already grown, so the value of the
learned model over that rule is explicit.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src import config
from src.features.pipeline import feature_columns
from src.models import dynamic
from src.models.dataset import Dataset, build_dataset

MODEL_PATH = config.MODELS_DIR / "risk_detector.joblib"
METRICS_PATH = config.REPORTS_DIR / "risk_metrics.json"
EXAMPLES_PATH = config.REPORTS_DIR / "risk_examples.csv"

_TOL = config.SILENT_OVERRUN_THRESHOLD_MIN
_ASSIGNED = "rider_assigned"
_PLACED = "order_placed"


def _split_frame(features: pd.DataFrame, dyn_model) -> pd.DataFrame:
    """Assemble the assignment-checkpoint training frame for one data split.

    Returns one row per order with the assignment-checkpoint features, the ETA
    growth already implied by the re-forecast, the placement and assignment
    ETAs, and the silent-overrun label.
    """
    cols = feature_columns(features)
    eta = features["event_offset_min"].to_numpy() + dyn_model.predict(features[cols])

    ledger = features[["order_id", "checkpoint", "checkpoint_index", "event_offset_min", "remaining_min"]].copy()
    ledger["eta"] = eta
    ledger["actual_total"] = ledger["event_offset_min"] + ledger["remaining_min"]

    eta_pivot = ledger.pivot_table(index="order_id", columns="checkpoint", values="eta")
    actual = ledger[ledger["checkpoint_index"] == 0].set_index("order_id")["actual_total"]

    assigned = features[features["checkpoint_index"] == 1].set_index("order_id")
    frame = assigned[cols].copy()
    frame["reforecast_growth"] = (eta_pivot[_ASSIGNED] - eta_pivot[_PLACED]).reindex(frame.index)

    frame["placement_eta"] = eta_pivot[_PLACED].reindex(frame.index)
    frame["assignment_eta"] = eta_pivot[_ASSIGNED].reindex(frame.index)
    frame["actual_total"] = actual.reindex(frame.index)
    frame["silent_overrun"] = (frame["actual_total"] - frame["placement_eta"] > _TOL).astype(int)
    return frame.reset_index()


def _feature_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    non_features = {
        "order_id",
        "placement_eta",
        "assignment_eta",
        "actual_total",
        "silent_overrun",
    }
    return frame[[c for c in frame.columns if c not in non_features]]


def _new_model(pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=5,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        random_state=config.SEED,
        n_jobs=-1,
    )


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    # precision_recall_curve returns one more precision/recall point than thresholds.
    return float(thresholds[np.argmax(f1[:-1])])


def _operating_point(y_true: np.ndarray, flags: np.ndarray) -> dict:
    return {
        "precision": float(precision_score(y_true, flags, zero_division=0)),
        "recall": float(recall_score(y_true, flags, zero_division=0)),
        "flagged_share": float(flags.mean()),
    }


def _threshold_for_budget(scores: np.ndarray, target_share: float) -> float:
    """Return the score threshold that flags approximately ``target_share`` of orders."""
    return float(np.quantile(scores, 1.0 - target_share))


def train(dataset: Dataset | None = None) -> tuple[XGBClassifier, dict, pd.DataFrame]:
    """Train the risk detector and evaluate it against a rule-based flag."""
    dataset = dataset or build_dataset()
    dyn_model, _ = dynamic.train(dataset)

    train_frame = _split_frame(dataset.features_train, dyn_model)
    test_frame = _split_frame(dataset.features_test, dyn_model)

    x_train = _feature_matrix(train_frame)
    y_train = train_frame["silent_overrun"].to_numpy()
    x_test = _feature_matrix(test_frame)
    y_test = test_frame["silent_overrun"].to_numpy()

    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    model = _new_model(pos_weight)
    model.fit(x_train, y_train)

    scores = model.predict_proba(x_test)[:, 1]
    threshold = _best_f1_threshold(y_train, model.predict_proba(x_train)[:, 1])
    model_flags = scores >= threshold

    # Rule baseline: flag when the ETA has already grown past the configured budget.
    rule_flags = test_frame["reforecast_growth"].to_numpy() > config.RISK_ETA_GROWTH_MIN

    # Budget-matched point: give the learned model the same number of nudges as the
    # rule, so precision and recall are compared at an equal operating cost.
    matched_threshold = _threshold_for_budget(scores, float(rule_flags.mean()))
    matched_flags = scores >= matched_threshold

    metrics = {
        "prevalence": float(y_test.mean()),
        "threshold": threshold,
        "pr_auc": float(average_precision_score(y_test, scores)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "learned_model": _operating_point(y_test, model_flags),
        "learned_model_matched_budget": _operating_point(y_test, matched_flags),
        "rule_baseline": _operating_point(y_test, rule_flags),
        "n_test": len(y_test),
    }

    test_frame = test_frame.assign(risk_score=scores, flagged=model_flags)
    return model, metrics, test_frame


def _plot_pr_curve(y_true: np.ndarray, scores: np.ndarray, rule_point: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_true, scores)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.plot(recall, precision, color="#2a9d8f", linewidth=2, label="Learned model")
    ax.scatter(
        [rule_point["recall"]], [rule_point["precision"]],
        color="#d1495b", zorder=5, label="ETA-growth rule",
    )
    ax.axhline(y_true.mean(), color="gray", linestyle="--", linewidth=1, label="Prevalence")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Risk detector: precision vs recall at rider assignment")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "risk_pr_curve.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Proactive-message examples
# --------------------------------------------------------------------------- #
def suggested_message(assignment_eta: float) -> str:
    minutes = int(round(assignment_eta / 5.0) * 5)
    return f"Your order is running later than expected. Updated ETA: about {minutes} minutes."


def _example_notifications(test_frame: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    flagged = test_frame[test_frame["flagged"] & (test_frame["silent_overrun"] == 1)]
    top = flagged.sort_values("reforecast_growth", ascending=False).head(n)
    return pd.DataFrame(
        {
            "order_id": top["order_id"].to_numpy(),
            "placement_eta_min": top["placement_eta"].round(1).to_numpy(),
            "assignment_eta_min": top["assignment_eta"].round(1).to_numpy(),
            "actual_delivery_min": top["actual_total"].round(1).to_numpy(),
            "risk_score": top["risk_score"].round(3).to_numpy(),
            "suggested_message": [suggested_message(v) for v in top["assignment_eta"]],
        }
    )


def _print_metrics(metrics: dict) -> None:
    print("Silent-overrun risk detector (flagged at rider assignment)")
    print(f"  prevalence:            {metrics['prevalence']:.1%}")
    print(f"  PR-AUC:                {metrics['pr_auc']:.3f}")
    print(f"  ROC-AUC:               {metrics['roc_auc']:.3f}")
    lm, mm, rb = (
        metrics["learned_model"],
        metrics["learned_model_matched_budget"],
        metrics["rule_baseline"],
    )
    print(f"  {'':<26}{'precision':>10}{'recall':>9}{'flagged':>9}")
    print(f"  learned (F1-optimal)      {lm['precision']:>10.1%}{lm['recall']:>9.1%}{lm['flagged_share']:>9.1%}")
    print(f"  learned (rule budget)     {mm['precision']:>10.1%}{mm['recall']:>9.1%}{mm['flagged_share']:>9.1%}")
    print(f"  ETA-growth rule           {rb['precision']:>10.1%}{rb['recall']:>9.1%}{rb['flagged_share']:>9.1%}")


def main() -> None:
    config.ensure_dirs()
    model, metrics, test_frame = train()

    joblib.dump(model, MODEL_PATH)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    examples = _example_notifications(test_frame)
    examples.to_csv(EXAMPLES_PATH, index=False)
    _plot_pr_curve(
        test_frame["silent_overrun"].to_numpy(),
        test_frame["risk_score"].to_numpy(),
        metrics["rule_baseline"],
    )

    _print_metrics(metrics)
    print(f"\nsaved model    -> {MODEL_PATH}")
    print(f"saved metrics  -> {METRICS_PATH}")
    print(f"saved examples -> {EXAMPLES_PATH}")


if __name__ == "__main__":
    main()
