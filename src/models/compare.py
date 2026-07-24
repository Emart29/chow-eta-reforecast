"""Baseline-versus-dynamic comparison and explainability.

Trains the static baseline and the dynamic re-forecaster on one shared split,
then quantifies the before/after story the project is built around:

  * the silent-overrun rate and accuracy at each lifecycle stage,
  * how the improvement varies by zone, peak timing, and delay type, and
  * which signals drive the re-forecast, via SHAP attributions that show the
    observed lifecycle measurements activating only once their stage arrives.

All numbers are written to ``reports/comparison.json`` and the figures to
``reports/figures/`` for use in the write-up.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import config
from src.features.pipeline import feature_columns, placement_feature_columns
from src.models import baseline, dynamic
from src.models.dataset import build_dataset
from src.models.metrics import silent_overrun_rate

COMPARISON_PATH = config.REPORTS_DIR / "comparison.json"

# Consistent colours: the static "before" against the dynamic "after".
_STATIC_COLOR = "#d1495b"
_DYNAMIC_COLOR = "#2a9d8f"
_ACCENT = "#457b9d"
_TOL = config.SILENT_OVERRUN_THRESHOLD_MIN


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 10,
        }
    )


# --------------------------------------------------------------------------- #
# Prediction assembly
# --------------------------------------------------------------------------- #
def _order_level_results(dataset) -> pd.DataFrame:
    """Assemble one row per test order with the static and per-stage dynamic ETAs."""
    base_model, _ = baseline.train(dataset)
    dyn_model, _ = dynamic.train(dataset)

    # Resolve the column lists from the clean table before adding derived columns.
    dyn_cols = feature_columns(dataset.features_test)
    static_cols = placement_feature_columns(dataset.features_test)

    test = dataset.features_test.copy()
    test["pred_eta"] = test["event_offset_min"] + dyn_model.predict(test[dyn_cols])
    test["actual_total"] = test["event_offset_min"] + test["remaining_min"]

    # Static ETA: one prediction from the placement rows.
    cp0 = test[test["checkpoint_index"] == 0]
    static = pd.DataFrame(
        {
            "order_id": cp0["order_id"].to_numpy(),
            "actual_total": cp0["actual_total"].to_numpy(),
            "static_eta": base_model.predict(cp0[static_cols]),
        }
    )

    # Dynamic ETA at each checkpoint, pivoted to columns.
    pivot = test.pivot_table(index="order_id", columns="checkpoint", values="pred_eta")
    pivot.columns = [f"dyn_{c}" for c in pivot.columns]
    results = static.merge(pivot, on="order_id", how="left")

    attrs = dataset.orders_test[
        ["order_id", "traffic_zone", "is_peak", "compounding_delay", "restaurant_high_volume"]
    ]
    results = results.merge(attrs, on="order_id", how="left")
    return results, dyn_model, dataset


def _overrun(results: pd.DataFrame, eta_col: str, mask: np.ndarray | None = None) -> float:
    frame = results if mask is None else results[mask]
    return silent_overrun_rate(frame[eta_col].to_numpy(), frame["actual_total"].to_numpy(), _TOL)


# --------------------------------------------------------------------------- #
# Segment breakdown
# --------------------------------------------------------------------------- #
def _segment_breakdown(results: pd.DataFrame) -> dict:
    """Silent-overrun rate by segment for the static and dynamic-at-pickup ETAs."""
    segments: dict[str, np.ndarray] = {"all orders": np.ones(len(results), dtype=bool)}
    for zone in config.ZONE_NAMES:
        segments[f"zone: {zone}"] = (results["traffic_zone"] == zone).to_numpy()
    segments["peak hours"] = results["is_peak"].to_numpy().astype(bool)
    segments["off-peak"] = ~results["is_peak"].to_numpy().astype(bool)
    segments["compounding delay"] = results["compounding_delay"].to_numpy().astype(bool)
    segments["normal orders"] = ~results["compounding_delay"].to_numpy().astype(bool)

    breakdown = {}
    for name, mask in segments.items():
        breakdown[name] = {
            "n": int(mask.sum()),
            "static": _overrun(results, "static_eta", mask),
            "dynamic_at_pickup": _overrun(results, "dyn_pickup_confirmed", mask),
        }
    return breakdown


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _fig_overrun_by_stage(results: pd.DataFrame) -> None:
    stages = ["static_eta", "dyn_rider_assigned", "dyn_pickup_confirmed", "dyn_enroute_midpoint"]
    labels = ["Static\n(placement)", "Dynamic\n(assigned)", "Dynamic\n(pickup)", "Dynamic\n(en route)"]
    rates = [_overrun(results, col) for col in stages]
    colors = [_STATIC_COLOR, _DYNAMIC_COLOR, _DYNAMIC_COLOR, _DYNAMIC_COLOR]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, [r * 100 for r in rates], color=colors)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{rate:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylabel("Silent-overrun rate (%)")
    ax.set_title(f"Silent overruns collapse as the ETA is re-forecast\n(> {_TOL:.0f} min past the shown ETA)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "overrun_by_stage.png")
    plt.close(fig)


def _fig_mae_by_checkpoint(dyn_metrics: dict) -> None:
    names = list(dyn_metrics["per_checkpoint"].keys())
    mae = [dyn_metrics["per_checkpoint"][n]["mae"] for n in names]
    labels = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(labels, mae, marker="o", color=_ACCENT, linewidth=2)
    for x, y in zip(labels, mae):
        ax.text(x, y + 0.3, f"{y:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Mean absolute error (min)")
    ax.set_title("ETA accuracy improves at each checkpoint")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "mae_by_checkpoint.png")
    plt.close(fig)


def _fig_overrun_by_zone(breakdown: dict) -> None:
    zones = [f"zone: {z}" for z in config.ZONE_NAMES]
    static = [breakdown[z]["static"] * 100 for z in zones]
    dynamic_ = [breakdown[z]["dynamic_at_pickup"] * 100 for z in zones]
    x = np.arange(len(zones))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x - width / 2, static, width, label="Static", color=_STATIC_COLOR)
    ax.bar(x + width / 2, dynamic_, width, label="Dynamic (at pickup)", color=_DYNAMIC_COLOR)
    ax.set_xticks(x, config.ZONE_NAMES)
    ax.set_ylabel("Silent-overrun rate (%)")
    ax.set_title("Before/after silent-overrun rate by zone")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "overrun_by_zone.png")
    plt.close(fig)


def _fig_shap_global(dyn_model, dataset, sample: int = 3000) -> pd.DataFrame:
    test = dataset.features_test
    cols = feature_columns(test)
    rows = test.sample(min(sample, len(test)), random_state=config.SEED)
    explainer = shap.TreeExplainer(dyn_model)
    shap_values = explainer.shap_values(rows[cols])
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=cols).sort_values(ascending=False)

    top = mean_abs.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top.index, top.values, color=_ACCENT)
    ax.set_xlabel("Mean |SHAP| (min of remaining-time impact)")
    ax.set_title("What drives the re-forecast")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_global_importance.png")
    plt.close(fig)
    return mean_abs


def _fig_shap_by_checkpoint(dyn_model, dataset, sample: int = 4000) -> None:
    """Show the observed lifecycle signals activating only from their checkpoint on."""
    test = dataset.features_test
    cols = feature_columns(test)
    rows = test.sample(min(sample, len(test)), random_state=config.SEED)
    explainer = shap.TreeExplainer(dyn_model)
    shap_values = np.abs(explainer.shap_values(rows[cols]))
    shap_df = pd.DataFrame(shap_values, columns=cols)
    shap_df["checkpoint_index"] = rows["checkpoint_index"].to_numpy()

    tracked = [
        ("obs_assignment_delay_min", "assignment delay"),
        ("obs_prep_overrun_min", "prep overrun"),
        ("obs_to_restaurant_min", "rider→restaurant"),
        ("expected_to_customer_min", "distance prior"),
    ]
    by_cp = shap_df.groupby("checkpoint_index").mean(numeric_only=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(config.CHECKPOINTS))
    for col, label in tracked:
        ax.plot(x, [by_cp.loc[i, col] if i in by_cp.index else 0 for i in x], marker="o", label=label)
    ax.set_xticks(x, [c.replace("_", "\n") for c in config.CHECKPOINTS])
    ax.set_ylabel("Mean |SHAP| (min)")
    ax.set_title("Observed signals gain influence as the order progresses")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shap_by_checkpoint.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    _style()
    config.ensure_dirs()
    dataset = build_dataset()

    _, base_metrics = baseline.train(dataset)
    _, dyn_metrics = dynamic.train(dataset)

    results, dyn_model, dataset = _order_level_results(dataset)
    breakdown = _segment_breakdown(results)

    _fig_overrun_by_stage(results)
    _fig_mae_by_checkpoint(dyn_metrics)
    _fig_overrun_by_zone(breakdown)
    shap_importance = _fig_shap_global(dyn_model, dataset)
    _fig_shap_by_checkpoint(dyn_model, dataset)

    comparison = {
        "static_silent_overrun": base_metrics["silent_overrun_rate"],
        "static_mae": base_metrics["mae"],
        "dynamic_per_checkpoint": dyn_metrics["per_checkpoint"],
        "segment_breakdown": breakdown,
        "shap_top_features": shap_importance.head(12).round(3).to_dict(),
    }
    with COMPARISON_PATH.open("w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)
    return comparison


def _print_summary(comparison: dict) -> None:
    static = comparison["static_silent_overrun"]
    pickup = comparison["dynamic_per_checkpoint"]["pickup_confirmed"]["silent_overrun_rate"]
    print("Before / after")
    print(f"  static silent-overrun rate:        {static:.1%}")
    print(f"  dynamic silent-overrun (at pickup): {pickup:.1%}")
    print(f"  relative reduction:                 {(1 - pickup / static):.0%}")
    print("\nTop re-forecast drivers (mean |SHAP|):")
    for feature, value in list(comparison["shap_top_features"].items())[:6]:
        print(f"  {feature:<28}{value:>7.2f} min")
    print(f"\nsaved comparison -> {COMPARISON_PATH}")
    print(f"saved figures    -> {config.FIGURES_DIR}")


def main() -> None:
    comparison = run()
    _print_summary(comparison)


if __name__ == "__main__":
    main()
