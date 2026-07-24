"""Monitoring dashboard for the ETA re-forecasting system.

Presents the before/after story interactively: the accuracy and silent-overrun
rate at each lifecycle checkpoint, how the improvement varies by segment, what
drives the re-forecast, and a live feed of orders currently flagged for a
proactive notification.

The dashboard reads the artefacts written by the training and evaluation
pipeline. Regenerate them first with::

    python -m src.models.compare
    python -m src.risk.detector

Run the dashboard with::

    streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Make the project importable when launched via `streamlit run dashboard/app.py`.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config

STATIC_COLOR = "#d1495b"
DYNAMIC_COLOR = "#2a9d8f"

st.set_page_config(page_title="Chow ETA Re-forecasting", page_icon="🛵", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data
def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@st.cache_data
def _load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _require(*artifacts) -> bool:
    if any(a is None for a in artifacts):
        st.error(
            "Report artefacts not found. Generate them first:\n\n"
            "```\npython -m src.models.compare\npython -m src.risk.detector\n```"
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
def _stage_chart(comparison: dict) -> alt.Chart:
    cp = comparison["dynamic_per_checkpoint"]
    rows = [{"stage": "Static (placement)", "rate": comparison["static_silent_overrun"], "kind": "Static"}]
    labels = {
        "rider_assigned": "Dynamic (assigned)",
        "pickup_confirmed": "Dynamic (pickup)",
        "enroute_midpoint": "Dynamic (en route)",
    }
    for key, label in labels.items():
        rows.append({"stage": label, "rate": cp[key]["silent_overrun_rate"], "kind": "Dynamic"})
    frame = pd.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("stage:N", sort=None, title=None),
            y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="Silent-overrun rate"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(domain=["Static", "Dynamic"], range=[STATIC_COLOR, DYNAMIC_COLOR]),
                legend=None,
            ),
            tooltip=[alt.Tooltip("rate:Q", format=".1%")],
        )
        .properties(height=320)
    )


def _mae_chart(comparison: dict) -> alt.Chart:
    cp = comparison["dynamic_per_checkpoint"]
    frame = pd.DataFrame(
        [{"checkpoint": name.replace("_", " "), "mae": cp[name]["mae"], "order": i}
         for i, name in enumerate(config.CHECKPOINTS)]
    )
    base = alt.Chart(frame).encode(
        x=alt.X("checkpoint:N", sort=alt.SortField("order"), title=None),
        y=alt.Y("mae:Q", title="Mean absolute error (min)"),
    )
    line = base.mark_line(color="#457b9d", point=True)
    text = base.mark_text(dy=-10, fontSize=11).encode(text=alt.Text("mae:Q", format=".1f"))
    return (line + text).properties(height=320)


def _segment_chart(comparison: dict) -> alt.Chart:
    breakdown = comparison["segment_breakdown"]
    segments = ["all orders", "peak hours", "off-peak", "compounding delay", "normal orders"]
    rows = []
    for seg in segments:
        rows.append({"segment": seg, "rate": breakdown[seg]["static"], "kind": "Static"})
        rows.append({"segment": seg, "rate": breakdown[seg]["dynamic_at_pickup"], "kind": "Dynamic (pickup)"})
    frame = pd.DataFrame(rows)
    return (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("kind:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("rate:Q", axis=alt.Axis(format="%"), title="Silent-overrun rate"),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Static", "Dynamic (pickup)"], range=[STATIC_COLOR, DYNAMIC_COLOR]
                ),
                title=None,
            ),
            column=alt.Column("segment:N", sort=segments, title=None,
                              header=alt.Header(labelOrient="bottom")),
            tooltip=[alt.Tooltip("rate:Q", format=".1%")],
        )
        .properties(height=300, width=90)
    )


def _shap_chart(comparison: dict) -> alt.Chart:
    shap = comparison["shap_top_features"]
    frame = pd.DataFrame({"feature": list(shap.keys()), "impact": list(shap.values())})
    return (
        alt.Chart(frame)
        .mark_bar(color="#457b9d")
        .encode(
            x=alt.X("impact:Q", title="Mean |SHAP| (min)"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("impact:Q", format=".2f")],
        )
        .properties(height=340)
    )


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🛵 Dynamic ETA Re-forecasting")
    st.caption(
        "Synthetic-data proof of concept. Delivery ETAs are re-forecast at every "
        "lifecycle checkpoint, and orders at risk of a silent overrun are flagged for "
        "a proactive customer notification."
    )

    comparison = _load_json(config.REPORTS_DIR / "comparison.json")
    risk = _load_json(config.REPORTS_DIR / "risk_metrics.json")
    examples = _load_csv(config.REPORTS_DIR / "risk_examples.csv")
    drift = _load_json(config.REPORTS_DIR / "drift_metrics.json")
    if not _require(comparison, risk):
        return

    static = comparison["static_silent_overrun"]
    pickup = comparison["dynamic_per_checkpoint"]["pickup_confirmed"]["silent_overrun_rate"]
    pickup_mae = comparison["dynamic_per_checkpoint"]["pickup_confirmed"]["mae"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Static silent-overrun", f"{static:.1%}")
    c2.metric("Dynamic (at pickup)", f"{pickup:.1%}", f"-{static - pickup:.1%}", delta_color="inverse")
    c3.metric("Relative reduction", f"{1 - pickup / static:.0%}")
    c4.metric("MAE: static → pickup", f"{comparison['static_mae']:.1f} → {pickup_mae:.1f} min")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.subheader("Silent overruns collapse as the ETA is re-forecast")
        st.altair_chart(_stage_chart(comparison), use_container_width=True)
    with right:
        st.subheader("Accuracy improves at each checkpoint")
        st.altair_chart(_mae_chart(comparison), use_container_width=True)

    st.subheader("Before / after by segment")
    st.altair_chart(_segment_chart(comparison), use_container_width=False)

    left2, right2 = st.columns([3, 2])
    with left2:
        st.subheader("What drives the re-forecast")
        st.altair_chart(_shap_chart(comparison), use_container_width=True)
    with right2:
        st.subheader("Risk detector at rider assignment")
        lm = risk["learned_model"]
        st.metric("PR-AUC", f"{risk['pr_auc']:.2f}", help=f"prevalence {risk['prevalence']:.1%}")
        st.metric("Precision / recall", f"{lm['precision']:.0%} / {lm['recall']:.0%}")
        st.caption(
            f"Flags {lm['flagged_share']:.0%} of orders at assignment, the earliest "
            "point the realised assignment delay is known."
        )

    if drift is not None:
        st.divider()
        st.subheader("📉 Model monitoring — data drift")
        st.caption(
            "A reference period from the normal regime versus a current period from a "
            "wetter, more congested 'rainy season'. Evidently flags the input shift; the "
            "production model, unaware of it, degrades."
        )
        overall = drift["drift"]["overall"]
        ref_perf, cur_perf = drift["performance"]["reference"], drift["performance"]["current"]
        d1, d2, d3 = st.columns(3)
        d1.metric("Drifted columns", f"{overall['drifted_columns']} ({overall['drifted_share']:.0%})")
        d2.metric(
            "Placement MAE", f"{cur_perf['mae']:.1f} min",
            f"+{cur_perf['mae'] - ref_perf['mae']:.1f}", delta_color="inverse",
        )
        d3.metric(
            "Silent-overrun rate", f"{cur_perf['silent_overrun_rate']:.1%}",
            f"+{cur_perf['silent_overrun_rate'] - ref_perf['silent_overrun_rate']:.1%}",
            delta_color="inverse",
        )
        st.caption(
            "Full Evidently report saved to `reports/drift_report.html` "
            "(regenerate with `python -m src.monitoring.drift`)."
        )

    st.divider()
    st.subheader("🔔 Live feed — orders flagged for proactive notification")
    if examples is None or examples.empty:
        st.info("No flagged-order examples available.")
        return

    show_n = st.slider("Orders to show", 1, len(examples), min(6, len(examples)))
    feed = examples.head(show_n)
    for _, row in feed.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 1, 1, 3])
            cols[0].metric("Placed ETA", f"{row['placement_eta_min']:.0f} min")
            cols[1].metric("New ETA", f"{row['assignment_eta_min']:.0f} min")
            cols[2].metric("Risk", f"{row['risk_score']:.2f}")
            cols[3].markdown(f"**Suggested message**\n\n{row['suggested_message']}")


if __name__ == "__main__":
    main()
