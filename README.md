# chow-eta-reforecast

**Dynamic ETA re-forecasting engine for last-mile delivery** — a synthetic-data proof of concept.

> ⚠️ This project is built entirely on **synthetic data**. It is a proof of concept exploring an
> approach to a problem, not a claim about any real company's production system.

## The problem

Most delivery ETAs are predicted **once**, at order placement, and never updated. When real delays
accumulate — a late rider assignment, a restaurant prep overrun, traffic — the customer-facing ETA
goes stale silently. Customers tend to tolerate delays far better than they tolerate *silent,
unexplained* ones. That makes stale ETAs a trust problem, not just a latency problem.

## The approach

Treat ETA as a **live, event-driven quantity**:

1. **Re-forecast** the ETA at every checkpoint in the order lifecycle
   (order placed → rider assigned → pickup confirmed → en-route midpoint), each prediction
   conditioned only on information available at that moment.
2. Add a **silent-overrun risk detector** that flags orders likely to blow past their current ETA,
   so the system can proactively notify the customer *before* they notice.

## Architecture

```text
Synthetic Data Engine
        │
Event Stream Simulator      (order lifecycle checkpoints w/ injected delays)
        │
Feature Engineering         (per-checkpoint, leakage-safe)
        │
ETA Model v1 (static)  vs  ETA Model v2 (dynamic re-forecast)
        │
Risk / Silent-Overrun Detector
        │
FastAPI Serving Layer       (/predict_eta, /reforecast, /risk_check)
        │
Monitoring Dashboard        (Streamlit: static vs dynamic error, drift, risk flags)
```

## Status

Scaffolding in place. Build proceeds phase by phase — see the build plan in
[`docs/chowdeck.md`](../docs/chowdeck.md) (Part B).

| Phase | Component | Status |
| --- | --- | --- |
| 1 | Synthetic data + lifecycle simulation | ⬜ |
| 2 | Static ETA baseline | ⬜ |
| 3 | Dynamic re-forecasting model | ⬜ |
| 4 | Silent-overrun risk detector | ⬜ |
| 5 | FastAPI serving layer | ⬜ |
| 6 | Monitoring dashboard + drift | ⬜ |
| 7 | Write-up | ⬜ |

## Quickstart

```bash
pip install -e ".[dev]"        # or: pip install -r requirements.txt
python -m src.simulate.orders --n-orders 10000
```

## Tech stack

Python · pandas · numpy · XGBoost/LightGBM · scikit-learn · SHAP · FastAPI · Evidently AI ·
Streamlit · Docker.

## License

MIT — see [LICENSE](LICENSE).
