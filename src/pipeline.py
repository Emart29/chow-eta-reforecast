"""End-to-end pipeline runner.

Runs every stage in order, from simulating the data to producing the trained
serving bundle and all evaluation artefacts:

    generate data -> baseline -> dynamic -> comparison -> risk -> bundle -> drift

Each stage writes its own outputs under ``data/``, ``models/``, and ``reports/``.
Run the whole thing with::

    python -m src.pipeline --n-orders 20000
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Callable

from src import config
from src.models import baseline, compare, dynamic
from src.monitoring import drift
from src.risk import detector
from src.serving import bundle
from src.simulate.lifecycle import simulate_lifecycle
from src.simulate.orders import generate_orders


def _generate_data(n_orders: int) -> None:
    orders = generate_orders(n_orders)
    enriched, events = simulate_lifecycle(orders)
    enriched.to_parquet(config.ORDERS_PATH, index=False)
    events.to_parquet(config.EVENTS_PATH, index=False)
    print(f"generated {len(enriched):,} orders and {len(events):,} events")


def _step(name: str, action: Callable[[], None]) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    start = time.perf_counter()
    action()
    print(f"[{name}] done in {time.perf_counter() - start:.1f}s")


def run(n_orders: int = 20_000) -> None:
    """Run the full pipeline, producing all data, models, and reports."""
    config.ensure_dirs()
    _step("1/7 simulate data", lambda: _generate_data(n_orders))
    _step("2/7 static baseline", baseline.main)
    _step("3/7 dynamic re-forecaster", dynamic.main)
    _step("4/7 comparison and explainability", compare.main)
    _step("5/7 risk detector", detector.main)
    _step("6/7 serving bundle", bundle.build_and_save)
    _step("7/7 drift monitoring", drift.main)
    print("\nPipeline complete. Artefacts written to data/, models/, and reports/.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full ETA re-forecasting pipeline.")
    parser.add_argument("-n", "--n-orders", type=int, default=20_000, help="number of orders to simulate")
    args = parser.parse_args()
    run(args.n_orders)


if __name__ == "__main__":
    main()
