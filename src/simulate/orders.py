"""Synthetic order generator (scaffold).

Fleshed out in Prompt 2 (base fields) and Prompt 3 (lifecycle + delay injection).
For now this only wires up the CLI entrypoint declared in pyproject.toml so the
package is runnable end-to-end from the very first commit.
"""
from __future__ import annotations

import argparse

from src import config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Chowdeck-style delivery orders.")
    parser.add_argument("-n", "--n-orders", type=int, default=10_000, help="number of orders to generate")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed")
    args = parser.parse_args()

    config.ensure_dirs()
    print(
        f"[scaffold] would generate {args.n_orders} orders (seed={args.seed}) "
        f"into {config.ORDERS_PATH}. Implemented in Prompt 2."
    )


if __name__ == "__main__":
    main()
