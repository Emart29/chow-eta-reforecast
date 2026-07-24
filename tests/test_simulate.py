"""Tests for the synthetic simulator: determinism and lifecycle invariants."""
from __future__ import annotations

from src import config
from src.simulate.lifecycle import simulate_lifecycle
from src.simulate.orders import generate_orders


def test_orders_are_deterministic():
    a = generate_orders(1500, seed=1)
    b = generate_orders(1500, seed=1)
    assert a.equals(b)


def test_lifecycle_is_deterministic():
    orders = generate_orders(1500, seed=2)
    e1, ev1 = simulate_lifecycle(orders, seed=2)
    e2, ev2 = simulate_lifecycle(orders, seed=2)
    assert e1.equals(e2)
    assert ev1.equals(ev2)


def test_distances_non_negative():
    orders = generate_orders(1500, seed=3)
    assert (orders["distance_km"] >= 0).all()


def test_restaurant_population_stable_across_seeds():
    # Restaurant identity must not depend on the order seed.
    a = generate_orders(1000, seed=10).drop_duplicates("restaurant_id").set_index("restaurant_id")
    b = generate_orders(1000, seed=20).drop_duplicates("restaurant_id").set_index("restaurant_id")
    shared = a.index.intersection(b.index)
    assert len(shared) > 0
    assert (a.loc[shared, "restaurant_zone"] == b.loc[shared, "restaurant_zone"]).all()


def test_delay_caps_respected():
    orders = generate_orders(4000, seed=4, rain_prob=0.6)
    enriched, _ = simulate_lifecycle(orders, seed=4, congestion_scale=1.5)
    assert enriched["assignment_delay_min"].max() <= config.DELAY.max_assignment_delay_min + 1e-6
    assert enriched["actual_prep_min"].max() <= config.DELAY.max_prep_min + 1e-6


def test_event_stream_structure():
    orders = generate_orders(1000, seed=5)
    _, events = simulate_lifecycle(orders, seed=5)

    # Exactly one row per order per checkpoint.
    assert (events.groupby("order_id").size() == len(config.CHECKPOINTS)).all()

    ordered = events.sort_values(["order_id", "checkpoint_index"])
    grouped = ordered.groupby("order_id")
    assert grouped["event_offset_min"].apply(lambda s: s.is_monotonic_increasing).all()
    assert grouped["remaining_min"].apply(lambda s: s.is_monotonic_decreasing).all()
    assert (events["remaining_min"] >= -1e-6).all()
