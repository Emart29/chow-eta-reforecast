"""Order-lifecycle and delay dynamics.

Takes the placement-time orders produced by :mod:`src.simulate.orders` and
advances each order through its full delivery lifecycle:

    order placed -> rider assigned -> pickup confirmed -> en route -> delivered

Each stage introduces realistic delay, drawn from distributions calibrated to
reproduce the failure modes seen in real last-mile delivery:

  * Preparation overruns concentrated on busy, high-volume restaurants.
  * A heavy-tailed rider-assignment delay that worsens during peaks, in the
    rain, and where rider supply is thin.
  * A subset of orders that suffer compounding delay in both preparation and
    assignment, producing the long tail of badly late deliveries.

The module emits two artefacts:

  * an enriched order table (one row per order) carrying every timing component
    and the final delivery time, and
  * an event stream (one row per order and lifecycle checkpoint) recording when
    each checkpoint occurs and how much delivery time remains from it. The event
    stream is the substrate for per-checkpoint re-forecasting.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config
from src.simulate.orders import generate_orders


# --------------------------------------------------------------------------- #
# Travel model
# --------------------------------------------------------------------------- #
def _travel_minutes(distance_km: np.ndarray, congestion: np.ndarray, rain: np.ndarray) -> np.ndarray:
    """Convert a distance to a travel time under zone congestion and rain.

    Effective speed is the free-flowing base speed divided by a congestion
    multiplier, with rain adding further slowdown.
    """
    effective_congestion = congestion + np.where(rain, config.DELAY.rain_congestion_extra, 0.0)
    speed = config.DELAY.base_speed_kmpm / effective_congestion
    return distance_km / speed


# --------------------------------------------------------------------------- #
# Lifecycle simulation
# --------------------------------------------------------------------------- #
def simulate_lifecycle(
    orders: pd.DataFrame, seed: int | None = None, congestion_scale: float = 1.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Advance every order through its lifecycle and return (orders, events).

    ``orders`` is the enriched order table with all timing components and the
    final delivery time; ``events`` is the long-format checkpoint stream. The
    result is deterministic for a given ``(orders, seed)``. ``congestion_scale``
    multiplies zone congestion to simulate a heavier-traffic regime.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    n = len(orders)
    d = config.DELAY

    zone_lookup = {z.name: z for z in config.ZONES}
    congestion = orders["traffic_zone"].map(lambda z: zone_lookup[z].congestion).to_numpy() * congestion_scale
    density = orders["traffic_zone"].map(lambda z: zone_lookup[z].rider_density).to_numpy()

    is_peak = orders["is_peak"].to_numpy()
    rain = orders["weather_rain"].to_numpy()
    high_volume = orders["restaurant_high_volume"].to_numpy()
    prep_estimate = orders["restaurant_prep_estimate"].to_numpy()
    distance_km = orders["distance_km"].to_numpy()
    riders_nearby = orders["rider_available_count_nearby"].to_numpy()

    # Orders that suffer compounding delay in both preparation and assignment.
    compounding = rng.random(n) < d.compounding_share

    # --- Preparation --------------------------------------------------------
    # Actual prep time is the estimate scaled by a multiplicative log-normal
    # shock. Busy restaurants and compounding orders carry a wider spread, so
    # their overruns are both more frequent and more severe.
    prep_sigma = (
        d.prep_noise_sigma
        + np.where(high_volume, d.prep_overrun_highvol_extra, 0.0)
        + np.where(compounding, d.prep_overrun_highvol_extra, 0.0)
    )
    prep_multiplier = np.exp(rng.normal(0.0, prep_sigma))
    actual_prep_min = np.minimum(prep_estimate * prep_multiplier, d.max_prep_min)

    # --- Rider assignment ---------------------------------------------------
    # Assignment delay is log-normal: usually quick, occasionally very slow.
    # Peaks, rain, thin supply, and compounding orders all push the mean up.
    assign_mu = (
        d.assign_mu
        + np.where(is_peak, d.assign_peak_mu_bump, 0.0)
        + np.where(rain, d.assign_rain_mu_bump, 0.0)
        + (1.0 - density) * d.assign_low_density_bump
        + np.where(compounding, d.assign_peak_mu_bump, 0.0)
    )
    assignment_delay_min = np.minimum(
        np.exp(rng.normal(assign_mu, d.assign_sigma)), d.max_assignment_delay_min
    )

    # --- Rider travel to the restaurant -------------------------------------
    # The assigned rider's distance to the restaurant grows as nearby supply
    # thins out. Travel time follows the shared congestion-aware model, with a
    # per-trip traffic shock so travel carries irreducible uncertainty.
    scarcity = np.clip(1.0 - riders_nearby / 12.0, 0.0, 1.0)
    rider_to_restaurant_km = rng.exponential(0.8 + 2.5 * scarcity)
    to_restaurant_noise = np.exp(rng.normal(0.0, d.travel_noise_sigma, size=n))
    to_restaurant_min = _travel_minutes(rider_to_restaurant_km, congestion, rain) * to_restaurant_noise

    # --- Pickup -------------------------------------------------------------
    # The order can be collected only once both the food is ready and the rider
    # has arrived, plus a short dwell at the counter.
    food_ready_min = actual_prep_min
    rider_arrival_min = assignment_delay_min + to_restaurant_min
    pickup_dwell_min = np.exp(rng.normal(np.log(d.pickup_dwell_mean), d.pickup_dwell_sigma))
    pickup_confirmed_min = np.maximum(food_ready_min, rider_arrival_min) + pickup_dwell_min

    # --- Delivery -----------------------------------------------------------
    to_customer_noise = np.exp(rng.normal(0.0, d.travel_noise_sigma, size=n))
    to_customer_min = _travel_minutes(distance_km, congestion, rain) * to_customer_noise
    delivered_min = pickup_confirmed_min + to_customer_min
    enroute_midpoint_min = pickup_confirmed_min + to_customer_min / 2.0

    placed_at = orders["order_placed_at"].to_numpy()

    def _at(offset_min: np.ndarray) -> np.ndarray:
        return placed_at + pd.to_timedelta(offset_min, unit="m").to_numpy()

    enriched = orders.copy()
    enriched["compounding_delay"] = compounding
    enriched["actual_prep_min"] = np.round(actual_prep_min, 2)
    enriched["assignment_delay_min"] = np.round(assignment_delay_min, 2)
    enriched["to_restaurant_min"] = np.round(to_restaurant_min, 2)
    enriched["pickup_dwell_min"] = np.round(pickup_dwell_min, 2)
    enriched["to_customer_min"] = np.round(to_customer_min, 2)
    enriched["rider_assigned_offset_min"] = np.round(assignment_delay_min, 2)
    enriched["pickup_confirmed_offset_min"] = np.round(pickup_confirmed_min, 2)
    enriched["enroute_midpoint_offset_min"] = np.round(enroute_midpoint_min, 2)
    enriched["actual_delivery_min"] = np.round(delivered_min, 2)
    enriched["rider_assigned_at"] = _at(assignment_delay_min)
    enriched["pickup_confirmed_at"] = _at(pickup_confirmed_min)
    enroute_at = _at(enroute_midpoint_min)
    enriched["enroute_midpoint_at"] = enroute_at
    enriched["delivered_at"] = _at(delivered_min)

    events = _build_event_stream(enriched)
    return enriched, events


def _build_event_stream(enriched: pd.DataFrame) -> pd.DataFrame:
    """Reshape the enriched order table into one row per checkpoint.

    Each row records when a checkpoint occurred and how much delivery time
    remains from it, giving each checkpoint a re-forecasting target.
    """
    offsets = {
        "order_placed": np.zeros(len(enriched)),
        "rider_assigned": enriched["rider_assigned_offset_min"].to_numpy(),
        "pickup_confirmed": enriched["pickup_confirmed_offset_min"].to_numpy(),
        "enroute_midpoint": enriched["enroute_midpoint_offset_min"].to_numpy(),
    }
    total = enriched["actual_delivery_min"].to_numpy()
    placed_at = enriched["order_placed_at"].to_numpy()

    frames = []
    for index, checkpoint in enumerate(config.CHECKPOINTS):
        offset = offsets[checkpoint]
        frames.append(
            pd.DataFrame(
                {
                    "order_id": enriched["order_id"].to_numpy(),
                    "checkpoint": checkpoint,
                    "checkpoint_index": index,
                    "event_offset_min": np.round(offset, 2),
                    "event_time": placed_at + pd.to_timedelta(offset, unit="m").to_numpy(),
                    "remaining_min": np.round(total - offset, 2),
                    "total_delivery_min": np.round(total, 2),
                }
            )
        )
    events = pd.concat(frames, ignore_index=True)
    return events.sort_values(["order_id", "checkpoint_index"], ignore_index=True)


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def _summarise(enriched: pd.DataFrame) -> str:
    prep_overrun = (enriched["actual_prep_min"] > enriched["restaurant_prep_estimate"]).mean()
    assign = enriched["assignment_delay_min"]
    delivery = enriched["actual_delivery_min"]
    lines = [
        f"orders:                    {len(enriched):,}",
        f"compounding-delay share:   {enriched['compounding_delay'].mean():.1%}",
        f"prep overrun vs estimate:  {prep_overrun:.1%}",
        (
            f"assignment delay  P50/P90/P99: {assign.quantile(0.5):.1f} / "
            f"{assign.quantile(0.9):.1f} / {assign.quantile(0.99):.1f} min"
        ),
        (
            f"delivery time     P50/P90/P99: {delivery.quantile(0.5):.1f} / "
            f"{delivery.quantile(0.9):.1f} / {delivery.quantile(0.99):.1f} min"
        ),
        f"deliveries over 60 min:    {(delivery > 60).mean():.1%}",
        f"deliveries over 90 min:    {(delivery > 90).mean():.1%}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the order lifecycle and delay dynamics.")
    parser.add_argument("-n", "--n-orders", type=int, default=20_000, help="number of orders to generate")
    parser.add_argument("--days", type=int, default=30, help="span of the simulated timeline in days")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed")
    args = parser.parse_args()

    config.ensure_dirs()
    orders = generate_orders(args.n_orders, days=args.days, seed=args.seed)
    enriched, events = simulate_lifecycle(orders, seed=args.seed)

    enriched.to_parquet(config.ORDERS_PATH, index=False)
    events.to_parquet(config.EVENTS_PATH, index=False)

    print(_summarise(enriched))
    print(f"\nwrote {len(enriched):,} orders  -> {config.ORDERS_PATH}")
    print(f"wrote {len(events):,} events  -> {config.EVENTS_PATH}")


if __name__ == "__main__":
    main()
