"""Synthetic order generation.

Produces a realistic population of restaurants and a stream of orders placed
across a simulated timeline. This module generates only the fields known at
order-placement time: geography, timing, environment, and local rider supply.
The downstream order lifecycle and delay dynamics (preparation, rider
assignment, travel) are modelled separately in ``src.simulate.lifecycle``.

Design goals:
  * Deterministic under ``config.SEED`` so a given configuration always yields
    the same dataset.
  * Domain-credible: real Lagos zones, meal-time demand peaks, rain, and rider
    scarcity that tightens during busy periods.
  * Free of target leakage: every field here is observable at placement time,
    so the placement-time model can consume these columns directly.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src import config


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
_EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Return the great-circle distance in km between two arrays of points (degrees)."""
    lat1r, lon1r, lat2r, lon2r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def _scatter_around(
    rng: np.random.Generator, lat: float, lon: float, radius_km: float, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``size`` points area-uniformly within ``radius_km`` of a centre point."""
    # sqrt of a uniform draw gives radial density proportional to area.
    r = radius_km * np.sqrt(rng.random(size))
    theta = rng.random(size) * 2.0 * math.pi
    dlat = (r * np.cos(theta)) / 111.0  # ~111 km per degree of latitude
    dlon = (r * np.sin(theta)) / (111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon


# --------------------------------------------------------------------------- #
# Restaurant population
# --------------------------------------------------------------------------- #
def build_restaurants(rng: np.random.Generator) -> pd.DataFrame:
    """Create a fixed population of restaurants distributed across the zones.

    A ``high_volume`` subset is flagged for the busiest, most popular vendors.
    These are the establishments most prone to preparation overruns once the
    lifecycle dynamics are applied downstream.
    """
    n = config.N_RESTAURANTS
    zones = config.ZONES

    # Assign each restaurant to a zone, weighting slightly toward denser zones.
    zone_weights = np.array([z.rider_density for z in zones], dtype=float)
    zone_weights /= zone_weights.sum()
    zone_idx = rng.choice(len(zones), size=n, p=zone_weights)

    lat = np.empty(n)
    lon = np.empty(n)
    for i, z in enumerate(zones):
        mask = zone_idx == i
        k = int(mask.sum())
        if k:
            lat[mask], lon[mask] = _scatter_around(rng, z.lat, z.lon, radius_km=3.0, size=k)

    # Each restaurant's typical quoted preparation time.
    base_prep = rng.uniform(config.DELAY.prep_estimate_min, config.DELAY.prep_estimate_max, size=n)

    # Popularity drives order volume; the top share is flagged high-volume.
    popularity = rng.random(n)
    high_vol_cut = np.quantile(popularity, 1.0 - config.HIGH_VOLUME_SHARE)
    high_volume = popularity >= high_vol_cut

    return pd.DataFrame(
        {
            "restaurant_id": [f"R{i:04d}" for i in range(n)],
            "restaurant_zone": [zones[i].name for i in zone_idx],
            "restaurant_lat": lat,
            "restaurant_lon": lon,
            "base_prep_estimate": base_prep,
            "popularity": popularity,
            "high_volume": high_volume,
        }
    )


# --------------------------------------------------------------------------- #
# Temporal and environment helpers
# --------------------------------------------------------------------------- #
def _is_peak(hour: np.ndarray) -> np.ndarray:
    """Return a mask marking hours that fall inside any configured peak window."""
    out = np.zeros_like(hour, dtype=bool)
    for start, end in config.PEAK_HOURS:
        out |= (hour >= start) & (hour < end)
    return out


def _demand_weight(hour: int, is_weekend: bool) -> float:
    """Return the relative order-arrival propensity for a given hour.

    Two Gaussian bumps model the lunch and dinner peaks over a low overnight
    floor; weekends carry slightly heavier demand.
    """
    lunch = math.exp(-((hour - 13) ** 2) / (2 * 1.3**2))
    dinner = math.exp(-((hour - 19.5) ** 2) / (2 * 1.6**2))
    base = 0.05 + lunch + 1.25 * dinner
    if is_weekend:
        base *= 1.15
    return base


def _sample_placed_at(rng: np.random.Generator, n: int, days: int) -> pd.Series:
    """Sample order-placement timestamps across ``days``, clustered at meal peaks."""
    start = pd.Timestamp("2026-05-01 00:00:00")
    total_hours = days * 24
    hours_index = pd.date_range(start, periods=total_hours, freq="h")
    weights = np.array(
        [_demand_weight(ts.hour, ts.dayofweek in config.WEEKEND_DAYS) for ts in hours_index]
    )
    weights /= weights.sum()
    chosen = rng.choice(total_hours, size=n, p=weights)
    # Spread each order uniformly within its chosen hour.
    minute_offset = rng.integers(0, 60, size=n)
    second_offset = rng.integers(0, 60, size=n)
    placed = (
        hours_index[chosen]
        + pd.to_timedelta(minute_offset, unit="m")
        + pd.to_timedelta(second_offset, unit="s")
    )
    return pd.Series(placed).sort_values(ignore_index=True)


# --------------------------------------------------------------------------- #
# Order generation
# --------------------------------------------------------------------------- #
def generate_orders(
    n_orders: int, days: int = 30, seed: int | None = None, rain_prob: float | None = None
) -> pd.DataFrame:
    """Generate ``n_orders`` orders with every field known at placement time.

    Returns a flat DataFrame. The result is deterministic for a given
    ``(n_orders, days, seed)`` combination. ``rain_prob`` overrides the baseline
    rain rate, which is used to simulate a wetter environmental regime.
    """
    rng = np.random.default_rng(config.SEED if seed is None else seed)
    rain_prob = config.RAIN_PROB if rain_prob is None else rain_prob
    # Restaurants come from their own fixed seed so their identity and history are
    # stable across order samples; only the orders themselves vary with ``seed``.
    restaurants = build_restaurants(np.random.default_rng(config.RESTAURANT_SEED))

    # Choose a restaurant per order, weighted by popularity.
    rest_p = restaurants["popularity"].to_numpy()
    rest_p = rest_p / rest_p.sum()
    r_idx = rng.choice(len(restaurants), size=n_orders, p=rest_p)
    chosen = restaurants.iloc[r_idx].reset_index(drop=True)

    placed_at = _sample_placed_at(rng, n_orders, days)
    hour = placed_at.dt.hour.to_numpy()
    dow = placed_at.dt.dayofweek.to_numpy()
    is_weekend = np.isin(dow, config.WEEKEND_DAYS)
    is_peak = _is_peak(hour)

    # Place each customer near the restaurant's zone, within a few kilometres.
    zone_lookup = {z.name: z for z in config.ZONES}
    cust_lat = np.empty(n_orders)
    cust_lon = np.empty(n_orders)
    for zname in chosen["restaurant_zone"].unique():
        mask = (chosen["restaurant_zone"] == zname).to_numpy()
        k = int(mask.sum())
        z = zone_lookup[zname]
        cust_lat[mask], cust_lon[mask] = _scatter_around(rng, z.lat, z.lon, radius_km=5.0, size=k)

    distance_km = haversine_km(
        chosen["restaurant_lat"].to_numpy(),
        chosen["restaurant_lon"].to_numpy(),
        cust_lat,
        cust_lon,
    )

    # An independent rain draw per order keeps this field observable at placement.
    weather_rain = rng.random(n_orders) < rain_prob

    # Nearby rider supply falls from a zone baseline during peaks and in the rain.
    zone_density = chosen["restaurant_zone"].map(lambda z: zone_lookup[z].rider_density).to_numpy()
    supply = 12.0 * zone_density
    supply *= np.where(is_peak, 0.55, 1.0)
    supply *= np.where(weather_rain, 0.75, 1.0)
    rider_available_count_nearby = np.maximum(0, rng.poisson(np.clip(supply, 0.5, None))).astype(int)

    return pd.DataFrame(
        {
            "order_id": [f"O{i:07d}" for i in range(n_orders)],
            "restaurant_id": chosen["restaurant_id"].to_numpy(),
            "restaurant_zone": chosen["restaurant_zone"].to_numpy(),
            "traffic_zone": chosen["restaurant_zone"].to_numpy(),
            "restaurant_lat": chosen["restaurant_lat"].to_numpy(),
            "restaurant_lon": chosen["restaurant_lon"].to_numpy(),
            "customer_lat": cust_lat,
            "customer_lon": cust_lon,
            "distance_km": np.round(distance_km, 3),
            "order_placed_at": placed_at.to_numpy(),
            "hour_of_day": hour,
            "day_of_week": dow,
            "is_weekend": is_weekend,
            "is_peak": is_peak,
            "weather_rain": weather_rain,
            "restaurant_high_volume": chosen["high_volume"].to_numpy(),
            "restaurant_prep_estimate": np.round(chosen["base_prep_estimate"].to_numpy(), 2),
            "rider_available_count_nearby": rider_available_count_nearby,
        }
    )


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def _summarise(orders: pd.DataFrame) -> str:
    lines = [
        f"orders:               {len(orders):,}",
        f"date range:           {orders['order_placed_at'].min()}  ->  {orders['order_placed_at'].max()}",
        f"unique restaurants:   {orders['restaurant_id'].nunique()}",
        f"peak-hour share:      {orders['is_peak'].mean():.1%}",
        f"rain share:           {orders['weather_rain'].mean():.1%}",
        f"high-volume share:    {orders['restaurant_high_volume'].mean():.1%} of orders",
        f"median distance:      {orders['distance_km'].median():.2f} km",
        f"median riders nearby: {orders['rider_available_count_nearby'].median():.0f}",
        "orders by zone:",
    ]
    for zone, cnt in orders["restaurant_zone"].value_counts().items():
        lines.append(f"    {zone:<10} {cnt:,}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic delivery orders.")
    parser.add_argument("-n", "--n-orders", type=int, default=20_000, help="number of orders to generate")
    parser.add_argument("--days", type=int, default=30, help="span of the simulated timeline in days")
    parser.add_argument("--seed", type=int, default=config.SEED, help="random seed")
    args = parser.parse_args()

    config.ensure_dirs()
    orders = generate_orders(args.n_orders, days=args.days, seed=args.seed)
    orders.to_parquet(config.ORDERS_PATH, index=False)

    print(_summarise(orders))
    print(f"\nwrote {len(orders):,} orders -> {config.ORDERS_PATH}")


if __name__ == "__main__":
    main()
