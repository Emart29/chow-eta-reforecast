"""Shared constants for the Chowdeck ETA re-forecasting demo.

Everything that governs the synthetic world lives here so the simulation is
reproducible and the modeling code has a single source of truth. All times are
in minutes unless otherwise noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

ORDERS_PATH = DATA_DIR / "orders.parquet"
EVENTS_PATH = DATA_DIR / "events.parquet"

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
# The restaurant population is a stable entity, generated from its own fixed seed
# so that a restaurant's identity and history are consistent across order samples
# drawn with different seeds (training, reference, and drift regimes alike).
RESTAURANT_SEED = 7

# --------------------------------------------------------------------------- #
# Lagos geography — real neighbourhoods for credibility.
# Approx (lat, lon) centroids; "congestion" is a baseline traffic multiplier
# applied to en-route travel time (1.0 = free-flowing).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Zone:
    name: str
    lat: float
    lon: float
    congestion: float  # baseline travel-time multiplier
    rider_density: float  # relative rider supply (higher = easier to assign)


ZONES: tuple[Zone, ...] = (
    Zone("Yaba", 6.5095, 3.3711, 1.25, 1.00),
    Zone("Surulere", 6.5006, 3.3587, 1.20, 0.90),
    Zone("Ikeja", 6.6018, 3.3515, 1.30, 1.10),
    Zone("Lekki", 6.4698, 3.5852, 1.45, 0.70),
    Zone("VI", 6.4281, 3.4219, 1.35, 0.80),  # Victoria Island
)
ZONE_NAMES: tuple[str, ...] = tuple(z.name for z in ZONES)

# --------------------------------------------------------------------------- #
# Temporal structure
# --------------------------------------------------------------------------- #
# Peak windows (24h clock). Demand and rider scarcity both spike here.
PEAK_HOURS: tuple[tuple[int, int], ...] = ((12, 14), (18, 21))  # lunch, dinner
WEEKEND_DAYS: tuple[int, ...] = (5, 6)  # Sat, Sun (Mon=0)

# --------------------------------------------------------------------------- #
# Delay-generating distributions (the heart of the simulation).
# Values chosen to reproduce the reviewed failure modes:
#   - most rider assignments are fast, but a heavy tail blows up (log-normal)
#   - prep overruns concentrate on high-volume restaurants
#   - peak hours + rain compound both
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DelayParams:
    # Restaurant prep (minutes)
    prep_estimate_min: float = 8.0
    prep_estimate_max: float = 30.0
    prep_noise_sigma: float = 0.28  # multiplicative log-normal sigma on actual vs estimate
    prep_overrun_highvol_extra: float = 0.35  # extra sigma for high-volume restaurants
    max_prep_min: float = 75.0  # ceiling; beyond this an order would realistically be escalated

    # Rider assignment delay (minutes) — log-normal(mu, sigma)
    assign_mu: float = 1.1  # ~exp(1.1) ≈ 3 min median off-peak
    assign_sigma: float = 0.75
    assign_peak_mu_bump: float = 0.85  # added to mu during peak windows
    assign_rain_mu_bump: float = 0.45
    assign_low_density_bump: float = 0.6  # scaled by (1 - rider_density)
    max_assignment_delay_min: float = 90.0  # ceiling; beyond this an order would be reassigned

    # Travel speed (km per minute) before congestion multiplier
    base_speed_kmpm: float = 0.5  # ~30 km/h free-flowing
    rain_congestion_extra: float = 0.20  # added to zone congestion when raining
    travel_noise_sigma: float = 0.22  # irreducible per-trip traffic variance (log-normal)

    # Pickup dwell (rider waits at restaurant), minutes
    pickup_dwell_mean: float = 3.5
    pickup_dwell_sigma: float = 0.4

    # Compounding failure: share of orders that get BOTH prep and rider tails
    compounding_share: float = 0.18


DELAY = DelayParams()

# --------------------------------------------------------------------------- #
# Weather / environment
# --------------------------------------------------------------------------- #
RAIN_PROB: float = 0.18  # baseline probability an order is placed during rain

# --------------------------------------------------------------------------- #
# Restaurant population
# --------------------------------------------------------------------------- #
N_RESTAURANTS: int = 120
HIGH_VOLUME_SHARE: float = 0.20  # top ~20% of restaurants are prep-overrun-prone

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# An order is a "silent overrun" when actual delivery exceeds the last shown ETA
# by more than this many minutes without a re-forecast having been issued.
SILENT_OVERRUN_THRESHOLD_MIN: float = 10.0

# Risk detector: re-forecast growth (minutes) since last shown ETA that trips a flag.
RISK_ETA_GROWTH_MIN: float = 8.0

# Lifecycle checkpoints, in order. Each is a re-forecast opportunity.
CHECKPOINTS: tuple[str, ...] = (
    "order_placed",
    "rider_assigned",
    "pickup_confirmed",
    "enroute_midpoint",
)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist."""
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
