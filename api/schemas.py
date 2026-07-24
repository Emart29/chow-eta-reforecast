"""Request and response models for the serving API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator

from src import config
from src.serving.service import OrderContext

_REFORECAST_CHECKPOINTS = tuple(c for c in config.CHECKPOINTS if c != config.CHECKPOINTS[0])


class OrderIn(BaseModel):
    """Placement-time attributes of an order."""

    order_id: str
    restaurant_id: str
    traffic_zone: str = Field(description=f"one of {list(config.ZONE_NAMES)}")
    distance_km: float = Field(gt=0)
    restaurant_prep_estimate: float = Field(gt=0, description="quoted preparation time in minutes")
    rider_available_count_nearby: int = Field(ge=0)
    restaurant_high_volume: bool = False
    weather_rain: bool = False
    order_placed_at: datetime

    @field_validator("traffic_zone")
    @classmethod
    def _known_zone(cls, value: str) -> str:
        if value not in config.ZONE_NAMES:
            raise ValueError(f"unknown zone {value!r}; expected one of {list(config.ZONE_NAMES)}")
        return value

    def to_context(self) -> OrderContext:
        return OrderContext(
            order_id=self.order_id,
            restaurant_id=self.restaurant_id,
            traffic_zone=self.traffic_zone,
            distance_km=self.distance_km,
            restaurant_prep_estimate=self.restaurant_prep_estimate,
            rider_available_count_nearby=self.rider_available_count_nearby,
            restaurant_high_volume=self.restaurant_high_volume,
            weather_rain=self.weather_rain,
            order_placed_at=self.order_placed_at,
        )


class ReforecastIn(OrderIn):
    """An order plus the signals observed at a lifecycle checkpoint."""

    checkpoint: str
    elapsed_min: float = Field(ge=0, description="minutes since the order was placed")
    assignment_delay_min: float | None = Field(default=None, ge=0)
    actual_prep_min: float | None = Field(default=None, ge=0)
    to_restaurant_min: float | None = Field(default=None, ge=0)

    @field_validator("checkpoint")
    @classmethod
    def _valid_checkpoint(cls, value: str) -> str:
        if value not in _REFORECAST_CHECKPOINTS:
            raise ValueError(
                f"checkpoint must be one of {list(_REFORECAST_CHECKPOINTS)}"
            )
        return value

    def observed(self) -> dict:
        fields = ("assignment_delay_min", "actual_prep_min", "to_restaurant_min")
        return {f: getattr(self, f) for f in fields if getattr(self, f) is not None}


class EtaResponse(BaseModel):
    order_id: str
    checkpoint: str
    eta_minutes: float
    predicted_delivery_at: datetime


class RiskResponse(BaseModel):
    order_id: str
    checkpoint: str
    risk_score: float
    flagged: bool
    placement_eta_minutes: float
    current_eta_minutes: float
    suggested_message: str | None


class ReforecastResponse(EtaResponse):
    risk: RiskResponse | None = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    checkpoints: list[str]
