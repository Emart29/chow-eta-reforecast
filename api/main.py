"""FastAPI serving layer for the ETA re-forecasting models.

Exposes the order lifecycle as three endpoints: an initial ETA at placement, a
revised ETA at each later checkpoint, and the current silent-overrun risk for an
order. Requests are validated with Pydantic, every response carries its handling
latency, and predictions share an in-memory order store so a risk check reflects
the most recent re-forecast.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from src import config
from src.serving.service import ModelService, OrderStore
from api.schemas import (
    EtaResponse,
    HealthResponse,
    OrderIn,
    ReforecastIn,
    ReforecastResponse,
    RiskResponse,
)

logger = logging.getLogger("chow.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the serving bundle once at startup and share it across requests."""
    state["service"] = ModelService()
    state["store"] = OrderStore()
    logger.info("serving bundle loaded; API ready")
    yield
    state.clear()


app = FastAPI(
    title="Chow ETA Re-forecasting API",
    description="Dynamic ETA re-forecasting and silent-overrun risk for last-mile delivery.",
    version="0.1.0",
    lifespan=lifespan,
)


def _service() -> ModelService:
    return state["service"]  # type: ignore[return-value]


def _store() -> OrderStore:
    return state["store"]  # type: ignore[return-value]


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %s (%.2f ms)", request.method, request.url.path, response.status_code, elapsed_ms)
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        models_loaded="service" in state,
        checkpoints=list(config.CHECKPOINTS),
    )


@app.post("/predict_eta", response_model=EtaResponse)
def predict_eta(order: OrderIn) -> EtaResponse:
    """Predict and record the ETA shown when the order is placed."""
    ctx = order.to_context()
    eta = _service().predict_eta(ctx)
    _store().upsert_placement(ctx.order_id, eta)
    return EtaResponse(
        order_id=ctx.order_id,
        checkpoint=config.CHECKPOINTS[0],
        eta_minutes=round(eta, 2),
        predicted_delivery_at=ModelService.delivery_time(ctx, eta),
    )


@app.post("/reforecast", response_model=ReforecastResponse)
def reforecast(update: ReforecastIn) -> ReforecastResponse:
    """Revise the ETA at a checkpoint, assessing risk at rider assignment."""
    ctx = update.to_context()
    service, store = _service(), _store()

    eta = service.reforecast(ctx, update.checkpoint, update.elapsed_min, update.observed())

    risk_response = None
    risk_assessment = None
    if update.checkpoint == "rider_assigned":
        record = store.get(ctx.order_id)
        placement_eta = record.placement_eta_minutes if record else eta
        risk_assessment = service.assess_risk(
            ctx, update.elapsed_min, update.observed(), placement_eta
        )
        risk_response = RiskResponse(**risk_assessment.__dict__)

    store.update_checkpoint(ctx.order_id, update.checkpoint, eta, risk_assessment)
    return ReforecastResponse(
        order_id=ctx.order_id,
        checkpoint=update.checkpoint,
        eta_minutes=round(eta, 2),
        predicted_delivery_at=ModelService.delivery_time(ctx, eta),
        risk=risk_response,
    )


@app.get("/risk_check/{order_id}", response_model=RiskResponse)
def risk_check(order_id: str) -> RiskResponse:
    """Return the most recent silent-overrun risk assessment for an order."""
    record = _store().get(order_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"unknown order {order_id!r}")
    if record.risk is None:
        raise HTTPException(
            status_code=409,
            detail=f"order {order_id!r} has no risk assessment yet; reforecast at rider assignment first",
        )
    return RiskResponse(**record.risk.__dict__)
