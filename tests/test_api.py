"""Tests for the FastAPI serving endpoints."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import src.serving.service as service_module
from api.main import app


@pytest.fixture
def client(serving_bundle, monkeypatch):
    # The app loads its bundle at startup; point that at the in-memory test bundle.
    monkeypatch.setattr(service_module, "load_bundle", lambda: serving_bundle)
    with TestClient(app) as test_client:
        yield test_client


def _order(order_id: str = "O-TEST-1") -> dict:
    return {
        "order_id": order_id,
        "restaurant_id": "R0007",
        "traffic_zone": "Lekki",
        "distance_km": 6.2,
        "restaurant_prep_estimate": 18.0,
        "rider_available_count_nearby": 2,
        "restaurant_high_volume": True,
        "weather_rain": True,
        "order_placed_at": "2026-05-15T19:30:00",
    }


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["models_loaded"] is True


def test_predict_eta(client):
    response = client.post("/predict_eta", json=_order())
    assert response.status_code == 200
    body = response.json()
    assert body["eta_minutes"] > 0
    assert "X-Process-Time-Ms" in response.headers


def test_reforecast_returns_risk_at_assignment(client):
    client.post("/predict_eta", json=_order())
    update = {**_order(), "checkpoint": "rider_assigned", "elapsed_min": 35.0, "assignment_delay_min": 35.0}
    body = client.post("/reforecast", json=update).json()
    assert body["checkpoint"] == "rider_assigned"
    assert body["risk"] is not None
    assert 0.0 <= body["risk"]["risk_score"] <= 1.0


def test_risk_check_flow(client):
    client.post("/predict_eta", json=_order("O-RISK"))
    update = {**_order("O-RISK"), "checkpoint": "rider_assigned", "elapsed_min": 40.0, "assignment_delay_min": 40.0}
    client.post("/reforecast", json=update)
    body = client.get("/risk_check/O-RISK").json()
    assert body["order_id"] == "O-RISK"
    assert isinstance(body["flagged"], bool)


def test_risk_check_unknown_order(client):
    assert client.get("/risk_check/does-not-exist").status_code == 404


def test_invalid_zone_rejected(client):
    bad = {**_order(), "traffic_zone": "Atlantis"}
    assert client.post("/predict_eta", json=bad).status_code == 422


def test_invalid_checkpoint_rejected(client):
    bad = {**_order(), "checkpoint": "order_placed", "elapsed_min": 0.0}
    assert client.post("/reforecast", json=bad).status_code == 422
