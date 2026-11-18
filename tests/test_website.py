"""
PromiseGuard Automated Web Route & API Test Suite
Tests FastAPI web server, HTML templates, and REST API endpoints.
"""

import pytest
from starlette.testclient import TestClient
from website.server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_web_routes_render(client):
    """Test that all public website pages render with HTTP 200."""
    routes = [
        "/",
        "/predict",
        "/reliability",
        "/uncertainty",
        "/promise-risk",
        "/satisfaction",
        "/explainability",
        "/simulation",
        "/audit",
        "/viva",
    ]
    for route in routes:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} failed with status {response.status_code}"
        assert "<!DOCTYPE html>" in response.text
        assert "PromiseGuard" in response.text

def test_api_health(client):
    """Test API health check endpoint."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_loaded"] is True

def test_api_metrics(client):
    """Test API metrics endpoint."""
    response = client.get("/api/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "reports" in data

def test_api_predict_endpoint(client):
    """Test live ML model inference via POST /api/predict."""
    payload = {
        "distance_km": 8.5,
        "Delivery_person_Age": 29.0,
        "Delivery_person_Ratings": 4.8,
        "multiple_deliveries": 1.0,
        "pickup_prep_delay_min": 10.0,
        "order_hour": 20,
        "Vehicle_condition": 1,
        "Weather_conditions": "Sunny",
        "Road_traffic_density": "High",
        "Type_of_order": "Meal",
        "Type_of_vehicle": "motorcycle",
        "Festival": "No",
        "City": "Metropolitian"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    res = body["data"]

    # Validate output types and ranges
    assert 5.0 <= res["predicted_eta"] <= 70.0
    assert res["pred_q_low"] <= res["pred_q_high"]
    assert res["conf_low"] <= res["conf_high"]
    assert 0.0 <= res["risk_probability"] <= 1.0
    assert res["risk_level"] in ["On-Time Delivery", "Elevated Uncertainty", "Severe Delay Risk"]
    assert len(res["contributors"]) >= 1

def test_api_simulate_endpoint(client):
    """Test dynamic threshold simulation via POST /api/simulate."""
    response = client.post("/api/simulate", json={"threshold": 0.57})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    d = body["data"]
    assert abs(d["threshold"] - 0.57) < 0.05
    assert d["alerts_sent"] > 0
    assert d["recall_pct"] > 80.0
