from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_valid_order():
    payload = {
        "AbsQuantity": 12,
        "Price": 6.95,
        "Hour": 14,
        "DayOfWeek": 2,
        "Month": 11,
        "Country": "United Kingdom"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_cancelled_pred" in data
    assert "cancellation_probability" in data
    assert 0.0 <= data["cancellation_probability"] <= 1.0

def test_predict_rejects_negative_price():
    bad_payload = {
        "AbsQuantity": 12,
        "Price": -5.0,
        "Hour": 14,
        "DayOfWeek": 2,
        "Month": 11,
        "Country": "United Kingdom"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation blocks bad data