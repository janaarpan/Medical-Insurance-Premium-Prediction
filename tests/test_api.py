from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "age": 36,
    "weight": 70,
    "height": 1.75,
    "income_lpa": 12.5,
    "smoker": False,
    "city": "Kolkata",
    "occupation": "private_job",
}

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_category"] in {"Low", "Medium", "High"}
    assert 0 <= body["confidence"] <= 1
    assert set(body["class_probabilities"]) == {"Low", "Medium", "High"}

def test_invalid_age_is_rejected():
    payload = {**VALID_PAYLOAD, "age": 0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
