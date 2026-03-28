import pytest
from fastapi.testclient import TestClient
from ml_service.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200

def test_predict_invalid_data():
    response = client.post("/predict", json={})
    assert response.status_code == 400
    assert "All input features are empty (json пуст)" in response.json()["detail"]

def test_predict_missing_specific_features():
    response = client.post("/predict", json={"age": 25})
    assert response.status_code in [200, 400] 

def test_update_model_error():
    response = client.post("/updateModel", json={"run_id": "non_existent_id"})
    assert response.status_code == 404