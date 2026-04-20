from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={"image": "dummy_data"})

    assert response.status_code in [200, 400]
