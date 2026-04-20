from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath("."))

from api.app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    print(response.text)  # helps debug in CI
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
