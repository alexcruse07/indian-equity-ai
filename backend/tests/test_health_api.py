"""Unit tests for `app/api/health.py`."""

from fastapi.testclient import TestClient

from app.main import app


def test_health_returns_up():
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
