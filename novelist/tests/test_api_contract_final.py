
from fastapi.testclient import TestClient
from src.server import app
import os
from unittest.mock import patch, MagicMock, AsyncMock

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_session_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        response = client.post(
            "/api/sessions",
            json={"topic": "test", "max_iterations": 1, "max_time": 10}
        )
        assert response.status_code == 400
        assert "Missing GEMINI_API_KEY" in response.json()["detail"]

def test_create_session_success():
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key"}):
        # Use AsyncMock for the background task
        with patch("src.server.run_session", new_callable=AsyncMock) as mock_run:
            response = client.post(
                "/api/sessions",
                json={"topic": "test topic", "max_iterations": 1}
            )
            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["status"] == "starting"

def test_get_session_status_404():
    response = client.get("/api/sessions/nonexistent-id")
    assert response.status_code == 404
