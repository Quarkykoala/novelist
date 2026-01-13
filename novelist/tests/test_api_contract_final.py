import json
import os
import shutil
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.contracts.schemas import SessionConstraints
from src.server import SESSIONS_DIR, app, running_tasks, sessions

client = TestClient(app)


def _reset_state():
    sessions.clear()
    running_tasks.clear()

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_create_session_missing_api_key():
    _reset_state()
    with patch.dict(os.environ, {}, clear=True):
        response = client.post(
            "/api/sessions",
            json={"topic": "test", "max_iterations": 1, "max_time": 10}
        )
        assert response.status_code == 400
        assert "Missing required API keys" in response.json()["detail"]

def test_create_session_success():
    _reset_state()
    payload = {
        "topic": "test topic",
        "max_iterations": 1,
        "constraints": {
            "domains": ["aging"],
            "modalities": ["simulation"],
            "timeline": "2 weeks",
            "dataset_links": ["https://example.com/dataset.csv"],
        },
    }
    with patch.dict(os.environ, {"GEMINI_API_KEY": "fake-key", "GROQ_API_KEY": "groq"}):
        with patch("src.server.run_session", new_callable=AsyncMock) as mock_run:
            response = client.post("/api/sessions", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["phase"] == "queued"
            assert data["constraints"]["domains"] == ["aging"]
            mock_call = mock_run.call_args
            assert mock_call is not None
            assert isinstance(mock_call.args[3], SessionConstraints)
            assert mock_call.args[3].domains == ["aging"]

def test_get_session_status_404():
    _reset_state()
    response = client.get("/api/sessions/nonexistent-id")
    assert response.status_code == 404


def test_resume_session_uses_summary_config():
    _reset_state()
    session_id = "resume11"
    summary_dir = SESSIONS_DIR / session_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "topic": "saved topic",
        "config": {"max_iterations": 2, "max_runtime_seconds": 120},
        "constraints": {"domains": ["chemistry"], "dataset_links": ["https://example.org"]},
        "status": "complete",
        "phase": "complete",
    }
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)

    try:
        with patch.dict(os.environ, {"GEMINI_API_KEY": "fake", "GROQ_API_KEY": "groq"}):
            with patch("src.server.run_session", new_callable=AsyncMock) as mock_run:
                response = client.post(f"/api/sessions/{session_id}/resume")
                assert response.status_code == 200
                data = response.json()
                assert data["origin_session_id"] == session_id
                assert mock_run.await_count == 1
    finally:
        shutil.rmtree(summary_dir, ignore_errors=True)


def test_list_sessions_includes_disk_metadata():
    _reset_state()
    session_id = "hist1234"
    summary_dir = SESSIONS_DIR / session_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "topic": "history topic",
        "status": "complete",
        "phase": "complete",
        "phase_history": [{"phase": "complete", "detail": "done", "timestamp": "2024-01-01T00:00:00"}],
        "constraints": {"domains": ["bio"]},
    }
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)

    try:
        resp = client.get("/api/sessions")
        assert resp.status_code == 200
        entries = resp.json()
        target = next((e for e in entries if e["id"] == session_id), None)
        assert target is not None
        assert target["phase"] == "complete"
        assert target["constraints"]["domains"] == ["bio"]
    finally:
        shutil.rmtree(summary_dir, ignore_errors=True)


def test_get_session_status_from_disk_includes_metadata():
    _reset_state()
    session_id = "disk1234"
    summary_dir = SESSIONS_DIR / session_id
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "topic": "disk topic",
        "status": "complete",
        "phase": "complete",
        "phase_history": [{"phase": "complete", "detail": "done", "timestamp": "2024-01-01T00:00:00"}],
        "constraints": {"domains": ["physics"], "dataset_links": ["https://example.com"]},
    }
    hypotheses = [
        {
            "id": "h1",
            "hypothesis": "Test",
            "rationale": "",
            "cross_disciplinary_connection": "",
            "experimental_design": ["step"],
            "expected_impact": "",
            "novelty_keywords": [],
            "evidence": {},
            "scores": {},
        }
    ]
    with open(summary_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(summary_dir / "hypotheses.json", "w", encoding="utf-8") as f:
        json.dump(hypotheses, f)

    try:
        resp = client.get(f"/api/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["constraints"]["domains"] == ["physics"]
        assert data["phase_history"][0]["phase"] == "complete"
        assert len(data["hypotheses"]) == 1
    finally:
        shutil.rmtree(summary_dir, ignore_errors=True)
