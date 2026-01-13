import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from src.contracts.schemas import RalphConfig, SessionPhase
from src.server import app, sessions

client = TestClient(app)

def test_get_knowledge_stats():
    resp = client.get("/api/knowledge/stats")
    assert resp.status_code == 200
    assert "papers_indexed" in resp.json()

def test_session_concept_map_exposure():
    session_id = "test_map"
    sessions[session_id] = {
        "id": session_id,
        "topic": "test",
        "status": "running",
        "phase": SessionPhase.MAPPING.value,
        "created_at": "2024-01-01T00:00:00",
        "hypotheses": [],
        "soulMessages": [],
        "concept_map": {
            "nodes": [{"id": "n1", "name": "Node 1", "type": "method"}],
            "edges": [],
            "gaps": [],
            "contradictions": []
        }
    }
    
    resp = client.get(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert "concept_map" in data
    assert data["concept_map"]["nodes"][0]["name"] == "Node 1"

async def test_orchestrator_emits_knowledge_updates():
    from src.ralph.orchestrator import RalphOrchestrator
    
    callback = AsyncMock()
    orch = RalphOrchestrator(callbacks={"on_knowledge_update": callback})
    orch.paper_store = {"p1": MagicMock()}
    
    # Simulate ingest update
    await orch._emit_knowledge_update({"papers_indexed": 1})
    
    callback.assert_awaited_with({"papers_indexed": 1})
