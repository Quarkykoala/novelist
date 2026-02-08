import json
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from src.contracts.schemas import Persona
from src.server import app, sessions

client = TestClient(app)

def test_persona_management_endpoints():
    session_id = "test_pers"
    sessions[session_id] = {
        "topic": "test",
        "orchestrator": AsyncMock(),
        "personas": [],
        "status": "running"
    }
    orchestrator = sessions[session_id]["orchestrator"]
    orchestrator.persona_roster = [
        {"id": "specialist", "name": "Dr. Smith", "locked": False, "weight": 0.4}
    ]

    # Test Lock
    resp = client.post(f"/api/sessions/{session_id}/personas/specialist/lock")
    assert resp.status_code == 200
    orchestrator.lock_persona.assert_awaited_once_with("specialist")

    # Test Unlock
    resp = client.post(f"/api/sessions/{session_id}/personas/specialist/unlock")
    assert resp.status_code == 200
    orchestrator.unlock_persona.assert_awaited_once_with("specialist")

    # Test Weight
    resp = client.post(
        f"/api/sessions/{session_id}/personas/specialist/weight",
        json={"weight": 0.5}
    )
    assert resp.status_code == 200
    orchestrator.update_persona_weight.assert_awaited_once_with("specialist", 0.5)

    # Test Regenerate
    resp = client.post(f"/api/sessions/{session_id}/personas/specialist/regenerate")
    assert resp.status_code == 200
    orchestrator.regenerate_persona.assert_awaited_once_with("specialist")

def test_regenerate_logic_in_orchestrator():
    from src.ralph.orchestrator import RalphOrchestrator
    from src.contracts.schemas import RalphConfig, SoulRole
    
    config = RalphConfig()
    orch = RalphOrchestrator(config=config)
    orch.topic = "energy storage"
    orch.persona_roster = [
        {
            "id": "specialist",
            "name": "Old Name",
            "role": "Old Role",
            "style": "Old Style",
            "objective": "Old Obj",
            "weight": 0.4,
            "soul_role": "creative",
            "locked": False
        }
    ]
    
    new_persona = Persona(
        id="specialist",
        name="New Name",
        role="New Role",
        style="New Style",
        objective="New Obj",
        weight=0.5,
        system_instruction="New Instruction",
        soul_role=SoulRole.CREATIVE,
        locked=False
    )
    
    orch.persona_forge = AsyncMock()
    orch.persona_forge.regenerate_persona.return_value = new_persona
    
    import asyncio
    asyncio.run(orch.regenerate_persona("specialist"))
    
    assert orch.persona_roster[0]["name"] == "New Name"
    assert orch.persona_roster[0]["weight"] == 0.5
    assert orch.collective.creative.custom_name == "New Name"
    assert orch.collective.creative.custom_instruction == "New Instruction"
