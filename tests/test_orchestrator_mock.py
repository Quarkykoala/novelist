
# Verification Script: Test Orchestrator Callbacks & Cost Tracking

import asyncio
import os
from unittest.mock import MagicMock, AsyncMock

from src.contracts.schemas import RalphConfig, GenerationResponse, TokenUsage, SessionPhase
from src.ralph.orchestrator import RalphOrchestrator
from src.soul.llm_client import LLMClient

# Mock LLM Client to return fake cost
async def mock_generate(*args, **kwargs):
    return GenerationResponse(
        content='{"test": "data"}', 
        usage=TokenUsage(total_tokens=100, cost_usd=0.05),
        model_name="test-model",
        provider="test-provider"
    )

async def test_orchestrator_flow():
    print("Testing Orchestrator Callbacks & Cost...")
    
    # Setup Mocks
    LLMClient.generate_content = mock_generate
    
    # Callbacks
    status_updates = []
    traces = []
    
    async def on_status(msg):
        print(f"Callback Status: {msg}")
        status_updates.append(msg)
        
    async def on_trace(trace):
        print(f"Callback Trace: Iteration {trace.iteration}, Cost: ${trace.cost_usd}")
        traces.append(trace)

    # Configure fast run
    config = RalphConfig(max_iterations=1, max_runtime_seconds=60)
    
    orch = RalphOrchestrator(
        config=config,
        callbacks={
            "on_status_change": on_status,
            "on_trace": on_trace
        }
    )
    
    # Inject fake gaps to force flow
    orch.gaps = [{"mock": "gap"}] 
    
    # Run (partially mocked)
    # We can't easily mock EVERYTHING, but we can check if callbacks fire during init
    # For a true unit test we'd need more DI, but this verifies the wiring.
    
    # Just run ingest papers phase (will fail on real API calls if not mocked properly, 
    # but we just want to see if our cost logic runs).
    # Easier: manually trigger _update_costs and callbacks
    
    orch.collective.creative.total_cost = 0.50
    orch._update_costs()
    print(f"Orchestrator Total Cost: {orch.total_cost}")
    assert orch.total_cost == 0.50
    
    await orch._emit_status(SessionPhase.DEBATING, "Test Status")
    assert any(update.get("detail") == "Test Status" for update in status_updates)
    
    print("Verification Passed!")

if __name__ == "__main__":
    asyncio.run(test_orchestrator_flow())
