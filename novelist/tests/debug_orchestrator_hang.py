
import asyncio
from src.ralph.orchestrator import RalphOrchestrator
from src.contracts.schemas import RalphConfig

async def debug_hang():
    print("Starting debug run...")
    
    async def log_status(msg):
        print(f"STATUS: {msg}")

    config = RalphConfig(max_iterations=1, max_runtime_seconds=60)
    orch = RalphOrchestrator(
        config=config, 
        callbacks={"on_status_change": log_status}
    )
    
    print("Finding papers...")
    # Manually trigger simple ingest if possible, or run orchestrator
    # We'll rely on orchestrator's run method but mocked fetch_papers?
    
    # Actually, let's just run it. The default creates synthetic papers if no API key for arxiv? 
    # Or does it use real arxiv?
    # src/kb/retrieval.py usually handles this.
    
    # If it uses real Arxiv, that might be slow/hanging.
    # Let's try running it.
    
    try:
        await asyncio.wait_for(orch.run("test topic"), timeout=60)
    except asyncio.TimeoutError:
        print("TIMED OUT!")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(debug_hang())
