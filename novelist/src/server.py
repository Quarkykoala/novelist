"""
FastAPI Backend Server — Scientific Hypothesis Synthesizer

Provides REST API endpoints for the frontend to communicate with the orchestrator.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.contracts.schemas import Hypothesis, IterationTrace, RalphConfig
from src.ralph.orchestrator import RalphOrchestrator

app = FastAPI(
    title="Scientific Hypothesis Synthesizer API",
    description="AI-powered generation and evaluation of scientific hypotheses",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, dict[str, Any]] = {}
running_tasks: dict[str, asyncio.Task] = {}
knowledge_stats: dict[str, int] = {
    "papers_indexed": 0,
    "concepts_extracted": 0,
    "relations_found": 0,
}

# Paths
BASE_DIR = Path(__file__).parent.parent
SESSIONS_DIR = BASE_DIR / "sessions"
WEB_DIR = BASE_DIR / "src" / "web"


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════

class SessionCreateRequest(BaseModel):
    topic: str
    max_iterations: int = 4
    max_time: int = 300
    superprompt: bool = True


class SessionResponse(BaseModel):
    id: str
    topic: str
    status: str
    created_at: str


class SessionStatusResponse(BaseModel):
    id: str
    topic: str
    status: str
    iteration: int
    phase: str
    complete: bool
    hypotheses: list[dict[str, Any]]
    soulMessages: list[dict[str, Any]]
    relevanceScore: float | None


# ═══════════════════════════════════════════════════════════════
# Background Task Runner
# ═══════════════════════════════════════════════════════════════

def _serialize_hypothesis(h: Hypothesis | dict[str, Any]) -> dict[str, Any]:
    """Shape hypothesis data for the frontend."""
    if isinstance(h, Hypothesis):
        return {
            "id": h.id,
            "statement": h.hypothesis,
            "rationale": h.rationale,
            "cross_disciplinary_connection": h.cross_disciplinary_connection,
            "experimental_design": h.experimental_design,
            "expected_impact": h.expected_impact,
            "novelty_keywords": h.novelty_keywords,
            "scores": h.scores.model_dump(),
            "evidence": h.evidence.model_dump(),
            "source_soul": h.source_soul.value if h.source_soul else None,
            "iteration": h.iteration,
        }

    # Dict fallback (loaded from disk)
    return {
        "id": h.get("id", ""),
        "statement": h.get("hypothesis") or h.get("statement") or h.get("text"),
        "rationale": h.get("rationale", ""),
        "cross_disciplinary_connection": h.get("cross_disciplinary_connection", ""),
        "experimental_design": h.get("experimental_design", []),
        "expected_impact": h.get("expected_impact", ""),
        "novelty_keywords": h.get("novelty_keywords", []),
        "scores": h.get("scores", {}),
        "evidence": h.get("evidence", {}),
        "source_soul": h.get("source_soul"),
        "iteration": h.get("iteration", 0),
    }


def _serialize_trace(trace: IterationTrace) -> dict[str, Any]:
    """Convert iteration traces into lightweight soul messages."""
    return {
        "soul": "synthesizer",
        "text": trace.thought or trace.observation,
        "timestamp": trace.timestamp.isoformat(),
        "highlighted": trace.avg_novelty >= 0.7 or trace.avg_feasibility >= 0.6,
    }


async def run_session(session_id: str, topic: str, config: RalphConfig):
    """Run the orchestrator in the background."""
    try:
        sessions[session_id]["status"] = "running"
        sessions[session_id]["phase"] = "Initializing"

        orchestrator = RalphOrchestrator(config=config)

        # Run the session (saves to sessions dir automatically)
        result = await orchestrator.run(topic, output_dir=SESSIONS_DIR)

        # Store results
        sessions[session_id]["status"] = "complete"
        sessions[session_id]["phase"] = result.stop_reason or "Complete"
        sessions[session_id]["complete"] = True
        sessions[session_id]["result"] = result
        sessions[session_id]["iteration"] = result.iterations_completed
        sessions[session_id]["hypotheses"] = [
            _serialize_hypothesis(h) for h in (result.final_hypotheses if result else [])
        ]
        sessions[session_id]["soulMessages"] = [
            _serialize_trace(t) for t in (result.traces if result else [])
        ]
        sessions[session_id]["relevanceScore"] = (
            result.concept_map and len(result.concept_map.nodes) / 10
        ) or None

        # Update knowledge stats from this run
        if result:
            knowledge_stats["papers_indexed"] = max(
                knowledge_stats.get("papers_indexed", 0), result.papers_ingested
            )
            if result.concept_map:
                knowledge_stats["concepts_extracted"] = max(
                    knowledge_stats.get("concepts_extracted", 0), len(result.concept_map.nodes)
                )
                knowledge_stats["relations_found"] = max(
                    knowledge_stats.get("relations_found", 0), len(result.concept_map.edges)
                )

    except Exception as e:
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = str(e)
        sessions[session_id]["complete"] = True
    finally:
        running_tasks.pop(session_id, None)


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    request: SessionCreateRequest,
    background_tasks: BackgroundTasks,
):
    """Start a new hypothesis generation session."""
    session_id = str(uuid.uuid4())[:8]
    
    config = RalphConfig(
        max_iterations=request.max_iterations,
        max_runtime_seconds=request.max_time,
    )

    # Initialize session
    sessions[session_id] = {
        "id": session_id,
        "topic": request.topic,
        "status": "starting",
        "created_at": datetime.now().isoformat(),
        "iteration": 0,
        "phase": "Starting",
        "complete": False,
        "hypotheses": [],
        "soulMessages": [],
        "result": None,
    }
    
    # Start background task
    task = asyncio.create_task(run_session(session_id, request.topic, config))
    running_tasks[session_id] = task

    return SessionResponse(
        id=session_id,
        topic=request.topic,
        status="starting",
        created_at=sessions[session_id]["created_at"],
    )


@app.get("/api/sessions/{session_id}")
async def get_session_status(session_id: str):
    """Get status of a running or completed session."""
    if session_id not in sessions:
        # Try to load from disk
        session_dir = SESSIONS_DIR / session_id
        if session_dir.exists():
            hypotheses_file = session_dir / "hypotheses.json"
            if hypotheses_file.exists():
                with open(hypotheses_file) as f:
                    hypotheses = json.load(f)
                serialized = [_serialize_hypothesis(h) for h in hypotheses]
                return {
                    "id": session_id,
                    "topic": "Loaded from disk",
                    "status": "complete",
                    "iteration": 0,
                    "phase": "Complete",
                    "complete": True,
                    "hypotheses": serialized,
                    "soulMessages": [],
                    "relevanceScore": None,
                }
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return {
        "id": session_id,
        "topic": session["topic"],
        "status": session["status"],
        "iteration": session.get("iteration", 0),
        "phase": session.get("phase", "Unknown"),
        "complete": session.get("complete", False),
        "hypotheses": session.get("hypotheses", []),
        "soulMessages": session.get("soulMessages", []),
        "relevanceScore": session.get("relevanceScore"),
    }


@app.post("/api/sessions/{session_id}/stop")
async def stop_session(session_id: str):
    """Stop a running session."""
    if session_id in running_tasks:
        running_tasks[session_id].cancel()
        del running_tasks[session_id]
    
    if session_id in sessions:
        sessions[session_id]["status"] = "stopped"
        sessions[session_id]["complete"] = True
    
    return {"status": "stopped"}


@app.get("/api/sessions")
async def list_sessions(limit: int = 20):
    """List recent sessions."""
    # Combine in-memory and on-disk sessions
    all_sessions = []
    
    # In-memory sessions
    for sid, session in sessions.items():
        all_sessions.append({
            "id": sid,
            "topic": session["topic"],
            "status": session["status"],
            "created_at": session["created_at"],
        })
    
    # On-disk sessions
    if SESSIONS_DIR.exists():
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True)[:limit]:
            if session_dir.is_dir() and session_dir.name not in sessions:
                all_sessions.append({
                    "id": session_dir.name,
                    "topic": "Loaded from disk",
                    "status": "complete",
                    "created_at": datetime.fromtimestamp(
                        session_dir.stat().st_mtime
                    ).isoformat(),
                })
    
    return all_sessions[:limit]


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics."""
    return knowledge_stats


# ═══════════════════════════════════════════════════════════════
# Static Files (Serve Frontend)
# ═══════════════════════════════════════════════════════════════

# Serve frontend static files
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
