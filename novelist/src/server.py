"""
FastAPI Backend Server — Scientific Hypothesis Synthesizer

Provides REST API endpoints for the frontend to communicate with the orchestrator.
"""

import asyncio
import json
import os
import sys
import traceback
import uuid
from datetime import datetime
from pathlib import Path
import tempfile
from typing import Any

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from src.contracts.schemas import (
    Hypothesis,
    IterationTrace,
    RalphConfig,
    SessionConstraints,
    SessionPhase,
)
from src.ralph.orchestrator import RalphOrchestrator

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
WEB_DIR = BASE_DIR / "ui" / "dist"


# ═══════════════════════════════════════════════════════════════
# Request/Response Models
# ═══════════════════════════════════════════════════════════════


def _require_api_keys() -> None:
    """Ensure required API keys are configured."""
    missing = []
    if not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")

    if missing:
        detail = (
            "Missing required API keys: " + ", ".join(missing) + ". "
            "Set them in your environment or .env file."
        )
        raise HTTPException(status_code=400, detail=detail)


def _serialize_constraints(data: SessionConstraints | dict[str, Any] | None) -> dict[str, Any] | None:
    if data is None:
        return None
    if isinstance(data, SessionConstraints):
        return data.model_dump()
    return data


def _append_phase_history(
    session_id: str,
    phase_value: str,
    detail: str | None,
    timestamp: str | None = None,
) -> None:
    session = sessions.get(session_id)
    if not session:
        return

    ts = timestamp or datetime.now().isoformat()
    history = session.setdefault("phase_history", [])
    if history and history[-1]["phase"] == phase_value:
        history[-1] = {"phase": phase_value, "detail": detail, "timestamp": ts}
    else:
        history.append({"phase": phase_value, "detail": detail, "timestamp": ts})


def _load_summary(session_id: str) -> dict[str, Any] | None:
    summary_path = SESSIONS_DIR / session_id / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

class SessionCreateRequest(BaseModel):
    topic: str
    max_iterations: int = 4
    max_time: int = 300
    superprompt: bool = True
    constraints: SessionConstraints = Field(default_factory=SessionConstraints)


class ChatRequest(BaseModel):
    message: str


class SessionResponse(BaseModel):
    id: str
    topic: str
    status: str
    created_at: str
    phase: SessionPhase
    constraints: dict[str, Any] | None = None
    origin_session_id: str | None = None


class SessionStatusResponse(BaseModel):
    id: str
    topic: str
    status: str
    iteration: int
    phase: SessionPhase
    complete: bool
    hypotheses: list[dict[str, Any]]
    soulMessages: list[dict[str, Any]]
    relevanceScore: float | None
    error: str | None = None
    created_at: str
    status_detail: str | None = None
    constraints: dict[str, Any] | None = None
    phase_history: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] | None = None
    personas: list[dict[str, Any]] | None = None


async def _spawn_session(
    request: SessionCreateRequest,
    *,
    origin_session_id: str | None = None,
) -> SessionResponse:
    _require_api_keys()

    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()
    constraints_dict = _serialize_constraints(request.constraints)

    config = RalphConfig(
        max_iterations=request.max_iterations,
        max_runtime_seconds=request.max_time,
        domains=request.constraints.domains,
        modalities=request.constraints.modalities,
        timeline_hint=request.constraints.timeline,
        dataset_links=[str(link) for link in request.constraints.dataset_links],
    )

    sessions[session_id] = {
        "id": session_id,
        "topic": request.topic,
        "status": "starting",
        "created_at": created_at,
        "iteration": 0,
        "phase": SessionPhase.QUEUED.value,
        "status_detail": "Queued",
        "complete": False,
        "hypotheses": [],
        "soulMessages": [],
        "gaps": [],
        "source_metadata": {},
        "result": None,
        "error": None,
        "constraints": constraints_dict,
        "config": config.model_dump(),
        "phase_history": [],
        "origin_session_id": origin_session_id,
        "personas": [],
    }

    _append_phase_history(session_id, SessionPhase.QUEUED.value, "Queued", created_at)

    task = asyncio.create_task(run_session(session_id, request.topic, config, request.constraints))
    running_tasks[session_id] = task

    return SessionResponse(
        id=session_id,
        topic=request.topic,
        status="starting",
        phase=SessionPhase.QUEUED,
        created_at=created_at,
        constraints=constraints_dict,
        origin_session_id=origin_session_id,
    )


# ═══════════════════════════════════════════════════════════════
# Background Task Runner
# ═══════════════════════════════════════════════════════════════

def _serialize_hypothesis(h: Hypothesis | dict[str, Any]) -> dict[str, Any]:
    """Shape hypothesis data for the frontend."""
    if isinstance(h, Hypothesis):
        sim_result = getattr(h, "simulation_result", None)
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
            "simulation_result": sim_result.model_dump() if sim_result else None,
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


def _serialize_dialogue(entry: Any) -> dict[str, Any]:
    """Convert a dialogue entry into a soul message."""
    return {
        "soul": entry.soul,
        "role": entry.role.value if hasattr(entry.role, 'value') else str(entry.role),
        "text": entry.message,
        "timestamp": entry.timestamp.isoformat(),
        "highlighted": entry.role in ["creative", "risk_taker"]
    }


def _serialize_trace(trace: IterationTrace) -> list[dict[str, Any]]:
    """Convert iteration trace into multiple soul messages if dialogue exists."""
    messages = []
    
    # Add the main BDI thought
    messages.append({
        "soul": "Synthesizer",
        "role": "synthesizer",
        "text": trace.thought,
        "timestamp": trace.timestamp.isoformat(),
        "highlighted": True
    })
    
    # Add detailed dialogue
    if hasattr(trace, 'dialogue') and trace.dialogue:
        for entry in trace.dialogue:
            messages.append(_serialize_dialogue(entry))
            
    # Add observation
    messages.append({
        "soul": "Orchestrator",
        "role": "synthesizer",
        "text": trace.observation,
        "timestamp": trace.timestamp.isoformat(),
        "highlighted": False
    })
    
    return messages


async def run_session(
    session_id: str,
    topic: str,
    config: RalphConfig,
    constraints: SessionConstraints | None = None,
):
    """Run the orchestrator in the background."""
    try:
        sessions[session_id]["status"] = "running"
        sessions[session_id]["phase"] = SessionPhase.FORGING.value
        sessions[session_id]["status_detail"] = "Initializing orchestrator"
        _append_phase_history(session_id, SessionPhase.FORGING.value, "Initializing orchestrator")

        # Define callbacks for live updates
        async def on_status_change(update: dict[str, Any]):
            if session_id not in sessions:
                return
            phase = update.get("phase", SessionPhase.DEBATING.value)
            detail = update.get("detail")
            timestamp = update.get("timestamp")
            sessions[session_id]["phase"] = phase
            sessions[session_id]["status_detail"] = detail
            _append_phase_history(session_id, phase, detail, timestamp)

        async def on_trace(trace: IterationTrace):
            if session_id in sessions:
                sessions[session_id]["iteration"] = trace.iteration
                # Append all messages from this trace
                for msg in _serialize_trace(trace):
                    sessions[session_id]["soulMessages"].append(msg)
                
                # Also update currently surviving hypotheses count if available in trace
                sessions[session_id]["hypotheses_count"] = trace.hypotheses_surviving
                sessions[session_id]["total_cost"] = trace.cost_usd

        async def on_personas(personas: list[dict[str, Any]]):
            if session_id in sessions:
                sessions[session_id]["personas"] = personas

        orchestrator = RalphOrchestrator(
            config=config,
            callbacks={
                "on_status_change": on_status_change,
                "on_trace": on_trace,
                "on_personas": on_personas,
            }
        )
        
        # Store instance for interaction
        sessions[session_id]["orchestrator"] = orchestrator

        # Run the session (saves to sessions dir automatically)
        result = await orchestrator.run(
            topic,
            output_dir=SESSIONS_DIR,
            session_id=session_id,
            constraints=constraints,
        )

        # Store results
        sessions[session_id]["status"] = "complete"
        sessions[session_id]["phase"] = SessionPhase.COMPLETE.value
        sessions[session_id]["status_detail"] = result.stop_reason or "Complete"
        sessions[session_id]["complete"] = True
        sessions[session_id]["result"] = result
        sessions[session_id]["iteration"] = result.iterations_completed
        sessions[session_id]["hypotheses"] = [
            _serialize_hypothesis(h) for h in (result.final_hypotheses if result else [])
        ]
        sessions[session_id]["soulMessages"] = [
            msg
            for trace in (result.traces if result else [])
            for msg in _serialize_trace(trace)
        ]
        sessions[session_id]["relevanceScore"] = (
            result.concept_map and len(result.concept_map.nodes) / 10
        ) or None
        sessions[session_id]["personas"] = orchestrator.persona_roster

        # Store gaps if available from orchestrator
        if hasattr(orchestrator, 'gaps') and orchestrator.gaps:
            sessions[session_id]["gaps"] = [
                {
                    "type": g.gap_type.value,
                    "description": g.description,
                    "concept_a": g.concept_a,
                    "concept_b": g.concept_b,
                    "potential_value": g.potential_value,
                }
                for g in orchestrator.gaps
            ]

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
        error_message = str(e)
        sessions[session_id]["status"] = "error"
        sessions[session_id]["error"] = error_message
        sessions[session_id]["phase"] = SessionPhase.ERROR.value
        sessions[session_id]["status_detail"] = error_message
        sessions[session_id]["complete"] = True
        _append_phase_history(session_id, SessionPhase.ERROR.value, error_message)
        try:
            session_dir = SESSIONS_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            with open(session_dir / "error.log", "w", encoding="utf-8") as f:
                f.write(traceback.format_exc())
        except Exception as log_error:
            print(f"[WARN] Failed to write error log: {log_error}")
    finally:
        session = sessions.get(session_id)
        if session:
            summary_data = _load_summary(session_id) or {}
            summary_data.update(
                {
                    "session_id": session_id,
                    "topic": session.get("topic"),
                    "started_at": session.get("created_at"),
                    "completed_at": datetime.now().isoformat(),
                    "phase": session.get("phase"),
                    "status": session.get("status"),
                    "status_detail": session.get("status_detail"),
                    "phase_history": session.get("phase_history", []),
                    "constraints": session.get("constraints"),
                    "config": session.get("config"),
                    "origin_session_id": session.get("origin_session_id"),
                }
            )
            summary_path = SESSIONS_DIR / session_id / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

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
    return await _spawn_session(request)


@app.get("/api/sessions/{session_id}")
async def get_session_status(session_id: str):
    """Get status of a running or completed session."""
    session = sessions.get(session_id)
    if not session:
        summary = _load_summary(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        hypotheses: list[dict[str, Any]] = []
        hypotheses_file = SESSIONS_DIR / session_id / "hypotheses.json"
        if hypotheses_file.exists():
            with open(hypotheses_file, "r", encoding="utf-8") as f:
                hypotheses = [_serialize_hypothesis(h) for h in json.load(f)]

        return {
            "id": session_id,
            "topic": summary.get("topic", "Loaded from disk"),
            "status": summary.get("status", "complete"),
            "iteration": summary.get("iterations", 0),
            "phase": summary.get("phase", SessionPhase.COMPLETE.value),
            "complete": summary.get("status", "complete") == "complete",
            "hypotheses": hypotheses,
            "soulMessages": [],
            "gaps": [],
            "source_metadata": {},
            "relevanceScore": None,
            "error": None,
            "created_at": summary.get("started_at"),
            "status_detail": summary.get("status_detail"),
            "constraints": summary.get("constraints"),
            "phase_history": summary.get("phase_history", []),
            "config": summary.get("config"),
            "personas": summary.get("personas"),
        }

    # Live session
    # Get live metadata if orchestrator is running
    metadata = session.get("source_metadata", {})
    orchestrator = session.get("orchestrator")
    if orchestrator and hasattr(orchestrator, 'paper_store'):
        metadata = {k: v.model_dump() for k, v in orchestrator.paper_store.items()}

    return {
        "id": session_id,
        "topic": session["topic"],
        "status": session["status"],
        "iteration": session.get("iteration", 0),
        "phase": session.get("phase", SessionPhase.DEBATING.value),
        "complete": session.get("complete", False),
        "hypotheses": session.get("hypotheses", []),
        "soulMessages": session.get("soulMessages", []),
        "gaps": session.get("gaps", []),
        "source_metadata": metadata,
        "relevanceScore": session.get("relevanceScore"),
        "error": session.get("error"),
        "created_at": session.get("created_at"),
        "status_detail": session.get("status_detail"),
        "constraints": session.get("constraints"),
        "phase_history": session.get("phase_history", []),
        "config": session.get("config"),
        "personas": session.get("personas"),
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


@app.post("/api/sessions/{session_id}/resume", response_model=SessionResponse)
async def resume_session(session_id: str, background_tasks: BackgroundTasks):
    """Resume a past session using its stored configuration."""
    summary = _load_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session summary not found")

    config_data = summary.get("config") or {}
    constraints_data = summary.get("constraints") or {}

    constraints = SessionConstraints(**constraints_data) if constraints_data else SessionConstraints()

    request = SessionCreateRequest(
        topic=summary.get("topic", "Untitled Session"),
        max_iterations=config_data.get("max_iterations", 4),
        max_time=config_data.get("max_runtime_seconds", 300),
        constraints=constraints,
    )

    return await _spawn_session(request, origin_session_id=session_id)


@app.post("/api/sessions/{session_id}/chat")
async def session_chat(session_id: str, request: ChatRequest):
    """Send a message to the running orchestrator."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active for this session")
    
    await orchestrator.inject_user_message(request.message)
    
    # Provide instant feedback in the Soul Feed
    sessions[session_id]["soulMessages"].append({
        "soul": "System",
        "role": "synthesizer",
        "text": f"Instruction logged: '{request.message}'. The collective will incorporate this into the next iteration.",
        "timestamp": datetime.now().isoformat(),
        "highlighted": True
    })
    
    return {"status": "message_injected"}


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
            "phase": session.get("phase"),
            "status_detail": session.get("status_detail"),
            "constraints": session.get("constraints"),
            "phase_history": session.get("phase_history", []),
            "origin_session_id": session.get("origin_session_id"),
            "personas": session.get("personas"),
        })
    
    # On-disk sessions
    if SESSIONS_DIR.exists():
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True)[:limit]:
            if session_dir.is_dir() and session_dir.name not in sessions:
                summary = _load_summary(session_dir.name) or {}
                all_sessions.append({
                    "id": session_dir.name,
                    "topic": summary.get("topic", "Loaded from disk"),
                    "status": summary.get("status", "complete"),
                    "created_at": summary.get("started_at")
                    or datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat(),
                    "phase": summary.get("phase"),
                    "status_detail": summary.get("status_detail"),
                    "constraints": summary.get("constraints"),
                    "phase_history": summary.get("phase_history", []),
                    "origin_session_id": summary.get("origin_session_id"),
                    "personas": summary.get("personas"),
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

# Mount plots directory
temp_plots_dir = Path(tempfile.gettempdir()) / "novelist_sims"
temp_plots_dir.mkdir(parents=True, exist_ok=True)
app.mount("/plots", StaticFiles(directory=temp_plots_dir), name="plots")


# ═══════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
