"""
FastAPI Backend Server — Scientific Hypothesis Synthesizer

Provides REST API endpoints for the frontend to communicate with the orchestrator.
"""

import asyncio
import json
import os
import re
import sys
import tempfile
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.contracts.schemas import (
    Hypothesis,
    IterationTrace,
    PersonaWeightRequest,
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

# CORS configuration:
# - default keeps localhost development working
# - production can override with CORS_ALLOW_ORIGINS / CORS_ALLOW_ORIGIN_REGEX
cors_allow_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "")
cors_allow_origins = [origin.strip() for origin in cors_allow_origins_env.split(",") if origin.strip()]
if not cors_allow_origins:
    cors_allow_origins = [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
cors_allow_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"^https?://(127\.0\.0\.1|localhost)(:\d+)?$",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allow_origins,
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=False,
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


def _require_api_keys(
    *,
    flash_model: str | None = None,
    pro_model: str | None = None,
) -> None:
    """Ensure required API keys are configured for the chosen models."""
    model_names = [m for m in [flash_model, pro_model] if m]
    if not model_names:
        model_names = ["gemini/gemini-3-flash"]

    need_gemini = any("gemini" in m.lower() for m in model_names)
    need_groq = any("groq" in m.lower() for m in model_names)
    need_cerebras = any("cerebras" in m.lower() for m in model_names)

    missing = []
    if need_gemini and not os.getenv("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    if need_groq and not os.getenv("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if need_cerebras and not os.getenv("CEREBRAS_API_KEY"):
        missing.append("CEREBRAS_API_KEY")

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


async def _load_json_async(path: Path) -> Any:
    """Load JSON from disk asynchronously to avoid blocking the event loop."""
    if not path.exists():
        return None
    try:
        def read():
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        return await asyncio.to_thread(read)
    except Exception:
        return None


async def _load_summary_async(session_id: str) -> dict[str, Any] | None:
    """Load session summary asynchronously."""
    summary_path = SESSIONS_DIR / session_id / "summary.json"
    return await _load_json_async(summary_path)


def _load_summary(session_id: str) -> dict[str, Any] | None:
    summary_path = SESSIONS_DIR / session_id / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with open(summary_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_hypotheses_from_disk(session_id: str) -> list[dict[str, Any]]:
    """Load hypotheses for an archived session."""
    hypotheses_file = SESSIONS_DIR / session_id / "hypotheses.json"
    if not hypotheses_file.exists():
        return []
    try:
        with open(hypotheses_file, encoding="utf-8") as f:
            return [_serialize_hypothesis(h) for h in json.load(f)]
    except Exception:
        return []


def _list_replay_candidates(limit: int = 20) -> list[dict[str, Any]]:
    """List completed sessions that can be used for judge replay."""
    candidates: list[dict[str, Any]] = []
    if not SESSIONS_DIR.exists():
        return candidates

    for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        session_id = session_dir.name
        summary = _load_summary(session_id) or {}
        if summary.get("status") != "complete":
            continue
        hypotheses = _load_hypotheses_from_disk(session_id)
        if not hypotheses:
            continue
        candidates.append(
            {
                "session_id": session_id,
                "topic": summary.get("topic", "Unknown"),
                "created_at": summary.get("started_at"),
                "hypotheses_count": len(hypotheses),
            }
        )
        if len(candidates) >= limit:
            break
    return candidates

from src.contracts.schemas import ConceptMap, ConceptNode
from src.kb.arxiv_client import ArxivClient
from src.kb.claim_extractor import ClaimExtractor


class SessionCreateRequest(BaseModel):
    topic: str
    max_iterations: int = 4
    max_time: int = 300
    superprompt: bool = True
    constraints: SessionConstraints = Field(default_factory=SessionConstraints)
    strict_grounding: bool = True
    enforce_numeric_citations: bool = True
    srsh_enabled: bool = True
    srsh_agents: int = 3
    srsh_iterations_per_agent: int = 3
    srsh_collisions: int = 2
    judge_replay: bool = False
    replay_session_id: str | None = None


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
    knowledge_sources: list[str] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    relevanceScore: float | None
    error: str | None = None
    created_at: str
    status_detail: str | None = None
    constraints: dict[str, Any] | None = None
    phase_history: list[dict[str, Any]] = Field(default_factory=list)
    config: dict[str, Any] | None = None
    personas: list[dict[str, Any]] | None = None
    directives: list[dict[str, Any]] | None = None
    impact_history: list[dict[str, Any]] | None = None
    strict_grounding: bool | None = None
    enforce_numeric_citations: bool | None = None
    judge_replay: bool | None = None
    replay_source_session_id: str | None = None


async def _spawn_session(
    request: SessionCreateRequest,
    *,
    origin_session_id: str | None = None,
) -> SessionResponse:
    session_id = str(uuid.uuid4())[:8]
    created_at = datetime.now().isoformat()
    constraints_dict = _serialize_constraints(request.constraints)

    config = RalphConfig(
        max_iterations=request.max_iterations,
        max_runtime_seconds=max(60, request.max_time),
        domains=request.constraints.domains,
        modalities=request.constraints.modalities,
        timeline_hint=request.constraints.timeline,
        dataset_links=[str(link) for link in request.constraints.dataset_links],
        strict_grounding=request.strict_grounding,
        enforce_numeric_citations=request.enforce_numeric_citations,
        srsh_enabled=request.srsh_enabled,
        srsh_agents=request.srsh_agents,
        srsh_iterations_per_agent=request.srsh_iterations_per_agent,
        srsh_collisions=request.srsh_collisions,
    )

    if request.judge_replay:
        return await _spawn_replay_session(
            request,
            config,
            created_at=created_at,
            session_id=session_id,
            constraints_dict=constraints_dict,
            origin_session_id=origin_session_id,
        )

    _require_api_keys(flash_model=config.flash_model, pro_model=config.pro_model)

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
        "knowledge_sources": [],
        "source_breakdown": {},
        "result": None,
        "error": None,
        "constraints": constraints_dict,
        "config": config.model_dump(),
        "phase_history": [],
        "origin_session_id": origin_session_id,
        "personas": [],
        "strict_grounding": request.strict_grounding,
        "enforce_numeric_citations": request.enforce_numeric_citations,
        "srsh_enabled": request.srsh_enabled,
        "srsh_agents": request.srsh_agents,
        "srsh_iterations_per_agent": request.srsh_iterations_per_agent,
        "srsh_collisions": request.srsh_collisions,
        "judge_replay": False,
        "replay_source_session_id": None,
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


async def _spawn_replay_session(
    request: SessionCreateRequest,
    config: RalphConfig,
    *,
    created_at: str,
    session_id: str,
    constraints_dict: dict[str, Any] | None,
    origin_session_id: str | None = None,
) -> SessionResponse:
    """Create a completed session by replaying archived results."""
    replay_source = request.replay_session_id
    if not replay_source:
        candidates = _list_replay_candidates(limit=1)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No replay sessions available. Run a complete session first.",
            )
        replay_source = candidates[0]["session_id"]

    source_summary = _load_summary(replay_source)
    if not source_summary:
        raise HTTPException(status_code=404, detail=f"Replay session not found: {replay_source}")

    hypotheses = _load_hypotheses_from_disk(replay_source)
    if not hypotheses:
        raise HTTPException(
            status_code=400,
            detail=f"Replay session has no hypotheses: {replay_source}",
        )

    sessions[session_id] = {
        "id": session_id,
        "topic": source_summary.get("topic", request.topic),
        "status": "complete",
        "created_at": created_at,
        "iteration": source_summary.get("iterations", 0),
        "phase": SessionPhase.COMPLETE.value,
        "status_detail": f"Replay completed from session {replay_source}",
        "complete": True,
        "hypotheses": hypotheses,
        "soulMessages": [],
        "gaps": [],
        "source_metadata": {},
        "knowledge_sources": [],
        "source_breakdown": {},
        "result": None,
        "error": None,
        "constraints": constraints_dict,
        "config": config.model_dump(),
        "phase_history": [],
        "origin_session_id": origin_session_id,
        "personas": source_summary.get("personas", []),
        "strict_grounding": request.strict_grounding,
        "enforce_numeric_citations": request.enforce_numeric_citations,
        "judge_replay": True,
        "replay_source_session_id": replay_source,
        "concept_map": source_summary.get("concept_map"),
    }

    _append_phase_history(session_id, SessionPhase.QUEUED.value, "Queued", created_at)
    _append_phase_history(
        session_id,
        SessionPhase.COMPLETE.value,
        f"Replay completed from session {replay_source}",
        datetime.now().isoformat(),
    )

    return SessionResponse(
        id=session_id,
        topic=sessions[session_id]["topic"],
        status="complete",
        phase=SessionPhase.COMPLETE,
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
        sim_history = getattr(h, "simulation_history", None)
        return {
            "id": h.id,
            "statement": h.hypothesis,
            "rationale": h.rationale,
            "cross_disciplinary_connection": h.cross_disciplinary_connection,
            "experimental_design": h.experimental_design,
            "expected_impact": h.expected_impact,
            "novelty_keywords": h.novelty_keywords,
            "supporting_papers": h.supporting_papers,
            "grounding_status": h.grounding_status,
            "citation_warnings": h.citation_warnings,
            "non_arxiv_sources": h.non_arxiv_sources,
            "evidence_trace": h.evidence_trace,
            "supported_facts": h.supported_facts,
            "novel_inference": h.novel_inference,
            "unsupported_parts": h.unsupported_parts,
            "evidence_spans": [
                span.model_dump() if hasattr(span, "model_dump") else span
                for span in (h.evidence_spans or [])
            ],
            "diagram": h.diagram,
            "scores": h.scores.model_dump(),
            "evidence": h.evidence.model_dump(),
            "source_soul": h.source_soul.value if h.source_soul else None,
            "simulation_result": sim_result.model_dump() if sim_result else None,
            "simulation_history": [
                result.model_dump() for result in (sim_history or [])
            ],
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
        "supporting_papers": h.get("supporting_papers", []),
        "grounding_status": h.get("grounding_status", "ungrounded"),
        "citation_warnings": h.get("citation_warnings", []),
        "non_arxiv_sources": h.get("non_arxiv_sources", []),
        "evidence_trace": h.get("evidence_trace", []),
        "supported_facts": h.get("supported_facts", []),
        "novel_inference": h.get("novel_inference", ""),
        "unsupported_parts": h.get("unsupported_parts", []),
        "evidence_spans": h.get("evidence_spans", []),
        "diagram": h.get("diagram"),
        "scores": h.get("scores", {}),
        "evidence": h.get("evidence", {}),
        "source_soul": h.get("source_soul"),
        "simulation_result": h.get("simulation_result"),
        "simulation_history": h.get("simulation_history", []),
        "iteration": h.get("iteration", 0),
    }


def _paper_abs_link(paper_id: str, metadata: dict[str, Any] | None = None) -> str:
    """Build a best-effort landing link for mixed-source citation IDs."""
    if metadata:
        abs_url = metadata.get("abs_url")
        if abs_url:
            return str(abs_url)

    raw = (paper_id or "").strip()
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered.startswith("pmid:"):
        pmid = raw.split(":", 1)[1].strip()
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if lowered.startswith("pmcid:"):
        pmcid = raw.split(":", 1)[1].strip().upper()
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    if lowered.startswith("doi:"):
        doi = raw.split(":", 1)[1].strip()
        return f"https://doi.org/{doi}"
    if re.match(r"^10\.\d{4,9}/\S+$", raw, flags=re.IGNORECASE):
        return f"https://doi.org/{raw}"
    if lowered.startswith("openalex:"):
        return f"https://openalex.org/{raw.split(':', 1)[1].strip()}"
    if lowered.startswith("s2:"):
        return f"https://www.semanticscholar.org/search?q={quote(raw)}"
    if re.match(r"^pmc\d+$", lowered):
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{raw.upper()}/"
    return f"https://arxiv.org/abs/{raw}"


def _serialize_dialogue(entry: Any) -> dict[str, Any]:
    """Convert a dialogue entry into a soul message."""
    role_value = entry.role.value if hasattr(entry.role, "value") else str(entry.role)
    return {
        "soul": entry.soul,
        "role": role_value,
        "text": entry.message,
        "timestamp": entry.timestamp.isoformat(),
        "highlighted": role_value in ["creative", "risk_taker", "skeptic", "methodical", "synthesizer"],
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

        async def on_knowledge_update(stats: dict[str, Any]):
            if session_id in sessions:
                # Update session-specific stats
                sessions[session_id]["papers_indexed"] = stats.get("papers_indexed", 0)
                sessions[session_id]["knowledge_sources"] = stats.get("sources", [])
                sessions[session_id]["source_breakdown"] = stats.get("source_breakdown", {})
                if "concept_map" in stats:
                    sessions[session_id]["concept_map"] = stats["concept_map"]

                # Update global stats
                knowledge_stats["papers_indexed"] = max(
                    knowledge_stats.get("papers_indexed", 0), stats.get("papers_indexed", 0)
                )
                knowledge_stats["concepts_extracted"] = max(
                    knowledge_stats.get("concepts_extracted", 0), stats.get("concepts_extracted", 0)
                )
                knowledge_stats["relations_found"] = max(
                    knowledge_stats.get("relations_found", 0), stats.get("relations_found", 0)
                )

        orchestrator = RalphOrchestrator(
            config=config,
            callbacks={
                "on_status_change": on_status_change,
                "on_trace": on_trace,
                "on_personas": on_personas,
                "on_knowledge_update": on_knowledge_update,
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

        # Flatten all traces into a single list of messages
        final_messages = []
        for trace in (result.traces if result else []):
            final_messages.extend(_serialize_trace(trace))
        sessions[session_id]["soulMessages"] = final_messages

        sessions[session_id]["relevanceScore"] = (
            result.concept_map and len(result.concept_map.nodes) / 10
        ) or None
        sessions[session_id]["personas"] = orchestrator.persona_roster
        sessions[session_id]["source_metadata"] = {
            key: value.model_dump() if hasattr(value, "model_dump") else value
            for key, value in (result.source_metadata or {}).items()
        }
        sessions[session_id]["knowledge_sources"] = sorted(
            {(paper.get("source") or "unknown") for paper in sessions[session_id]["source_metadata"].values()}
        )
        sessions[session_id]["source_breakdown"] = {
            source: sum(
                1 for paper in sessions[session_id]["source_metadata"].values()
                if (paper.get("source") or "unknown") == source
            )
            for source in sessions[session_id]["knowledge_sources"]
        }

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
        error_message = str(e).strip() or f"{e.__class__.__name__}: {repr(e)}"
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
                    "strict_grounding": session.get("strict_grounding"),
                    "enforce_numeric_citations": session.get("enforce_numeric_citations"),
                    "judge_replay": session.get("judge_replay", False),
                    "replay_source_session_id": session.get("replay_source_session_id"),
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
        summary = await _load_summary_async(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        hypotheses: list[dict[str, Any]] = []
        hypotheses_file = SESSIONS_DIR / session_id / "hypotheses.json"

        if hypotheses_file.exists():
            def load_and_serialize():
                with open(hypotheses_file, encoding="utf-8") as f:
                    return [_serialize_hypothesis(h) for h in json.load(f)]

            try:
                hypotheses = await asyncio.to_thread(load_and_serialize)
            except Exception:
                hypotheses = []

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
            "knowledge_sources": summary.get("knowledge_sources", []),
            "source_breakdown": summary.get("source_breakdown", {}),
            "relevanceScore": None,
            "error": None,
            "created_at": summary.get("started_at"),
            "status_detail": summary.get("status_detail"),
            "constraints": summary.get("constraints"),
            "phase_history": summary.get("phase_history", []),
            "config": summary.get("config"),
            "personas": summary.get("personas"),
            "concept_map": summary.get("concept_map"),
            "strict_grounding": summary.get("strict_grounding"),
            "enforce_numeric_citations": summary.get("enforce_numeric_citations"),
            "judge_replay": summary.get("judge_replay", False),
            "replay_source_session_id": summary.get("replay_source_session_id"),
        }

    # Live session
    # Get live metadata if orchestrator is running
    metadata = session.get("source_metadata", {})
    orchestrator = session.get("orchestrator")
    directives = []
    impact_history = []
    if orchestrator:
        if hasattr(orchestrator, 'paper_store'):
            metadata = {k: v.model_dump() for k, v in orchestrator.paper_store.items()}
        directives = getattr(orchestrator.memory.working, 'pinned_directives', [])
        impact_history = getattr(orchestrator.memory.working, 'directive_impact_history', [])

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
        "knowledge_sources": session.get("knowledge_sources", []),
        "source_breakdown": session.get("source_breakdown", {}),
        "relevanceScore": session.get("relevanceScore"),
        "error": session.get("error"),
        "created_at": session.get("created_at"),
        "status_detail": session.get("status_detail"),
        "constraints": session.get("constraints"),
        "phase_history": session.get("phase_history", []),
        "config": session.get("config"),
        "personas": session.get("personas"),
        "directives": directives,
        "impact_history": impact_history,
        "concept_map": session.get("concept_map"),
        "strict_grounding": session.get("strict_grounding"),
        "enforce_numeric_citations": session.get("enforce_numeric_citations"),
        "judge_replay": session.get("judge_replay", False),
        "replay_source_session_id": session.get("replay_source_session_id"),
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
    summary = await _load_summary_async(session_id)
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
        strict_grounding=config_data.get("strict_grounding", True),
        enforce_numeric_citations=config_data.get("enforce_numeric_citations", True),
        srsh_enabled=config_data.get("srsh_enabled", True),
        srsh_agents=config_data.get("srsh_agents", 3),
        srsh_iterations_per_agent=config_data.get("srsh_iterations_per_agent", 3),
        srsh_collisions=config_data.get("srsh_collisions", 2),
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


@app.post("/api/sessions/{session_id}/chat/pin")
async def pin_directive(session_id: str, request: ChatRequest):
    """Pin a directive."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.pin_directive(request.message)
    return {"status": "pinned"}


@app.post("/api/sessions/{session_id}/chat/unpin")
async def unpin_directive(session_id: str, request: ChatRequest):
    """Unpin a directive."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.unpin_directive(request.message)
    return {"status": "unpinned"}


@app.post("/api/sessions/{session_id}/personas/{persona_id}/lock")
async def lock_persona(session_id: str, persona_id: str):
    """Lock a persona."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.lock_persona(persona_id)
    return {"status": "locked"}


@app.post("/api/sessions/{session_id}/personas/{persona_id}/unlock")
async def unlock_persona(session_id: str, persona_id: str):
    """Unlock a persona."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.unlock_persona(persona_id)
    return {"status": "unlocked"}


@app.post("/api/sessions/{session_id}/personas/{persona_id}/weight")
async def update_persona_weight(session_id: str, persona_id: str, request: PersonaWeightRequest):
    """Update persona weight."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.update_persona_weight(persona_id, request.weight)
    return {"status": "weight_updated"}


@app.post("/api/sessions/{session_id}/personas/{persona_id}/regenerate")
async def regenerate_persona(session_id: str, persona_id: str):
    """Regenerate a persona."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.regenerate_persona(persona_id)
    return {"status": "regenerated"}


@app.post("/api/sessions/{session_id}/hypotheses/{hypothesis_id}/vote")
async def vote_hypothesis(session_id: str, hypothesis_id: str, request: dict[str, Any]):
    """Vote on a hypothesis (up/down)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    direction = request.get("direction", "up")
    await orchestrator.vote_hypothesis(hypothesis_id, direction)
    return {"status": "voted"}


@app.post("/api/sessions/{session_id}/hypotheses/{hypothesis_id}/investigate")
async def investigate_hypothesis(session_id: str, hypothesis_id: str):
    """Mark a hypothesis for deeper investigation."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    await orchestrator.investigate_hypothesis(hypothesis_id)
    return {"status": "investigation_queued"}


@app.post("/api/sessions/{session_id}/hypotheses/{hypothesis_id}/rerun")
async def rerun_simulation(session_id: str, hypothesis_id: str, request: dict[str, Any] = None):
    """Rerun simulation for a hypothesis."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    custom_code = request.get("code") if request else None
    await orchestrator.rerun_simulation(hypothesis_id, custom_code)
    return {"status": "rerun_queued"}


# ═══════════════════════════════════════════════════════════════
# SRSH Experimental Endpoint
# ═══════════════════════════════════════════════════════════════

class SRSHRequest(BaseModel):
    """Request for SRSH experimental mode."""
    topic: str
    n_agents: int = Field(default=3, ge=2, le=5)
    iterations_per_agent: int = Field(default=3, ge=1, le=10)
    n_collisions: int = Field(default=2, ge=1, le=5)
    search_limit: int = Field(default=5, ge=0, le=20, description="Papers to fetch (0 for abstract mode)")


@app.get("/api/replay/sessions")
async def list_replay_sessions(limit: int = 20):
    """List archived complete sessions available for judge replay mode."""
    return {"sessions": _list_replay_candidates(limit=limit)}


@app.post("/api/experimental/srsh")
async def run_srsh_experiment(request: SRSHRequest):
    """
    Run Stress-Responsive Semantic Hypermutation experiment.
    
    This is an experimental endpoint for testing cellular SOS-inspired
    hypothesis generation.
    """
    try:
        _require_api_keys()

        # 1. Initialize ConceptMap with topic node
        concept_map = ConceptMap(nodes=[ConceptNode(id="topic", name=request.topic, category="topic")])
        claims = []

        # 2. Grounding: Fetch real papers if requested
        if request.search_limit > 0:
            print(f"SRSH Grounding: Searching for {request.search_limit} papers on '{request.topic}'...")
            async with ArxivClient() as client:
                papers = await client.search(request.topic, max_results=request.search_limit)

            print(f"SRSH Grounding: Found {len(papers)} papers. Extracting claims...")
            extractor = ClaimExtractor()

            # Simple parallel extraction limited by client concurrency mostly, but loop is fine for 5 papers
            for paper in papers:
                try:
                    paper_claims = await extractor.extract_claims(
                        paper_id=paper.arxiv_id,
                        title=paper.title,
                        abstract=paper.abstract
                    )
                    claims.extend(paper_claims)

                    # Populate ConceptMap nodes from entities
                    for claim in paper_claims:
                        for entity in claim.entities_mentioned:
                            node_id = entity.strip().lower().replace(" ", "_")
                            # Add if unique
                            if not any(n.id == node_id for n in concept_map.nodes):
                                concept_map.nodes.append(ConceptNode(
                                    id=node_id,
                                    name=entity,
                                    category=claim.claim_type.value if hasattr(claim.claim_type, 'value') else "extracted"
                                ))
                except Exception as e:
                    print(f"Error processing paper {paper.arxiv_id}: {e}")

            print(f"SRSH Grounding: Extracted {len(claims)} claims and {len(concept_map.nodes)} concepts.")

        from src.kb.grounded_generator import GroundedHypothesisGenerator
        from src.soul.srsh_orchestrator import SRSHConfig, SRSHOrchestrator

        # Helper classes imported

        # Create generator and SRSH orchestrator
        generator = GroundedHypothesisGenerator()
        config = SRSHConfig(
            enabled=True,
            n_agents=request.n_agents,
            iterations_per_agent=request.iterations_per_agent,
            n_collisions=request.n_collisions,
        )

        status_updates = []
        def status_callback(msg: str):
            status_updates.append({"timestamp": datetime.now().isoformat(), "message": msg})

        srsh = SRSHOrchestrator(
            generator=generator,
            config=config,
            status_callback=status_callback,
        )

        result = await srsh.run(
            topic=request.topic,
            concept_map=concept_map,
            claims=[],  # No pre-existing claims
        )

        return {
            "status": "complete",
            "topic": request.topic,
            "metrics": result.metrics,
            "status_updates": status_updates,
            "stress_zones": [
                {
                    "type": z.stress_type.value if hasattr(z.stress_type, 'value') else str(z.stress_type),
                    "concepts": z.concepts,
                    "intensity": z.intensity,
                    "description": z.description,
                }
                for z in result.stress_zones
            ],
            "hypotheses": [
                {
                    "id": h.id,
                    "statement": h.hypothesis,
                    "rationale": h.rationale,
                    "source": h.cross_disciplinary_connection,
                    "novelty_keywords": h.novelty_keywords,
                    "scores": h.scores.model_dump() if h.scores else {},
                }
                for h in result.hypotheses[:10]  # Top 10
            ],
            "collision_narratives": [
                {
                    "narrative": c.collision_narrative,
                    "bridged_domains": c.bridged_domains,
                    "source_agents": c.source_agents,
                }
                for batch in result.collision_batches
                for c in batch.collisions
            ],
        }
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, format: str = "json"):
    """Export session data in specified format."""
    # Try to get from live sessions first, then from disk
    session = sessions.get(session_id)
    summary = None
    hypotheses = []
    metadata: dict[str, Any] = {}

    if session:
        summary = {
            "id": session_id,
            "topic": session["topic"],
            "status": session["status"],
            "created_at": session["created_at"],
            "constraints": session.get("constraints"),
            "strict_grounding": session.get("strict_grounding"),
            "enforce_numeric_citations": session.get("enforce_numeric_citations"),
            "judge_replay": session.get("judge_replay", False),
            "replay_source_session_id": session.get("replay_source_session_id"),
            "knowledge_sources": session.get("knowledge_sources", []),
        }
        hypotheses = session.get("hypotheses", [])
        metadata = session.get("source_metadata", {}) or {}
    else:
        summary = await _load_summary_async(session_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Session not found")

        hypotheses_file = SESSIONS_DIR / session_id / "hypotheses.json"

        if hypotheses_file.exists():
            def load_and_serialize():
                with open(hypotheses_file, encoding="utf-8") as f:
                    return [_serialize_hypothesis(h) for h in json.load(f)]

            try:
                hypotheses = await asyncio.to_thread(load_and_serialize)
            except Exception:
                hypotheses = []

    if format == "json":
        return {"summary": summary, "hypotheses": hypotheses}

    if format == "markdown":
        md = f"# Research Report: {summary.get('topic')}\n\n"
        md += f"Session ID: {session_id}\n"
        created_at = summary.get("created_at") or summary.get("started_at")
        md += f"Date: {created_at}\n\n"
        md += "## Grounding Policy\n\n"
        md += f"- Strict grounding: {summary.get('strict_grounding')}\n"
        md += f"- Numeric citation enforcement: {summary.get('enforce_numeric_citations')}\n"
        if summary.get("knowledge_sources"):
            md += f"- Knowledge sources: {', '.join(summary.get('knowledge_sources', []))}\n"
        if summary.get("judge_replay"):
            md += f"- Replay source: {summary.get('replay_source_session_id')}\n"
        md += "\n"

        md += "## Ranked Hypotheses\n\n"
        for i, h in enumerate(hypotheses):
            md += f"### {i+1}. {h.get('statement')}\n"
            md += f"**Rationale:** {h.get('rationale')}\n\n"
            md += f"**Novelty Keywords:** {', '.join(h.get('novelty_keywords', []))}\n\n"
            papers = h.get("supporting_papers", [])
            if papers:
                md += "**Direct Citations:**\n"
                for paper_id in papers:
                    paper_meta = metadata.get(paper_id, {})
                    link = _paper_abs_link(paper_id, paper_meta)
                    label = paper_meta.get("title") or paper_id
                    md += f"- [{label}]({link})\n"
                md += "\n"
            else:
                md += "**Direct Citations:** None linked\n\n"

            warnings = h.get("citation_warnings", [])
            if warnings:
                md += "**Citation Warnings:**\n"
                for warning in warnings:
                    md += f"- {warning}\n"
                md += "\n"

            trace = h.get("evidence_trace", [])
            if trace:
                md += "**Evidence Trace:**\n"
                for line in trace[:6]:
                    md += f"- {line}\n"
                md += "\n"

            supported_facts = h.get("supported_facts", [])
            if supported_facts:
                md += "**Supported Facts:**\n"
                for fact in supported_facts[:8]:
                    md += f"- {fact}\n"
                md += "\n"

            novel_inference = (h.get("novel_inference") or "").strip()
            if novel_inference:
                md += f"**Novel Inference:** {novel_inference}\n\n"

            unsupported_parts = h.get("unsupported_parts", [])
            if unsupported_parts:
                md += "**Unsupported Parts:**\n"
                for part in unsupported_parts[:8]:
                    md += f"- {part}\n"
                md += "\n"

            if h.get('simulation_result'):
                res = h['simulation_result']
                md += f"**Simulation Verdict:** {'SUCCESS' if res.get('success') and res.get('supports_hypothesis') else 'FAILURE'}\n"
                if res.get('vision_commentary'):
                    md += f"\n*Vision Analysis:* {res['vision_commentary']}\n"
            md += "\n---\n\n"

        return {"content": md}

    raise HTTPException(status_code=400, detail="Unsupported format")


@app.post("/api/sessions/{session_id}/hypotheses/{hypothesis_id}/graveyard")
async def bury_hypothesis(session_id: str, hypothesis_id: str, request: dict[str, Any]):
    """Push a hypothesis into the Graveyard memory store."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    orchestrator = sessions[session_id].get("orchestrator")
    if not orchestrator:
        raise HTTPException(status_code=400, detail="Orchestrator not active")

    # Find the hypothesis
    target = next((h for h in orchestrator.hypotheses if h.id == hypothesis_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="Hypothesis not found")

    reason = request.get("reason", "Manual burial by researcher")
    orchestrator.memory.graveyard.bury(target.hypothesis, reason, orchestrator.topic)

    return {"status": "buried"}


@app.post("/api/sessions/{session_id}/retry")
async def retry_phase(session_id: str, phase: str = None):
    """Retry a failed session phase."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Only allow retry if error or complete
    if not session.get("error") and not session.get("complete"):
        raise HTTPException(status_code=400, detail="Session is still running")

    # Reset error state
    session["status"] = "running"
    session["error"] = None
    session["complete"] = False

    orchestrator = session.get("orchestrator")
    if not orchestrator:
        # If orchestrator is gone (e.g. restart), we can't easily resume state in memory
        # For now, we only support retries if the orchestrator instance is alive in memory
        # Or if we implement full state rehydration (which is complex).
        # We'll fail gracefully.
        raise HTTPException(status_code=400, detail="Cannot retry: Session state lost (server restarted?)")

    # Determine phase to retry
    target_phase = phase or session.get("phase")

    # Logic to restart the loop from the current state
    # This is simplified: we just clear the error and let the loop continue if it was paused
    # But orchestrator.run() loop might have exited.

    # We need to re-trigger the loop.
    # The orchestrator.run() has a while True loop. If it broke due to exception, we need to call run again?
    # BUT run() initializes everything.

    # We will trigger a new background task that calls _run_iteration loop directly?
    # Or better, we call run() again but pass the existing state?
    # Orchestrator doesn't support re-entry easily yet.

    # MVP approach: We'll just set status to "running" and if the loop was stuck in a retryable error,
    # we might need to manually trigger the next step.

    # Actually, US-108 implies we should be able to retry specific failed actions.
    # Let's just restart the main loop if it stopped.

    task = asyncio.create_task(run_session(
        session_id,
        session["topic"],
        RalphConfig(**session["config"]),
        SessionConstraints(**session["constraints"]) if session.get("constraints") else None
    ))
    running_tasks[session_id] = task

    return {"status": "retrying", "phase": target_phase}


@app.get("/api/sessions/{session_id}/logs")
async def get_session_logs(session_id: str):
    """Get error logs for a session."""
    log_path = SESSIONS_DIR / session_id / "error.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="No logs found")

    with open(log_path, encoding="utf-8") as f:
        return {"content": f.read()}


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
            "strict_grounding": session.get("strict_grounding"),
            "enforce_numeric_citations": session.get("enforce_numeric_citations"),
            "knowledge_sources": session.get("knowledge_sources", []),
            "source_breakdown": session.get("source_breakdown", {}),
            "judge_replay": session.get("judge_replay", False),
            "replay_source_session_id": session.get("replay_source_session_id"),
        })

    # On-disk sessions
    if SESSIONS_DIR.exists():
        for session_dir in sorted(SESSIONS_DIR.iterdir(), reverse=True)[:limit]:
            if session_dir.is_dir() and session_dir.name not in sessions:
                summary = await _load_summary_async(session_dir.name) or {}
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
                    "strict_grounding": summary.get("strict_grounding"),
                    "enforce_numeric_citations": summary.get("enforce_numeric_citations"),
                    "knowledge_sources": summary.get("knowledge_sources", []),
                    "source_breakdown": summary.get("source_breakdown", {}),
                    "judge_replay": summary.get("judge_replay", False),
                    "replay_source_session_id": summary.get("replay_source_session_id"),
                })

    return all_sessions[:limit]


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """Get knowledge base statistics."""
    return knowledge_stats


# ═══════════════════════════════════════════════════════════════
# Feedback Learning API
# ═══════════════════════════════════════════════════════════════

from src.soul.feedback import get_feedback_store


class FeedbackRequest(BaseModel):
    """Request to record feedback on a hypothesis."""
    hypothesis_id: str
    action: str  # accept, reject, star, edit
    context: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/sessions/{session_id}/feedback")
async def record_feedback(session_id: str, request: FeedbackRequest):
    """Record user feedback on a hypothesis.
    
    Actions:
    - accept: User explicitly approves/uses the hypothesis
    - reject: User dismisses the hypothesis
    - star: User marks as particularly interesting
    - edit: User modified the hypothesis
    """
    feedback_store = get_feedback_store()

    try:
        feedback_store.record(
            hypothesis_id=request.hypothesis_id,
            action=request.action,
            context=request.context,
            session_id=session_id,
        )
        return {"status": "recorded", "action": request.action}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/sessions/{session_id}/preferences")
async def get_preferences(session_id: str):
    """Get learned preference weights from user feedback."""
    feedback_store = get_feedback_store()
    return {
        "weights": feedback_store.get_preference_weights(),
        "stats": feedback_store.get_feedback_stats(),
    }


@app.get("/api/feedback/stats")
async def get_global_feedback_stats():
    """Get global feedback statistics across all sessions."""
    feedback_store = get_feedback_store()
    return {
        "stats": feedback_store.get_feedback_stats(),
        "recent": feedback_store.get_recent_feedback(limit=10),
    }


# ═══════════════════════════════════════════════════════════════
# Real-Time Collaboration (WebSocket)
# ═══════════════════════════════════════════════════════════════

from fastapi import WebSocket, WebSocketDisconnect


class ConnectionManager:
    """Manages WebSocket connections for real-time collaboration."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}
        self.user_presence: dict[str, dict[str, str]] = {}  # session_id -> {ws_id: username}

    async def connect(self, websocket: WebSocket, session_id: str, username: str = "Anonymous"):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
            self.user_presence[session_id] = {}

        self.active_connections[session_id].append(websocket)
        ws_id = str(id(websocket))
        self.user_presence[session_id][ws_id] = username

        # Broadcast user joined
        await self.broadcast(session_id, {
            "type": "user_joined",
            "username": username,
            "users": list(self.user_presence.get(session_id, {}).values()),
        })

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            try:
                self.active_connections[session_id].remove(websocket)
            except ValueError:
                pass

            ws_id = str(id(websocket))
            username = self.user_presence.get(session_id, {}).pop(ws_id, "Anonymous")

            if not self.active_connections[session_id]:
                del self.active_connections[session_id]
                self.user_presence.pop(session_id, None)

            return username
        return "Anonymous"

    async def broadcast(self, session_id: str, message: dict[str, Any]):
        """Broadcast a message to all connections for a session."""
        if session_id not in self.active_connections:
            return

        dead_connections = []
        for connection in self.active_connections[session_id]:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        # Clean up dead connections
        for dead in dead_connections:
            self.disconnect(dead, session_id)

    def get_users(self, session_id: str) -> list[str]:
        """Get list of users in a session."""
        return list(self.user_presence.get(session_id, {}).values())


# Global connection manager
ws_manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str, username: str = "Anonymous"):
    """WebSocket endpoint for real-time session updates."""
    await ws_manager.connect(websocket, session_id, username)

    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()

            # Handle different message types
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            elif msg_type == "cursor":
                # User cursor position (for collaborative editing)
                await ws_manager.broadcast(session_id, {
                    "type": "cursor",
                    "username": username,
                    "position": data.get("position"),
                })

            elif msg_type == "chat":
                # Collaborative chat message
                await ws_manager.broadcast(session_id, {
                    "type": "chat",
                    "username": username,
                    "message": data.get("message", ""),
                    "timestamp": datetime.now().isoformat(),
                })

            else:
                # Echo unknown messages back for debugging
                await websocket.send_json({"type": "echo", "data": data})

    except WebSocketDisconnect:
        username = ws_manager.disconnect(websocket, session_id)
        await ws_manager.broadcast(session_id, {
            "type": "user_left",
            "username": username,
            "users": ws_manager.get_users(session_id),
        })


@app.get("/api/sessions/{session_id}/users")
async def get_session_users(session_id: str):
    """Get list of users connected to a session."""
    return {"users": ws_manager.get_users(session_id)}


# Helper function to broadcast session updates (call from run_session)
async def broadcast_session_update(session_id: str, update_type: str, data: dict[str, Any]):
    """Broadcast a session update to all connected clients."""
    await ws_manager.broadcast(session_id, {
        "type": update_type,
        "timestamp": datetime.now().isoformat(),
        **data,
    })


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
