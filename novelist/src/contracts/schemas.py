"""Core Pydantic schemas for the hypothesis synthesizer.

These schemas define the data contracts for:
- Hypotheses and their components (evidence, scores)
- BDI (Beliefs, Desires, Intentions) state
- Iteration traces for debugging and visualization
- arXiv papers and concept maps
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class SoulRole(str, Enum):
    """Roles for the multi-soul collective."""

    CREATIVE = "creative"
    SKEPTIC = "skeptic"
    METHODICAL = "methodical"
    RISK_TAKER = "risk_taker"
    SYNTHESIZER = "synthesizer"


class CritiqueVerdict(str, Enum):
    """Verdict from the Skeptic soul's critique."""

    FATAL = "fatal"  # Hypothesis should be removed
    MODERATE = "moderate"  # Needs significant revision
    MINOR = "minor"  # Small improvements needed
    PASS = "pass"  # Good to go


class GenerationMode(str, Enum):
    """Modes for hypothesis generation."""

    GAP_HUNT = "gap_hunt"  # Find gaps in the concept map
    CONTRADICTION_HUNT = "contradiction_hunt"  # Find contradictions
    ANALOGY_TRANSFER = "analogy_transfer"  # Cross-domain analogies
    CONSTRAINT_RELAX = "constraint_relax"  # Invert assumptions
    RANDOM_INJECTION = "random_injection"  # Inject random domain


# =============================================================================
# Evidence and Scoring
# =============================================================================


class EvidenceBlock(BaseModel):
    """Evidence supporting novelty claims for a hypothesis."""

    arxiv_hits: int = Field(default=0, ge=0, description="Number of arXiv search hits")
    closest_arxiv_titles: list[str] = Field(
        default_factory=list, description="Titles of closest matching arXiv papers"
    )
    pubmed_hits: int = Field(default=0, ge=0, description="Number of PubMed search hits")
    arxiv_query: str = Field(default="", description="Query used for arXiv search")


class ScoreBlock(BaseModel):
    """Scores for a hypothesis across multiple dimensions."""

    novelty: float = Field(default=0.0, ge=0.0, le=1.0, description="Novelty score (0-1)")
    feasibility: float = Field(default=0.0, ge=0.0, le=1.0, description="Feasibility score (0-1)")
    impact: float = Field(default=0.0, ge=0.0, le=1.0, description="Impact score (0-1)")
    cross_domain: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Cross-domain connection score (0-1)"
    )

    @property
    def aggregate(self) -> float:
        """Calculate aggregate score as weighted average."""
        weights = {"novelty": 0.3, "feasibility": 0.25, "impact": 0.25, "cross_domain": 0.2}
        return sum(getattr(self, k) * v for k, v in weights.items())


# =============================================================================
# Hypothesis
# =============================================================================


class Hypothesis(BaseModel):
    """A scientific hypothesis with full metadata."""

    id: str = Field(default="", description="Unique identifier for this hypothesis")
    hypothesis: str = Field(..., min_length=10, description="One clear sentence hypothesis")
    rationale: str = Field(..., description="Why it could be true + why it's interesting")
    cross_disciplinary_connection: str = Field(
        ..., description="Fields connected and how they relate"
    )
    experimental_design: list[str] = Field(
        ..., min_length=1, description="Step-by-step experimental protocol"
    )
    expected_impact: str = Field(..., description="What changes if this hypothesis is true")
    novelty_keywords: list[str] = Field(
        ..., min_length=1, description="Keywords for novelty search"
    )
    evidence: EvidenceBlock = Field(default_factory=EvidenceBlock)
    scores: ScoreBlock = Field(default_factory=ScoreBlock)

    # Metadata
    source_soul: SoulRole | None = Field(default=None, description="Which soul generated this")
    iteration: int = Field(default=0, ge=0, description="Which iteration this was generated in")
    created_at: datetime = Field(default_factory=datetime.now)

    def meets_thresholds(
        self,
        novelty: float = 0.7,
        feasibility: float = 0.6,
        impact: float = 0.5,
    ) -> bool:
        """Check if hypothesis meets quality thresholds."""
        return (
            self.scores.novelty >= novelty
            and self.scores.feasibility >= feasibility
            and self.scores.impact >= impact
        )


class Critique(BaseModel):
    """Critique of a hypothesis from the Skeptic soul."""

    hypothesis_id: str = Field(..., description="ID of the hypothesis being critiqued")
    verdict: CritiqueVerdict = Field(..., description="Overall verdict")
    issues: list[str] = Field(default_factory=list, description="List of identified issues")
    suggestions: list[str] = Field(
        default_factory=list, description="Specific improvement suggestions"
    )
    principle_violations: list[str] = Field(
        default_factory=list, description="Which constitutional principles were violated"
    )


# =============================================================================
# BDI State
# =============================================================================


class BeliefStore(BaseModel):
    """Beliefs about the current state of the world."""

    topic: str = Field(default="", description="Research topic/query")
    domain_tags: list[str] = Field(default_factory=list, description="Detected domain tags")
    papers_ingested: int = Field(default=0, ge=0)
    concept_map_nodes: int = Field(default=0, ge=0)
    concept_map_edges: int = Field(default=0, ge=0)
    identified_gaps: int = Field(default=0, ge=0)
    last_novelty_scores: list[float] = Field(default_factory=list)
    last_feasibility_scores: list[float] = Field(default_factory=list)
    stagnation_count: int = Field(default=0, ge=0, description="Iterations without improvement")
    known_failure_modes: list[str] = Field(
        default_factory=list, description="Patterns that have failed"
    )


class DesireSet(BaseModel):
    """Desired goals and thresholds."""

    target_novelty: float = Field(default=0.7, ge=0.0, le=1.0)
    target_feasibility: float = Field(default=0.6, ge=0.0, le=1.0)
    target_impact: float = Field(default=0.5, ge=0.0, le=1.0)
    target_cross_domain: float = Field(default=0.5, ge=0.0, le=1.0)
    min_hypotheses: int = Field(default=10, ge=1)
    max_hypotheses: int = Field(default=15, ge=1)

    def check_met(self, scores: ScoreBlock, hypothesis_count: int) -> dict[str, bool]:
        """Check which desires are currently met."""
        return {
            "novelty": scores.novelty >= self.target_novelty,
            "feasibility": scores.feasibility >= self.target_feasibility,
            "impact": scores.impact >= self.target_impact,
            "cross_domain": scores.cross_domain >= self.target_cross_domain,
            "hypothesis_count": hypothesis_count >= self.min_hypotheses,
        }


class BDIState(BaseModel):
    """Complete BDI (Beliefs, Desires, Intentions) state."""

    beliefs: BeliefStore = Field(default_factory=BeliefStore)
    desires: DesireSet = Field(default_factory=DesireSet)
    intentions: list[str] = Field(
        default_factory=lambda: ["ingest_papers", "extract_concepts", "generate", "debate", "verify"]
    )
    current_mode: GenerationMode = Field(default=GenerationMode.GAP_HUNT)
    exhausted_modes: list[GenerationMode] = Field(default_factory=list)


# =============================================================================
# Iteration Trace
# =============================================================================


class IterationTrace(BaseModel):
    """Trace of a single Ralph loop iteration for debugging and visualization."""

    iteration: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    thought: str = Field(..., description="BDI deliberation reasoning")
    action: str = Field(..., description="What action was taken")
    observation: str = Field(..., description="What was observed after action")
    bdi_snapshot: BDIState = Field(..., description="BDI state at end of iteration")
    hypotheses_generated: int = Field(default=0, ge=0)
    hypotheses_surviving: int = Field(default=0, ge=0)
    avg_novelty: float = Field(default=0.0, ge=0.0, le=1.0)
    avg_feasibility: float = Field(default=0.0, ge=0.0, le=1.0)
    mode_used: GenerationMode = Field(default=GenerationMode.GAP_HUNT)
    tokens_used: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


# =============================================================================
# arXiv and Knowledge Base
# =============================================================================


class ArxivPaper(BaseModel):
    """Structured representation of an arXiv paper."""

    arxiv_id: str = Field(..., description="arXiv identifier (e.g., '2301.12345')")
    title: str = Field(...)
    abstract: str = Field(...)
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = Field(default=None)
    updated: datetime | None = Field(default=None)
    pdf_url: str = Field(default="")
    abs_url: str = Field(default="")


class PaperSummary(BaseModel):
    """Structured summary of a paper generated by LLM."""

    arxiv_id: str = Field(...)
    problem: str = Field(..., description="What problem does this address?")
    method: str = Field(..., description="What method/approach is used?")
    key_result: str = Field(..., description="What is the main result?")
    limitations: str = Field(default="", description="What are the limitations?")
    future_work: str = Field(default="", description="What future work is suggested?")
    extracted_entities: list[str] = Field(
        default_factory=list, description="Key entities (methods, datasets, etc.)"
    )


# =============================================================================
# Literature-First Pipeline: Structured Claims
# =============================================================================


class ClaimType(str, Enum):
    """Types of claims that can be extracted from papers."""

    RESULT = "result"  # Empirical finding
    METHOD = "method"  # Technique or approach used
    LIMITATION = "limitation"  # Known constraints or failures
    FUTURE_WORK = "future_work"  # Suggested next steps
    BASELINE = "baseline"  # Current state-of-art numbers


class QuantitativeData(BaseModel):
    """Quantitative data from a paper claim."""

    metric: str = Field(..., description="What is being measured (e.g., cycle_life, accuracy)")
    value: float = Field(..., description="Numeric value")
    unit: str = Field(default="", description="Unit of measurement")
    conditions: str = Field(default="", description="Under what conditions")
    is_sota: bool = Field(default=False, description="Is this state-of-the-art?")


class ExtractedClaim(BaseModel):
    """A structured claim extracted from a research paper."""

    paper_id: str = Field(..., description="arXiv ID of source paper")
    claim_type: ClaimType = Field(..., description="Type of claim")
    statement: str = Field(..., min_length=10, description="The claim in plain language")
    evidence: str = Field(default="", description="How the paper supports this claim")
    quantitative_data: QuantitativeData | None = Field(
        default=None, description="Numeric data if applicable"
    )
    entities_mentioned: list[str] = Field(
        default_factory=list, description="Key entities (compounds, methods, etc.)"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Extraction confidence"
    )


# =============================================================================
# Literature-First Pipeline: Gap Identification
# =============================================================================


class GapType(str, Enum):
    """Types of gaps that can be identified in the literature."""

    MISSING_CONNECTION = "missing_connection"  # Concepts not yet linked
    CONTRADICTION = "contradiction"  # Conflicting claims
    UNEXPLORED_RANGE = "unexplored_range"  # Parameter tested in narrow range
    CROSS_DOMAIN = "cross_domain"  # Method from field A, not tried in B
    SCALE_GAP = "scale_gap"  # Works in lab, not tested at scale
    MECHANISM_UNKNOWN = "mechanism_unknown"  # Effect observed, mechanism unclear


class IdentifiedGap(BaseModel):
    """A gap in the literature that could be addressed by research."""

    gap_type: GapType = Field(..., description="Category of gap")
    description: str = Field(..., min_length=20, description="Clear statement of the gap")
    concept_a: str = Field(..., description="First concept involved")
    concept_b: str = Field(default="", description="Second concept if applicable")
    supporting_evidence: list[str] = Field(
        default_factory=list, description="Paper IDs showing the gap exists"
    )
    potential_value: str = Field(..., description="Why filling this gap matters")
    difficulty: str = Field(
        default="medium", description="Estimated difficulty: low, medium, high"
    )
    related_claims: list[str] = Field(
        default_factory=list, description="IDs of claims related to this gap"
    )


# =============================================================================
# Literature-First Pipeline: Grounded Hypothesis
# =============================================================================


class MechanismStep(BaseModel):
    """A single step in a causal mechanism chain."""

    cause: str = Field(..., description="The cause in this step")
    effect: str = Field(..., description="The effect in this step")
    evidence_paper: str = Field(default="", description="Paper ID supporting this step")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionBounds(BaseModel):
    """Quantitative bounds for a hypothesis prediction."""

    metric: str = Field(..., description="What is being predicted")
    lower_bound: float = Field(..., description="Minimum expected value")
    upper_bound: float = Field(..., description="Maximum expected value")
    unit: str = Field(default="", description="Unit of measurement")
    baseline_value: float = Field(..., description="Current state-of-art value")
    baseline_source: str = Field(default="", description="Paper ID for baseline")


class SuggestedExperiment(BaseModel):
    """A concrete experiment to test the hypothesis."""

    description: str = Field(..., description="What experiment to run")
    controls: list[str] = Field(default_factory=list, description="Control conditions")
    measurements: list[str] = Field(default_factory=list, description="What to measure")
    expected_timeline: str = Field(default="", description="Estimated time to complete")
    required_resources: list[str] = Field(
        default_factory=list, description="Equipment, materials, etc."
    )


class GroundedHypothesis(BaseModel):
    """A hypothesis grounded in literature with full justification."""

    id: str = Field(default="", description="Unique identifier")
    
    # Core claim (falsifiable)
    claim: str = Field(
        ..., min_length=20,
        description="If X, then Y, because Z — one falsifiable sentence"
    )
    
    # Mechanism chain (required)
    mechanism: list[MechanismStep] = Field(
        ..., min_length=1, description="Causal chain: A→B→C"
    )
    
    # Quantitative prediction
    prediction: str = Field(..., description="What we expect to observe")
    prediction_bounds: PredictionBounds | None = Field(
        default=None, description="Quantitative prediction if applicable"
    )
    
    # Falsification
    null_result: str = Field(
        ..., description="What observation would reject this hypothesis"
    )
    
    # Grounding
    gap_addressed: str = Field(..., description="Which gap this hypothesis fills")
    supporting_papers: list[str] = Field(
        default_factory=list, description="Paper IDs supporting the mechanism"
    )
    contradicting_papers: list[str] = Field(
        default_factory=list, description="Paper IDs that might challenge this"
    )
    
    # Experiment
    suggested_experiments: list[SuggestedExperiment] = Field(
        default_factory=list, description="How to test this hypothesis"
    )
    
    # Scores and metadata
    scores: ScoreBlock = Field(default_factory=ScoreBlock)
    source_soul: SoulRole | None = Field(default=None)
    iteration: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)

    def is_well_formed(self) -> bool:
        """Check if hypothesis has all required components."""
        return (
            len(self.claim) >= 20
            and len(self.mechanism) >= 1
            and len(self.null_result) >= 10
            and len(self.gap_addressed) >= 5
        )


class ConceptNode(BaseModel):
    """A node in the concept map."""

    id: str = Field(...)
    name: str = Field(...)
    type: str = Field(..., description="method, dataset, metric, entity, etc.")
    frequency: int = Field(default=1, ge=1, description="How often this appears")
    source_papers: list[str] = Field(default_factory=list, description="arXiv IDs")


class ConceptEdge(BaseModel):
    """An edge in the concept map."""

    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    relation: str = Field(..., description="Relationship type (improves, uses, etc.)")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_papers: list[str] = Field(default_factory=list)


class ConceptMap(BaseModel):
    """Complete concept map extracted from papers."""

    nodes: list[ConceptNode] = Field(default_factory=list)
    edges: list[ConceptEdge] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(
        default_factory=list, description="Identified gaps (missing edges)"
    )
    contradictions: list[dict[str, Any]] = Field(
        default_factory=list, description="Conflicting edges"
    )


# =============================================================================
# Session and Configuration
# =============================================================================


class RalphConfig(BaseModel):
    """Configuration for a Ralph loop session."""

    max_iterations: int = Field(default=20, ge=1, le=100)
    max_cost_usd: float = Field(default=5.0, ge=0.0)
    max_runtime_seconds: int = Field(default=600, ge=60)
    stagnation_threshold: int = Field(default=3, ge=1)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    target_novelty: float = Field(default=0.7, ge=0.0, le=1.0)
    target_feasibility: float = Field(default=0.6, ge=0.0, le=1.0)
    min_hypotheses: int = Field(default=10, ge=1)
    max_hypotheses: int = Field(default=15, ge=1)
    flash_model: str = Field(default="gemini-2.0-flash")
    pro_model: str = Field(default="gemini-2.0-flash")


class SessionResult(BaseModel):
    """Final result of a hypothesis generation session."""

    session_id: str = Field(...)
    topic: str = Field(...)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: datetime | None = Field(default=None)
    iterations_completed: int = Field(default=0, ge=0)
    stop_reason: str = Field(default="")
    final_hypotheses: list[Hypothesis] = Field(default_factory=list)
    traces: list[IterationTrace] = Field(default_factory=list)
    total_tokens_used: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    papers_ingested: int = Field(default=0, ge=0)
    concept_map: ConceptMap | None = Field(default=None)
