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

from pydantic import AnyHttpUrl, BaseModel, Field

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


class SessionPhase(str, Enum):
    """Deterministic lifecycle states for a session."""

    QUEUED = "queued"
    FORGING = "forging"
    MAPPING = "mapping"
    DEBATING = "debating"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    ERROR = "error"


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


class EvidenceSpan(BaseModel):
    """Structured evidence link for one claim fragment."""

    claim_text: str = Field(..., min_length=5, description="Claim sentence/fragment being supported")
    citation_id: str = Field(..., min_length=3, description="Supporting paper ID")
    quote: str = Field(..., min_length=12, description="Quoted/supporting span from paper title/abstract")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Model confidence in this linkage")




# =============================================================================
# Hypothesis
# =============================================================================


class SimulationResult(BaseModel):
    """Result of an in-silico verification simulation."""

    code: str = Field(..., description="The Python code generated for the simulation")
    success: bool = Field(..., description="Whether the simulation ran without errors")
    supports_hypothesis: bool = Field(..., description="Whether the simulation result supports the claim")
    output_log: str = Field(default="", description="Stdout/Stderr from the execution")
    plot_path: str | None = Field(default=None, description="Path to generated plot image")
    metrics: dict[str, float] = Field(default_factory=dict, description="Key metrics from the simulation")
    vision_commentary: str | None = Field(default=None, description="Gemini Vision analysis of the plot")
    status: str = Field(default="complete", description="queued, running, complete, error")
    timestamp: datetime = Field(default_factory=datetime.now)

    # Retry tracking for simulation reliability
    retry_count: int = Field(default=0, ge=0, description="Number of retry attempts")
    validation_errors: list[str] = Field(default_factory=list, description="Code validation errors encountered")


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
    supporting_papers: list[str] = Field(
        default_factory=list, description="IDs of papers that support or inspired this hypothesis"
    )
    grounding_status: str = Field(
        default="ungrounded",
        description="grounded | ungrounded | mixed",
    )
    citation_warnings: list[str] = Field(
        default_factory=list, description="Notes about invalid or missing citations"
    )
    non_arxiv_sources: list[str] = Field(
        default_factory=list,
        description="Non-arXiv source IDs kept for traceability",
    )
    evidence_trace: list[str] = Field(
        default_factory=list,
        description="Claim/paper snippets used to derive this hypothesis",
    )
    supported_facts: list[str] = Field(
        default_factory=list,
        description="Directly supported fact snippets tied to citations",
    )
    novel_inference: str = Field(
        default="",
        description="Explicit synthesis step that combines supported facts into a new claim",
    )
    unsupported_parts: list[str] = Field(
        default_factory=list,
        description="Claim fragments or assertions lacking direct support in cited evidence",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Structured claim->citation->quote mappings used for grounding checks",
    )
    diagram: str = Field(default="", description="SVG code for mechanism visualization")
    evidence: EvidenceBlock = Field(default_factory=EvidenceBlock)
    scores: ScoreBlock = Field(default_factory=ScoreBlock)

    # Metadata
    source_soul: SoulRole | None = Field(default=None, description="Which soul generated this")
    iteration: int = Field(default=0, ge=0, description="Which iteration this was generated in")
    created_at: datetime = Field(default_factory=datetime.now)

    # Simulation
    simulation_result: "SimulationResult | None" = Field(default=None)
    simulation_history: "list[SimulationResult]" = Field(default_factory=list)

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


class DialogueEntry(BaseModel):
    """A single message in the soul-to-soul debate."""
    soul: str = Field(..., description="Name of the soul (e.g., 'Dr. Vance')")
    role: SoulRole = Field(..., description="Role of the soul")
    message: str = Field(..., description="The content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)


class IterationTrace(BaseModel):
    """Trace of a single Ralph loop iteration for debugging and visualization."""

    iteration: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
    thought: str = Field(..., description="BDI deliberation reasoning")
    action: str = Field(..., description="What action was taken")
    observation: str = Field(..., description="What was observed after action")
    bdi_snapshot: BDIState = Field(..., description="BDI state at end of iteration")
    dialogue: list[DialogueEntry] = Field(default_factory=list, description="Granular messages from souls")
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
    """Structured representation of a research paper.
    
    Note: Named ArxivPaper for historical reasons, but supports papers from
    multiple sources (arXiv, PubMed, Semantic Scholar).
    """

    arxiv_id: str = Field(..., description="Identifier (arXiv ID, PMID, or S2 paper ID)")
    title: str = Field(...)
    abstract: str = Field(...)
    authors: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    published: datetime | None = Field(default=None)
    updated: datetime | None = Field(default=None)
    pdf_url: str = Field(default="")
    abs_url: str = Field(default="")
    source: str = Field(default="arxiv", description="Paper source: arxiv, pubmed, or semantic_scholar")
    citation_count: int | None = Field(default=None, description="Citation count if available")
    embedding: list[float] | None = Field(default=None, description="Dense embedding vector for semantic search")



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
# LLM & Generation
# =============================================================================


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)


class GenerationResponse(BaseModel):
    """Standardized response from LLM providers."""

    content: str = Field(..., description="Generated text content")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="Usage statistics")
    model_name: str = Field(default="", description="Model used for generation")
    provider: str = Field(default="", description="Provider used")


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

    description: str = Field(..., description="Detailed description of the experiment")
    controls: list[str] = Field(..., description="Necessary control groups")
    measurements: list[str] = Field(..., description="Specific variables to measure")
    expected_timeline: str = Field(..., description="Estimated time to complete")
    required_resources: list[str] = Field(..., description="Equipment/reagents needed")


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
    source_claims: list[str] = Field(
        default_factory=list,
        description="Claim snippets (with paper IDs) used to derive this hypothesis",
    )
    evidence_spans: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Structured claim->citation->quote mappings for grounded verification",
    )

    # Experiment
    suggested_experiments: list[SuggestedExperiment] = Field(
        default_factory=list, description="Concrete experiments to test the hypothesis"
    )

    # Validation & Simulation
    simulation_result: SimulationResult | None = Field(
        default=None, description="Result of in-silico verification"
    )
    simulation_history: list[SimulationResult] = Field(default_factory=list)

    # Scores and metadata
    scores: ScoreBlock | dict[str, Any] = Field(default_factory=dict, description="Scoring breakdown")
    source_soul: SoulRole | None = Field(default=None)
    iteration: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.now)

    # Lineage tracking for hypothesis evolution
    parent_id: str = Field(default="", description="ID of parent hypothesis if refined")
    lineage: list[str] = Field(default_factory=list, description="Chain of ancestor IDs (oldest first)")
    refinement_instruction: str = Field(default="", description="Instruction that produced this refinement")

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
    overlaps: list[dict[str, Any]] = Field(
        default_factory=list, description="Embedding-based overlaps between papers"
    )


# =============================================================================
# Persona Forge
# =============================================================================


class Persona(BaseModel):
    """A generated expert persona."""

    id: str = Field(..., description="Unique ID for this persona")
    name: str = Field(..., description="Name of the agent (e.g., 'Dr. Vance')")
    role: str = Field(..., description="Specialized role (e.g., 'Plasma Physicist')")
    style: str = Field(..., description="Communication style (e.g., 'Concise', 'Radical')")
    objective: str = Field(..., description="Primary objective for this persona")
    weight: float = Field(default=0.33, ge=0.0, le=1.0, description="Relative weight in debate")
    system_instruction: str = Field(..., description="The full system prompt for this persona")
    locked: bool = Field(default=False, description="Whether this persona is locked from regeneration")
    soul_role: SoulRole | None = Field(default=None, description="Mapping to a collective soul role")


class PersonaWeightRequest(BaseModel):
    """Request to update a persona's weight."""

    weight: float = Field(..., ge=0.0, le=1.0)


class PersonaLockRequest(BaseModel):
    """Request to lock/unlock a persona."""

    locked: bool


# =============================================================================
# Session and Configuration
# =============================================================================


class SessionConstraints(BaseModel):
    """Optional researcher-provided constraints for a session."""

    domains: list[str] = Field(default_factory=list, description="Preferred domains or fields")
    modalities: list[str] = Field(default_factory=list, description="Desired modalities (e.g., wet lab, simulation)")
    timeline: str | None = Field(default=None, description="Timeline pressure or deadline phrasing")
    dataset_links: list[AnyHttpUrl] = Field(
        default_factory=list,
        description="External dataset links to ground the run",
    )


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
    flash_model: str = Field(default="groq/llama-3.3-70b-versatile")
    pro_model: str = Field(default="gemini/gemini-3-flash")
    domains: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=list)
    timeline_hint: str | None = Field(default=None)
    dataset_links: list[str] = Field(default_factory=list)
    strict_grounding: bool = Field(
        default=False,
        description="If true, hypotheses without usable citations are removed from results.",
    )
    enforce_numeric_citations: bool = Field(
        default=True,
        description="If true, numeric claims must be supported by cited paper text.",
    )
    srsh_enabled: bool = Field(
        default=True,
        description="If true, run SRSH parallel-stream collision generation on iteration 1.",
    )
    srsh_agents: int = Field(
        default=3,
        ge=2,
        le=5,
        description="Number of isolated SRSH idea streams.",
    )
    srsh_iterations_per_agent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Iterations per SRSH stream before collision.",
    )
    srsh_collisions: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of collision synthesis attempts per stress zone.",
    )


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
    source_metadata: dict[str, ArxivPaper] = Field(default_factory=dict)
    constraints: SessionConstraints | None = Field(default=None)


# =============================================================================
# Experiment Protocols
# =============================================================================


class HazardLevel(str, Enum):
    """Hazard classification levels."""
    NONE = "none"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


class Reagent(BaseModel):
    """A chemical reagent or biological material."""

    name: str = Field(..., description="Reagent name (e.g., 'Sodium Chloride')")
    cas_number: str = Field(default="", description="CAS registry number")
    quantity: str = Field(..., description="Required quantity (e.g., '50 mL', '10 mg')")
    concentration: str = Field(default="", description="Concentration if applicable")
    hazard_level: HazardLevel = Field(default=HazardLevel.LOW)
    storage: str = Field(default="room temperature", description="Storage conditions")
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    supplier: str = Field(default="", description="Recommended supplier")
    notes: str = Field(default="")


class Equipment(BaseModel):
    """Laboratory equipment required."""

    name: str = Field(..., description="Equipment name")
    specifications: str = Field(default="", description="Required specs")
    quantity: int = Field(default=1, ge=1)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0, description="Cost if purchase needed")
    is_common: bool = Field(default=True, description="Commonly available in labs")
    alternatives: list[str] = Field(default_factory=list)


class SafetyWarning(BaseModel):
    """Safety warning for a protocol step."""

    hazard_type: str = Field(..., description="e.g., 'chemical', 'biological', 'radiation'")
    level: HazardLevel = Field(default=HazardLevel.MODERATE)
    description: str = Field(...)
    ppe_required: list[str] = Field(default_factory=list, description="Personal protective equipment")
    emergency_procedure: str = Field(default="")


class ProtocolStep(BaseModel):
    """A single step in an experiment protocol."""

    step_number: int = Field(..., ge=1)
    action: str = Field(..., description="What to do")
    duration: str = Field(default="", description="Expected time (e.g., '5 min', '2 hours')")
    temperature: str = Field(default="", description="Temperature if relevant")
    expected_result: str = Field(default="", description="What should happen")
    tips: list[str] = Field(default_factory=list, description="Helpful tips")
    warnings: list[SafetyWarning] = Field(default_factory=list)
    is_critical: bool = Field(default=False, description="Step is critical for success")


class ExperimentProtocol(BaseModel):
    """Complete experimental protocol for testing a hypothesis.
    
    Designed for compatibility with lab automation systems.
    """

    id: str = Field(default="", description="Protocol identifier")
    title: str = Field(..., description="Protocol title")
    hypothesis_id: str = Field(default="", description="Associated hypothesis ID")
    version: str = Field(default="1.0")

    # Overview
    objective: str = Field(..., description="What this protocol aims to test/demonstrate")
    background: str = Field(default="", description="Brief scientific background")
    expected_duration: str = Field(default="", description="Total time estimate")
    difficulty: str = Field(default="intermediate", description="beginner/intermediate/advanced")

    # Materials
    reagents: list[Reagent] = Field(default_factory=list)
    equipment: list[Equipment] = Field(default_factory=list)

    # Procedure
    steps: list[ProtocolStep] = Field(default_factory=list)

    # Safety
    overall_hazard_level: HazardLevel = Field(default=HazardLevel.MODERATE)
    safety_summary: str = Field(default="")
    institutional_approval_required: bool = Field(default=False)

    # Analysis
    success_criteria: list[str] = Field(default_factory=list, description="How to determine success")
    data_collection: list[str] = Field(default_factory=list, description="What data to collect")
    analysis_methods: list[str] = Field(default_factory=list)

    # Cost
    estimated_materials_cost_usd: float = Field(default=0.0, ge=0.0)
    estimated_equipment_cost_usd: float = Field(default=0.0, ge=0.0)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    references: list[str] = Field(default_factory=list, description="Paper IDs or DOIs")
