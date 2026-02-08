"""
Grounded Hypothesis Generator — Synthesize hypotheses from identified gaps.

This module generates hypotheses that:
- Address specific identified gaps
- Are grounded in literature with citations
- Include mechanism chains
- Have quantitative predictions bounded by existing data
- Suggest concrete experiments
"""

import json
import re
import uuid

from src.contracts.schemas import (
    ExtractedClaim,
    GapType,
    GroundedHypothesis,
    IdentifiedGap,
    MechanismStep,
    PredictionBounds,
    ScoreBlock,
    SoulRole,
    SuggestedExperiment,
)
from src.soul.llm_client import LLMClient



HYPOTHESIS_GENERATION_PROMPT = """You are a senior research scientist generating BREAKTHROUGH hypotheses. Your task is to create a SPECIFIC, TECHNICAL, ACTIONABLE hypothesis that bridges disciplines.

ANTI-VAGUENESS RULES (CRITICAL):
1. NO GENERIC STATEMENTS like "applying principles from X to Y" — be SPECIFIC about WHICH principle
2. NO HAND-WAVING like "novel materials" — specify the EXACT material or class (e.g., "graphene oxide nanosheets")
3. EVERY claim must have a MECHANISM with named components (molecules, forces, algorithms, etc.)
4. PREDICTIONS must include NUMBERS with units (e.g., "30% improvement", "10^6 cycles")
5. CITE only paper IDs from the pulled literature provided below (arXiv/PMID/DOI/S2)

CROSS-DISCIPLINARY EXAMPLES of GOOD vs BAD:
- BAD: "Biomimicry can improve battery design"
- GOOD: "The hierarchical pore structure in diatom frustules (silica exoskeletons) can template Li-ion cathode synthesis, increasing surface area from 50 to 200 m²/g and improving charge/discharge rates by 40%"

- BAD: "Machine learning can enhance drug discovery"  
- GOOD: "Applying graph neural networks trained on protein-ligand binding affinities (as in 2103.09430) to screen 10^6 candidates for SARS-CoV-2 Mpro inhibition, predicting Ki values with <100 nM threshold"

THE GAP TO ADDRESS:
{gap_description}

Gap type: {gap_type}
Concept A: {concept_a}
Concept B: {concept_b}

RELEVANT CLAIMS FROM LITERATURE (USE THESE PAPER IDs):
{relevant_claims}

STATE-OF-THE-ART BASELINES (YOUR PREDICTIONS MUST BEAT OR EXPLAIN THESE):
{baselines}

Generate a hypothesis in this EXACT JSON format:

```json
{{
  "claim": "If [specific intervention], then [specific outcome], because [mechanism summary]",
  "mechanism": [
    {{"cause": "A", "effect": "B", "evidence_paper": "paper_id or empty"}},
    {{"cause": "B", "effect": "C", "evidence_paper": "paper_id or empty"}},
    {{"cause": "C", "effect": "final outcome", "evidence_paper": ""}}
  ],
  "prediction": "We expect to observe [specific measurable outcome]",
  "prediction_bounds": {{
    "metric": "what we measure",
    "lower_bound": 10,
    "upper_bound": 30,
    "unit": "%",
    "baseline_value": 5,
    "baseline_source": "paper_id"
  }},
  "null_result": "If we observe [specific observation], the hypothesis is rejected",
  "supporting_papers": ["paper_id_1", "paper_id_2"],
  "suggested_experiments": [
    {{
      "description": "What to do",
      "controls": ["control 1", "control 2"],
      "measurements": ["what to measure"],
      "expected_timeline": "2-4 weeks",
      "required_resources": ["equipment", "materials"]
    }}
  ]
}}
```

Generate a grounded hypothesis:"""


HYPOTHESIS_REFINEMENT_PROMPT = """You are a scientific hypothesis refiner. Your task is to IMPROVE an existing hypothesis based on a specific instruction.

THE ORIGINAL HYPOTHESIS:
{original_hypothesis}

INSTRUCTION FOR REFINEMENT:
"{instruction}"

RELEVANT CLAIMS FROM LITERATURE:
{relevant_claims}

CRITICAL RULES:
1. Keep the core idea if it's sound, but enhance it as requested.
2. Ensure the JSON structure remains valid.
3. If asked to deepen mechanism, add specific steps.
4. If asked to add experiments, provide concrete details.

Generate the refined hypothesis in the EXACT same JSON format as the original:
```json
{{
  ...
}}
```
"""


class GroundedHypothesisGenerator:
    """Generates hypotheses grounded in literature that address identified gaps."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = LLMClient(model=model)

    async def generate_from_gap(
        self,
        gap: IdentifiedGap,
        claims: list[ExtractedClaim],
        iteration: int = 0,
    ) -> GroundedHypothesis | None:
        """Generate a grounded hypothesis that addresses a specific gap."""
        
        # Find relevant claims
        relevant_claims = self._find_relevant_claims(gap, claims)
        baselines = self._find_baselines(claims)
        
        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            gap_description=gap.description,
            gap_type=gap.gap_type.value,
            concept_a=gap.concept_a,
            concept_b=gap.concept_b,
            relevant_claims=self._format_claims(relevant_claims),
            baselines=self._format_baselines(baselines),
        )
        
        response = await self.client.generate_content(prompt)
        if not response:
            return None
        # Handle GenerationResponse
        if hasattr(response, "content"):
            response = response.content
        if not response:
            return None
            
        hypothesis = self._parse_hypothesis(response, gap, iteration)
        if not hypothesis:
            return None

        # Enforce grounding: restrict citations to known claim sources
        valid_sources = [c.paper_id for c in relevant_claims if c.paper_id]
        if valid_sources:
            allowed = set(valid_sources)
            filtered = [pid for pid in hypothesis.supporting_papers if pid in allowed]
            if filtered:
                hypothesis.supporting_papers = list(dict.fromkeys(filtered))
            else:
                # Fallback: seed with a few relevant claim sources
                hypothesis.supporting_papers = list(dict.fromkeys(valid_sources))[:3]

        # Attach provenance snippets for traceability
        source_claims = [
            f"[{c.paper_id}] {c.statement}"
            for c in relevant_claims
            if c.paper_id and c.statement
        ]
        if source_claims:
            hypothesis.source_claims = source_claims[:8]

        return hypothesis

    async def generate_from_description(
        self,
        gap_description: str,
        claims: list[ExtractedClaim],
        iteration: int = 0,
    ) -> GroundedHypothesis | None:
        """Generate hypothesis from a text description (for SRSH parallel streams).
        
        This creates a synthetic IdentifiedGap from the description and extracts
        concepts for claim matching.
        """
        # Extract key concepts from the description for claim matching
        words = gap_description.lower().split()
        # Filter to meaningful terms (>3 chars, not common stopwords)
        stopwords = {'the', 'and', 'for', 'with', 'that', 'this', 'from', 'are', 
                     'was', 'were', 'have', 'has', 'been', 'will', 'would', 'could',
                     'should', 'must', 'what', 'when', 'where', 'which', 'who', 'how'}
        concepts = [w for w in words if len(w) > 3 and w not in stopwords][:10]
        
        # Create synthetic gap
        synthetic_gap = IdentifiedGap(
            gap_type=GapType.MECHANISM_UNKNOWN,
            description=gap_description,
            concept_a=concepts[0] if concepts else "unknown",
            concept_b=concepts[1] if len(concepts) > 1 else "target",
            source_papers=[c.paper_id for c in claims[:5] if c.paper_id],
            priority=0.7,
        )
        
        return await self.generate_from_gap(synthetic_gap, claims, iteration)

    async def refine_hypothesis(
        self,
        hypothesis: GroundedHypothesis,
        instruction: str,
        claims: list[ExtractedClaim],
    ) -> GroundedHypothesis | None:
        """Refine an existing hypothesis based on an instruction.
        
        Tracks lineage: the refined hypothesis stores its parent_id and
        builds a lineage chain for evolution tracking.
        """
        
        # Format claims for context
        claims_text = "\n".join(
            [f"- {c.statement} (Source: {c.paper_id})" for c in claims[:5]]
        )
        
        # Prepare hypothesis JSON string
        hyp_json = json.dumps(
            hypothesis.model_dump(
                mode="json",
                exclude={"scores", "source_soul", "iteration", "parent_id", "lineage", "refinement_instruction"},
            ),
            indent=2,
        )

        prompt = HYPOTHESIS_REFINEMENT_PROMPT.format(
            original_hypothesis=hyp_json,
            instruction=instruction,
            relevant_claims=claims_text,
        )

        try:
            response = await self.client.generate_content(prompt)
            if not response:
                return None
            
            # Handle GenerationResponse
            if hasattr(response, "content"):
                response = response.content

            # Extract JSON
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Build lineage chain: append parent's ID to lineage
            new_lineage = list(hypothesis.lineage)  # Copy existing lineage
            if hypothesis.id:
                new_lineage.append(hypothesis.id)
            
            # Map back to GroundedHypothesis with lineage tracking
            refined = GroundedHypothesis(
                id=str(uuid.uuid4())[:8],  # Generate new ID for the refined node
                claim=data.get("claim", hypothesis.claim),
                mechanism=[MechanismStep(**m) for m in data.get("mechanism", [])],
                prediction=data.get("prediction", ""),
                prediction_bounds=PredictionBounds(**data.get("prediction_bounds", {})) if data.get("prediction_bounds") else None,
                null_result=data.get("null_result", ""),
                gap_addressed=hypothesis.gap_addressed,  # Keep original gap addressed
                supporting_papers=data.get("supporting_papers", []),
                contradicting_papers=hypothesis.contradicting_papers,  # Keep original
                suggested_experiments=[SuggestedExperiment(**e) for e in data.get("suggested_experiments", [])],
                scores=hypothesis.scores,  # Needs re-scoring later
                source_soul=hypothesis.source_soul,  # Keep original source soul
                iteration=hypothesis.iteration + 1,  # Increment iteration
                # Lineage tracking
                parent_id=hypothesis.id,  # Track parent
                lineage=new_lineage,  # Chain of ancestors
                refinement_instruction=instruction,  # Store instruction that produced this
                source_claims=[
                    f"[{c.paper_id}] {c.statement}"
                    for c in claims[:8]
                    if c.paper_id and c.statement
                ],
            )
            
            return refined

        except Exception as e:
            print(f"Error refining hypothesis: {e}")
            return None


    async def generate_batch(
        self,
        gaps: list[IdentifiedGap],
        claims: list[ExtractedClaim],
        max_hypotheses: int = 5,
        iteration: int = 0,
    ) -> list[GroundedHypothesis]:
        """Generate hypotheses for multiple gaps."""
        hypotheses = []
        
        for gap in gaps[:max_hypotheses]:
            hyp = await self.generate_from_gap(gap, claims, iteration)
            if hyp and hyp.is_well_formed():
                hypotheses.append(hyp)
        
        return hypotheses

    def _find_relevant_claims(
        self,
        gap: IdentifiedGap,
        claims: list[ExtractedClaim],
    ) -> list[ExtractedClaim]:
        """Find claims relevant to a gap."""
        relevant = []
        gap_terms = set(gap.concept_a.lower().split() + gap.concept_b.lower().split())
        
        for claim in claims:
            claim_terms = set(claim.statement.lower().split())
            claim_terms.update(e.lower() for e in claim.entities_mentioned)
            
            # Check for overlap
            if gap_terms & claim_terms:
                relevant.append(claim)
        
        return relevant[:10]  # Limit for prompt size

    def _find_baselines(self, claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
        """Find claims with quantitative baseline data."""
        baselines = []
        for claim in claims:
            if claim.quantitative_data:
                baselines.append(claim)
        return baselines[:5]

    def _format_claims(self, claims: list[ExtractedClaim]) -> str:
        """Format claims for prompt."""
        if not claims:
            return "No directly relevant claims found."
        
        lines = []
        for claim in claims:
            lines.append(f"- [{claim.paper_id}] {claim.statement}")
            if claim.quantitative_data:
                qd = claim.quantitative_data
                lines.append(f"  Data: {qd.metric} = {qd.value} {qd.unit}")
        
        return "\n".join(lines)

    def _format_baselines(self, baselines: list[ExtractedClaim]) -> str:
        """Format baseline data for prompt."""
        if not baselines:
            return "No quantitative baselines available."
        
        lines = []
        for claim in baselines:
            if claim.quantitative_data:
                qd = claim.quantitative_data
                sota_marker = " [SOTA]" if qd.is_sota else ""
                lines.append(f"- {qd.metric}: {qd.value} {qd.unit} ({qd.conditions}){sota_marker} [{claim.paper_id}]")
        
        return "\n".join(lines)

    def _parse_hypothesis(
        self,
        response: str,
        gap: IdentifiedGap,
        iteration: int,
    ) -> GroundedHypothesis | None:
        """Parse LLM response into GroundedHypothesis."""
        if not response or not isinstance(response, str):
            return None
        
        json_match = re.search(r'\{[\s\S]*\}', response)
        if not json_match:
            return None
        
        try:
            raw = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None
        
        try:
            # Parse mechanism steps
            mechanism = []
            for step in raw.get("mechanism", []):
                mechanism.append(MechanismStep(
                    cause=step.get("cause", ""),
                    effect=step.get("effect", ""),
                    evidence_paper=step.get("evidence_paper", ""),
                    confidence=0.6,
                ))
            
            # Parse prediction bounds
            bounds_data = raw.get("prediction_bounds")
            prediction_bounds = None
            if bounds_data:
                prediction_bounds = PredictionBounds(
                    metric=bounds_data.get("metric", "unknown"),
                    lower_bound=float(bounds_data.get("lower_bound", 0)),
                    upper_bound=float(bounds_data.get("upper_bound", 0)),
                    unit=bounds_data.get("unit", ""),
                    baseline_value=float(bounds_data.get("baseline_value", 0)),
                    baseline_source=bounds_data.get("baseline_source", ""),
                )
            
            # Parse experiments
            experiments = []
            for exp in raw.get("suggested_experiments", []):
                experiments.append(SuggestedExperiment(
                    description=exp.get("description", ""),
                    controls=exp.get("controls", []),
                    measurements=exp.get("measurements", []),
                    expected_timeline=exp.get("expected_timeline", ""),
                    required_resources=exp.get("required_resources", []),
                ))
            
            # Collect all paper IDs from both explicit list AND mechanism evidence
            all_papers: set[str] = set()
            
            # From the explicit list
            for paper in raw.get("supporting_papers", []):
                if paper and isinstance(paper, str) and paper.strip():
                    all_papers.add(paper.strip())
            
            # From mechanism evidence_paper fields
            for step in mechanism:
                if step.evidence_paper and step.evidence_paper.strip():
                    all_papers.add(step.evidence_paper.strip())
            
            # From prediction bounds baseline_source
            if prediction_bounds and prediction_bounds.baseline_source:
                all_papers.add(prediction_bounds.baseline_source)
            
            hypothesis = GroundedHypothesis(
                id=str(uuid.uuid4())[:8],
                claim=raw.get("claim", ""),
                mechanism=mechanism if mechanism else [MechanismStep(cause="A", effect="B")],
                prediction=raw.get("prediction", ""),
                prediction_bounds=prediction_bounds,
                null_result=raw.get("null_result", ""),
                gap_addressed=gap.description,
                supporting_papers=list(all_papers),  # Combined paper references
                contradicting_papers=[],
                suggested_experiments=experiments,
                scores=ScoreBlock(),
                source_soul=SoulRole.CREATIVE,
                iteration=iteration,
            )
            
            return hypothesis
            
        except Exception:
            return None
