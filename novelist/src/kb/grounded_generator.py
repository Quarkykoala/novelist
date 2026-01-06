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


HYPOTHESIS_GENERATION_PROMPT = """You are a scientific hypothesis generator. Your task is to create a GROUNDED, TESTABLE hypothesis that addresses the identified gap.

CRITICAL RULES:
1. The hypothesis must be FALSIFIABLE — it must be possible to prove wrong
2. The mechanism must be a CAUSAL CHAIN — A causes B, B causes C, C produces outcome
3. Predictions must be QUANTITATIVE and BOUNDED by existing data
4. NO FANTASY NUMBERS — predictions must be within 2x of cited baselines
5. Include at least one concrete EXPERIMENT to test the hypothesis

THE GAP TO ADDRESS:
{gap_description}

Gap type: {gap_type}
Concept A: {concept_a}
Concept B: {concept_b}

RELEVANT CLAIMS FROM LITERATURE:
{relevant_claims}

STATE-OF-THE-ART BASELINES:
{baselines}

Generate a hypothesis in this EXACT JSON format:

```json
{{
  "claim": "If [specific intervention], then [specific outcome], because [mechanism summary]",
  "mechanism": [
    {{"cause": "A", "effect": "B", "evidence_paper": "arxiv_id or empty"}},
    {{"cause": "B", "effect": "C", "evidence_paper": "arxiv_id or empty"}},
    {{"cause": "C", "effect": "final outcome", "evidence_paper": ""}}
  ],
  "prediction": "We expect to observe [specific measurable outcome]",
  "prediction_bounds": {{
    "metric": "what we measure",
    "lower_bound": 10,
    "upper_bound": 30,
    "unit": "%",
    "baseline_value": 5,
    "baseline_source": "arxiv_id"
  }},
  "null_result": "If we observe [specific observation], the hypothesis is rejected",
  "supporting_papers": ["arxiv_id_1", "arxiv_id_2"],
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
        hypothesis = self._parse_hypothesis(response, gap, iteration)
        
        return hypothesis

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
            
            hypothesis = GroundedHypothesis(
                id=str(uuid.uuid4())[:8],
                claim=raw.get("claim", ""),
                mechanism=mechanism if mechanism else [MechanismStep(cause="A", effect="B")],
                prediction=raw.get("prediction", ""),
                prediction_bounds=prediction_bounds,
                null_result=raw.get("null_result", ""),
                gap_addressed=gap.description,
                supporting_papers=raw.get("supporting_papers", []),
                contradicting_papers=[],
                suggested_experiments=experiments,
                scores=ScoreBlock(),
                source_soul=SoulRole.CREATIVE,
                iteration=iteration,
            )
            
            return hypothesis
            
        except Exception:
            return None
