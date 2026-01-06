"""Feasibility and Impact scoring using LLM.

Uses Gemini to score hypotheses on:
- Feasibility: Can this experiment actually be done?
- Impact: How significant would the results be?
"""

import asyncio
import json
import os
from typing import Any

from dotenv import load_dotenv

from src.contracts.schemas import Hypothesis
from src.contracts.validators import extract_json_from_response

load_dotenv()

FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash")

SCORING_PROMPT = """You are a scientific reviewer scoring research hypotheses.

<hypothesis>
Statement: {hypothesis}
Experimental Design:
{experiment}
Expected Impact: {impact}
Cross-Domain Connection: {cross_domain}
</hypothesis>

Score this hypothesis on three dimensions:

1. FEASIBILITY (0-10): How achievable is the proposed experiment?
   +2: Uses existing tools, datasets, or methods
   +2: Has clear, measurable outcomes
   -5: Is fundamentally unfalsifiable

2. IMPACT (0-10): How significant would validated results be?
   +3: Addresses a major bottleneck or high-value problem
   +3: Enables genuinely new capabilities
   -2: Only incremental improvement

3. CROSS-DOMAIN SYNERGY (0-10): How well does it merge distinct fields?
   +4: Finds a structural isomorphism between unrelated fields (e.g. Fungi -> Neuroscience)
   +3: Applies a proven method from Field A to solve a blocked problem in Field B
   +2: Uses analogies that reveal new mechanisms
   -3: Merely uses a tool from another field (e.g. "AI for X") without conceptual blending
   -5: Buzzword soup with no real connection

Respond with valid JSON:
```json
{{
  "feasibility_score": 7,
  "feasibility_reasoning": "Brief explanation",
  "impact_score": 8,
  "impact_reasoning": "Brief explanation",
  "cross_domain_score": 9,
  "cross_domain_reasoning": "Brief explanation"
}}
```
"""


class ScoringService:
    """Scores hypotheses on feasibility and impact."""

    def __init__(self, model: str = FLASH_MODEL):
        self.model = model
        from src.soul.llm_client import LLMClient
        self.client = LLMClient(model=model)

    async def score(self, hypothesis: Hypothesis) -> tuple[float, float, float]:
        """Score a hypothesis on feasibility, impact, and cross-domain synergy.

        Args:
            hypothesis: Hypothesis to score

        Returns:
            Tuple of (feasibility, impact, cross_domain) normalized to 0-1
        """
        experiment = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(hypothesis.experimental_design))

        prompt = SCORING_PROMPT.format(
            hypothesis=hypothesis.hypothesis,
            experiment=experiment,
            impact=hypothesis.expected_impact,
            cross_domain=hypothesis.cross_disciplinary_connection,
        )

        try:
            response_text = await self.client.generate_content(prompt)
            if not response_text:
                return 0.5, 0.5, 0.0

            json_str = extract_json_from_response(response_text)
            if not json_str:
                return 0.5, 0.5, 0.0

            data = json.loads(json_str)

            # Normalize from 0-10 to 0-1
            feasibility = min(10, max(0, data.get("feasibility_score", 5))) / 10
            impact = min(10, max(0, data.get("impact_score", 5))) / 10
            cross_domain = min(10, max(0, data.get("cross_domain_score", 0))) / 10

            return feasibility, impact, cross_domain
        except Exception:
            return 0.5, 0.5, 0.0

    async def batch_score(
        self,
        hypotheses: list[Hypothesis],
        max_concurrent: int = 5,
    ) -> list[Hypothesis]:
        """Score multiple hypotheses with concurrency control.

        Args:
            hypotheses: List of hypotheses to score
            max_concurrent: Maximum concurrent API calls

        Returns:
            Hypotheses with updated scores
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def score_with_semaphore(h: Hypothesis) -> Hypothesis:
            async with semaphore:
                feasibility, impact, cross_domain = await self.score(h)
                h.scores.feasibility = feasibility
                h.scores.impact = impact
                h.scores.cross_domain = cross_domain
                return h

        tasks = [score_with_semaphore(h) for h in hypotheses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scored: list[Hypothesis] = []
        for result in results:
            if isinstance(result, Hypothesis):
                scored.append(result)
            elif isinstance(result, Exception):
                # Keep original on error
                pass

        return scored


# Simple heuristic scoring (fallback)


def heuristic_feasibility(hypothesis: Hypothesis) -> float:
    """Simple heuristic feasibility scoring without LLM."""
    score = 0.5  # Base score

    # More experimental steps = more concrete = more feasible
    steps = len(hypothesis.experimental_design)
    if steps >= 5:
        score += 0.2
    elif steps >= 3:
        score += 0.1

    # Longer rationale suggests more thought
    if len(hypothesis.rationale) > 200:
        score += 0.1

    # Check for common feasibility indicators in experiment
    experiment_text = " ".join(hypothesis.experimental_design).lower()
    if any(kw in experiment_text for kw in ["dataset", "existing", "available", "standard"]):
        score += 0.1
    if any(kw in experiment_text for kw in ["measure", "quantify", "compare", "baseline"]):
        score += 0.1

    return min(1.0, score)


def heuristic_impact(hypothesis: Hypothesis) -> float:
    """Simple heuristic impact scoring without LLM."""
    score = 0.5  # Base score

    # Check impact keywords
    impact_text = hypothesis.expected_impact.lower()
    high_impact_words = ["paradigm", "revolution", "breakthrough", "transform", "enable", "fundamental"]
    medium_impact_words = ["improve", "enhance", "reduce", "increase", "optimize"]

    if any(w in impact_text for w in high_impact_words):
        score += 0.3
    elif any(w in impact_text for w in medium_impact_words):
        score += 0.1

    # Cross-domain = higher potential impact
    if hypothesis.cross_disciplinary_connection:
        score += 0.1

    return min(1.0, score)
