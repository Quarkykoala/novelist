"""Skeptic Soul - harsh critique and quality control.

The Skeptic soul evaluates hypotheses using Constitutional AI-style principles.
It identifies fatal flaws, moderate issues, and minor improvements.
"""

import asyncio
import json
from typing import Any

from src.contracts.schemas import Critique, CritiqueVerdict, Hypothesis, SoulRole
from src.contracts.validators import extract_json_from_response
from src.soul.prompts.base import BaseSoul
from src.kb.arxiv_client import ArxivClient


class SkepticSoul(BaseSoul):
    """The Skeptic soul critiques hypotheses harshly but fairly."""

    role = SoulRole.SKEPTIC
    description = """You are the SKEPTIC voice in the research collective.
Your role is quality control through rigorous critique.
You are harsh but fair. You look for:
- Logical flaws and circular reasoning
- Unfalsifiable claims
- Vague or undefined terms
- Missing experimental steps
- Overclaiming or unsupported conclusions
- Prior art that makes this "not novel"

A PASS verdict should be rare - you have high standards."""

    # Constitutional AI-style principles for critique
    CONSTITUTION = """
## Critique Principles

1. FALSIFIABILITY: The hypothesis must be testable. If it cannot be proven wrong, it's not scientific.

2. SPECIFICITY: All terms must be precisely defined. "Improves performance" is too vague.

3. NOVELTY: The idea should not be obviously covered by existing literature.

4. COHERENCE: The hypothesis must not contradict itself or known scientific facts.

5. MECHANISM: There should be a plausible mechanism explaining why this might work.

6. TESTABILITY: The experimental design must actually test the hypothesis, not something else.

7. MINIMALITY: The experimental steps should be concrete and achievable.

8. SCOPE: Claims should be proportional to the evidence that could be gathered.
"""

    async def critique(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
    ) -> list[Critique]:
        """Critique a list of hypotheses.

        Args:
            hypotheses: Hypotheses to critique
            topic: Research topic for context

        Returns:
            List of Critique objects
        """
        # Phase 1: Real-world Verification (Deep Skeptic)
        verification_context = await self._verify_claims(hypotheses)
        
        # Phase 2: LLM Critique
        prompt = self._build_critique_prompt(hypotheses, topic, verification_context)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return []

        critiques = self._parse_critiques(response_text, hypotheses)
        
        # Phase 3: Attach verification papers to hypotheses (traceability)
        # We update the hypothesis objects in place
        for h in hypotheses:
            if h.id in verification_context:
                # Add found papers to supporting_papers if not already there
                found_papers = [p['id'] for p in verification_context[h.id]]
                for pid in found_papers:
                    if pid not in h.supporting_papers:
                        h.supporting_papers.append(pid)

        return critiques

    async def _verify_claims(self, hypotheses: list[Hypothesis]) -> dict[str, list[dict[str, str]]]:
        """Search arXiv to verify claims for each hypothesis."""
        results = {}
        async with ArxivClient() as client:
            for h in hypotheses:
                # Construct a targeted query from keywords
                query = " AND ".join(f'"{k}"' for k in h.novelty_keywords[:3])
                papers = await client.search(query, max_results=3)
                
                results[h.id] = [
                    {"id": p.arxiv_id, "title": p.title, "summary": p.abstract[:200]} 
                    for p in papers
                ]
        return results

    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: Any,
    ) -> list[Hypothesis]:
        """Skeptic doesn't generate hypotheses, only critiques."""
        return []

    def _build_critique_prompt(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
        verification_context: dict[str, list[dict[str, str]]] = {},
    ) -> str:
        """Build the critique prompt."""

        hypotheses_text = ""
        for i, h in enumerate(hypotheses):
            evidence = verification_context.get(h.id, [])
            evidence_text = "\n".join([f"- [{p['id']}] {p['title']}" for p in evidence])
            
            hypotheses_text += f"""
### Hypothesis {i + 1} (ID: {h.id})
Statement: {h.hypothesis}
Rationale: {h.rationale}
Cross-domain: {h.cross_disciplinary_connection}
Experiment: {' → '.join(h.experimental_design[:3])}...
Keywords: {', '.join(h.novelty_keywords)}
FOUND LITERATURE (Real-time Search):
{evidence_text or "No direct matches found."}
"""

        return f"""{self.get_persona_prompt()}

{self.CONSTITUTION}

<research_topic>
{topic}
</research_topic>

<hypotheses_to_critique>
{hypotheses_text}
</hypotheses_to_critique>

<task>
Critique each hypothesis according to the Constitution principles above.
USE THE "FOUND LITERATURE" to fact-check the claims.
If the literature contradicts the hypothesis, mark it as FATAL.
If the literature supports it but the hypothesis claims to be "novel", mark it as MODERATE (needs to distinguish itself).

For each hypothesis, provide:
1. Verdict: fatal / moderate / minor / pass
   - FATAL: Fundamental flaw, hypothesis should be removed
   - MODERATE: Significant issues but salvageable with revision
   - MINOR: Small improvements needed
   - PASS: Good quality, ready to proceed

2. Issues: List specific problems found

3. Suggestions: Concrete ways to fix the issues

4. Principle violations: Which Constitution principles were violated

Be HARSH but CONSTRUCTIVE. Science advances through rigorous critique.
</task>

Respond with valid JSON:
```json
[
  {{
    "hypothesis_id": "creative_0",
    "verdict": "moderate",
    "issues": ["Issue 1", "Issue 2"],
    "suggestions": ["How to fix 1", "How to fix 2"],
    "principle_violations": ["SPECIFICITY", "MECHANISM"]
  }}
]
```
"""

    def _parse_critiques(
        self,
        response_text: str,
        hypotheses: list[Hypothesis],
    ) -> list[Critique]:
        """Parse critique response into Critique objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            critiques = []
            for item in data:
                try:
                    verdict_str = item.get("verdict", "moderate").lower()
                    verdict = CritiqueVerdict(verdict_str)

                    critique = Critique(
                        hypothesis_id=item.get("hypothesis_id", ""),
                        verdict=verdict,
                        issues=item.get("issues", []),
                        suggestions=item.get("suggestions", []),
                        principle_violations=item.get("principle_violations", []),
                    )
                    critiques.append(critique)
                except Exception:
                    continue

            return critiques
        except json.JSONDecodeError:
            return []
