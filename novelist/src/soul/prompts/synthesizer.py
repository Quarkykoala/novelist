"""Synthesizer Soul - merges and refines hypotheses into a coherent set.

The Synthesizer combines overlapping ideas and produces the final output.
"""

import asyncio
import json
from typing import Any

from src.contracts.schemas import GenerationMode, Hypothesis, SoulRole
from src.contracts.validators import extract_json_from_response
from src.soul.prompts.base import BaseSoul


class SynthesizerSoul(BaseSoul):
    """The Synthesizer soul merges hypotheses into a coherent set."""

    role = SoulRole.SYNTHESIZER
    description = """You are the SYNTHESIZER voice in the research collective.
Your role is to merge, refine, and harmonize hypotheses.
You look for:
- Overlapping ideas that can be combined
- Complementary hypotheses that form a research agenda
- Gaps in coverage that need new hypotheses
- Inconsistencies that need resolution

You produce the final, coherent set of hypotheses."""

    def __init__(self, model: str = "gemini-3-pro"):
        """Synthesizer uses the Pro model for better reasoning."""
        super().__init__(model)

    async def synthesize(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
        target_count: int = 10,
    ) -> list[Hypothesis]:
        """Synthesize hypotheses into a final coherent set.

        Args:
            hypotheses: All surviving hypotheses
            topic: Research topic
            target_count: Target number of final hypotheses

        Returns:
            Synthesized, merged list of hypotheses
        """
        prompt = self._build_synthesis_prompt(hypotheses, topic, target_count)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return hypotheses[:target_count]

        return self._parse_response(response_text, target_count)

    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> list[Hypothesis]:
        """Synthesizer doesn't generate, only synthesizes."""
        return []

    def _build_synthesis_prompt(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
        target_count: int,
    ) -> str:
        """Build the synthesis prompt."""

        hyp_text = ""
        for i, h in enumerate(hypotheses):
            hyp_text += f"""
### Hypothesis {i + 1} (ID: {h.id}, Source: {h.source_soul.value if h.source_soul else 'unknown'})
Statement: {h.hypothesis}
Rationale: {h.rationale}
Cross-domain: {h.cross_disciplinary_connection}
Impact: {h.expected_impact}
Keywords: {', '.join(h.novelty_keywords)}
"""

        return f"""{self.get_persona_prompt()}

<research_topic>
{topic}
</research_topic>

<hypotheses_to_synthesize>
{hyp_text}
</hypotheses_to_synthesize>

<task>
Synthesize these {len(hypotheses)} hypotheses into {target_count} final hypotheses.

Your goals:
1. MERGE overlapping hypotheses into stronger combined versions
2. PRESERVE diversity - keep different approaches represented
3. PRIORITIZE by potential impact and novelty
4. ENSURE coherence - final set should form a logical research agenda
5. REFINE wording for clarity and precision

For each final hypothesis:
- Take the best elements from similar sources
- Ensure the experimental design is complete
- Verify cross-disciplinary connections are clear
- Keep novelty keywords comprehensive

Output exactly {target_count} synthesized hypotheses.
</task>

Respond with valid JSON:
```json
[
  {{
    "hypothesis": "Clear, testable statement",
    "rationale": "Combined rationale from merged sources",
    "cross_disciplinary_connection": "Fields connected",
    "experimental_design": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
    "expected_impact": "Combined impact statement",
    "novelty_keywords": ["keyword1", "keyword2", "keyword3"],
    "source_ids": ["id1", "id2"]
  }}
]
```
"""

    def _parse_response(
        self,
        response_text: str,
        target_count: int,
    ) -> list[Hypothesis]:
        """Parse synthesis response into Hypothesis objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            hypotheses = []
            for i, item in enumerate(data[:target_count]):
                try:
                    design = item.get("experimental_design", [])
                    if not isinstance(design, list):
                        design = []
                    if not design:
                        design = ["Design a controlled experiment to test the hypothesis."]

                    keywords = item.get("novelty_keywords", [])
                    if not isinstance(keywords, list):
                        keywords = []
                    if not keywords:
                        keywords = ["novelty", "research"]

                    h = Hypothesis(
                        id=f"final_{i}",
                        hypothesis=item.get("hypothesis", ""),
                        rationale=item.get("rationale", ""),
                        cross_disciplinary_connection=item.get("cross_disciplinary_connection", ""),
                        experimental_design=design,
                        expected_impact=item.get("expected_impact", ""),
                        novelty_keywords=keywords,
                        source_soul=self.role,
                    )
                    hypotheses.append(h)
                except Exception:
                    continue

            return hypotheses
        except json.JSONDecodeError:
            return []
