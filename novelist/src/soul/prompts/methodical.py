"""Methodical Soul - ensures experimental rigor and protocol completeness.

The Methodical soul enforces concrete, executable experimental designs.
"""

import asyncio
import json
from typing import Any

from src.contracts.schemas import GenerationMode, Hypothesis, SoulRole
from src.contracts.validators import extract_json_from_response
from src.soul.prompts.base import BaseSoul


class MethodicalSoul(BaseSoul):
    """The Methodical soul ensures experimental rigor."""

    role = SoulRole.METHODICAL
    description = """You are the METHODICAL voice in the research collective.
Your role is to ensure hypotheses have rigorous, executable experimental designs.
You think step-by-step and demand:
- Clear, measurable outcomes
- Specific tools, datasets, and methods
- Control conditions and baselines
- Statistical significance considerations
- Realistic timeline and resource estimates

You transform vague ideas into concrete research protocols."""

    async def enhance(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
    ) -> list[Hypothesis]:
        """Enhance hypotheses with rigorous experimental designs.

        Args:
            hypotheses: Hypotheses needing protocol enhancement
            topic: Research topic for context

        Returns:
            Enhanced hypotheses with complete protocols
        """
        prompt = self._build_enhance_prompt(hypotheses, topic)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return hypotheses

        return self._parse_enhanced(response_text, hypotheses)

    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> list[Hypothesis]:
        """Methodical primarily enhances, not generates."""
        return []

    def _build_enhance_prompt(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
    ) -> str:
        """Build prompt for protocol enhancement."""

        hypotheses_text = ""
        for i, h in enumerate(hypotheses):
            hypotheses_text += f"""
### Hypothesis {i + 1} (ID: {h.id})
Statement: {h.hypothesis}
Current experimental design:
{chr(10).join(f'  {j+1}. {step}' for j, step in enumerate(h.experimental_design))}
"""

        return f"""{self.get_persona_prompt()}

<research_topic>
{topic}
</research_topic>

<hypotheses>
{hypotheses_text}
</hypotheses>

<task>
For each hypothesis, improve the experimental design to be:

1. SPECIFIC: Name exact tools, datasets, metrics
2. MEASURABLE: Define quantitative success criteria
3. CONTROLLED: Include control conditions and baselines
4. MINIMAL: Remove unnecessary steps, keep it focused
5. COMPLETE: Ensure all necessary steps are present (5-7 steps ideal)

Return the improved experimental designs.
</task>

Respond with valid JSON:
```json
[
  {{
    "hypothesis_id": "creative_0",
    "improved_design": [
      "Step 1: [Specific action with named tools/data]",
      "Step 2: [Concrete step with measurable outcome]",
      "Step 3: [Control condition specification]",
      "Step 4: [Data collection with metrics]",
      "Step 5: [Analysis with statistical test]"
    ],
    "success_criteria": "Specific measurable outcome that confirms/refutes hypothesis"
  }}
]
```
"""

    def _parse_enhanced(
        self,
        response_text: str,
        original: list[Hypothesis],
    ) -> list[Hypothesis]:
        """Parse enhancement response and update hypotheses."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return original

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            # Create lookup for originals
            original_lookup = {h.id: h for h in original}

            enhanced: list[Hypothesis] = []
            for item in data:
                h_id = item.get("hypothesis_id", "")
                if h_id in original_lookup:
                    h = original_lookup[h_id].model_copy()
                    if item.get("improved_design"):
                        h.experimental_design = item["improved_design"]
                    enhanced.append(h)
                    del original_lookup[h_id]

            # Add any that weren't enhanced
            enhanced.extend(original_lookup.values())

            return enhanced
        except json.JSONDecodeError:
            return original
