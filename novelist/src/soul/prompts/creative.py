"""Creative Soul - wild cross-domain hypothesis generation.

The Creative soul pushes for unexpected connections and novel ideas.
Uses SuperPrompt-inspired techniques for divergent thinking.
"""

import asyncio
import json
from typing import Any

from src.contracts.schemas import GenerationMode, Hypothesis, SoulRole
from src.contracts.validators import extract_json_from_response
from src.soul.prompts.base import (
    BaseSoul,
    HYPOTHESIS_OUTPUT_SCHEMA,
    format_context_for_prompt,
)


class CreativeSoul(BaseSoul):
    """The Creative soul generates wild, cross-domain hypotheses."""

    role = SoulRole.CREATIVE
    description = """You are the CREATIVE voice in the research collective.
Your superpower is making unexpected connections across disciplines.
You think in analogies, metaphors, and cross-domain transfers.
You are NOT afraid to propose bold, unconventional ideas.
You draw inspiration from diverse fields: biology, physics, computer science, 
psychology, economics, art, and more.

Your hypotheses should surprise and inspire, even if they seem unusual at first."""

    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> list[Hypothesis]:
        """Generate creative, cross-domain hypotheses."""
        prompt = self._build_prompt(topic, context, mode)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return []

        return self._parse_response(response_text)

    def _build_prompt(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> str:
        """Build the generation prompt with SuperPrompt-style metadata."""

        # SuperPrompt-inspired metadata for creative thinking
        metadata = """<prompt_metadata>
Type: Cross-Domain Scientific Synthesis
Purpose: Generate Novel Research Hypotheses
Paradigm: Analogical Reasoning + Conceptual Blending
Constraints: Must be falsifiable, must connect 2+ domains
Objective: Surprise with unexpected yet plausible connections
</prompt_metadata>"""

        context_str = format_context_for_prompt(context)

        # Fallback: If context is empty, force random injection to ensure output
        if context_str.strip() == "No prior context available." and mode != GenerationMode.RANDOM_INJECTION:
            mode = GenerationMode.RANDOM_INJECTION
            context_str += "\n(Context is empty, switching to RANDOM_INJECTION mode)"

        mode_instruction = self._get_mode_instruction(mode)

        return f"""{metadata}

{self.get_persona_prompt()}

<research_topic>
{topic}
</research_topic>

<context>
{context_str}
</context>

<generation_mode>
{mode_instruction}
</generation_mode>

<task>
Generate 3-5 creative, cross-domain scientific hypotheses about this topic.

Think like this:
1. What patterns from other fields might apply here?
2. What if we inverted a common assumption?
3. What would a physicist/economist/artist see that a domain expert might miss?
4. What surprising analogy could reveal new insights?

Be BOLD. The Skeptic will filter out bad ideas later.
</task>

{HYPOTHESIS_OUTPUT_SCHEMA}
"""

    def _get_mode_instruction(self, mode: GenerationMode) -> str:
        """Get specific instructions based on generation mode."""
        instructions = {
            GenerationMode.GAP_HUNT: """Focus on GAPS in the research landscape.
What hasn't been connected? What obvious combination is missing?""",

            GenerationMode.CONTRADICTION_HUNT: """Focus on CONTRADICTIONS.
What assumptions might be wrong? What if the opposite were true?""",

            GenerationMode.ANALOGY_TRANSFER: """Focus on ANALOGIES from other fields.
What patterns from physics/economics/ecology/etc. might apply here?""",

            GenerationMode.CONSTRAINT_RELAX: """Focus on RELAXING CONSTRAINTS.
What if a key limitation didn't exist? What becomes possible?""",

            GenerationMode.RANDOM_INJECTION: """Focus on UNEXPECTED CONNECTIONS.
Bring in ideas from completely unrelated fields. Be wild!""",
        }
        return instructions.get(mode, instructions[GenerationMode.GAP_HUNT])

    def _parse_response(self, response_text: str) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            hypotheses = []
            for i, item in enumerate(data[:5]):  # Max 5 hypotheses
                try:
                    h = Hypothesis(
                        id=f"creative_{i}",
                        hypothesis=item.get("hypothesis", ""),
                        rationale=item.get("rationale", ""),
                        cross_disciplinary_connection=item.get("cross_disciplinary_connection", ""),
                        experimental_design=item.get("experimental_design", []),
                        expected_impact=item.get("expected_impact", ""),
                        novelty_keywords=item.get("novelty_keywords", []),
                        source_soul=self.role,
                    )
                    hypotheses.append(h)
                except Exception:
                    continue

            return hypotheses
        except json.JSONDecodeError:
            return []
