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
        count: int = 3,
    ) -> list[Hypothesis]:
        """Generate creative, cross-domain hypotheses."""
        prompt = self._build_prompt(topic, context, mode, count)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return []

        return self._parse_response(response_text, count)

    def _build_prompt(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
        count: int = 3,
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
Generate exactly {count} SPECIFIC, TECHNICAL cross-domain scientific hypotheses.

ANTI-VAGUENESS RULES:
- NO "applying principles from X" → SAY WHICH principle (e.g., "Carnot efficiency limits")
- NO "novel materials" → NAME them (e.g., "MXene Ti3C2Tx nanosheets")
- NO "machine learning" → SPECIFY the architecture (e.g., "transformer with 12 attention heads")
- EVERY hypothesis needs NUMBERS: percentages, concentrations, energies, rates

CROSS-DISCIPLINARY THINKING (Be like this):
- "The bioluminescent luciferin-luciferase system from Photinus pyralis could replace LEDs in low-power IoT sensors, achieving 0.02 W illumination at 560nm with 41% quantum yield"
- "Applying the Nash equilibrium from game theory to bacterial quorum sensing predicts that Vibrio fischeri will defect from cooperation when population density exceeds 10^7 CFU/mL"

Ask yourself:
1. What SPECIFIC mechanism from another field could work here?
2. What EXACT materials, molecules, or algorithms would implement this?
3. What QUANTITATIVE prediction would prove/disprove it?
4. What UNEXPECTED domain (biology, economics, physics, music theory) has solved this problem?

Grounding rules:
- Use only paper IDs from the pulled paper list in context (arXiv/PMID/DOI/S2)
- Each hypothesis must include 1-3 supporting_papers IDs
- Each hypothesis must include evidence_trace lines prefixed with [paper_id]

The Skeptic will reject anything vague. Be BOLD but also be SPECIFIC.
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

    def _parse_response(self, response_text: str, count: int = 3) -> list[Hypothesis]:
        """Parse LLM response into Hypothesis objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            hypotheses = []
            for i, item in enumerate(data[:count]):  # Use count instead of hardcoded limit
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

                    supporting_papers = item.get("supporting_papers", [])
                    if not isinstance(supporting_papers, list):
                        supporting_papers = []
                    supporting_papers = [
                        str(pid).strip() for pid in supporting_papers
                        if str(pid).strip()
                    ]

                    evidence_trace = item.get("evidence_trace", [])
                    if not isinstance(evidence_trace, list):
                        evidence_trace = []
                    evidence_trace = [
                        str(line).strip() for line in evidence_trace
                        if str(line).strip()
                    ][:8]

                    evidence_spans = item.get("evidence_spans", [])
                    if not isinstance(evidence_spans, list):
                        evidence_spans = []
                    normalized_spans = []
                    for span in evidence_spans:
                        if not isinstance(span, dict):
                            continue
                        claim_text = str(span.get("claim_text", "")).strip()
                        citation_id = str(span.get("citation_id", "")).strip()
                        quote = str(span.get("quote", "")).strip()
                        confidence = span.get("confidence", 0.5)
                        if not claim_text or not citation_id or not quote:
                            continue
                        try:
                            confidence_value = float(confidence)
                        except Exception:
                            confidence_value = 0.5
                        normalized_spans.append(
                            {
                                "claim_text": claim_text,
                                "citation_id": citation_id,
                                "quote": quote,
                                "confidence": max(0.0, min(1.0, confidence_value)),
                            }
                        )

                    supported_facts = item.get("supported_facts", [])
                    if not isinstance(supported_facts, list):
                        supported_facts = []
                    supported_facts = [str(f).strip() for f in supported_facts if str(f).strip()][:8]

                    novel_inference = str(item.get("novel_inference", "")).strip()

                    unsupported_parts = item.get("unsupported_parts", [])
                    if not isinstance(unsupported_parts, list):
                        unsupported_parts = []
                    unsupported_parts = [str(p).strip() for p in unsupported_parts if str(p).strip()][:8]

                    h = Hypothesis(
                        id=f"creative_{i}",
                        hypothesis=item.get("hypothesis", ""),
                        rationale=item.get("rationale", ""),
                        cross_disciplinary_connection=item.get("cross_disciplinary_connection", ""),
                        experimental_design=design,
                        expected_impact=item.get("expected_impact", ""),
                        novelty_keywords=keywords,
                        supporting_papers=supporting_papers,
                        evidence_trace=evidence_trace,
                        supported_facts=supported_facts,
                        novel_inference=novel_inference,
                        unsupported_parts=unsupported_parts,
                        evidence_spans=normalized_spans,
                        source_soul=self.role,
                    )
                    hypotheses.append(h)
                except Exception:
                    continue

            return hypotheses
        except json.JSONDecodeError:
            return []
