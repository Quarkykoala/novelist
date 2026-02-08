"""Risk-Taker Soul - pushes for bold, high-upside hypotheses.

The Risk-Taker challenges the collective to think bigger and bolder.
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


class RiskTakerSoul(BaseSoul):
    """The Risk-Taker soul pushes for bold hypotheses."""

    role = SoulRole.RISK_TAKER
    description = """You are the RISK-TAKER voice in the research collective.
Your role is to push boundaries and champion high-upside ideas.
You challenge "safe" hypotheses and ask:
- What if we 10x the ambition?
- What paradigm-shifting discovery could this enable?
- What would make this worthy of a Nature paper?

You value potential impact over probability of success.
Better to propose something revolutionary that might fail 
than something incremental that's guaranteed to work."""

    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
        count: int = 3,
    ) -> list[Hypothesis]:
        """Generate bold, high-upside hypotheses."""
        prompt = self._build_prompt(topic, context, mode, count)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return []

        return self._parse_response(response_text, count)

    async def amplify(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
    ) -> list[Hypothesis]:
        """Amplify existing hypotheses to be bolder.

        Takes safe/incremental hypotheses and pushes them to be more ambitious.
        """
        prompt = self._build_amplify_prompt(hypotheses, topic)

        response_text = await self._call_model_with_retry(prompt)
        if not response_text:
            return hypotheses

        return self._parse_response(response_text, len(hypotheses))

    def _build_prompt(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
        count: int = 3,
    ) -> str:
        """Build generation prompt for bold hypotheses."""

        context_str = format_context_for_prompt(context)

        # Fallback: If context is empty, force random injection
        if context_str.strip() == "No prior context available." and mode != GenerationMode.RANDOM_INJECTION:
            mode = GenerationMode.RANDOM_INJECTION
            context_str += "\n(Context is empty, switching to RANDOM_INJECTION mode)"

        return f"""{self.get_persona_prompt()}

<research_topic>
{topic}
</research_topic>

<context>
{context_str}
</context>

<task>
Generate exactly {count} BOLD, HIGH-UPSIDE scientific hypotheses.

Requirements:
- Each hypothesis should have paradigm-shifting potential
- Think about what could lead to a breakthrough discovery
- Consider hypotheses that most researchers would consider "too ambitious"
- Focus on potential IMPACT more than probability of success
- Use only paper IDs from the pulled paper list in context (arXiv/PMID/DOI/S2)
- Include 1-3 supporting_papers IDs per hypothesis
- Include evidence_trace lines prefixed with [paper_id]

Ask yourself:
- "What discovery would fundamentally change this field?"
- "What 'impossible' thing might actually be possible?"
- "What would win a Nobel Prize if validated?"

Be audacious. The Skeptic will filter out bad ideas later.
</task>

{HYPOTHESIS_OUTPUT_SCHEMA}
"""

    def _build_amplify_prompt(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
    ) -> str:
        """Build prompt to amplify existing hypotheses."""

        hyp_text = ""
        for i, h in enumerate(hypotheses):
            hyp_text += f"""
{i+1}. {h.hypothesis}
   Impact: {h.expected_impact}
"""

        return f"""{self.get_persona_prompt()}

<research_topic>
{topic}
</research_topic>

<current_hypotheses>
{hyp_text}
</current_hypotheses>

<task>
These hypotheses feel too SAFE. Amplify them!

For each hypothesis, create a BOLDER version that:
- Increases the scope or ambition by 10x
- Targets a more fundamental mechanism
- Could enable a paradigm shift if true
- Has higher potential impact (even if harder to achieve)

Transform incremental ideas into revolutionary ones.
</task>

{HYPOTHESIS_OUTPUT_SCHEMA}
"""

    def _parse_response(self, response_text: str, count: int = 3) -> list[Hypothesis]:
        """Parse response into Hypothesis objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            hypotheses = []
            for i, item in enumerate(data[:count]):
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
                        id=f"risktaker_{i}",
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
