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

        return self._parse_response(response_text, target_count, hypotheses)

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
Supporting papers: {', '.join(h.supporting_papers) if h.supporting_papers else 'none'}
Evidence trace:
{chr(10).join(h.evidence_trace[:4]) if h.evidence_trace else 'none'}
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
6. PRESERVE GROUNDING - final hypotheses must keep or improve citation grounding

For each final hypothesis:
- Take the best elements from similar sources
- Ensure the experimental design is complete
- Verify cross-disciplinary connections are clear
- Keep novelty keywords comprehensive
- Include supporting_papers with pulled paper IDs
- Include evidence_trace entries with [paper_id] prefix

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
    "supporting_papers": ["paper_id_1", "paper_id_2"],
    "evidence_trace": ["[paper_id_1] concise supporting claim"],
    "source_ids": ["id1", "id2"]
  }}
]
```
"""

    def _parse_response(
        self,
        response_text: str,
        target_count: int,
        source_hypotheses: list[Hypothesis],
    ) -> list[Hypothesis]:
        """Parse synthesis response into Hypothesis objects."""
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]

            source_lookup: dict[str, Hypothesis] = {h.id: h for h in source_hypotheses if h.id}

            def _normalize_list(value: Any) -> list[str]:
                if not isinstance(value, list):
                    return []
                return [str(v).strip() for v in value if str(v).strip()]

            def _dedupe(items: list[str]) -> list[str]:
                return list(dict.fromkeys(items))

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

                    source_ids = _normalize_list(item.get("source_ids", []))
                    supporting_papers = _normalize_list(item.get("supporting_papers", []))
                    evidence_trace = _normalize_list(item.get("evidence_trace", []))
                    evidence_spans_raw = item.get("evidence_spans", [])
                    if not isinstance(evidence_spans_raw, list):
                        evidence_spans_raw = []
                    evidence_spans = []
                    for span in evidence_spans_raw:
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
                        evidence_spans.append(
                            {
                                "claim_text": claim_text,
                                "citation_id": citation_id,
                                "quote": quote,
                                "confidence": max(0.0, min(1.0, confidence_value)),
                            }
                        )
                    supported_facts = _normalize_list(item.get("supported_facts", []))
                    novel_inference = str(item.get("novel_inference", "")).strip()
                    unsupported_parts = _normalize_list(item.get("unsupported_parts", []))

                    # Merge grounding provenance from source hypotheses.
                    for source_id in source_ids:
                        src = source_lookup.get(source_id)
                        if not src:
                            continue
                        supporting_papers.extend(src.supporting_papers or [])
                        evidence_trace.extend(src.evidence_trace or [])
                        supported_facts.extend(src.supported_facts or [])
                        unsupported_parts.extend(src.unsupported_parts or [])
                        if not novel_inference and src.novel_inference:
                            novel_inference = src.novel_inference
                        evidence_spans.extend(
                            [
                                span.model_dump() if hasattr(span, "model_dump") else span
                                for span in (src.evidence_spans or [])
                                if isinstance(span, dict) or hasattr(span, "model_dump")
                            ]
                        )

                    supporting_papers = _dedupe(supporting_papers)
                    evidence_trace = _dedupe(evidence_trace)[:10]
                    supported_facts = _dedupe(supported_facts)[:8]
                    unsupported_parts = _dedupe(unsupported_parts)[:8]

                    h = Hypothesis(
                        id=f"final_{i}",
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
                        evidence_spans=evidence_spans[:10],
                        source_soul=self.role,
                    )
                    hypotheses.append(h)
                except Exception:
                    continue

            return hypotheses
        except json.JSONDecodeError:
            return []
