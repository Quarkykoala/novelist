"""
Claim Extractor — Structured extraction of claims from research papers.

This module extracts structured claims (results, methods, limitations, etc.)
from paper abstracts and stores them for gap analysis and hypothesis grounding.
"""

import json
import re
from typing import Any

from src.contracts.schemas import (
    ClaimType,
    ExtractedClaim,
    PaperSummary,
    QuantitativeData,
)
from src.soul.llm_client import LLMClient


CLAIM_EXTRACTION_PROMPT = """You are a scientific claim extractor. Analyze this paper abstract and extract ALL claims.

For each claim, identify:
1. claim_type: "result" (findings), "method" (techniques), "limitation" (constraints), "future_work" (suggestions), or "baseline" (state-of-art numbers)
2. statement: The claim in plain language (one sentence)
3. evidence: How the paper supports this (brief)
4. quantitative_data: If there are numbers, extract: metric, value, unit, conditions
5. entities_mentioned: Key technical terms (compounds, methods, materials, etc.)

IMPORTANT:
- Extract ALL quantitative data (percentages, cycles, temperatures, etc.)
- Be precise with numbers — these will bound future hypothesis predictions
- Mark significant results as baseline if they represent current state-of-art

Return a JSON array of claims. Example:

```json
[
  {{
    "claim_type": "result",
    "statement": "LiFePO4/graphene composite cathode achieves 1200 cycles with 92% capacity retention",
    "evidence": "Electrochemical testing with coin cells over 6 months",
    "quantitative_data": {{
      "metric": "cycle_life",
      "value": 1200,
      "unit": "cycles",
      "conditions": "0.5C charge/discharge, 25C",
      "is_sota": true
    }},
    "entities_mentioned": ["LiFePO4", "graphene", "cathode", "coin cell"]
  }},
  {{
    "claim_type": "limitation",
    "statement": "Fast charging above 2C causes significant capacity fade",
    "evidence": "Capacity dropped to 70% after 200 cycles at 3C",
    "quantitative_data": {{
      "metric": "capacity_retention",
      "value": 70,
      "unit": "%",
      "conditions": "3C charge rate, 200 cycles",
      "is_sota": false
    }},
    "entities_mentioned": ["fast charging", "capacity fade"]
  }}
]
```

Paper ID: {paper_id}
Title: {title}
Abstract:
{abstract}

Extract all claims as JSON:"""


class ClaimExtractor:
    """Extracts structured claims from research papers."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = LLMClient(model=model)
        self.extraction_cache: dict[str, list[ExtractedClaim]] = {}

    async def extract_claims(
        self,
        paper_id: str,
        title: str,
        abstract: str,
    ) -> list[ExtractedClaim]:
        """Extract structured claims from a paper abstract."""
        
        # Check cache
        if paper_id in self.extraction_cache:
            return self.extraction_cache[paper_id]

        prompt = CLAIM_EXTRACTION_PROMPT.format(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
        )

        response = await self.client.generate_content(prompt)
        claims = self._parse_claims(paper_id, response)
        
        # Cache results
        self.extraction_cache[paper_id] = claims
        return claims

    def _parse_claims(self, paper_id: str, response: str) -> list[ExtractedClaim]:
        """Parse LLM response into ExtractedClaim objects."""
        claims = []
        
        # Extract JSON from response
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return claims

        try:
            raw_claims = json.loads(json_match.group())
        except json.JSONDecodeError:
            return claims

        for raw in raw_claims:
            try:
                # Parse quantitative data if present
                quant_data = None
                if raw.get("quantitative_data"):
                    qd = raw["quantitative_data"]
                    quant_data = QuantitativeData(
                        metric=qd.get("metric", "unknown"),
                        value=float(qd.get("value", 0)),
                        unit=qd.get("unit", ""),
                        conditions=qd.get("conditions", ""),
                        is_sota=qd.get("is_sota", False),
                    )

                # Map claim type
                claim_type_str = raw.get("claim_type", "result").lower()
                claim_type_map = {
                    "result": ClaimType.RESULT,
                    "method": ClaimType.METHOD,
                    "limitation": ClaimType.LIMITATION,
                    "future_work": ClaimType.FUTURE_WORK,
                    "baseline": ClaimType.BASELINE,
                }
                claim_type = claim_type_map.get(claim_type_str, ClaimType.RESULT)

                claim = ExtractedClaim(
                    paper_id=paper_id,
                    claim_type=claim_type,
                    statement=raw.get("statement", ""),
                    evidence=raw.get("evidence", ""),
                    quantitative_data=quant_data,
                    entities_mentioned=raw.get("entities_mentioned", []),
                    confidence=0.8,  # Default confidence for LLM extraction
                )
                claims.append(claim)

            except Exception:
                # Skip malformed claims
                continue

        return claims

    async def extract_from_summary(
        self,
        summary: PaperSummary,
        abstract: str,
    ) -> list[ExtractedClaim]:
        """Extract claims using existing summary for context."""
        # Combine summary fields with abstract for richer extraction
        enhanced_abstract = f"""
{abstract}

Key findings from analysis:
- Problem: {summary.problem}
- Method: {summary.method}
- Result: {summary.key_result}
- Limitations: {summary.limitations}
- Future work: {summary.future_work}
"""
        return await self.extract_claims(
            paper_id=summary.arxiv_id,
            title="",  # Title often embedded in abstract
            abstract=enhanced_abstract,
        )

    def get_sota_claims(self) -> list[ExtractedClaim]:
        """Get all claims marked as state-of-the-art."""
        sota = []
        for claims in self.extraction_cache.values():
            for claim in claims:
                if claim.quantitative_data and claim.quantitative_data.is_sota:
                    sota.append(claim)
        return sota

    def get_claims_by_type(self, claim_type: ClaimType) -> list[ExtractedClaim]:
        """Get all claims of a specific type."""
        result = []
        for claims in self.extraction_cache.values():
            for claim in claims:
                if claim.claim_type == claim_type:
                    result.append(claim)
        return result

    def get_all_entities(self) -> set[str]:
        """Get all unique entities mentioned across claims."""
        entities = set()
        for claims in self.extraction_cache.values():
            for claim in claims:
                entities.update(claim.entities_mentioned)
        return entities
