"""
Gap Analyzer — Identify research gaps from extracted claims.

This module analyzes extracted claims and concept maps to identify:
- Missing connections between concepts
- Contradictions between papers
- Unexplored parameter ranges
- Cross-domain opportunities
"""

import json
import re
from collections import defaultdict
from itertools import combinations

from src.contracts.schemas import (
    ClaimType,
    ConceptMap,
    ExtractedClaim,
    GapType,
    IdentifiedGap,
)
from src.soul.llm_client import LLMClient


GAP_IDENTIFICATION_PROMPT = """You are a research gap analyst. Given these claims from multiple papers, identify research gaps.

Types of gaps to find:
1. MISSING_CONNECTION: Two concepts are each studied but never combined
2. CONTRADICTION: Papers disagree on a finding
3. UNEXPLORED_RANGE: A parameter is only tested in a narrow range
4. CROSS_DOMAIN: A method from one field could apply to another
5. MECHANISM_UNKNOWN: An effect is observed but the mechanism is unclear

For each gap, provide:
- gap_type: one of the types above
- description: Clear statement of what's missing (20+ chars)
- concept_a: First concept involved
- concept_b: Second concept (if applicable)
- potential_value: Why this matters
- difficulty: low, medium, or high

Claims from papers:
{claims_text}

Known entities in this domain:
{entities}

Return a JSON array of gaps. Be specific — vague gaps are useless.

Example output:
```json
[
  {{
    "gap_type": "MISSING_CONNECTION",
    "description": "Graphene-based anodes have been studied for Li-ion batteries, and ZnO has been used to suppress dendrites in Li-metal batteries, but no study combines ZnO with graphene anodes in Li-ion cells",
    "concept_a": "graphene anode",
    "concept_b": "ZnO dendrite suppression",
    "potential_value": "Could extend cycle life by 50%+ without switching to Li-metal",
    "difficulty": "medium"
  }}
]
```

Identify all gaps:"""


class GapAnalyzer:
    """Identifies research gaps from extracted claims and concept maps."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = LLMClient(model=model)
        self.claims: list[ExtractedClaim] = []
        self.gaps: list[IdentifiedGap] = []

    def add_claims(self, claims: list[ExtractedClaim]) -> None:
        """Add extracted claims to the analyzer."""
        self.claims.extend(claims)

    async def analyze(
        self,
        concept_map: ConceptMap | None = None,
    ) -> list[IdentifiedGap]:
        """Analyze claims and concept map to find gaps."""
        
        # First, find structural gaps from concept map
        structural_gaps = self._find_structural_gaps(concept_map) if concept_map else []
        
        # Then, use LLM to find semantic gaps from claims
        semantic_gaps = await self._find_semantic_gaps()
        
        # Combine and deduplicate
        self.gaps = self._merge_gaps(structural_gaps, semantic_gaps)
        return self.gaps

    def _find_structural_gaps(self, concept_map: ConceptMap) -> list[IdentifiedGap]:
        """Find gaps based on concept map structure."""
        gaps = []
        
        # Build adjacency set
        connected = set()
        for edge in concept_map.edges:
            connected.add((edge.source, edge.target))
            connected.add((edge.target, edge.source))  # Bidirectional
        
        # Find high-frequency nodes that aren't connected
        node_freq = {n.id: n.frequency for n in concept_map.nodes}
        high_freq_nodes = [n for n in concept_map.nodes if n.frequency >= 2]
        
        for node_a, node_b in combinations(high_freq_nodes, 2):
            if (node_a.id, node_b.id) not in connected:
                # Potential missing connection
                gaps.append(IdentifiedGap(
                    gap_type=GapType.MISSING_CONNECTION,
                    description=f"'{node_a.name}' and '{node_b.name}' are both frequently discussed but no paper connects them directly",
                    concept_a=node_a.name,
                    concept_b=node_b.name,
                    supporting_evidence=list(set(node_a.source_papers + node_b.source_papers)),
                    potential_value=f"Connecting these could bridge {len(node_a.source_papers)} + {len(node_b.source_papers)} papers",
                    difficulty="medium",
                ))
        
        # Add any gaps already identified in concept map
        for gap_dict in concept_map.gaps:
            gaps.append(IdentifiedGap(
                gap_type=GapType.MISSING_CONNECTION,
                description=gap_dict.get("description", "Missing connection identified"),
                concept_a=gap_dict.get("concept_a", ""),
                concept_b=gap_dict.get("concept_b", ""),
                potential_value=gap_dict.get("potential_value", "Unknown"),
                difficulty="medium",
            ))
        
        # Add contradictions from concept map
        for contra_dict in concept_map.contradictions:
            gaps.append(IdentifiedGap(
                gap_type=GapType.CONTRADICTION,
                description=contra_dict.get("description", "Contradiction found"),
                concept_a=contra_dict.get("claim_a", ""),
                concept_b=contra_dict.get("claim_b", ""),
                supporting_evidence=contra_dict.get("papers", []),
                potential_value="Resolving contradictions can clarify the field",
                difficulty="high",
            ))
        
        return gaps

    async def _find_semantic_gaps(self) -> list[IdentifiedGap]:
        """Use LLM to find semantic gaps from claims."""
        if not self.claims:
            return []
        
        # Format claims for prompt
        claims_text = self._format_claims_for_prompt()
        entities = self._get_all_entities()
        
        prompt = GAP_IDENTIFICATION_PROMPT.format(
            claims_text=claims_text,
            entities=", ".join(list(entities)[:50]),  # Limit entities
        )
        
        response = await self.client.generate_content(prompt)
        return self._parse_gaps(response)

    def _format_claims_for_prompt(self) -> str:
        """Format claims for LLM prompt."""
        lines = []
        
        # Group by paper
        by_paper: dict[str, list[ExtractedClaim]] = defaultdict(list)
        for claim in self.claims:
            by_paper[claim.paper_id].append(claim)
        
        for paper_id, claims in list(by_paper.items())[:10]:  # Limit papers
            lines.append(f"\n## Paper {paper_id}")
            for claim in claims[:5]:  # Limit claims per paper
                type_str = claim.claim_type.value.upper()
                lines.append(f"- [{type_str}] {claim.statement}")
                if claim.quantitative_data:
                    qd = claim.quantitative_data
                    lines.append(f"  → {qd.metric}: {qd.value} {qd.unit} ({qd.conditions})")
        
        return "\n".join(lines)

    def _get_all_entities(self) -> set[str]:
        """Get all unique entities from claims."""
        entities = set()
        for claim in self.claims:
            entities.update(claim.entities_mentioned)
        return entities

    def _parse_gaps(self, response: str) -> list[IdentifiedGap]:
        """Parse LLM response into IdentifiedGap objects."""
        gaps = []
        
        json_match = re.search(r'\[[\s\S]*\]', response)
        if not json_match:
            return gaps
        
        try:
            raw_gaps = json.loads(json_match.group())
        except json.JSONDecodeError:
            return gaps
        
        gap_type_map = {
            "MISSING_CONNECTION": GapType.MISSING_CONNECTION,
            "CONTRADICTION": GapType.CONTRADICTION,
            "UNEXPLORED_RANGE": GapType.UNEXPLORED_RANGE,
            "CROSS_DOMAIN": GapType.CROSS_DOMAIN,
            "SCALE_GAP": GapType.SCALE_GAP,
            "MECHANISM_UNKNOWN": GapType.MECHANISM_UNKNOWN,
        }
        
        for raw in raw_gaps:
            try:
                gap_type_str = raw.get("gap_type", "MISSING_CONNECTION").upper()
                gap_type = gap_type_map.get(gap_type_str, GapType.MISSING_CONNECTION)
                
                gap = IdentifiedGap(
                    gap_type=gap_type,
                    description=raw.get("description", ""),
                    concept_a=raw.get("concept_a", ""),
                    concept_b=raw.get("concept_b", ""),
                    potential_value=raw.get("potential_value", ""),
                    difficulty=raw.get("difficulty", "medium"),
                )
                
                if len(gap.description) >= 20:  # Only keep substantive gaps
                    gaps.append(gap)
                    
            except Exception:
                continue
        
        return gaps

    def _merge_gaps(
        self,
        structural: list[IdentifiedGap],
        semantic: list[IdentifiedGap],
    ) -> list[IdentifiedGap]:
        """Merge and deduplicate gaps from different sources."""
        all_gaps = structural + semantic
        
        # Simple deduplication based on concept overlap
        seen = set()
        unique = []
        
        for gap in all_gaps:
            key = (gap.concept_a.lower(), gap.concept_b.lower(), gap.gap_type)
            if key not in seen:
                seen.add(key)
                unique.append(gap)
        
        return unique

    def get_gaps_by_type(self, gap_type: GapType) -> list[IdentifiedGap]:
        """Get gaps of a specific type."""
        return [g for g in self.gaps if g.gap_type == gap_type]

    def get_high_value_gaps(self, limit: int = 5) -> list[IdentifiedGap]:
        """Get gaps sorted by potential value (heuristic)."""
        # Prioritize: CONTRADICTION > MISSING_CONNECTION > others
        priority = {
            GapType.CONTRADICTION: 0,
            GapType.MISSING_CONNECTION: 1,
            GapType.MECHANISM_UNKNOWN: 2,
            GapType.CROSS_DOMAIN: 3,
            GapType.UNEXPLORED_RANGE: 4,
            GapType.SCALE_GAP: 5,
        }
        
        sorted_gaps = sorted(self.gaps, key=lambda g: priority.get(g.gap_type, 10))
        return sorted_gaps[:limit]
