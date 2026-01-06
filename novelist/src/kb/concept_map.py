"""Concept map extraction and management.

Builds a knowledge graph from paper summaries:
- Nodes: Entities (methods, datasets, metrics, etc.)
- Edges: Relations (improves, uses, compares, etc.)
- Gaps: Missing expected connections
- Contradictions: Conflicting claims
"""

import asyncio
import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.contracts.schemas import (
    ConceptEdge,
    ConceptMap,
    ConceptNode,
    PaperSummary,
)
from src.contracts.validators import extract_json_from_response

load_dotenv()

FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash")

RELATION_EXTRACTION_PROMPT = """Analyze these paper summaries and extract relationships between entities.

<papers>
{papers_text}
</papers>

For each paper, identify relationships between entities (methods, datasets, concepts).
Common relation types:
- "improves": X improves upon Y
- "uses": X uses/applies Y  
- "extends": X extends/builds on Y
- "compares": X is compared to Y
- "enables": X enables/allows Y
- "outperforms": X outperforms Y
- "combines": X combines with Y
- "addresses": X addresses problem Y

Respond with valid JSON.
Example:
```json
{{
  "relations": [
    {{"source": "bert", "target": "sentiment_analysis", "relation": "improves", "paper_id": "2301.001"}},
    {{"source": "f1_score", "target": "accuracy", "relation": "compares", "paper_id": "2301.001"}}
  ]
}}
```

Requirements:
1. Use exact entity names as they appear in the entities list.
2. Relation must be one of the types above or similar.
3. Extract 3-5 relations per paper.
"""


class ConceptMapBuilder:
    """Builds and manages concept maps from paper summaries."""

    def __init__(self, model: str = FLASH_MODEL):
        self.model = model
        from src.soul.llm_client import LLMClient
        self.client = LLMClient(model=model)
        self.concept_map = ConceptMap()
        self.total_tokens = 0
        self.total_cost = 0.0

    def add_entities_from_summaries(self, summaries: list[PaperSummary]) -> None:
        """Add entities from paper summaries to the concept map.

        Args:
            summaries: List of paper summaries with extracted entities
        """
        for summary in summaries:
            for entity in summary.extracted_entities:
                self._add_or_update_node(entity, summary.arxiv_id)

    def _add_or_update_node(self, entity: str, paper_id: str) -> None:
        """Add a new node or update frequency of existing node."""
        # Normalize entity name
        normalized = entity.strip().lower()
        node_id = normalized.replace(" ", "_")

        # Check if node exists
        existing = next((n for n in self.concept_map.nodes if n.id == node_id), None)

        if existing:
            existing.frequency += 1
            if paper_id not in existing.source_papers:
                existing.source_papers.append(paper_id)
        else:
            node = ConceptNode(
                id=node_id,
                name=entity,
                type=self._infer_node_type(entity),
                frequency=1,
                source_papers=[paper_id],
            )
            self.concept_map.nodes.append(node)

    def _infer_node_type(self, entity: str) -> str:
        """Infer the type of an entity based on naming patterns."""
        entity_lower = entity.lower()

        # Dataset patterns
        if any(kw in entity_lower for kw in ["dataset", "benchmark", "corpus"]):
            return "dataset"

        # Metric patterns
        if any(kw in entity_lower for kw in ["accuracy", "f1", "bleu", "rouge", "score", "loss"]):
            return "metric"

        # Method/model patterns
        if any(kw in entity_lower for kw in ["net", "bert", "gpt", "gan", "model", "algorithm"]):
            return "method"

        # Biological patterns
        if any(kw in entity_lower for kw in ["crispr", "cas", "gene", "protein", "cell", "dna", "rna"]):
            return "biological"

        return "concept"

    async def extract_relations(self, summaries: list[PaperSummary]) -> list[ConceptEdge]:
        """Use LLM to extract relations between entities.

        Args:
            summaries: Paper summaries to analyze

        Returns:
            List of extracted edges
        """
        if not summaries:
            return []

        # Format papers for prompt
        papers_text = ""
        for summary in summaries[:10]:  # Limit to 10 papers per call
            papers_text += f"""
Paper [{summary.arxiv_id}]:
- Problem: {summary.problem}
- Method: {summary.method}
- Result: {summary.key_result}
- Entities: {', '.join(summary.extracted_entities)}
"""

        prompt = RELATION_EXTRACTION_PROMPT.format(papers_text=papers_text)

        try:
            # Simple retry loop
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response_obj = await self.client.generate_content(prompt)
                    # Check for GenerationResponse vs legacy string
                    if hasattr(response_obj, 'content'):
                        response_text = response_obj.content
                        if hasattr(response_obj, 'usage'):
                             self.total_tokens += response_obj.usage.total_tokens
                             self.total_cost += response_obj.usage.cost_usd
                    else:
                        response_text = str(response_obj) # Fallback

                    if not response_text:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return []

                    json_str = extract_json_from_response(response_text)
                    if json_str:
                        data = json.loads(json_str)
                        relations = data.get("relations", [])

                        edges: list[ConceptEdge] = []
                        for rel in relations:
                            edge = ConceptEdge(
                                source=rel.get("source", "").lower().replace(" ", "_"),
                                target=rel.get("target", "").lower().replace(" ", "_"),
                                relation=rel.get("relation", "related"),
                                confidence=0.7,
                                source_papers=[rel.get("paper_id", "")],
                            )
                            edges.append(edge)
                            self.concept_map.edges.append(edge)
                        return edges
                    else:
                         # JSON extraction failed
                         print(f"[WARN] Failed to extract JSON from relation extraction (Attempt {attempt+1})")
                
                except Exception as e:
                    print(f"[WARN] Error in relation extraction (Attempt {attempt+1}): {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise e
        except Exception:
            pass

        fallback_edges = self._fallback_relations(summaries)
        self.concept_map.edges.extend(fallback_edges)
        return fallback_edges

    def _fallback_relations(self, summaries: list[PaperSummary]) -> list[ConceptEdge]:
        """Heuristic relation extraction when LLM JSON is missing/invalid."""
        edges: list[ConceptEdge] = []
        seen_pairs: set[tuple[str, str]] = set()

        for summary in summaries:
            entities = [e.strip() for e in summary.extracted_entities if e.strip()]
            if len(entities) < 2:
                continue

            for a, b in combinations(entities[:8], 2):
                src = a.lower().replace(" ", "_")
                tgt = b.lower().replace(" ", "_")
                key = tuple(sorted((src, tgt)))

                if key in seen_pairs:
                    continue

                seen_pairs.add(key)
                edge = ConceptEdge(
                    source=src,
                    target=tgt,
                    relation="co_occurs",
                    confidence=0.4,
                    source_papers=[summary.arxiv_id],
                )
                edges.append(edge)

        return edges

    def identify_gaps(self) -> list[dict[str, Any]]:
        """Identify potential gaps in the concept map.

        Gaps are pairs of entities that might have a relationship
        but no edge exists between them.
        """
        gaps: list[dict[str, Any]] = []

        # Find high-frequency nodes that aren't connected
        high_freq_nodes = [n for n in self.concept_map.nodes if n.frequency >= 2]

        # Check for missing connections between high-frequency nodes
        connected_pairs: set[tuple[str, str]] = set()
        for edge in self.concept_map.edges:
            connected_pairs.add((edge.source, edge.target))
            connected_pairs.add((edge.target, edge.source))  # Both directions

        for i, node1 in enumerate(high_freq_nodes):
            for node2 in high_freq_nodes[i + 1:]:
                # Check if they share papers (likely related but no edge)
                shared_papers = set(node1.source_papers) & set(node2.source_papers)
                if shared_papers and (node1.id, node2.id) not in connected_pairs:
                    gaps.append({
                        "node1": node1.name,
                        "node2": node2.name,
                        "shared_papers": list(shared_papers),
                        "potential": "high" if len(shared_papers) > 1 else "medium",
                    })

        self.concept_map.gaps = gaps
        return gaps

    def identify_contradictions(self) -> list[dict[str, Any]]:
        """Identify potential contradictions in the concept map.

        Contradictions are conflicting edges (e.g., A improves B and B outperforms A).
        """
        contradictions: list[dict[str, Any]] = []

        conflicting_relations = {
            ("improves", "outperforms"),
            ("outperforms", "improves"),
        }

        edge_lookup: dict[tuple[str, str], list[ConceptEdge]] = {}
        for edge in self.concept_map.edges:
            key = (edge.source, edge.target)
            if key not in edge_lookup:
                edge_lookup[key] = []
            edge_lookup[key].append(edge)

        for (src, tgt), edges in edge_lookup.items():
            reverse_key = (tgt, src)
            if reverse_key in edge_lookup:
                for edge1 in edges:
                    for edge2 in edge_lookup[reverse_key]:
                        rel_pair = (edge1.relation, edge2.relation)
                        if rel_pair in conflicting_relations:
                            contradictions.append({
                                "edge1": {"source": src, "target": tgt, "relation": edge1.relation},
                                "edge2": {"source": tgt, "target": src, "relation": edge2.relation},
                                "conflict_type": "reverse_claim",
                            })

        self.concept_map.contradictions = contradictions
        return contradictions

    async def build_from_summaries(self, summaries: list[PaperSummary]) -> ConceptMap:
        """Build complete concept map from paper summaries.

        Args:
            summaries: List of paper summaries

        Returns:
            Complete ConceptMap with nodes, edges, gaps, and contradictions
        """
        # Add entities
        self.add_entities_from_summaries(summaries)

        # Extract relations
        await self.extract_relations(summaries)

        # Identify gaps and contradictions
        self.identify_gaps()
        self.identify_contradictions()

        return self.concept_map

    def save(self, path: Path) -> None:
        """Save concept map to JSON files.

        Args:
            path: Directory to save to
        """
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "concepts.json", "w") as f:
            nodes_data = [n.model_dump() for n in self.concept_map.nodes]
            json.dump(nodes_data, f, indent=2, default=str)

        with open(path / "edges.json", "w") as f:
            edges_data = [e.model_dump() for e in self.concept_map.edges]
            json.dump(edges_data, f, indent=2, default=str)

        with open(path / "gaps.json", "w") as f:
            json.dump(self.concept_map.gaps, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ConceptMapBuilder":
        """Load concept map from JSON files.

        Args:
            path: Directory to load from

        Returns:
            ConceptMapBuilder with loaded map
        """
        builder = cls()

        concepts_file = path / "concepts.json"
        if concepts_file.exists():
            with open(concepts_file) as f:
                nodes_data = json.load(f)
                builder.concept_map.nodes = [ConceptNode(**n) for n in nodes_data]

        edges_file = path / "edges.json"
        if edges_file.exists():
            with open(edges_file) as f:
                edges_data = json.load(f)
                builder.concept_map.edges = [ConceptEdge(**e) for e in edges_data]

        gaps_file = path / "gaps.json"
        if gaps_file.exists():
            with open(gaps_file) as f:
                builder.concept_map.gaps = json.load(f)

        return builder
