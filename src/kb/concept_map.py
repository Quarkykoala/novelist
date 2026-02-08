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
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any
from collections import Counter, defaultdict

from dotenv import load_dotenv

from src.contracts.schemas import (
    ConceptEdge,
    ConceptMap,
    ConceptNode,
    PaperSummary,
)
from src.contracts.validators import extract_json_from_response
from src.kb.embedding_service import EmbeddingService

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

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "into", "using", "use", "used", "via", "than", "over", "under", "between",
    "within", "without", "across", "through", "into", "onto", "can", "may",
    "our", "their", "these", "those", "which", "such", "also", "here", "there",
    "paper", "study", "method", "results", "show", "shows", "based", "model",
}


class ConceptMapBuilder:
    """Builds and manages concept maps from paper summaries."""

    def __init__(self, model: str = FLASH_MODEL, use_real_embeddings: bool = True):
        self.model = model
        from src.soul.llm_client import LLMClient
        self.client = LLMClient(model=model)
        self.concept_map = ConceptMap()
        self.total_tokens = 0
        self.total_cost = 0.0
        self.use_real_embeddings = use_real_embeddings
        self._embedding_service: EmbeddingService | None = None
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

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

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text for lightweight embedding similarity."""
        tokens = re.findall(r"[a-z0-9]{3,}", text.lower())
        return [t for t in tokens if t not in _STOPWORDS]

    async def build_global_map_from_abstracts(self, papers: list[Any]) -> ConceptMap:
        """Gemini 3 Competition Feature: Long-Context Global Mapping.
        Processes all paper abstracts at once to find 'Trans-Paper Gaps'.
        """
        if not papers:
            return self.concept_map

        # Format all papers into a single massive context block
        full_context = ""
        for i, p in enumerate(papers):
            full_context += f"PAPER [{p.arxiv_id}]:\nTitle: {p.title}\nAbstract: {p.abstract}\n\n"

        prompt = f"""You are a Strategic Research Analyst. Analyze the following collection of {len(papers)} research papers to build a GLOBAL CONCEPT MAP.

RESEARCH LANDSCAPE:
{full_context}

TASK:
1. Extract the top 20 most important ENTITIES (methods, chemicals, biological targets, etc.) across ALL papers.
2. Extract the 30 most critical RELATIONSHIPS (A uses B, X contradicts Y, M improves N).
3. Identify 5 'TRANS-PAPER GAPS': Connections between concepts that are implied by the data but NO SINGLE paper has explicitly explored yet.

Respond with valid JSON:
```json
{{
  "nodes": [
    {{"id": "node_id", "name": "Entity Name", "type": "method|biological|metric|etc"}},
    ...
  ],
  "edges": [
    {{"source": "id1", "target": "id2", "relation": "uses", "paper_ids": ["arxiv1", "arxiv2"]}},
    ...
  ],
  "trans_paper_gaps": [
    {{"description": "Gap description", "node_a": "id1", "node_b": "id2", "logic": "Why this connection is implied"}}
  ]
}}
```
"""

        try:
            response_obj = await self.client.generate_content(prompt)
            response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
            
            json_str = extract_json_from_response(response_text)
            if json_str:
                data = json.loads(json_str)
                
                # Update nodes
                for n in data.get("nodes", []):
                    self.concept_map.nodes.append(ConceptNode(
                        id=n["id"].lower().replace(" ", "_"),
                        name=n["name"],
                        type=n.get("type", "concept"),
                        frequency=1,
                        source_papers=[]
                    ))
                
                # Update edges
                for e in data.get("edges", []):
                    self.concept_map.edges.append(ConceptEdge(
                        source=e["source"].lower().replace(" ", "_"),
                        target=e["target"].lower().replace(" ", "_"),
                        relation=e["relation"],
                        confidence=0.8,
                        source_papers=e.get("paper_ids", [])
                    ))
                
                # Store gaps (we'll map these to IdentifiedGap in orchestrator later)
                self.concept_map.gaps = data.get("trans_paper_gaps", [])
                
                if hasattr(response_obj, 'usage'):
                    self.total_tokens += response_obj.usage.total_tokens
                    self.total_cost += response_obj.usage.cost_usd

        except Exception as e:
            print(f"[ERROR] Global Mapping failed: {e}")
            # Fallback to empty map or partial results

        # Always compute embedding-based overlaps (no LLM)
        self.identify_overlaps(papers)

        return self.concept_map

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

    def identify_overlaps(
        self,
        papers: list[Any],
        *,
        top_k: int = 8,
        min_similarity: float = 0.25,
    ) -> list[dict[str, Any]]:
        """Identify overlapping ideas using embedding similarity over abstracts."""
        if not papers or len(papers) < 2:
            self.concept_map.overlaps = []
            return []

        docs = []
        for p in papers:
            text = f"{getattr(p, 'title', '')} {getattr(p, 'abstract', '')}"
            tokens = self._tokenize(text)
            docs.append({
                "paper": p,
                "tokens": tokens,
            })

        # Build document frequencies
        df: dict[str, int] = defaultdict(int)
        for doc in docs:
            unique_tokens = set(doc["tokens"])
            for t in unique_tokens:
                df[t] += 1

        # Build TF-IDF vectors (sparse dicts)
        n_docs = len(docs)
        vectors: list[dict[str, float]] = []
        tf_maps: list[dict[str, float]] = []
        for doc in docs:
            tf = Counter(doc["tokens"])
            tf_maps.append(tf)
            vec: dict[str, float] = {}
            for token, count in tf.items():
                if df[token] < 2:
                    continue  # focus on overlap terms
                idf = math.log((n_docs + 1) / (df[token] + 1)) + 1.0
                vec[token] = float(count) * idf
            vectors.append(vec)

        # Compute pairwise cosine similarity
        def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
            if not a or not b:
                return 0.0
            dot = 0.0
            for t, v in a.items():
                if t in b:
                    dot += v * b[t]
            norm_a = math.sqrt(sum(v * v for v in a.values()))
            norm_b = math.sqrt(sum(v * v for v in b.values()))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        overlaps: list[dict[str, Any]] = []
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                sim = _cosine(vectors[i], vectors[j])
                if sim < min_similarity:
                    continue
                paper_a = docs[i]["paper"]
                paper_b = docs[j]["paper"]
                tokens_a = set(docs[i]["tokens"])
                tokens_b = set(docs[j]["tokens"])
                shared = tokens_a & tokens_b
                shared_terms = sorted(
                    shared,
                    key=lambda t: tf_maps[i].get(t, 0) + tf_maps[j].get(t, 0),
                    reverse=True,
                )[:8]
                cats_a = set(getattr(paper_a, "categories", []) or [])
                cats_b = set(getattr(paper_b, "categories", []) or [])
                cross_domain = len(cats_a & cats_b) == 0
                overlaps.append({
                    "paper_a": getattr(paper_a, "arxiv_id", ""),
                    "paper_b": getattr(paper_b, "arxiv_id", ""),
                    "similarity": round(sim, 3),
                    "shared_terms": shared_terms,
                    "cross_domain": cross_domain,
                    "categories_a": list(cats_a),
                    "categories_b": list(cats_b),
                })

        overlaps.sort(key=lambda o: o["similarity"], reverse=True)
        overlaps = overlaps[:top_k]
        self.concept_map.overlaps = overlaps
        return overlaps

    async def identify_overlaps_embedding(
        self,
        papers: list[Any],
        *,
        top_k: int = 8,
        min_similarity: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Identify overlapping ideas using dense embeddings from EmbeddingService.
        
        This method uses the Gemini text-embedding-004 model for high-quality
        semantic similarity instead of TF-IDF. Falls back to TF-IDF on error.
        
        Args:
            papers: List of papers with title and abstract
            top_k: Maximum number of overlaps to return
            min_similarity: Minimum cosine similarity threshold
            
        Returns:
            List of overlap dicts with paper pairs and similarity scores
        """
        if not papers or len(papers) < 2:
            self.concept_map.overlaps = []
            return []
        
        if not self.use_real_embeddings:
            # Fall back to TF-IDF-based method
            return self.identify_overlaps(papers, top_k=top_k, min_similarity=min_similarity)
        
        try:
            # Generate embeddings for all papers
            texts = [
                f"{getattr(p, 'title', '')} {getattr(p, 'abstract', '')}"
                for p in papers
            ]
            embeddings = await self.embedding_service.embed_batch(texts)
            
            # Compute pairwise cosine similarities
            overlaps: list[dict[str, Any]] = []
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    sim = EmbeddingService.cosine_similarity(embeddings[i], embeddings[j])
                    if sim < min_similarity:
                        continue
                    
                    paper_a = papers[i]
                    paper_b = papers[j]
                    cats_a = set(getattr(paper_a, "categories", []) or [])
                    cats_b = set(getattr(paper_b, "categories", []) or [])
                    cross_domain = len(cats_a & cats_b) == 0
                    
                    overlaps.append({
                        "paper_a": getattr(paper_a, "arxiv_id", ""),
                        "paper_b": getattr(paper_b, "arxiv_id", ""),
                        "similarity": round(sim, 3),
                        "shared_terms": [],  # Not available for embedding-based
                        "cross_domain": cross_domain,
                        "categories_a": list(cats_a),
                        "categories_b": list(cats_b),
                        "method": "embedding",
                    })
            
            overlaps.sort(key=lambda o: o["similarity"], reverse=True)
            overlaps = overlaps[:top_k]
            self.concept_map.overlaps = overlaps
            return overlaps
            
        except Exception as e:
            print(f"[ConceptMapBuilder] Embedding overlap detection failed: {e}, using TF-IDF fallback")
            return self.identify_overlaps(papers, top_k=top_k, min_similarity=min_similarity)

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

        with open(path / "overlaps.json", "w") as f:
            json.dump(self.concept_map.overlaps, f, indent=2)

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

        overlaps_file = path / "overlaps.json"
        if overlaps_file.exists():
            with open(overlaps_file) as f:
                builder.concept_map.overlaps = json.load(f)

        return builder

    # =========================================================================
    # Cross-Session Persistence (SQLite)
    # =========================================================================

    def save_to_db(self, db_path: Path | None = None) -> None:
        """Persist concept map to SQLite database.
        
        Args:
            db_path: Path to database file (default: sessions/knowledge.db)
        """
        import sqlite3
        from datetime import datetime
        
        if db_path is None:
            db_path = Path("sessions/knowledge.db")
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables if not exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                name TEXT,
                type TEXT,
                frequency INTEGER DEFAULT 1,
                source_papers TEXT  -- JSON list
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source TEXT,
                target TEXT,
                relation TEXT,
                confidence REAL,
                source_papers TEXT,  -- JSON list
                PRIMARY KEY (source, target, relation)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                topic TEXT,
                node_count INTEGER,
                edge_count INTEGER
            )
        """)
        
        # Upsert nodes
        for node in self.concept_map.nodes:
            cursor.execute("""
                INSERT INTO nodes (id, name, type, frequency, source_papers)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    frequency = frequency + excluded.frequency,
                    source_papers = excluded.source_papers
            """, (
                node.id,
                node.name,
                node.type,
                node.frequency,
                json.dumps(node.source_papers),
            ))
        
        # Upsert edges
        for edge in self.concept_map.edges:
            cursor.execute("""
                INSERT INTO edges (source, target, relation, confidence, source_papers)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, target, relation) DO UPDATE SET
                    confidence = MAX(confidence, excluded.confidence),
                    source_papers = excluded.source_papers
            """, (
                edge.source,
                edge.target,
                edge.relation,
                edge.confidence,
                json.dumps(edge.source_papers),
            ))
        
        # Log session
        import uuid
        cursor.execute("""
            INSERT INTO sessions (id, timestamp, topic, node_count, edge_count)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(uuid.uuid4())[:8],
            datetime.now().isoformat(),
            "auto",
            len(self.concept_map.nodes),
            len(self.concept_map.edges),
        ))
        
        conn.commit()
        conn.close()

    def load_from_db(self, db_path: Path | None = None) -> None:
        """Load concept map from SQLite database.
        
        Args:
            db_path: Path to database file (default: sessions/knowledge.db)
        """
        import sqlite3
        
        if db_path is None:
            db_path = Path("sessions/knowledge.db")
        
        if not db_path.exists():
            return
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Load nodes
        try:
            cursor.execute("SELECT id, name, type, frequency, source_papers FROM nodes")
            for row in cursor.fetchall():
                node = ConceptNode(
                    id=row[0],
                    name=row[1],
                    type=row[2] or "concept",
                    frequency=row[3] or 1,
                    source_papers=json.loads(row[4]) if row[4] else [],
                )
                # Check if already exists
                existing = next((n for n in self.concept_map.nodes if n.id == node.id), None)
                if existing:
                    existing.frequency += node.frequency
                else:
                    self.concept_map.nodes.append(node)
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet
        
        # Load edges
        try:
            cursor.execute("SELECT source, target, relation, confidence, source_papers FROM edges")
            for row in cursor.fetchall():
                edge = ConceptEdge(
                    source=row[0],
                    target=row[1],
                    relation=row[2],
                    confidence=row[3] or 0.5,
                    source_papers=json.loads(row[4]) if row[4] else [],
                )
                # Check if already exists
                existing = next(
                    (e for e in self.concept_map.edges 
                     if e.source == edge.source and e.target == edge.target and e.relation == edge.relation),
                    None
                )
                if not existing:
                    self.concept_map.edges.append(edge)
        except sqlite3.OperationalError:
            pass
        
        conn.close()

    def merge(self, other: "ConceptMap") -> None:
        """Merge another concept map into this one.
        
        - Nodes with same ID get frequency merged
        - New edges are added, existing edges update confidence
        
        Args:
            other: ConceptMap to merge from
        """
        # Merge nodes
        existing_node_ids = {n.id for n in self.concept_map.nodes}
        for node in other.nodes:
            if node.id in existing_node_ids:
                # Update existing node
                existing = next(n for n in self.concept_map.nodes if n.id == node.id)
                existing.frequency += node.frequency
                for paper in node.source_papers:
                    if paper not in existing.source_papers:
                        existing.source_papers.append(paper)
            else:
                # Add new node
                self.concept_map.nodes.append(node)
                existing_node_ids.add(node.id)
        
        # Merge edges
        existing_edges = {(e.source, e.target, e.relation) for e in self.concept_map.edges}
        for edge in other.edges:
            key = (edge.source, edge.target, edge.relation)
            if key in existing_edges:
                # Update confidence to max
                existing = next(
                    e for e in self.concept_map.edges 
                    if e.source == edge.source and e.target == edge.target and e.relation == edge.relation
                )
                existing.confidence = max(existing.confidence, edge.confidence)
            else:
                self.concept_map.edges.append(edge)
                existing_edges.add(key)

    def get_db_stats(self, db_path: Path | None = None) -> dict[str, Any]:
        """Get statistics from the persistent knowledge base.
        
        Returns:
            Dict with node_count, edge_count, session_count
        """
        import sqlite3
        
        if db_path is None:
            db_path = Path("sessions/knowledge.db")
        
        if not db_path.exists():
            return {"node_count": 0, "edge_count": 0, "session_count": 0}
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        stats = {}
        try:
            cursor.execute("SELECT COUNT(*) FROM nodes")
            stats["node_count"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM edges")
            stats["edge_count"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM sessions")
            stats["session_count"] = cursor.fetchone()[0]
        except:
            stats = {"node_count": 0, "edge_count": 0, "session_count": 0}
        
        conn.close()
        return stats

