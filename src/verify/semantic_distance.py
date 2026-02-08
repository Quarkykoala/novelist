"""
Semantic Distance Scorer — Measure how "far" concepts in a hypothesis are from each other.

The core insight: truly novel ideas bridge semantically distant concepts.
"Bioluminescence + hydrogen storage" is more novel than "hydrogen + fuel cells".

This module computes semantic distance using embeddings and rewards hypotheses
that make far-bridging connections while remaining coherent.
"""

import asyncio
import re
from dataclasses import dataclass

from src.soul.llm_client import LLMClient
from src.kb.embedding_service import EmbeddingService


@dataclass
class SemanticDistanceResult:
    """Result of semantic distance analysis."""
    
    concepts: list[str]  # Extracted concepts
    pairwise_distances: dict[tuple[str, str], float]  # Concept pair → distance
    average_distance: float  # Mean of all pairwise distances
    max_distance: float  # Highest single distance
    bridging_pair: tuple[str, str] | None  # The two most distant concepts
    novelty_boost: float  # Multiplier for scoring (1.0 = normal, 2.0 = very novel)


class SemanticDistanceScorer:
    """
    Measure semantic distance between concepts in hypotheses.
    
    Uses LLM embeddings to calculate how "far apart" connected concepts are.
    Higher distance = more novel connection.
    """
    
    def __init__(self, model: str = "gemini-2.0-flash", use_real_embeddings: bool = True):
        self.client = LLMClient(model=model)
        self._embedding_cache: dict[str, list[float]] = {}
        self.use_real_embeddings = use_real_embeddings
        self._embedding_service: EmbeddingService | None = None
    
    @property
    def embedding_service(self) -> EmbeddingService:
        """Lazy initialization of embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service
    
    async def score_hypothesis(self, hypothesis_text: str) -> SemanticDistanceResult:
        """
        Score a hypothesis by the semantic distance of its concepts.
        
        Args:
            hypothesis_text: The hypothesis claim text
            
        Returns:
            SemanticDistanceResult with distance metrics
        """
        # 1. Extract key concepts from hypothesis
        concepts = await self._extract_concepts(hypothesis_text)
        
        if len(concepts) < 2:
            return SemanticDistanceResult(
                concepts=concepts,
                pairwise_distances={},
                average_distance=0.0,
                max_distance=0.0,
                bridging_pair=None,
                novelty_boost=1.0,
            )
        
        # 2. Get embeddings for each concept
        embeddings = await self._get_embeddings(concepts)
        
        # 3. Calculate pairwise distances
        pairwise = self._calculate_pairwise_distances(concepts, embeddings)
        
        # 4. Compute metrics
        distances = list(pairwise.values())
        avg_dist = sum(distances) / len(distances) if distances else 0.0
        max_dist = max(distances) if distances else 0.0
        
        # Find the bridging pair (most distant)
        bridging_pair = None
        if pairwise:
            bridging_pair = max(pairwise.keys(), key=lambda k: pairwise[k])
        
        # Calculate novelty boost based on distance
        novelty_boost = self._calculate_novelty_boost(avg_dist)
        
        return SemanticDistanceResult(
            concepts=concepts,
            pairwise_distances=pairwise,
            average_distance=avg_dist,
            max_distance=max_dist,
            bridging_pair=bridging_pair,
            novelty_boost=novelty_boost,
        )
    
    async def _extract_concepts(self, text: str) -> list[str]:
        """Extract key scientific/technical concepts from text."""
        prompt = f"""Extract the 3-5 most important scientific/technical CONCEPTS from this hypothesis.

Return ONLY a comma-separated list of concepts (nouns/noun phrases).
Focus on domain-specific terms, not generic words.

Hypothesis: {text}

Concepts:"""
        
        response = await self.client.generate_content(prompt)
        if not response:
            return self._fallback_extract(text)
        
        # Handle GenerationResponse
        if hasattr(response, "content"):
            response = response.content
        
        if not response:
            return self._fallback_extract(text)
        
        # Parse comma-separated list
        concepts = [c.strip() for c in response.split(",") if c.strip()]
        return concepts[:5]  # Limit to 5
    
    def _fallback_extract(self, text: str) -> list[str]:
        """Fallback extraction using simple heuristics."""
        # Extract capitalized noun phrases and technical terms
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[a-z]+)*\b|\b[a-z]+(?:-[a-z]+)+\b', text)
        # Deduplicate while preserving order
        seen = set()
        concepts = []
        for w in words:
            w_lower = w.lower()
            if w_lower not in seen and len(w) > 3:
                seen.add(w_lower)
                concepts.append(w)
        return concepts[:5]
    
    async def _get_embeddings(self, concepts: list[str]) -> dict[str, list[float]]:
        """Get embeddings for concepts (with caching).
        
        Uses real embeddings from EmbeddingService when available,
        falls back to LLM pseudo-embeddings otherwise.
        """
        embeddings = {}
        
        # Check cache first
        to_fetch = []
        for concept in concepts:
            if concept in self._embedding_cache:
                embeddings[concept] = self._embedding_cache[concept]
            else:
                to_fetch.append(concept)
        
        # Fetch missing embeddings
        if to_fetch:
            if self.use_real_embeddings:
                # Use real embeddings from EmbeddingService (Gemini API)
                try:
                    real_embeddings = await self.embedding_service.embed_batch(to_fetch)
                    for concept, embedding in zip(to_fetch, real_embeddings):
                        self._embedding_cache[concept] = embedding
                        embeddings[concept] = embedding
                except Exception as e:
                    print(f"[SemanticDistanceScorer] Real embeddings failed: {e}, using fallback")
                    for concept in to_fetch:
                        embedding = await self._generate_pseudo_embedding(concept)
                        self._embedding_cache[concept] = embedding
                        embeddings[concept] = embedding
            else:
                # Use LLM to generate pseudo-embeddings via similarity proxy
                for concept in to_fetch:
                    embedding = await self._generate_pseudo_embedding(concept)
                    self._embedding_cache[concept] = embedding
                    embeddings[concept] = embedding
        
        return embeddings
    
    async def _generate_pseudo_embedding(self, concept: str) -> list[float]:
        """
        Generate a pseudo-embedding by asking LLM to rate concept on dimensions.
        
        This is a creative workaround when we don't have embedding API access.
        Uses semantic dimensions that capture cross-domain properties.
        """
        dimensions = [
            "physical_vs_abstract",      # Is this a physical object or abstract idea?
            "biological_vs_mechanical",  # Living systems or engineered?
            "micro_vs_macro",            # Small scale or large scale?
            "energy_vs_information",     # About energy transfer or information?
            "static_vs_dynamic",         # Steady state or changing?
            "natural_vs_artificial",     # Occurs naturally or human-made?
            "theoretical_vs_applied",    # Pure theory or practical application?
            "chemical_vs_physical",      # Chemistry-based or physics-based?
        ]
        
        prompt = f"""Rate the concept "{concept}" on each dimension from 0.0 to 1.0.

Dimensions:
1. physical_vs_abstract (0=abstract, 1=physical)
2. biological_vs_mechanical (0=mechanical, 1=biological)
3. micro_vs_macro (0=micro, 1=macro)
4. energy_vs_information (0=information, 1=energy)
5. static_vs_dynamic (0=static, 1=dynamic)
6. natural_vs_artificial (0=artificial, 1=natural)
7. theoretical_vs_applied (0=theoretical, 1=applied)
8. chemical_vs_physical (0=physical, 1=chemical)

Return ONLY 8 numbers separated by commas (e.g., 0.3,0.7,0.5,0.2,0.8,0.4,0.6,0.9):"""
        
        response = await self.client.generate_content(prompt)
        if not response:
            return [0.5] * len(dimensions)  # Default to middle
        
        if hasattr(response, "content"):
            response = response.content
        
        if not response:
            return [0.5] * len(dimensions)
        
        # Parse numbers
        try:
            numbers = re.findall(r'[\d.]+', response)
            embedding = [float(n) for n in numbers[:len(dimensions)]]
            # Pad if needed
            while len(embedding) < len(dimensions):
                embedding.append(0.5)
            return embedding
        except (ValueError, IndexError):
            return [0.5] * len(dimensions)
    
    def _calculate_pairwise_distances(
        self, 
        concepts: list[str], 
        embeddings: dict[str, list[float]]
    ) -> dict[tuple[str, str], float]:
        """Calculate cosine distances between all concept pairs."""
        import math
        
        def cosine_distance(a: list[float], b: list[float]) -> float:
            """1 - cosine_similarity (higher = more different)."""
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            
            if norm_a == 0 or norm_b == 0:
                return 1.0  # Max distance
            
            similarity = dot / (norm_a * norm_b)
            return 1.0 - similarity
        
        pairwise = {}
        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i+1:]:
                emb_a = embeddings.get(concept_a, [0.5] * 8)
                emb_b = embeddings.get(concept_b, [0.5] * 8)
                distance = cosine_distance(emb_a, emb_b)
                pairwise[(concept_a, concept_b)] = distance
        
        return pairwise
    
    def _calculate_novelty_boost(self, avg_distance: float) -> float:
        """
        Convert average distance to novelty boost multiplier.
        
        Distance 0.0-0.2: boost 1.0 (normal)
        Distance 0.2-0.5: boost 1.0-1.5 (interesting)
        Distance 0.5-0.8: boost 1.5-2.0 (novel)
        Distance 0.8-1.0: boost 2.0+ (breakthrough)
        """
        if avg_distance < 0.2:
            return 1.0
        elif avg_distance < 0.5:
            return 1.0 + (avg_distance - 0.2) * (0.5 / 0.3)
        elif avg_distance < 0.8:
            return 1.5 + (avg_distance - 0.5) * (0.5 / 0.3)
        else:
            return 2.0 + (avg_distance - 0.8) * (0.5 / 0.2)
    
    async def batch_score(
        self, 
        hypotheses: list[str]
    ) -> list[SemanticDistanceResult]:
        """Score multiple hypotheses."""
        tasks = [self.score_hypothesis(h) for h in hypotheses]
        return await asyncio.gather(*tasks)
