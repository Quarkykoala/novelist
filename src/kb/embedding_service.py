"""Embedding service for generating dense vector embeddings.

Provides a unified interface for embedding text using various providers:
- Gemini (default, free tier)
- OpenAI (optional, higher quality)
- Local sentence-transformers (optional, no API calls)

Features:
- Async batch embedding for efficiency
- In-memory caching with optional disk persistence
- Cosine similarity calculation
"""

import asyncio
import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import google.genai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")

# Embedding dimensions by model
EMBEDDING_DIMS = {
    "text-embedding-004": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using Gemini's text-embedding-004 model."""

    def __init__(self, model: str = "text-embedding-004"):
        self.model = model
        self._dim = EMBEDDING_DIMS.get(model, 768)
        self._client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text using Gemini."""
        if not self._client:
            print("[EmbeddingService] Gemini API key not configured")
            return [0.0] * self._dim

        try:
            # Run in executor since genai is synchronous
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.models.embed_content(
                    model=self.model,
                    contents=text,
                    config={"task_type": "retrieval_document"},
                ),
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"[EmbeddingService] Gemini embedding error: {e}")
            # Return zero vector on error
            return [0.0] * self._dim

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []

        if not self._client:
            print("[EmbeddingService] Gemini API key not configured")
            return [[0.0] * self._dim for _ in texts]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config={"task_type": "retrieval_document"},
                ),
            )
            return [e.values for e in result.embeddings]
        except Exception as e:
            print(f"[EmbeddingService] Gemini batch embedding error: {e}")
            # Fallback to individual embeddings
            return [await self.embed(text) for text in texts]


class EmbeddingService:
    """Unified embedding service with caching and multiple provider support.

    Usage:
        service = EmbeddingService()
        embedding = await service.embed("battery dendrite formation")
        similarities = service.similarity(emb1, emb2)
    """

    def __init__(
        self,
        provider: EmbeddingProvider | None = None,
        cache_path: Path | None = None,
    ):
        """Initialize embedding service.

        Args:
            provider: Embedding provider (defaults to Gemini)
            cache_path: Path to cache file for persistent caching
        """
        self.provider = provider or GeminiEmbeddingProvider()
        self.cache_path = cache_path
        self._cache: dict[str, list[float]] = {}

        # Load cache from disk if exists
        if cache_path and cache_path.exists():
            self._load_cache()

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self.provider.dimension

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            if self.cache_path and self.cache_path.exists():
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
                print(f"[EmbeddingService] Loaded {len(self._cache)} cached embeddings")
        except Exception as e:
            print(f"[EmbeddingService] Cache load error: {e}")
            self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        if self.cache_path:
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "w") as f:
                    json.dump(self._cache, f)
            except Exception as e:
                print(f"[EmbeddingService] Cache save error: {e}")

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text with caching.

        Args:
            text: Text to embed

        Returns:
            Dense embedding vector
        """
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        embedding = await self.provider.embed(text)
        self._cache[key] = embedding

        # Periodic cache save (every 100 new embeddings)
        if len(self._cache) % 100 == 0:
            self._save_cache()

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts with caching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Check cache for each text
        results: list[list[float] | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Fetch uncached embeddings
        if uncached_texts:
            new_embeddings = await self.provider.embed_batch(uncached_texts)
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                key = self._cache_key(text)
                self._cache[key] = new_embeddings[i]
                results[idx] = new_embeddings[i]

            self._save_cache()

        return [r for r in results if r is not None]

    @staticmethod
    def similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            a: First embedding
            b: Second embedding

        Returns:
            Cosine similarity score (0 to 1)
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    @staticmethod
    def distance(a: list[float], b: list[float]) -> float:
        """Calculate cosine distance (1 - similarity) between embeddings.

        Args:
            a: First embedding
            b: Second embedding

        Returns:
            Cosine distance (0 to 1, higher = more different)
        """
        return 1.0 - EmbeddingService.similarity(a, b)

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache = {}
        if self.cache_path and self.cache_path.exists():
            self.cache_path.unlink()


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test embedding service")
    parser.add_argument("--text", "-t", default="battery dendrite formation", help="Text to embed")
    parser.add_argument("--compare", "-c", default="lithium ion battery", help="Text to compare")
    args = parser.parse_args()

    async def main():
        service = EmbeddingService()
        
        print(f"Embedding dimension: {service.dimension}")
        
        emb1 = await service.embed(args.text)
        print(f"Embedding for '{args.text}': {len(emb1)} dims, first 5: {emb1[:5]}")
        
        emb2 = await service.embed(args.compare)
        print(f"Embedding for '{args.compare}': {len(emb2)} dims, first 5: {emb2[:5]}")
        
        sim = service.similarity(emb1, emb2)
        print(f"Similarity: {sim:.4f}")

    asyncio.run(main())
