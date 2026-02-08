"""Vector store abstraction for semantic paper search.

Provides a unified interface for storing and searching embeddings:
- InMemoryVectorStore: Fast, ephemeral, good for testing and small datasets
- ChromaVectorStore: Local persistent storage (requires chromadb)
- SupabaseVectorStore: Cloud pgvector storage (requires supabase)

All stores support:
- Upsert: Add or update embeddings with metadata
- Search: Find top-k most similar embeddings
- Delete: Remove embeddings by ID
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SearchResult:
    """Result from vector similarity search."""

    id: str
    score: float  # Similarity score (0 to 1, higher = more similar)
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an embedding.

        Args:
            id: Unique identifier for the embedding
            embedding: Dense embedding vector
            metadata: Optional metadata to store with the embedding
        """
        ...

    @abstractmethod
    async def upsert_batch(
        self,
        items: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> None:
        """Add or update multiple embeddings.

        Args:
            items: List of (id, embedding, metadata) tuples
        """
        ...

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of SearchResult sorted by similarity (highest first)
        """
        ...

    @abstractmethod
    async def get(self, id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get embedding and metadata by ID.

        Args:
            id: Unique identifier

        Returns:
            Tuple of (embedding, metadata) or None if not found
        """
        ...

    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an embedding by ID.

        Args:
            id: Unique identifier

        Returns:
            True if deleted, False if not found
        """
        ...

    @abstractmethod
    async def count(self) -> int:
        """Return the number of stored embeddings."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Remove all embeddings from the store."""
        ...


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for testing and small datasets.

    Features:
    - Fast brute-force similarity search
    - Optional JSON persistence
    - No external dependencies
    """

    def __init__(self, persist_path: Path | None = None):
        """Initialize in-memory vector store.

        Args:
            persist_path: Optional path to persist store as JSON
        """
        self.persist_path = persist_path
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

        # Load from disk if exists
        if persist_path and persist_path.exists():
            self._load()

    def _load(self) -> None:
        """Load store from disk."""
        try:
            if self.persist_path and self.persist_path.exists():
                with open(self.persist_path) as f:
                    data = json.load(f)
                    self._embeddings = data.get("embeddings", {})
                    self._metadata = data.get("metadata", {})
                print(f"[VectorStore] Loaded {len(self._embeddings)} vectors")
        except Exception as e:
            print(f"[VectorStore] Load error: {e}")

    def _save(self) -> None:
        """Save store to disk."""
        if self.persist_path:
            try:
                self.persist_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.persist_path, "w") as f:
                    json.dump({
                        "embeddings": self._embeddings,
                        "metadata": self._metadata,
                    }, f)
            except Exception as e:
                print(f"[VectorStore] Save error: {e}")

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an embedding."""
        self._embeddings[id] = embedding
        self._metadata[id] = metadata or {}
        self._save()

    async def upsert_batch(
        self,
        items: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> None:
        """Add or update multiple embeddings."""
        for id, embedding, metadata in items:
            self._embeddings[id] = embedding
            self._metadata[id] = metadata or {}
        self._save()

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings using brute-force cosine similarity."""
        results: list[tuple[str, float]] = []

        for id, embedding in self._embeddings.items():
            # Apply metadata filter if specified
            if filter_metadata:
                meta = self._metadata.get(id, {})
                if not all(meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            score = self._cosine_similarity(query_embedding, embedding)
            results.append((id, score))

        # Sort by score descending and take top_k
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[:top_k]

        return [
            SearchResult(id=id, score=score, metadata=self._metadata.get(id, {}))
            for id, score in results
        ]

    async def get(self, id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get embedding and metadata by ID."""
        if id not in self._embeddings:
            return None
        return (self._embeddings[id], self._metadata.get(id, {}))

    async def delete(self, id: str) -> bool:
        """Delete an embedding by ID."""
        if id not in self._embeddings:
            return False
        del self._embeddings[id]
        self._metadata.pop(id, None)
        self._save()
        return True

    async def count(self) -> int:
        """Return the number of stored embeddings."""
        return len(self._embeddings)

    async def clear(self) -> None:
        """Remove all embeddings from the store."""
        self._embeddings = {}
        self._metadata = {}
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()


# =============================================================================
# Optional: ChromaDB implementation (requires chromadb package)
# =============================================================================

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store for local persistent storage.

    Requires: pip install chromadb
    """

    def __init__(
        self,
        collection_name: str = "papers",
        persist_directory: Path | None = None,
    ):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb is required: pip install chromadb")

        self.collection_name = collection_name

        if persist_directory:
            self._client = chromadb.PersistentClient(
                path=str(persist_directory),
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client(
                settings=Settings(anonymized_telemetry=False),
            )

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def upsert(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update an embedding."""
        # ChromaDB requires metadata values to be strings, ints, floats, or bools
        clean_metadata = self._clean_metadata(metadata or {})
        self._collection.upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[clean_metadata],
        )

    async def upsert_batch(
        self,
        items: list[tuple[str, list[float], dict[str, Any] | None]],
    ) -> None:
        """Add or update multiple embeddings."""
        if not items:
            return

        ids = [item[0] for item in items]
        embeddings = [item[1] for item in items]
        metadatas = [self._clean_metadata(item[2] or {}) for item in items]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar embeddings."""
        where = None
        if filter_metadata:
            where = {k: v for k, v in filter_metadata.items()}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results["distances"] else [0] * len(ids)
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(ids)

            for id, distance, metadata in zip(ids, distances, metadatas):
                # ChromaDB returns distances, convert to similarity
                score = 1.0 - distance
                search_results.append(SearchResult(id=id, score=score, metadata=metadata or {}))

        return search_results

    async def get(self, id: str) -> tuple[list[float], dict[str, Any]] | None:
        """Get embedding and metadata by ID."""
        results = self._collection.get(
            ids=[id],
            include=["embeddings", "metadatas"],
        )
        if not results["ids"]:
            return None
        embedding = results["embeddings"][0] if results["embeddings"] else []
        metadata = results["metadatas"][0] if results["metadatas"] else {}
        return (embedding, metadata)

    async def delete(self, id: str) -> bool:
        """Delete an embedding by ID."""
        try:
            self._collection.delete(ids=[id])
            return True
        except Exception:
            return False

    async def count(self) -> int:
        """Return the number of stored embeddings."""
        return self._collection.count()

    async def clear(self) -> None:
        """Remove all embeddings from the store."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Clean metadata for ChromaDB (only primitive types allowed)."""
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, list):
                # Convert lists to JSON strings
                clean[k] = json.dumps(v)
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# =============================================================================
# Factory function
# =============================================================================


def create_vector_store(
    backend: str = "memory",
    persist_path: Path | None = None,
    **kwargs: Any,
) -> VectorStore:
    """Create a vector store instance.

    Args:
        backend: "memory" or "chroma"
        persist_path: Path for persistence
        **kwargs: Additional backend-specific arguments

    Returns:
        VectorStore instance
    """
    if backend == "memory":
        return InMemoryVectorStore(persist_path=persist_path)
    elif backend == "chroma":
        return ChromaVectorStore(
            persist_directory=persist_path,
            collection_name=kwargs.get("collection_name", "papers"),
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def main():
        store = InMemoryVectorStore()

        # Add some test embeddings
        await store.upsert("paper1", [0.1, 0.2, 0.3], {"title": "Battery research"})
        await store.upsert("paper2", [0.15, 0.22, 0.28], {"title": "Lithium ion"})
        await store.upsert("paper3", [0.9, 0.1, 0.1], {"title": "Machine learning"})

        print(f"Store count: {await store.count()}")

        # Search
        results = await store.search([0.12, 0.21, 0.29], top_k=2)
        print("Search results:")
        for r in results:
            print(f"  {r.id}: score={r.score:.4f}, metadata={r.metadata}")

    asyncio.run(main())
