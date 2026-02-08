"""Unit tests for VectorStore implementations.

Tests InMemoryVectorStore and VectorStore abstraction.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestInMemoryVectorStore:
    """Tests for InMemoryVectorStore implementation."""

    @pytest.fixture
    def store(self):
        """Create a fresh InMemoryVectorStore for each test."""
        from src.kb.vector_store import InMemoryVectorStore
        return InMemoryVectorStore()

    @pytest.mark.asyncio
    async def test_upsert_and_get(self, store):
        """Test basic upsert and retrieval."""
        embedding = [0.1, 0.2, 0.3]
        metadata = {"title": "Test Paper"}
        
        await store.upsert("paper1", embedding, metadata)
        result = await store.get("paper1")
        
        assert result is not None
        emb, meta = result
        assert emb == embedding
        assert meta == metadata

    @pytest.mark.asyncio
    async def test_upsert_updates_existing(self, store):
        """Test that upsert updates an existing entry."""
        await store.upsert("paper1", [0.1, 0.2], {"v": 1})
        await store.upsert("paper1", [0.3, 0.4], {"v": 2})
        
        result = await store.get("paper1")
        emb, meta = result
        assert emb == [0.3, 0.4]
        assert meta["v"] == 2

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a non-existent ID returns None."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test deleting an entry."""
        await store.upsert("paper1", [0.1, 0.2], None)
        deleted = await store.delete("paper1")
        
        assert deleted is True
        assert await store.get("paper1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting non-existent returns False."""
        result = await store.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_count(self, store):
        """Test counting entries."""
        assert await store.count() == 0
        
        await store.upsert("paper1", [0.1], None)
        await store.upsert("paper2", [0.2], None)
        
        assert await store.count() == 2

    @pytest.mark.asyncio
    async def test_clear(self, store):
        """Test clearing all entries."""
        await store.upsert("paper1", [0.1], None)
        await store.upsert("paper2", [0.2], None)
        
        await store.clear()
        
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_upsert_batch(self, store):
        """Test batch upserting."""
        items = [
            ("paper1", [0.1, 0.2], {"title": "Paper 1"}),
            ("paper2", [0.3, 0.4], {"title": "Paper 2"}),
            ("paper3", [0.5, 0.6], None),
        ]
        
        await store.upsert_batch(items)
        
        assert await store.count() == 3
        emb, meta = await store.get("paper1")
        assert meta["title"] == "Paper 1"


class TestInMemoryVectorStoreSearch:
    """Tests for InMemoryVectorStore search functionality."""

    @pytest.fixture
    async def populated_store(self):
        """Create a store with test data."""
        from src.kb.vector_store import InMemoryVectorStore
        store = InMemoryVectorStore()
        
        # Add test vectors - normalized for cosine similarity
        await store.upsert("paper1", [1.0, 0.0, 0.0], {"category": "physics"})
        await store.upsert("paper2", [0.0, 1.0, 0.0], {"category": "chemistry"})
        await store.upsert("paper3", [0.707, 0.707, 0.0], {"category": "physics"})  # Between 1 and 2
        
        return store

    @pytest.mark.asyncio
    async def test_search_returns_sorted_by_similarity(self, populated_store):
        """Search should return results sorted by similarity (descending)."""
        store = populated_store
        
        # Query vector closest to paper1
        query = [0.9, 0.1, 0.0]
        results = await store.search(query, top_k=3)
        
        assert len(results) == 3
        # paper1 should be most similar
        assert results[0].id == "paper1"
        # Scores should be descending
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_search_top_k(self, populated_store):
        """Search should respect top_k limit."""
        store = populated_store
        
        results = await store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self, populated_store):
        """Search should filter by metadata."""
        store = populated_store
        
        results = await store.search(
            [1.0, 0.0, 0.0], 
            top_k=10,
            filter_metadata={"category": "physics"}
        )
        
        assert len(results) == 2
        assert all(r.metadata["category"] == "physics" for r in results)

    @pytest.mark.asyncio
    async def test_search_empty_store(self):
        """Search on empty store should return empty list."""
        from src.kb.vector_store import InMemoryVectorStore
        store = InMemoryVectorStore()
        
        results = await store.search([1.0, 0.0], top_k=10)
        assert results == []


class TestVectorStorePersistence:
    """Tests for vector store persistence."""

    @pytest.mark.asyncio
    async def test_json_persistence(self, tmp_path):
        """Test saving and loading from JSON."""
        from src.kb.vector_store import InMemoryVectorStore
        
        persist_path = tmp_path / "store.json"
        
        # Create and populate store
        store1 = InMemoryVectorStore(persist_path=persist_path)
        await store1.upsert("paper1", [0.1, 0.2], {"title": "Test"})
        await store1.upsert("paper2", [0.3, 0.4], None)
        
        # Create new store from same file
        store2 = InMemoryVectorStore(persist_path=persist_path)
        
        assert await store2.count() == 2
        result = await store2.get("paper1")
        emb, meta = result
        assert meta["title"] == "Test"


class TestCreateVectorStore:
    """Tests for the factory function."""

    def test_create_memory_store(self):
        """Factory should create InMemoryVectorStore for 'memory' backend."""
        from src.kb.vector_store import create_vector_store, InMemoryVectorStore
        
        store = create_vector_store("memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_create_invalid_backend(self):
        """Factory should raise error for unknown backend."""
        from src.kb.vector_store import create_vector_store
        
        with pytest.raises(ValueError, match="Unknown backend"):
            create_vector_store("invalid_backend")
