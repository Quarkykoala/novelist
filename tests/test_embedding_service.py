"""Unit tests for EmbeddingService.

Tests embedding generation, caching, and similarity calculations.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import math


class TestEmbeddingServiceInit:
    """Tests for EmbeddingService initialization."""

    def test_default_initialization(self):
        """Test default initialization without cache path."""
        from src.kb.embedding_service import EmbeddingService
        service = EmbeddingService()
        assert service.provider.model == "text-embedding-004"
        assert service.dimension == 768
        assert service._cache == {}

    def test_initialization_with_cache_path(self, tmp_path):
        """Test initialization with disk cache path."""
        with patch("src.kb.embedding_service.genai.Client"):
            from src.kb.embedding_service import EmbeddingService
            cache_file = tmp_path / "embedding_cache.json"
            service = EmbeddingService(cache_path=cache_file)
            assert service.cache_path == cache_file


class TestCosineSimilarity:
# ... (rest of the class)
    """Tests for static cosine similarity/distance calculations."""

    def test_cosine_similarity_identical_vectors(self):
        """Identical vectors should have similarity 1.0."""
        from src.kb.embedding_service import EmbeddingService
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert EmbeddingService.similarity(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should have similarity 0.0."""
        from src.kb.embedding_service import EmbeddingService
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert EmbeddingService.similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self):
        """Opposite vectors should have similarity -1.0."""
        from src.kb.embedding_service import EmbeddingService
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        assert EmbeddingService.similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_distance_identical(self):
        """Identical vectors should have distance 0.0."""
        from src.kb.embedding_service import EmbeddingService
        a = [1.0, 2.0, 3.0]
        b = [1.0, 2.0, 3.0]
        assert EmbeddingService.distance(a, b) == pytest.approx(0.0)

    def test_cosine_distance_orthogonal(self):
        """Orthogonal vectors should have distance 1.0."""
        from src.kb.embedding_service import EmbeddingService
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert EmbeddingService.distance(a, b) == pytest.approx(1.0)

    def test_cosine_similarity_zero_vector(self):
        """Zero vector should return 0.0 similarity."""
        from src.kb.embedding_service import EmbeddingService
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert EmbeddingService.similarity(a, b) == pytest.approx(0.0)


class TestEmbeddingCaching:
    """Tests for embedding cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_hit(self):
        """Previously embedded text should be returned from cache."""
        with patch("src.kb.embedding_service.genai.Client") as mock_client_class:
            from src.kb.embedding_service import EmbeddingService
            
            # Setup mock
            mock_client = mock_client_class.return_value
            mock_embedding = MagicMock()
            mock_embedding.values = [0.1] * 768
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.models.embed_content.return_value = mock_response
            
            service = EmbeddingService()
            
            # First call - should call API
            result1 = await service.embed("test text")
            assert len(result1) == 768
            
            # Second call - should use cache
            result2 = await service.embed("test text")
            assert result1 == result2
            
            # API should only be called once
            assert mock_client.models.embed_content.call_count == 1

    @pytest.mark.asyncio
    async def test_clear_cache(self):
        """clear_cache should empty the cache."""
        with patch("src.kb.embedding_service.genai.Client"):
            from src.kb.embedding_service import EmbeddingService
            service = EmbeddingService()
            key = service._cache_key("test")
            service._cache[key] = [0.1] * 768
            
            service.clear_cache()
            
            assert service._cache == {}


class TestBatchEmbedding:
    """Tests for batch embedding functionality."""

    @pytest.mark.asyncio
    async def test_batch_embed_multiple_texts(self):
        """Batch embedding should return embeddings for all texts."""
        with patch("src.kb.embedding_service.genai.Client") as mock_client_class:
            from src.kb.embedding_service import EmbeddingService
            
            # Setup mock
            mock_client = mock_client_class.return_value
            
            def create_mock_embedding(val):
                m = MagicMock()
                m.values = [val] * 768
                return m

            mock_response = MagicMock()
            mock_response.embeddings = [
                create_mock_embedding(0.1),
                create_mock_embedding(0.2),
                create_mock_embedding(0.3),
            ]
            mock_client.models.embed_content.return_value = mock_response
            
            service = EmbeddingService()
            results = await service.embed_batch(["text1", "text2", "text3"])
            
            assert len(results) == 3
            assert all(len(emb) == 768 for emb in results)
            assert results[0][0] == 0.1
            assert results[1][0] == 0.2
            assert results[2][0] == 0.3

    @pytest.mark.asyncio
    async def test_batch_embed_empty_list(self):
        """Batch embedding empty list should return empty list."""
        with patch("src.kb.embedding_service.genai.Client"):
            from src.kb.embedding_service import EmbeddingService
            service = EmbeddingService()
            
            results = await service.embed_batch([])
            assert results == []

    @pytest.mark.asyncio
    async def test_batch_embed_uses_cache(self):
        """Batch embed should use cached embeddings when available."""
        with patch("src.kb.embedding_service.genai.Client") as mock_client_class:
            from src.kb.embedding_service import EmbeddingService
            
            service = EmbeddingService()
            # Pre-populate cache
            cached_embedding = [0.5] * 768
            key = service._cache_key("cached_text")
            service._cache[key] = cached_embedding
            
            # Setup mock for new text
            mock_client = mock_client_class.return_value
            mock_embedding = MagicMock()
            mock_embedding.values = [0.9] * 768
            mock_response = MagicMock()
            mock_response.embeddings = [mock_embedding]
            mock_client.models.embed_content.return_value = mock_response
            
            results = await service.embed_batch(["cached_text", "new_text"])
            
            assert len(results) == 2
            assert results[0] == cached_embedding  # From cache
            assert results[1][0] == 0.9  # From API
