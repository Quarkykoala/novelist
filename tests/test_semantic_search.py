"""Integration tests for SemanticPaperSearch.

Tests the high-level semantic search API that combines
EmbeddingService and VectorStore.
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch


class TestSemanticPaperSearchInit:
    """Tests for SemanticPaperSearch initialization."""

    def test_default_initialization(self):
        """Test default initialization creates services."""
        with patch("src.kb.semantic_search.EmbeddingService"):
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            assert search.embedding_service is not None
            assert search.vector_store is not None
            assert search._paper_cache == {}

    def test_is_initialized_empty(self):
        """Empty search should not be initialized."""
        with patch("src.kb.semantic_search.EmbeddingService"):
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            assert search.is_initialized is False


class TestSemanticPaperSearchIndexing:
    """Tests for paper indexing functionality."""

    @pytest.fixture
    def mock_paper(self):
        """Create a mock ArxivPaper."""
        from src.contracts.schemas import ArxivPaper
        return ArxivPaper(
            arxiv_id="2401.001",
            title="Test Paper on Batteries",
            abstract="This paper discusses lithium-ion battery improvements.",
            source="arxiv",
            categories=["cond-mat.mtrl-sci"],
            published=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_index_single_paper(self, mock_paper):
        """Test indexing a single paper."""
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.embed = AsyncMock(return_value=[0.1] * 768)
            MockEmbed.return_value = mock_service
            
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            
            await search.index_paper(mock_paper)
            
            assert search.is_initialized
            assert mock_paper.arxiv_id in search._paper_cache
            mock_service.embed.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_multiple_papers(self, mock_paper):
        """Test batch indexing papers."""
        from src.contracts.schemas import ArxivPaper
        
        papers = [
            mock_paper,
            ArxivPaper(
                arxiv_id="2401.002",
                title="Another Paper",
                abstract="About fuel cells.",
                source="arxiv",
                categories=["physics"],
                published=datetime.now(),
            ),
        ]
        
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.embed_batch = AsyncMock(return_value=[[0.1] * 768, [0.2] * 768])
            MockEmbed.return_value = mock_service
            
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            
            count = await search.index_papers(papers)
            
            assert count == 2
            assert len(search._paper_cache) == 2

    @pytest.mark.asyncio
    async def test_index_empty_list(self):
        """Indexing empty list should return 0."""
        with patch("src.kb.semantic_search.EmbeddingService"):
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            
            count = await search.index_papers([])
            assert count == 0


class TestSemanticPaperSearchSearch:
    """Tests for search functionality."""

    @pytest.fixture
    async def populated_search(self):
        """Create search with indexed papers."""
        from src.contracts.schemas import ArxivPaper
        from src.kb.semantic_search import SemanticPaperSearch
        from src.kb.vector_store import InMemoryVectorStore
        
        # Create real in-memory store
        store = InMemoryVectorStore()
        
        # Mock embedding service
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.embed = AsyncMock(side_effect=lambda t: [0.5] * 768)
            mock_service.embed_batch = AsyncMock(return_value=[
                [1.0, 0.0, 0.0] + [0.0] * 765,
                [0.0, 1.0, 0.0] + [0.0] * 765,
            ])
            MockEmbed.return_value = mock_service
            
            search = SemanticPaperSearch(vector_store=store)
            search.embedding_service = mock_service
            
            papers = [
                ArxivPaper(
                    arxiv_id="paper1",
                    title="Battery Research",
                    abstract="Lithium dendrites.",
                    source="arxiv",
                    categories=["physics"],
                    published=datetime.now(),
                ),
                ArxivPaper(
                    arxiv_id="paper2", 
                    title="Fuel Cells",
                    abstract="Hydrogen storage.",
                    source="pubmed",
                    categories=["chemistry"],
                    published=datetime.now(),
                ),
            ]
            
            await search.index_papers(papers)
            return search

    @pytest.mark.asyncio
    async def test_search_returns_papers_with_scores(self, populated_search):
        """Search should return papers with similarity scores."""
        search = populated_search
        
        results = await search.search_similar("battery", top_k=10)
        
        assert len(results) > 0
        for paper, score in results:
            assert hasattr(paper, "arxiv_id")
            assert isinstance(score, float)
            assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, populated_search):
        """Search should respect top_k limit."""
        search = populated_search
        
        results = await search.search_similar("test", top_k=1)
        assert len(results) <= 1

    @pytest.mark.asyncio
    async def test_get_paper(self, populated_search):
        """Should retrieve indexed paper by ID."""
        search = populated_search
        
        paper = await search.get_paper("paper1")
        assert paper is not None
        assert paper.arxiv_id == "paper1"

    @pytest.mark.asyncio
    async def test_get_paper_nonexistent(self, populated_search):
        """Should return None for non-existent paper."""
        search = populated_search
        
        paper = await search.get_paper("nonexistent")
        assert paper is None


class TestSemanticPaperSearchFindRelated:
    """Tests for finding related papers."""

    @pytest.mark.asyncio
    async def test_find_related_excludes_self(self):
        """Finding related papers should exclude the input paper."""
        from src.contracts.schemas import ArxivPaper
        from src.kb.semantic_search import SemanticPaperSearch
        from src.kb.vector_store import InMemoryVectorStore
        
        store = InMemoryVectorStore()
        
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.embed = AsyncMock(return_value=[0.5] * 768)
            mock_service.embed_batch = AsyncMock(return_value=[[0.5] * 768, [0.6] * 768])
            MockEmbed.return_value = mock_service
            
            search = SemanticPaperSearch(vector_store=store)
            search.embedding_service = mock_service
            
            paper1 = ArxivPaper(
                arxiv_id="paper1",
                title="Test",
                abstract="Test abstract",
                source="arxiv",
                published=datetime.now(),
            )
            paper2 = ArxivPaper(
                arxiv_id="paper2",
                title="Related",
                abstract="Related abstract",
                source="arxiv",
                published=datetime.now(),
            )
            
            await search.index_papers([paper1, paper2])
            related = await search.find_related_papers(paper1, top_k=5)
            
            # paper1 should not be in results
            paper_ids = [p.arxiv_id for p, _ in related]
            assert "paper1" not in paper_ids


class TestSemanticPaperSearchStats:
    """Tests for search statistics."""

    def test_get_stats(self):
        """Should return statistics about the index."""
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.dimension = 768
            MockEmbed.return_value = mock_service
            
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            
            stats = search.get_stats()
            
            assert "papers_indexed" in stats
            assert "embedding_dimension" in stats
            assert "vector_store_type" in stats

    @pytest.mark.asyncio
    async def test_count(self):
        """Count should return number of indexed papers."""
        with patch("src.kb.semantic_search.EmbeddingService"):
            from src.kb.semantic_search import SemanticPaperSearch
            search = SemanticPaperSearch()
            
            count = await search.count()
            assert count == 0

    @pytest.mark.asyncio
    async def test_clear(self):
        """Clear should remove all indexed papers."""
        with patch("src.kb.semantic_search.EmbeddingService") as MockEmbed:
            mock_service = MagicMock()
            mock_service.embed = AsyncMock(return_value=[0.1] * 768)
            mock_service.clear_cache = MagicMock()
            MockEmbed.return_value = mock_service
            
            from src.contracts.schemas import ArxivPaper
            from src.kb.semantic_search import SemanticPaperSearch
            
            search = SemanticPaperSearch()
            paper = ArxivPaper(
                arxiv_id="test",
                title="Test",
                abstract="Test",
                source="arxiv",
                published=datetime.now(),
            )
            
            await search.index_paper(paper)
            assert search.is_initialized
            
            await search.clear()
            assert not search.is_initialized
