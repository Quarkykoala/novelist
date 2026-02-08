"""Semantic paper search using dense embeddings.

High-level API for:
- Indexing papers with their embeddings
- Searching for papers by semantic similarity
- Finding related papers

Usage:
    from src.kb.semantic_search import SemanticPaperSearch
    
    search = SemanticPaperSearch()
    await search.index_paper(paper)
    results = await search.search_similar("battery dendrite prevention", top_k=10)
"""

from pathlib import Path
from typing import Any

from src.contracts.schemas import ArxivPaper
from src.kb.embedding_service import EmbeddingService
from src.kb.vector_store import InMemoryVectorStore, SearchResult, VectorStore, create_vector_store


class SemanticPaperSearch:
    """High-level semantic search API for papers.

    Combines embedding service and vector store to provide
    paper-centric search operations.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        persist_dir: Path | None = None,
    ):
        """Initialize semantic paper search.

        Args:
            embedding_service: Embedding service (defaults to Gemini)
            vector_store: Vector store (defaults to InMemoryVectorStore)
            persist_dir: Directory for persistence (used if no services provided)
        """
        self.persist_dir = persist_dir

        # Initialize embedding service
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            cache_path = persist_dir / "embedding_cache.json" if persist_dir else None
            self.embedding_service = EmbeddingService(cache_path=cache_path)

        # Initialize vector store
        if vector_store:
            self.vector_store = vector_store
        else:
            store_path = persist_dir / "vector_store.json" if persist_dir else None
            self.vector_store = InMemoryVectorStore(persist_path=store_path)

        # Track indexed papers for retrieval
        self._paper_cache: dict[str, ArxivPaper] = {}

    @property
    def is_initialized(self) -> bool:
        """Check if the search index has any papers."""
        # Fast check without async
        return len(self._paper_cache) > 0

    async def index_paper(self, paper: ArxivPaper) -> None:
        """Index a paper for semantic search.

        Args:
            paper: ArxivPaper to index
        """
        # Generate embedding from title + abstract
        text = f"{paper.title}\n\n{paper.abstract}"
        embedding = await self.embedding_service.embed(text)

        # Store metadata
        metadata = {
            "title": paper.title,
            "source": paper.source,
            "categories": ",".join(paper.categories) if paper.categories else "",
            "citation_count": paper.citation_count or 0,
        }

        # Upsert to vector store
        await self.vector_store.upsert(paper.arxiv_id, embedding, metadata)

        # Cache paper for retrieval
        self._paper_cache[paper.arxiv_id] = paper

    async def index_papers(self, papers: list[ArxivPaper]) -> int:
        """Index multiple papers for semantic search.

        Args:
            papers: List of papers to index

        Returns:
            Number of papers indexed
        """
        if not papers:
            return 0

        # Generate embeddings in batch
        texts = [f"{p.title}\n\n{p.abstract}" for p in papers]
        embeddings = await self.embedding_service.embed_batch(texts)

        # Prepare batch upsert
        items: list[tuple[str, list[float], dict[str, Any] | None]] = []
        for paper, embedding in zip(papers, embeddings):
            metadata = {
                "title": paper.title,
                "source": paper.source,
                "categories": ",".join(paper.categories) if paper.categories else "",
                "citation_count": paper.citation_count or 0,
            }
            items.append((paper.arxiv_id, embedding, metadata))
            self._paper_cache[paper.arxiv_id] = paper

        await self.vector_store.upsert_batch(items)
        return len(papers)

    async def search_similar(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
        source_filter: str | None = None,
    ) -> list[tuple[ArxivPaper, float]]:
        """Search for papers semantically similar to a query.

        Args:
            query: Search query text
            top_k: Maximum number of results
            min_score: Minimum similarity score threshold
            source_filter: Optional filter by source (arxiv, pubmed, semantic_scholar)

        Returns:
            List of (paper, score) tuples sorted by similarity
        """
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(query)

        # Build filter
        filter_metadata = None
        if source_filter:
            filter_metadata = {"source": source_filter}

        # Search vector store
        results = await self.vector_store.search(
            query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Convert to papers with scores
        paper_results: list[tuple[ArxivPaper, float]] = []
        for result in results:
            if result.score < min_score:
                continue
            paper = self._paper_cache.get(result.id)
            if paper:
                paper_results.append((paper, result.score))

        return paper_results

    async def find_related_papers(
        self,
        paper: ArxivPaper,
        top_k: int = 5,
        exclude_same_id: bool = True,
    ) -> list[tuple[ArxivPaper, float]]:
        """Find papers similar to a given paper.

        Args:
            paper: Reference paper
            top_k: Maximum number of results
            exclude_same_id: Whether to exclude the input paper from results

        Returns:
            List of (paper, score) tuples sorted by similarity
        """
        # Get or generate embedding
        text = f"{paper.title}\n\n{paper.abstract}"
        embedding = await self.embedding_service.embed(text)

        # Search with extra result if excluding self
        search_k = top_k + 1 if exclude_same_id else top_k
        results = await self.vector_store.search(embedding, top_k=search_k)

        # Convert to papers
        paper_results: list[tuple[ArxivPaper, float]] = []
        for result in results:
            if exclude_same_id and result.id == paper.arxiv_id:
                continue
            related_paper = self._paper_cache.get(result.id)
            if related_paper:
                paper_results.append((related_paper, result.score))
            if len(paper_results) >= top_k:
                break

        return paper_results

    async def get_paper(self, paper_id: str) -> ArxivPaper | None:
        """Get a paper by ID from the index.

        Args:
            paper_id: Paper identifier

        Returns:
            ArxivPaper if found, None otherwise
        """
        return self._paper_cache.get(paper_id)

    async def count(self) -> int:
        """Return the number of indexed papers."""
        return await self.vector_store.count()

    async def clear(self) -> None:
        """Clear all indexed papers."""
        await self.vector_store.clear()
        self._paper_cache.clear()
        self.embedding_service.clear_cache()

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the search index."""
        return {
            "papers_indexed": len(self._paper_cache),
            "embedding_dimension": self.embedding_service.dimension,
            "vector_store_type": type(self.vector_store).__name__,
        }


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    async def main():
        # Create test papers
        papers = [
            ArxivPaper(
                arxiv_id="2401.001",
                title="Battery Dendrite Prevention Using Solid Electrolytes",
                abstract="We investigate solid-state electrolytes to prevent lithium dendrite formation in batteries. Our approach uses ceramic materials to block dendrite growth.",
                source="arxiv",
                categories=["cond-mat.mtrl-sci"],
                published=datetime.now(),
            ),
            ArxivPaper(
                arxiv_id="2401.002",
                title="Machine Learning for Material Discovery",
                abstract="Deep learning models are used to predict material properties and accelerate discovery of new compounds for various applications.",
                source="arxiv",
                categories=["cs.LG"],
                published=datetime.now(),
            ),
            ArxivPaper(
                arxiv_id="2401.003",
                title="Lithium Ion Transport in Polymer Electrolytes",
                abstract="We study ion transport mechanisms in polymer-based electrolytes for next-generation battery applications.",
                source="pubmed",
                categories=["q-bio.BM"],
                published=datetime.now(),
            ),
        ]

        # Initialize search
        search = SemanticPaperSearch()

        # Index papers
        count = await search.index_papers(papers)
        print(f"Indexed {count} papers")
        print(f"Stats: {search.get_stats()}")

        # Search
        print("\n--- Search: 'dendrite prevention' ---")
        results = await search.search_similar("dendrite prevention", top_k=3)
        for paper, score in results:
            print(f"  [{score:.4f}] {paper.title}")

        # Find related
        print("\n--- Related to first paper ---")
        related = await search.find_related_papers(papers[0], top_k=2)
        for paper, score in related:
            print(f"  [{score:.4f}] {paper.title}")

    asyncio.run(main())
