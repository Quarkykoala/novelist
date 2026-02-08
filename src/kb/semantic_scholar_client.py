"""Semantic Scholar API client for fetching papers.

Uses Semantic Scholar Academic Graph API:
- Free tier: 100 requests per 5 minutes (no key required)
- Includes citation counts for impact scoring
- API docs: https://api.semanticscholar.org/api-docs/
"""

import asyncio
import os
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from src.contracts.schemas import ArxivPaper

load_dotenv()

# Optional API key for higher rate limits
S2_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
S2_RATE_LIMIT = 3.0  # Conservative: ~20 req/min to stay well under 100/5min

# Global rate limiter
_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


class SemanticScholarClient:
    """Async client for the Semantic Scholar Academic Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, rate_limit: float = S2_RATE_LIMIT):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "SemanticScholarClient":
        headers = {"Accept": "application/json"}
        if S2_API_KEY:
            headers["x-api-key"] = S2_API_KEY
        self._client = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limits globally."""
        global _LAST_REQUEST_TIME
        async with _GLOBAL_LOCK:
            now = asyncio.get_event_loop().time()
            elapsed = now - _LAST_REQUEST_TIME
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            _LAST_REQUEST_TIME = asyncio.get_event_loop().time()

    async def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a request to the S2 API with rate limiting."""
        await self._rate_limit_wait()

        if self._client is None:
            headers = {"Accept": "application/json"}
            if S2_API_KEY:
                headers["x-api-key"] = S2_API_KEY
            self._client = httpx.AsyncClient(timeout=30.0, headers=headers)

        url = f"{self.BASE_URL}/{endpoint}"
        response = await self._client.get(url, params=params or {})
        response.raise_for_status()
        return response.json()

    async def search(
        self,
        query: str,
        max_results: int = 30,
        year: str | None = None,
    ) -> list[ArxivPaper]:
        """Search Semantic Scholar for papers matching the query.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            year: Optional year filter (e.g., "2020-2024" or "2023")

        Returns:
            List of ArxivPaper objects with source='semantic_scholar'
        """
        try:
            params = {
                "query": query,
                "limit": min(max_results, 100),  # S2 max is 100
                "fields": "paperId,title,abstract,authors,year,citationCount,externalIds,url",
            }
            if year:
                params["year"] = year

            data = await self._make_request("paper/search", params)
            papers = []

            for item in data.get("data", []):
                paper = self._parse_paper(item)
                if paper:
                    papers.append(paper)

            return papers

        except Exception as e:
            print(f"[ERROR] Semantic Scholar search failed: {e}")
            return []

    async def get_by_paper_id(self, paper_id: str) -> ArxivPaper | None:
        """Fetch a single paper by S2 paper ID, DOI, or arXiv ID.
        
        Args:
            paper_id: S2 paper ID, DOI (doi:xxx), or arXiv ID (arXiv:xxx)
        """
        try:
            data = await self._make_request(
                f"paper/{paper_id}",
                {"fields": "paperId,title,abstract,authors,year,citationCount,externalIds,url"}
            )
            return self._parse_paper(data)
        except Exception as e:
            print(f"[ERROR] Failed to fetch paper {paper_id}: {e}")
            return None

    async def get_novelty_hits(
        self,
        keywords: list[str],
        max_results: int = 10,
    ) -> tuple[int, list[str]]:
        """Search for keywords to assess novelty.

        Returns:
            Tuple of (total hit count, list of closest titles)
        """
        query = " ".join(keywords)
        try:
            params = {
                "query": query,
                "limit": max_results,
                "fields": "paperId,title,citationCount",
            }

            data = await self._make_request("paper/search", params)
            total_count = data.get("total", 0)
            titles = [p.get("title", "") for p in data.get("data", [])]

            return total_count, titles

        except Exception as e:
            print(f"[ERROR] Semantic Scholar novelty check failed: {e}")
            return 0, []

    async def get_citations_for_paper(self, paper_id: str, limit: int = 50) -> list[ArxivPaper]:
        """Get papers that cite a given paper."""
        try:
            data = await self._make_request(
                f"paper/{paper_id}/citations",
                {"fields": "paperId,title,abstract,authors,year,citationCount", "limit": limit}
            )
            papers = []
            for item in data.get("data", []):
                citing_paper = item.get("citingPaper", {})
                paper = self._parse_paper(citing_paper)
                if paper:
                    papers.append(paper)
            return papers
        except Exception:
            return []

    def _parse_paper(self, data: dict[str, Any]) -> ArxivPaper | None:
        """Parse a single paper from S2 API response."""
        try:
            paper_id = data.get("paperId", "")
            title = data.get("title", "")
            abstract = data.get("abstract") or ""

            # Skip papers without abstract (usually incomplete records)
            if not title:
                return None

            # Authors
            authors = []
            for author in data.get("authors", [])[:10]:
                name = author.get("name", "")
                if name:
                    authors.append(name)

            # Year to date
            year = data.get("year")
            pub_date = datetime(year, 1, 1) if year else None

            # External IDs
            external_ids = data.get("externalIds", {}) or {}
            arxiv_id = external_ids.get("ArXiv", "")
            doi = external_ids.get("DOI", "")

            # Create identifier
            if arxiv_id:
                identifier = arxiv_id  # Use arXiv ID if available
            elif doi:
                identifier = f"DOI:{doi}"
            else:
                identifier = f"S2:{paper_id}"

            # URL
            url = data.get("url", "")
            if not url and paper_id:
                url = f"https://www.semanticscholar.org/paper/{paper_id}"

            return ArxivPaper(
                arxiv_id=identifier,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=[],  # S2 doesn't provide categories in basic search
                published=pub_date,
                abs_url=url,
                source="semantic_scholar",
                citation_count=data.get("citationCount"),
            )

        except Exception as e:
            print(f"[WARN] Failed to parse S2 paper: {e}")
            return None


# =============================================================================
# CLI testing utility
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Semantic Scholar")
    parser.add_argument("--query", "-q", default="battery dendrite formation", help="Search query")
    parser.add_argument("--max-results", "-n", type=int, default=5, help="Max results")
    args = parser.parse_args()

    async def main() -> None:
        async with SemanticScholarClient() as client:
            papers = await client.search(args.query, max_results=args.max_results)
            print(f"\nFound {len(papers)} papers:\n")
            for p in papers:
                citations = f"({p.citation_count} citations)" if p.citation_count else ""
                print(f"  [{p.arxiv_id}] {p.title[:50]}... {citations}")
                print(f"    Authors: {', '.join(p.authors[:3])}")
                print()

    asyncio.run(main())
