"""arXiv API client for fetching and searching papers.

Uses the arXiv API (https://info.arxiv.org/help/api/index.html).
- Free, no API key required
- Rate limit: 3 requests per second (we use 1 req/s to be safe)
- Returns Atom XML which we parse into structured data
"""

import asyncio
import xml.etree.ElementTree as ET
from collections import OrderedDict
from datetime import datetime
from typing import Any

import httpx

from src.contracts.schemas import ArxivPaper

# arXiv API configuration
ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0  # Respect standard 3s interval
ARXIV_MAX_RESULTS = 100

# XML namespaces used by arXiv Atom feed
NAMESPACES = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

# arXiv category mappings for neighbor detection
CATEGORY_NEIGHBORS: dict[str, list[str]] = {
    "cs.AI": ["cs.LG", "cs.CL", "cs.CV", "stat.ML"],
    "cs.LG": ["cs.AI", "stat.ML", "cs.CV", "cs.NE"],
    "cs.CL": ["cs.AI", "cs.LG", "cs.IR"],
    "cs.CV": ["cs.LG", "cs.AI", "eess.IV"],
    "q-bio.BM": ["q-bio.MN", "q-bio.GN", "physics.bio-ph"],
    "q-bio.GN": ["q-bio.BM", "q-bio.MN", "cs.LG"],
    "q-bio.NC": ["q-bio.QM", "cs.AI", "cs.NE"],
    "physics.bio-ph": ["q-bio.BM", "cond-mat.soft"],
    "stat.ML": ["cs.LG", "cs.AI", "math.ST"],
}


# Global rate limiter state
_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


class SimpleLRUCache:
    def __init__(self, capacity: int = 128):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: Any) -> Any | None:
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: Any, value: Any) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

_SEARCH_CACHE = SimpleLRUCache(capacity=100)


class ArxivClient:
    """Async client for the arXiv API."""

    def __init__(self, rate_limit: float = ARXIV_RATE_LIMIT_SECONDS):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ArxivClient":
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Novelist/1.0 (research-bot; python)"}
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _rate_limit_wait(self) -> None:
        """Wait if needed to respect rate limits globally."""
        global _LAST_REQUEST_TIME
        
        # Ensure lock is valid for current loop
        # (Handling the case where loop might have restarted or lock created in different loop context)
        # But for uvicorn single worker, global lock is usually fine if initialized in module?
        # Actually, asyncio.Lock() creates a future attached to the loop. 
        # If imported before loop start, it might be bound to a different loop (or none).
        # Safe pattern: logic inside the method.
        
        async with _GLOBAL_LOCK:
            now = asyncio.get_running_loop().time()
            elapsed = now - _LAST_REQUEST_TIME
            if elapsed < self.rate_limit:
                wait_time = self.rate_limit - elapsed
                await asyncio.sleep(wait_time)
            _LAST_REQUEST_TIME = asyncio.get_running_loop().time()

    async def _make_request(self, params: dict[str, Any]) -> httpx.Response:
        """Make a request to the arXiv API with rate limiting and retries."""
        if not self._client:
            raise RuntimeError("Client not initialized. Use 'async with' context.")

        retries = 4
        backoff = 10.0  # Start with 10 seconds for 429

        for attempt in range(retries + 1):
            await self._rate_limit_wait()
            try:
                response = await self._client.get(ARXIV_API_BASE, params=params)
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                # Handle 429 (Too Many Requests) and 503 (Service Unavailable)
                if e.response.status_code in (429, 503):
                    if attempt < retries:
                        wait_time = backoff * (2**attempt)
                        print(
                            f"[ArxivClient] Got {e.response.status_code}. Retrying in {wait_time}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                raise e

    async def search(
        self,
        query: str,
        max_results: int = 30,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[ArxivPaper]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search query (supports arXiv query syntax)
            max_results: Maximum number of results to return
            sort_by: Sort field (relevance, lastUpdatedDate, submittedDate)
            sort_order: Sort order (ascending, descending)

        Returns:
            List of ArxivPaper objects
        """
        # Build query parameters
        query_text = query.strip()
        needs_group = any(op in query_text.lower() for op in [" or ", " and ", " not "])
        search_query = f"all:({query_text})" if needs_group else f"all:{query_text}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(max_results, ARXIV_MAX_RESULTS),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        # Check cache
        cache_key = (params["search_query"], params["max_results"], params["sortBy"], params["sortOrder"])
        cached = _SEARCH_CACHE.get(cache_key)
        if cached:
            return cached

        response = await self._make_request(params)
        result = self._parse_atom_response(response.text)

        _SEARCH_CACHE.put(cache_key, result)
        return result

    async def search_by_category(
        self,
        category: str,
        query: str = "",
        max_results: int = 20,
    ) -> list[ArxivPaper]:
        """Search within a specific arXiv category.

        Args:
            category: arXiv category (e.g., 'cs.AI', 'q-bio.BM')
            query: Additional search terms
            max_results: Maximum results

        Returns:
            List of ArxivPaper objects
        """
        if query:
            query_text = query.strip()
            needs_group = any(op in query_text.lower() for op in [" or ", " and ", " not "])
            all_query = f"all:({query_text})" if needs_group else f"all:{query_text}"
            full_query = f"cat:{category} AND {all_query}"
        else:
            full_query = f"cat:{category}"

        params = {
            "search_query": full_query,
            "start": 0,
            "max_results": min(max_results, ARXIV_MAX_RESULTS),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        response = await self._make_request(params)
        return self._parse_atom_response(response.text)

    async def get_novelty_hits(
        self,
        keywords: list[str],
        max_results: int = 10,
    ) -> tuple[int, list[str]]:
        """Search for keywords to assess novelty.

        Args:
            keywords: List of novelty keywords to search
            max_results: Max papers to fetch for titles

        Returns:
            Tuple of (total hit count estimate, list of closest titles)
        """
        query = " AND ".join(f'"{kw}"' for kw in keywords[:5])  # Limit to 5 keywords
        papers = await self.search(query, max_results=max_results)

        # The API doesn't give us a total count, so we estimate from results
        # If we got max_results, there are likely more
        hit_count = len(papers)
        if hit_count == max_results:
            hit_count = max_results * 2  # Conservative estimate

        titles = [p.title for p in papers[:5]]
        return hit_count, titles

    def _parse_atom_response(self, xml_text: str) -> list[ArxivPaper]:
        """Parse arXiv Atom XML response into ArxivPaper objects."""
        papers: list[ArxivPaper] = []

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return papers

        for entry in root.findall("atom:entry", NAMESPACES):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> ArxivPaper | None:
        """Parse a single Atom entry into an ArxivPaper."""
        try:
            # Extract arxiv ID from the id URL
            id_elem = entry.find("atom:id", NAMESPACES)
            if id_elem is None or id_elem.text is None:
                return None

            arxiv_id = id_elem.text.split("/abs/")[-1]

            # Title
            title_elem = entry.find("atom:title", NAMESPACES)
            title = title_elem.text.strip().replace("\n", " ") if title_elem is not None and title_elem.text else ""

            # Abstract (summary)
            summary_elem = entry.find("atom:summary", NAMESPACES)
            abstract = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None and summary_elem.text else ""

            # Authors
            authors: list[str] = []
            for author in entry.findall("atom:author", NAMESPACES):
                name_elem = author.find("atom:name", NAMESPACES)
                if name_elem is not None and name_elem.text:
                    authors.append(name_elem.text)

            # Categories
            categories: list[str] = []
            for cat in entry.findall("arxiv:primary_category", NAMESPACES):
                term = cat.get("term")
                if term:
                    categories.append(term)
            for cat in entry.findall("atom:category", NAMESPACES):
                term = cat.get("term")
                if term and term not in categories:
                    categories.append(term)

            # Dates
            published = self._parse_date(entry.find("atom:published", NAMESPACES))
            updated = self._parse_date(entry.find("atom:updated", NAMESPACES))

            # Links
            pdf_url = ""
            abs_url = ""
            for link in entry.findall("atom:link", NAMESPACES):
                link_type = link.get("type", "")
                href = link.get("href", "")
                if "pdf" in link_type:
                    pdf_url = href
                elif link.get("rel") == "alternate":
                    abs_url = href

            return ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                abs_url=abs_url,
            )
        except Exception:
            return None

    def _parse_date(self, elem: ET.Element | None) -> datetime | None:
        """Parse an ISO date from an XML element."""
        if elem is None or elem.text is None:
            return None
        try:
            # arXiv uses ISO format: 2023-01-15T12:34:56Z
            return datetime.fromisoformat(elem.text.replace("Z", "+00:00"))
        except ValueError:
            return None


def get_neighbor_categories(category: str) -> list[str]:
    """Get related arXiv categories for domain injection.

    Args:
        category: Primary arXiv category

    Returns:
        List of related category codes
    """
    return CATEGORY_NEIGHBORS.get(category, [])


def detect_categories_from_query(query: str) -> list[str]:
    """Detect likely arXiv categories from a research query.

    This is a simple keyword-based heuristic.
    """
    query_lower = query.lower()
    categories: list[str] = []

    # Machine learning / AI keywords
    if any(kw in query_lower for kw in ["machine learning", "deep learning", "neural", "ai", "llm", "transformer"]):
        categories.extend(["cs.LG", "cs.AI"])

    # NLP keywords
    if any(kw in query_lower for kw in ["language", "nlp", "text", "semantic", "embedding"]):
        categories.append("cs.CL")

    # Biology keywords
    if any(kw in query_lower for kw in ["crispr", "gene", "dna", "rna", "protein", "cell", "molecular"]):
        categories.extend(["q-bio.BM", "q-bio.GN"])

    # Neuroscience keywords
    if any(kw in query_lower for kw in ["neural tissue", "brain", "neuron", "cognitive", "neuro"]):
        categories.append("q-bio.NC")

    # Physics keywords
    if any(kw in query_lower for kw in ["quantum", "physics", "particle"]):     
        categories.append("physics.bio-ph" if "bio" in query_lower else "physics.gen-ph")

    # Energy storage / batteries keywords
    if any(kw in query_lower for kw in ["battery", "batteries", "lithium", "electrolyte", "anode", "cathode", "solid-state", "solid state", "energy storage"]):
        categories.extend(["cond-mat.mtrl-sci", "cond-mat.mes-hall", "physics.chem-ph", "physics.app-ph"])

    # Remove duplicates while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cat in categories:
        if cat not in seen:
            seen.add(cat)
            unique.append(cat)

    return unique[:3]  # Return top 3 most relevant


# =============================================================================
# CLI testing utility
# =============================================================================

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Test arXiv API client")
    parser.add_argument("--query", "-q", default="CRISPR delivery", help="Search query")
    parser.add_argument("--max-results", "-n", type=int, default=5, help="Max results")
    args = parser.parse_args()

    async def main() -> None:
        async with ArxivClient() as client:
            print(f"Searching arXiv for: {args.query}")
            papers = await client.search(args.query, max_results=args.max_results)
            print(f"Found {len(papers)} papers:\n")
            for i, paper in enumerate(papers, 1):
                print(f"{i}. [{paper.arxiv_id}] {paper.title[:80]}...")
                print(f"   Categories: {', '.join(paper.categories)}")
                print()

    asyncio.run(main())
