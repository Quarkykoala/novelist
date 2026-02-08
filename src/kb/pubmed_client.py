"""PubMed E-utilities API client for fetching papers.

Uses NCBI E-utilities (https://www.ncbi.nlm.nih.gov/books/NBK25500/):
- esearch: Search for PMIDs
- efetch: Fetch paper details
- Free tier: 3 requests/second without API key, 10 req/s with key
"""

import asyncio
import os
from datetime import datetime
from typing import Any
from xml.etree import ElementTree as ET

import httpx
from dotenv import load_dotenv

from src.contracts.schemas import ArxivPaper

load_dotenv()

# Optional API key for higher rate limits
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")
PUBMED_RATE_LIMIT = 0.35 if NCBI_API_KEY else 1.0  # seconds between requests

# Global rate limiter
_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


class PubMedClient:
    """Async client for the NCBI PubMed E-utilities API."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, rate_limit: float = PUBMED_RATE_LIMIT):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PubMedClient":
        self._client = httpx.AsyncClient(timeout=30.0)
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

    async def _make_request(self, endpoint: str, params: dict[str, Any]) -> str:
        """Make a request to the E-utilities API with rate limiting."""
        await self._rate_limit_wait()

        if NCBI_API_KEY:
            params["api_key"] = NCBI_API_KEY

        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)

        url = f"{self.BASE_URL}/{endpoint}"
        response = await self._client.get(url, params=params)
        response.raise_for_status()
        return response.text

    async def search(
        self,
        query: str,
        max_results: int = 30,
        sort: str = "relevance",
    ) -> list[ArxivPaper]:
        """Search PubMed for papers matching the query.

        Args:
            query: Search query (supports PubMed query syntax)
            max_results: Maximum number of results to return
            sort: Sort order ('relevance' or 'date')

        Returns:
            List of ArxivPaper objects with source='pubmed'
        """
        try:
            # Step 1: Search for PMIDs
            sort_param = "relevance" if sort == "relevance" else "pub+date"
            search_xml = await self._make_request("esearch.fcgi", {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "sort": sort_param,
                "retmode": "xml",
            })

            pmids = self._parse_search_results(search_xml)
            if not pmids:
                return []

            # Step 2: Fetch paper details
            fetch_xml = await self._make_request("efetch.fcgi", {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            })

            return self._parse_fetch_results(fetch_xml)

        except Exception as e:
            print(f"[ERROR] PubMed search failed: {e}")
            return []

    async def get_by_pmid(self, pmid: str) -> ArxivPaper | None:
        """Fetch a single paper by PMID."""
        papers = await self.search(f"{pmid}[uid]", max_results=1)
        return papers[0] if papers else None

    async def get_novelty_hits(
        self,
        keywords: list[str],
        max_results: int = 10,
    ) -> tuple[int, list[str]]:
        """Search for keywords to assess novelty.

        Returns:
            Tuple of (total hit count, list of closest titles)
        """
        query = " AND ".join(keywords)
        try:
            search_xml = await self._make_request("esearch.fcgi", {
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "xml",
            })

            # Parse count
            root = ET.fromstring(search_xml)
            count_elem = root.find(".//Count")
            total_count = int(count_elem.text) if count_elem is not None and count_elem.text else 0

            # Get titles
            pmids = self._parse_search_results(search_xml)
            if not pmids:
                return total_count, []

            papers = await self.search(query, max_results=max_results)
            titles = [p.title for p in papers]

            return total_count, titles

        except Exception as e:
            print(f"[ERROR] PubMed novelty check failed: {e}")
            return 0, []

    def _parse_search_results(self, xml_text: str) -> list[str]:
        """Parse esearch XML to extract PMIDs."""
        try:
            root = ET.fromstring(xml_text)
            pmids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
            return pmids
        except Exception:
            return []

    def _parse_fetch_results(self, xml_text: str) -> list[ArxivPaper]:
        """Parse efetch XML to extract paper details."""
        papers = []
        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                paper = self._parse_article(article)
                if paper:
                    papers.append(paper)

        except Exception as e:
            print(f"[ERROR] Failed to parse PubMed response: {e}")

        return papers

    def _parse_article(self, article: ET.Element) -> ArxivPaper | None:
        """Parse a single PubmedArticle element."""
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None and pmid_elem.text else ""

            # Article info
            article_elem = medline.find(".//Article")
            if article_elem is None:
                return None

            # Title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None and title_elem.text else ""

            # Abstract
            abstract_parts = []
            for abstract_text in article_elem.findall(".//AbstractText"):
                if abstract_text.text:
                    label = abstract_text.get("Label", "")
                    if label:
                        abstract_parts.append(f"{label}: {abstract_text.text}")
                    else:
                        abstract_parts.append(abstract_text.text)
            abstract = " ".join(abstract_parts)

            # Authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last = author.find("LastName")
                fore = author.find("ForeName")
                if last is not None and last.text:
                    name = last.text
                    if fore is not None and fore.text:
                        name = f"{fore.text} {last.text}"
                    authors.append(name)

            # Date
            pub_date = None
            date_elem = article_elem.find(".//PubDate")
            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                day = date_elem.find("Day")
                if year is not None and year.text:
                    try:
                        y = int(year.text)
                        m = int(month.text) if month is not None and month.text.isdigit() else 1
                        d = int(day.text) if day is not None and day.text.isdigit() else 1
                        pub_date = datetime(y, m, d)
                    except:
                        pass

            # MeSH categories
            categories = []
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    categories.append(mesh.text)

            return ArxivPaper(
                arxiv_id=f"PMID:{pmid}",
                title=title,
                abstract=abstract,
                authors=authors[:10],  # Limit authors
                categories=categories[:5],
                published=pub_date,
                abs_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                source="pubmed",
            )

        except Exception as e:
            print(f"[WARN] Failed to parse PubMed article: {e}")
            return None


# =============================================================================
# CLI testing utility
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search PubMed")
    parser.add_argument("--query", "-q", default="CRISPR delivery", help="Search query")
    parser.add_argument("--max-results", "-n", type=int, default=5, help="Max results")
    args = parser.parse_args()

    async def main() -> None:
        async with PubMedClient() as client:
            papers = await client.search(args.query, max_results=args.max_results)
            print(f"\nFound {len(papers)} papers:\n")
            for p in papers:
                print(f"  [{p.arxiv_id}] {p.title[:60]}...")
                print(f"    Authors: {', '.join(p.authors[:3])}")
                print(f"    Published: {p.published}")
                print()

    asyncio.run(main())
