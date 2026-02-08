"""OpenAlex API client for open bibliographic retrieval.

Uses OpenAlex Works API:
- Free, no API key required
- Includes DOI metadata, citation counts, and concept tags
- Docs: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import httpx

from src.contracts.schemas import ArxivPaper

OPENALEX_BASE_URL = "https://api.openalex.org/works"
OPENALEX_RATE_LIMIT_SECONDS = 0.5

_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


def _decode_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Convert OpenAlex abstract_inverted_index into plain text."""
    if not inverted_index:
        return ""
    tokens: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions or []:
            if isinstance(pos, int):
                tokens.append((pos, word))
    if not tokens:
        return ""
    tokens.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in tokens)


class OpenAlexClient:
    """Async client for OpenAlex Works search."""

    def __init__(self, rate_limit: float = OPENALEX_RATE_LIMIT_SECONDS):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenAlexClient":
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Novelist/1.0 (research-bot; openalex)"},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def _rate_limit_wait(self) -> None:
        global _LAST_REQUEST_TIME
        async with _GLOBAL_LOCK:
            now = asyncio.get_running_loop().time()
            elapsed = now - _LAST_REQUEST_TIME
            if elapsed < self.rate_limit:
                await asyncio.sleep(self.rate_limit - elapsed)
            _LAST_REQUEST_TIME = asyncio.get_running_loop().time()

    async def _make_request(self, params: dict[str, Any]) -> dict[str, Any]:
        await self._rate_limit_wait()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        response = await self._client.get(OPENALEX_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    async def search(self, query: str, max_results: int = 30) -> list[ArxivPaper]:
        """Search OpenAlex works by full-text query."""
        try:
            data = await self._make_request(
                {
                    "search": query,
                    "per-page": min(max_results, 50),
                    "sort": "cited_by_count:desc",
                    "mailto": "novelist@local.invalid",
                }
            )
        except Exception as e:
            print(f"[WARN] OpenAlex search failed: {e}")
            return []

        papers: list[ArxivPaper] = []
        for item in data.get("results", []):
            paper = self._parse_work(item)
            if paper:
                papers.append(paper)
        return papers

    def _parse_work(self, item: dict[str, Any]) -> ArxivPaper | None:
        try:
            title = (item.get("display_name") or "").strip()
            if not title:
                return None

            doi_url = (item.get("doi") or "").strip()
            doi = doi_url.replace("https://doi.org/", "").replace("http://doi.org/", "")
            openalex_id = (item.get("id") or "").split("/")[-1]

            if doi:
                identifier = f"DOI:{doi}"
            elif openalex_id:
                identifier = f"OPENALEX:{openalex_id}"
            else:
                return None

            abstract = _decode_abstract(item.get("abstract_inverted_index"))
            if not abstract:
                abstract = (item.get("abstract") or "").strip()

            authors: list[str] = []
            for authorship in item.get("authorships", [])[:10]:
                author = authorship.get("author", {}) if isinstance(authorship, dict) else {}
                name = (author.get("display_name") or "").strip()
                if name:
                    authors.append(name)

            categories: list[str] = []
            for concept in item.get("concepts", [])[:5]:
                name = (concept.get("display_name") or "").strip()
                if name:
                    categories.append(name)

            publication_date = item.get("publication_date")
            published = None
            if publication_date:
                try:
                    published = datetime.fromisoformat(publication_date)
                except ValueError:
                    published = None

            primary_location = item.get("primary_location") or {}
            abs_url = (
                primary_location.get("landing_page_url")
                or primary_location.get("pdf_url")
                or (item.get("id") or "")
            )
            pdf_url = primary_location.get("pdf_url") or ""

            return ArxivPaper(
                arxiv_id=identifier,
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published,
                abs_url=abs_url,
                pdf_url=pdf_url,
                source="openalex",
                citation_count=item.get("cited_by_count"),
            )
        except Exception:
            return None
