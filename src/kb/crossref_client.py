"""Crossref API client for DOI-centric literature discovery.

Uses Crossref Works API:
- Free, no API key required
- Good coverage for DOI metadata across domains
- Docs: https://api.crossref.org/swagger-ui/index.html
"""

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime
from typing import Any

import httpx

from src.contracts.schemas import ArxivPaper

CROSSREF_BASE_URL = "https://api.crossref.org/works"
CROSSREF_RATE_LIMIT_SECONDS = 1.0

_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


def _strip_html(raw: str) -> str:
    if not raw:
        return ""
    text = re.sub(r"<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


class CrossrefClient:
    """Async client for Crossref works search."""

    def __init__(self, rate_limit: float = CROSSREF_RATE_LIMIT_SECONDS):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "CrossrefClient":
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "Novelist/1.0 (mailto:novelist@local.invalid)"},
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
        response = await self._client.get(CROSSREF_BASE_URL, params=params)
        response.raise_for_status()
        return response.json()

    async def search(self, query: str, max_results: int = 30) -> list[ArxivPaper]:
        """Search Crossref works by bibliographic query."""
        try:
            data = await self._make_request(
                {
                    "query.bibliographic": query,
                    "rows": min(max_results, 50),
                    "select": (
                        "DOI,title,author,issued,subject,abstract,URL,"
                        "is-referenced-by-count"
                    ),
                    "sort": "relevance",
                    "order": "desc",
                }
            )
        except Exception as e:
            print(f"[WARN] Crossref search failed: {e}")
            return []

        items = (data.get("message") or {}).get("items", [])
        papers: list[ArxivPaper] = []
        for item in items:
            paper = self._parse_work(item)
            if paper:
                papers.append(paper)
        return papers

    def _parse_work(self, item: dict[str, Any]) -> ArxivPaper | None:
        try:
            titles = item.get("title") or []
            title = (titles[0] if titles else "").strip()
            doi = (item.get("DOI") or "").strip()
            if not title or not doi:
                return None

            authors: list[str] = []
            for author in item.get("author", [])[:10]:
                given = (author.get("given") or "").strip()
                family = (author.get("family") or "").strip()
                full = " ".join(part for part in [given, family] if part)
                if full:
                    authors.append(full)

            categories: list[str] = []
            for subject in item.get("subject", [])[:5]:
                if isinstance(subject, str) and subject.strip():
                    categories.append(subject.strip())

            published = None
            date_parts = ((item.get("issued") or {}).get("date-parts") or [])
            if date_parts and isinstance(date_parts[0], list):
                parts = date_parts[0]
                year = int(parts[0]) if len(parts) > 0 else None
                month = int(parts[1]) if len(parts) > 1 else 1
                day = int(parts[2]) if len(parts) > 2 else 1
                if year:
                    published = datetime(year, month, day)

            abstract = _strip_html(item.get("abstract") or "")
            url = (item.get("URL") or "").strip()

            return ArxivPaper(
                arxiv_id=f"DOI:{doi}",
                title=title,
                abstract=abstract,
                authors=authors,
                categories=categories,
                published=published,
                abs_url=url,
                source="crossref",
                citation_count=item.get("is-referenced-by-count"),
            )
        except Exception:
            return None
