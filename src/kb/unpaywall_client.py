"""Unpaywall API client for open-access full-text resolution by DOI.

Notes:
- Unpaywall requires an email query parameter.
- Set UNPAYWALL_EMAIL in environment to enable enrichment.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"
UNPAYWALL_EMAIL = os.getenv("UNPAYWALL_EMAIL", "").strip()
UNPAYWALL_RATE_LIMIT_SECONDS = 0.35

_LAST_REQUEST_TIME = 0.0
_GLOBAL_LOCK = asyncio.Lock()


class UnpaywallClient:
    """Async client for Unpaywall DOI lookups."""

    def __init__(self, rate_limit: float = UNPAYWALL_RATE_LIMIT_SECONDS):
        self.rate_limit = rate_limit
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(UNPAYWALL_EMAIL)

    async def __aenter__(self) -> "UnpaywallClient":
        self._client = httpx.AsyncClient(timeout=20.0)
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

    async def _make_request(self, doi: str) -> dict[str, Any]:
        if not self.enabled:
            return {}
        await self._rate_limit_wait()
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=20.0)
        url = f"{UNPAYWALL_BASE_URL}/{doi}"
        response = await self._client.get(url, params={"email": UNPAYWALL_EMAIL})
        if response.status_code == 404:
            return {}
        response.raise_for_status()
        return response.json()

    async def lookup(self, doi: str) -> dict[str, str]:
        """Return OA links for a DOI, if available."""
        clean = (doi or "").strip().lower()
        if not clean:
            return {}
        data = await self._make_request(clean)
        if not data:
            return {}

        best = data.get("best_oa_location") or {}
        abs_url = (
            best.get("url_for_landing_page")
            or best.get("url")
            or ""
        )
        pdf_url = best.get("url_for_pdf") or ""
        pmh_id = (best.get("pmh_id") or "").strip()
        pmcid = ""
        if pmh_id.startswith("oai:pmc.ncbi.nlm.nih.gov:"):
            pmcid = pmh_id.split(":")[-1]
        elif pmh_id.upper().startswith("PMC"):
            pmcid = pmh_id.upper()

        out: dict[str, str] = {}
        if abs_url:
            out["abs_url"] = abs_url
        if pdf_url:
            out["pdf_url"] = pdf_url
        if pmcid:
            out["pmcid"] = pmcid
        return out
