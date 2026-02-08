"""Citation validation utilities for grounding hypotheses in arXiv papers."""

from __future__ import annotations

from typing import Iterable

from src.kb.arxiv_client import ArxivClient, is_arxiv_id, normalize_arxiv_id
from src.contracts.schemas import ArxivPaper


def _build_arxiv_index(paper_store: dict[str, ArxivPaper]) -> dict[str, str]:
    """Map normalized arXiv base IDs -> stored IDs."""
    index: dict[str, str] = {}
    for stored_id in paper_store.keys():
        base = normalize_arxiv_id(stored_id, keep_version=False)
        if base and base not in index:
            index[base] = stored_id
    return index


async def validate_supporting_papers(
    paper_ids: Iterable[str],
    paper_store: dict[str, ArxivPaper],
    arxiv_client: ArxivClient,
    *,
    max_fetch: int = 15,
) -> tuple[list[str], list[str], list[str]]:
    """Validate supporting paper IDs against arXiv.

    Returns:
        (validated_arxiv_ids, invalid_arxiv_ids, non_arxiv_ids)
    """
    if not paper_ids:
        return [], [], []

    index = _build_arxiv_index(paper_store)
    normalized_inputs: list[str] = []
    non_arxiv: list[str] = []
    invalid: list[str] = []

    # Normalize and collect arXiv IDs to check
    to_fetch: list[str] = []
    seen_fetch: set[str] = set()
    for raw in paper_ids:
        raw_id = (raw or "").strip()
        if not raw_id:
            continue
        if not is_arxiv_id(raw_id):
            non_arxiv.append(raw_id)
            continue
        base = normalize_arxiv_id(raw_id, keep_version=False)
        if not base:
            invalid.append(raw_id)
            continue
        normalized_inputs.append(base)
        if base not in index and base not in seen_fetch and len(to_fetch) < max_fetch:
            seen_fetch.add(base)
            to_fetch.append(base)

    # Fetch unknown IDs from arXiv
    if to_fetch:
        fetched = await arxiv_client.fetch_by_ids(to_fetch)
        for base_id, paper in fetched.items():
            paper_store[paper.arxiv_id] = paper
            index[base_id] = paper.arxiv_id

    # Build validated list in input order (deduplicated)
    validated: list[str] = []
    seen_valid: set[str] = set()
    for base in normalized_inputs:
        stored_id = index.get(base)
        if stored_id:
            if stored_id not in seen_valid:
                seen_valid.add(stored_id)
                validated.append(stored_id)
        else:
            invalid.append(base)

    return validated, invalid, non_arxiv
