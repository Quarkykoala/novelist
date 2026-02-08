"""Novelty verification using arXiv search.

Checks how many papers match a hypothesis's novelty keywords.
Lower hit count = more novel.
"""

import httpx

from src.contracts.schemas import EvidenceBlock, Hypothesis
from src.kb.arxiv_client import ArxivClient

ARXIV_UNAVAILABLE_MARKER = "[arxiv-unavailable]"


async def verify_novelty_arxiv(
    hypothesis: Hypothesis,
    arxiv_client: ArxivClient | None = None,
) -> EvidenceBlock:
    """Verify novelty by searching arXiv for similar work.

    Args:
        hypothesis: Hypothesis to verify
        arxiv_client: Optional existing client (creates one if None)

    Returns:
        EvidenceBlock with hit count and closest titles
    """
    keywords = hypothesis.novelty_keywords[:5]  # Max 5 keywords

    if not keywords:
        return EvidenceBlock(
            arxiv_hits=0,
            closest_arxiv_titles=[],
            arxiv_query="(no keywords provided)",
        )

    # Build query from keywords
    query = " AND ".join(f'"{kw}"' for kw in keywords)

    try:
        if arxiv_client:
            hit_count, titles = await arxiv_client.get_novelty_hits(keywords)
        else:
            async with ArxivClient() as client:
                hit_count, titles = await client.get_novelty_hits(keywords)
    except httpx.HTTPError as e:
        print(f"[WARN] arXiv novelty check failed: {e}")
        return EvidenceBlock(
            arxiv_hits=0,
            closest_arxiv_titles=[],
            arxiv_query=f"{ARXIV_UNAVAILABLE_MARKER} {query}",
        )

    return EvidenceBlock(
        arxiv_hits=hit_count,
        closest_arxiv_titles=titles,
        arxiv_query=query,
    )


def calculate_novelty_score(evidence: EvidenceBlock) -> float:
    """Calculate novelty score from evidence.

    Returns:
        Score from 0.0 (not novel) to 1.0 (highly novel)
    """
    # Score based on arXiv hits
    # < 5 hits = very novel (0.9-1.0)
    # 5-20 hits = somewhat novel (0.6-0.9)
    # 20-50 hits = moderate (0.3-0.6)
    # 50-100 hits = crowded (0.1-0.3)
    # > 100 hits = very crowded (0.0-0.1)

    if evidence.arxiv_query.startswith(ARXIV_UNAVAILABLE_MARKER):
        return 0.5

    hits = evidence.arxiv_hits

    if hits < 5:
        return 0.9 + (5 - hits) * 0.02  # 0.9-1.0
    elif hits < 20:
        return 0.9 - (hits - 5) * 0.02  # 0.6-0.9
    elif hits < 50:
        return 0.6 - (hits - 20) * 0.01  # 0.3-0.6
    elif hits < 100:
        return 0.3 - (hits - 50) * 0.004  # 0.1-0.3
    else:
        return max(0.0, 0.1 - (hits - 100) * 0.001)


async def batch_verify_novelty(
    hypotheses: list[Hypothesis],
) -> list[tuple[Hypothesis, EvidenceBlock, float]]:
    """Verify novelty for multiple hypotheses.

    Args:
        hypotheses: List of hypotheses to verify

    Returns:
        List of (hypothesis, evidence, score) tuples
    """
    results: list[tuple[Hypothesis, EvidenceBlock, float]] = []

    async with ArxivClient() as client:
        for h in hypotheses:
            evidence = await verify_novelty_arxiv(h, client)
            score = calculate_novelty_score(evidence)

            # Update hypothesis evidence and score
            h.evidence = evidence
            h.scores.novelty = score

            results.append((h, evidence, score))

    return results
