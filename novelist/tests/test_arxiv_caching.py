
import pytest
import time
from src.kb.arxiv_client import ArxivClient

@pytest.mark.asyncio
async def test_arxiv_search_caching():
    """Verify that repeated searches are cached and fast."""
    query = "machine learning"

    async with ArxivClient() as client:
        # First call - should hit API (slow)
        start1 = time.time()
        res1 = await client.search(query, max_results=5)
        end1 = time.time()
        duration1 = end1 - start1

        # Second call - should hit cache (fast)
        start2 = time.time()
        res2 = await client.search(query, max_results=5)
        end2 = time.time()
        duration2 = end2 - start2

        # Check results are consistent
        assert len(res1) == len(res2)
        if res1:
            assert res1[0].title == res2[0].title

        # Verify speedup
        # 2nd call should be significantly faster, basically instant
        assert duration2 < 0.1, f"Cached call took too long: {duration2}s"

        # Verify 1st call was slower (assuming it actually hit network/rate limit)
        # Note: If rate limit state was fresh, 1st call might be fast too,
        # but 2nd call definitely shouldn't wait for rate limit.
        # If caching wasn't working, 2nd call would wait ~3s due to rate limit.

    # Verify cache persistence across client instances (since we used global cache)
    async with ArxivClient() as client2:
        start3 = time.time()
        res3 = await client2.search(query, max_results=5)
        end3 = time.time()
        duration3 = end3 - start3

        assert len(res1) == len(res3)
        assert duration3 < 0.1, f"Cross-client cached call took too long: {duration3}s"
