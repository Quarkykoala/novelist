
import asyncio
from src.kb.arxiv_client import ArxivClient

async def test_arxiv():
    print("Testing ArxivClient...")
    try:
        async with ArxivClient() as client:
            print("Client created. Searching...")
            papers = await client.search("machine learning", max_results=5)
            print(f"Found {len(papers)} papers.")
            for p in papers:
                 print(f"- {p.title}")
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_arxiv())
