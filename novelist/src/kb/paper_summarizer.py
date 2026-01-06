"""Paper summarization using Gemini.

Summarizes arXiv paper abstracts into structured format:
- What problem does this address?
- What method/approach is used?
- What is the main result?
- What are the limitations?
- What future work is suggested?
"""

import asyncio
import os
from typing import Any

from dotenv import load_dotenv

from src.contracts.schemas import ArxivPaper, PaperSummary
from src.contracts.validators import extract_json_from_response

load_dotenv()

# Gemini configuration
FLASH_MODEL = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.0-flash")

SUMMARIZE_PROMPT = """You are a scientific paper analyst. Summarize this paper abstract into a structured format.

<paper>
Title: {title}
Abstract: {abstract}
Categories: {categories}
</paper>

Extract the following information. Be concise but precise.

Respond with valid JSON:
```json
{{
  "arxiv_id": "{arxiv_id}",
  "problem": "What problem or question does this paper address? (1-2 sentences)",
  "method": "What method, approach, or technique is used? (1-2 sentences)",
  "key_result": "What is the main finding or contribution? (1-2 sentences)",
  "limitations": "What limitations are mentioned or implied? (1 sentence, or empty string if none)",
  "future_work": "What future work is suggested? (1 sentence, or empty string if none)",
  "extracted_entities": ["list", "of", "key", "entities", "methods", "datasets", "metrics"]
}}
```

Focus on extracting factual information. The extracted_entities should include:
- Method/technique names (e.g., "CRISPR-Cas9", "transformer", "gradient descent")
- Dataset names if mentioned
- Metrics or evaluation methods
- Biological targets, chemicals, or other domain-specific entities
"""


class PaperSummarizer:
    """Summarizes papers using LLM API."""

    def __init__(self, model: str = FLASH_MODEL):
        self.model = model
        from src.soul.llm_client import LLMClient
        self.client = LLMClient(model=model)

    async def summarize(self, paper: ArxivPaper) -> PaperSummary | None:
        """Summarize a single paper.

        Args:
            paper: ArxivPaper to summarize

        Returns:
            PaperSummary or None if summarization fails
        """
        prompt = SUMMARIZE_PROMPT.format(
            title=paper.title,
            abstract=paper.abstract,
            categories=", ".join(paper.categories),
            arxiv_id=paper.arxiv_id,
        )

        try:
            response_text = await self.client.generate_content(prompt)
            if not response_text:
                return None

            json_str = extract_json_from_response(response_text)
            if not json_str:
                return None

            import json
            data = json.loads(json_str)

            return PaperSummary(
                arxiv_id=paper.arxiv_id,
                problem=data.get("problem", ""),
                method=data.get("method", ""),
                key_result=data.get("key_result", ""),
                limitations=data.get("limitations", ""),
                future_work=data.get("future_work", ""),
                extracted_entities=data.get("extracted_entities", []),
            )
        except Exception:
            return None

    async def summarize_batch(
        self,
        papers: list[ArxivPaper],
        max_concurrent: int = 5,
    ) -> list[PaperSummary]:
        """Summarize multiple papers with concurrency control.

        Args:
            papers: List of papers to summarize
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of successful summaries
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def summarize_with_semaphore(paper: ArxivPaper) -> PaperSummary | None:
            async with semaphore:
                return await self.summarize(paper)

        tasks = [summarize_with_semaphore(paper) for paper in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        summaries: list[PaperSummary] = []
        for result in results:
            if isinstance(result, PaperSummary):
                summaries.append(result)

        return summaries


# =============================================================================
# Fallback summarizer (no LLM, just extraction)
# =============================================================================


def extract_entities_simple(abstract: str) -> list[str]:
    """Simple entity extraction without LLM.

    Uses basic heuristics to extract potential entities.
    """
    import re

    entities: list[str] = []

    # Common patterns for methods/techniques
    method_patterns = [
        r"\b([A-Z][a-z]+(?:-[A-Z]?[a-z]+)+)\b",  # CamelCase-with-hyphens (e.g., CRISPR-Cas9)
        r"\b([A-Z]{2,}(?:-[A-Z0-9]+)?)\b",  # Acronyms (e.g., GAN, BERT, GPT-4)
        r"\b(\w+(?:Net|GAN|BERT|GPT|LLM|CNN|RNN))\b",  # Neural network names
    ]

    for pattern in method_patterns:
        matches = re.findall(pattern, abstract)
        entities.extend(matches)

    # Remove common false positives
    stopwords = {"The", "This", "That", "These", "Our", "We", "In", "On", "For"}
    entities = [e for e in entities if e not in stopwords and len(e) > 2]

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for e in entities:
        if e.lower() not in seen:
            seen.add(e.lower())
            unique.append(e)

    return unique[:15]  # Limit to 15 entities


def create_simple_summary(paper: ArxivPaper) -> PaperSummary:
    """Create a simple summary without LLM.

    Useful as a fallback when the API is unavailable.
    """
    # Extract first sentence for problem (if contains question words or problem indicators)
    sentences = paper.abstract.split(". ")
    problem = sentences[0] if sentences else paper.abstract[:200]

    # Try to find method sentence
    method_keywords = ["propose", "present", "introduce", "develop", "use", "apply"]
    method = ""
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in method_keywords):
            method = sentence
            break

    # Try to find result sentence
    result_keywords = ["result", "show", "demonstrate", "achieve", "outperform", "improve"]
    key_result = ""
    for sentence in sentences:
        if any(kw in sentence.lower() for kw in result_keywords):
            key_result = sentence
            break

    return PaperSummary(
        arxiv_id=paper.arxiv_id,
        problem=problem,
        method=method,
        key_result=key_result,
        limitations="",
        future_work="",
        extracted_entities=extract_entities_simple(paper.abstract),
    )
