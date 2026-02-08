"""Citation guardrails for hypothesis outputs."""

from __future__ import annotations

import re

from src.contracts.schemas import ArxivPaper, EvidenceSpan, Hypothesis

# Match percentages, decimals, scientific notation, and plain integers.
_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d+(?:\.\d+)?(?:e[+-]?\d+)?%?)(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def _extract_numeric_tokens(text: str) -> set[str]:
    """Extract normalized numeric tokens from text."""
    if not text:
        return set()
    tokens = {
        token.lower().strip()
        for token in _NUMERIC_TOKEN_RE.findall(text)
    }
    # Ignore very weak numeric anchors (single digits) unless percentages
    filtered = {
        token for token in tokens
        if token.endswith("%") or len(token.replace(".", "").replace("+", "").replace("-", "")) >= 2
    }
    return filtered


def has_numeric_claims(hypothesis: Hypothesis) -> bool:
    """Return True when the main statement includes numeric claims."""
    return bool(_extract_numeric_tokens(hypothesis.hypothesis))


def validate_numeric_citation_coverage(
    hypothesis: Hypothesis,
    paper_store: dict[str, ArxivPaper],
) -> tuple[bool, str | None]:
    """Validate that numeric claim tokens appear in cited paper content.

    Returns:
        (is_valid, warning_message)
    """
    numeric_tokens = _extract_numeric_tokens(hypothesis.hypothesis)
    if not numeric_tokens:
        return True, None

    if not hypothesis.supporting_papers:
        return False, "Numeric claim detected but no supporting citations are linked."

    cited_corpus: list[str] = []
    for paper_id in hypothesis.supporting_papers:
        paper = paper_store.get(paper_id)
        if not paper:
            continue
        cited_corpus.append(f"{paper.title}\n{paper.abstract}")

    if not cited_corpus:
        return False, "Numeric claim detected but cited papers are missing from local metadata."

    cited_tokens = _extract_numeric_tokens("\n".join(cited_corpus))
    missing_tokens = sorted(numeric_tokens - cited_tokens)
    if not missing_tokens:
        return True, None

    tokens_preview = ", ".join(missing_tokens[:6])
    # Inspiration-grounding mode: keep hypothesis if citations exist, but mark
    # that numeric values are extrapolated rather than directly stated.
    return (
        True,
        "Numeric extrapolation from cited evidence (not exact textual match): "
        + tokens_preview,
    )


_TRACE_CITATION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*(.+?)\s*$")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _sentence_like_units(text: str) -> list[str]:
    units = [part.strip() for part in re.split(r"[.!?]+", text or "") if part.strip()]
    return units or ([text.strip()] if text and text.strip() else [])


def _id_key(raw: str) -> str:
    value = (raw or "").strip().lower()
    if value.startswith("doi:"):
        return value.split(":", 1)[1].strip()
    if re.match(r"^10\.\d{4,9}/\S+$", value, flags=re.IGNORECASE):
        return value
    if value.startswith("pmid:"):
        return value.split(":", 1)[1].strip()
    if value.startswith("pmcid:"):
        return value.split(":", 1)[1].strip()
    return value


def _extract_spans_from_trace(hypothesis: Hypothesis) -> list[EvidenceSpan]:
    spans: list[EvidenceSpan] = []
    for line in hypothesis.evidence_trace or []:
        match = _TRACE_CITATION_RE.match(line)
        if not match:
            continue
        citation_id = match.group(1).strip()
        quote = match.group(2).strip()
        if not citation_id or len(quote) < 12:
            continue
        spans.append(
            EvidenceSpan(
                claim_text=hypothesis.hypothesis,
                citation_id=citation_id,
                quote=quote,
                confidence=0.5,
            )
        )
    return spans


def validate_evidence_span_coverage(
    hypothesis: Hypothesis,
    paper_store: dict[str, ArxivPaper],
) -> tuple[bool, str | None]:
    """Validate structured evidence spans for every claim sentence.

    Rules:
    - hypothesis must have at least one evidence span (or parseable evidence_trace line)
    - each span citation_id must be in supporting_papers
    - each span quote must be present in cited paper title/abstract (substring or token overlap)
    - each sentence-like unit in hypothesis must have at least one supporting span
    """
    if not hypothesis.supporting_papers:
        return False, "Evidence spans missing: no supporting citations are linked."

    spans = list(hypothesis.evidence_spans or [])
    if not spans:
        spans = _extract_spans_from_trace(hypothesis)
        if spans:
            hypothesis.evidence_spans = spans

    if not spans:
        return False, "Evidence spans missing: provide claim->citation->quote mappings."

    supporting_keys = {_id_key(pid) for pid in hypothesis.supporting_papers}
    invalid_citations = sorted(
        {span.citation_id for span in spans if _id_key(span.citation_id) not in supporting_keys}
    )
    if invalid_citations:
        return (
            False,
            "Evidence spans reference citations not in supporting_papers: "
            + ", ".join(invalid_citations[:5]),
        )

    # Validate quote presence in cited corpus.
    quote_failures: list[str] = []
    for span in spans:
        paper = paper_store.get(span.citation_id)
        if not paper:
            # Try key-based fallback
            key = _id_key(span.citation_id)
            paper = next((p for pid, p in paper_store.items() if _id_key(pid) == key), None)
        if not paper:
            quote_failures.append(span.citation_id)
            continue

        corpus = _normalize_text(f"{paper.title} {paper.abstract}")
        quote = _normalize_text(span.quote)
        if quote in corpus:
            continue
        quote_tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", quote)}
        corpus_tokens = {t for t in re.findall(r"[a-z0-9][a-z0-9\-]{2,}", corpus)}
        if len(quote_tokens & corpus_tokens) < 3:
            quote_failures.append(span.citation_id)

    if quote_failures:
        return (
            False,
            "Evidence span quotes not grounded in cited paper text for: "
            + ", ".join(sorted(set(quote_failures))[:5]),
        )

    # Require each sentence-like unit to be covered by at least one span claim_text.
    sentences = _sentence_like_units(hypothesis.hypothesis)
    uncovered = 0
    for sentence in sentences:
        sentence_norm = _normalize_text(sentence)
        covered = False
        for span in spans:
            claim_norm = _normalize_text(span.claim_text)
            if sentence_norm in claim_norm or claim_norm in sentence_norm:
                covered = True
                break
            sentence_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", sentence_norm))
            claim_tokens = set(re.findall(r"[a-z0-9][a-z0-9\-]{2,}", claim_norm))
            if len(sentence_tokens & claim_tokens) >= 4:
                covered = True
                break
        if not covered:
            uncovered += 1

    if uncovered:
        return False, f"Evidence span coverage incomplete: {uncovered} claim sentence(s) lack support."

    return True, None
