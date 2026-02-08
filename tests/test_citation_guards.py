from src.contracts.schemas import ArxivPaper, Hypothesis
from src.verify.citation_guards import (
    has_numeric_claims,
    validate_evidence_span_coverage,
    validate_numeric_citation_coverage,
)


def _make_hypothesis(
    statement: str,
    papers: list[str] | None = None,
    evidence_trace: list[str] | None = None,
    evidence_spans: list[dict] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id="h1",
        hypothesis=statement,
        rationale="Rationale",
        cross_disciplinary_connection="A-B",
        experimental_design=["Step 1"],
        expected_impact="Impact",
        novelty_keywords=["battery"],
        supporting_papers=papers or [],
        evidence_trace=evidence_trace or [],
        evidence_spans=evidence_spans or [],
    )


def test_has_numeric_claims_detects_numbers() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 25% higher power density at 100 mA/cm^2."
    )
    assert has_numeric_claims(h) is True


def test_numeric_claim_without_citation_fails() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 25% higher power density at 100 mA/cm^2."
    )
    ok, warning = validate_numeric_citation_coverage(h, {})
    assert ok is False
    assert warning is not None


def test_numeric_claim_with_matching_citation_passes() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 25% higher power density at 100 mA/cm^2.",
        papers=["2401.12345"],
    )
    paper_store = {
        "2401.12345": ArxivPaper(
            arxiv_id="2401.12345",
            title="Battery study",
            abstract="We report 25% gain under 100 mA/cm^2 conditions.",
            authors=[],
            categories=[],
        )
    }
    ok, warning = validate_numeric_citation_coverage(h, paper_store)
    assert ok is True
    assert warning is None


def test_numeric_claim_missing_tokens_becomes_extrapolation_warning() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 1500 Wh/kg and 5000 W/kg.",
        papers=["2401.12345"],
    )
    paper_store = {
        "2401.12345": ArxivPaper(
            arxiv_id="2401.12345",
            title="Battery study",
            abstract="We report 1500 Wh/kg in a prototype system.",
            authors=[],
            categories=[],
        )
    }
    ok, warning = validate_numeric_citation_coverage(h, paper_store)
    assert ok is True
    assert warning is not None
    assert "Numeric extrapolation" in warning
    assert "5000" in warning


def test_evidence_span_coverage_accepts_trace_backfill() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 25% higher power density at 100 mA/cm^2.",
        papers=["2401.12345"],
        evidence_trace=["[2401.12345] We report 25% gain under 100 mA/cm^2 conditions."],
    )
    paper_store = {
        "2401.12345": ArxivPaper(
            arxiv_id="2401.12345",
            title="Battery study",
            abstract="We report 25% gain under 100 mA/cm^2 conditions.",
            authors=[],
            categories=[],
        )
    }
    ok, warning = validate_evidence_span_coverage(h, paper_store)
    assert ok is True
    assert warning is None
    assert h.evidence_spans


def test_evidence_span_coverage_rejects_missing_spans() -> None:
    h = _make_hypothesis(
        "Cell design can deliver 25% higher power density at 100 mA/cm^2.",
        papers=["2401.12345"],
    )
    paper_store = {
        "2401.12345": ArxivPaper(
            arxiv_id="2401.12345",
            title="Battery study",
            abstract="We report 25% gain under 100 mA/cm^2 conditions.",
            authors=[],
            categories=[],
        )
    }
    ok, warning = validate_evidence_span_coverage(h, paper_store)
    assert ok is False
    assert warning is not None
    assert warning.startswith("Evidence spans missing")
