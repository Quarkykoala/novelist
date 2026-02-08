import pytest

from src.contracts.schemas import ArxivPaper, Hypothesis, RalphConfig
from src.kb.arxiv_client import normalize_arxiv_id
from src.ralph.orchestrator import RalphOrchestrator
from src.soul.prompts.synthesizer import SynthesizerSoul


def _make_hypothesis(
    hypothesis_id: str,
    statement: str,
    supporting_papers: list[str] | None = None,
    evidence_trace: list[str] | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        hypothesis=statement,
        rationale="Rationale",
        cross_disciplinary_connection="A-B",
        experimental_design=["Step 1"],
        expected_impact="Impact",
        novelty_keywords=["electrode", "electrolyte"],
        supporting_papers=supporting_papers or [],
        evidence_trace=evidence_trace or [],
    )


def test_synthesizer_preserves_citations_from_source_ids() -> None:
    source_hypotheses = [
        _make_hypothesis(
            "h1",
            "First source hypothesis.",
            supporting_papers=["2401.11111"],
            evidence_trace=["[2401.11111] electrode conductivity improves under stress."],
        ),
        _make_hypothesis(
            "h2",
            "Second source hypothesis.",
            supporting_papers=["2402.22222"],
            evidence_trace=["[2402.22222] redox kinetics are improved by MXene interfaces."],
        ),
    ]

    response_text = """
    [
      {
        "hypothesis": "Merged grounded hypothesis.",
        "rationale": "Combines source mechanisms.",
        "cross_disciplinary_connection": "materials + electrochemistry",
        "experimental_design": ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5"],
        "expected_impact": "Higher power density",
        "novelty_keywords": ["mxene", "redox"],
        "source_ids": ["h1", "h2"]
      }
    ]
    """

    soul = SynthesizerSoul(model="gemini/gemini-3-flash")
    final_hypotheses = soul._parse_response(response_text, 1, source_hypotheses)

    assert len(final_hypotheses) == 1
    final = final_hypotheses[0]
    assert "2401.11111" in final.supporting_papers
    assert "2402.22222" in final.supporting_papers
    assert any("2401.11111" in line for line in final.evidence_trace)
    assert any("2402.22222" in line for line in final.evidence_trace)


class _DummySemanticSearch:
    is_initialized = True

    async def search_similar(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return []


@pytest.mark.asyncio
async def test_backfill_uses_lexical_overlap_when_semantic_search_is_empty() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig())
    paper = ArxivPaper(
        arxiv_id="2501.12345",
        title="MXene Ti3C2Tx electrodes for vanadium redox systems",
        abstract="We report improved charge transfer and power density in hydrogen battery settings.",
        authors=[],
        categories=[],
    )
    orchestrator.paper_store = {paper.arxiv_id: paper}
    orchestrator.pulled_paper_bases = {
        normalize_arxiv_id(paper.arxiv_id, keep_version=False) or paper.arxiv_id
    }
    orchestrator._semantic_search = _DummySemanticSearch()  # type: ignore[attr-defined]

    hypothesis = _make_hypothesis(
        "h3",
        "MXene Ti3C2Tx with vanadium redox electrolyte increases power density.",
    )

    await orchestrator._backfill_citations([hypothesis], top_k=2, min_score=0.2)

    assert hypothesis.supporting_papers == ["2501.12345"]
    assert hypothesis.grounding_status == "grounded"
    assert any("lexical overlap" in warning for warning in hypothesis.citation_warnings)


def test_citation_policy_restricts_to_pulled_corpus() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig(strict_grounding=False))
    orchestrator.pulled_paper_bases = {"2401.11111"}
    hypothesis = _make_hypothesis(
        "h4",
        "A grounded statement without numeric claims.",
        supporting_papers=["9999.99999"],
    )

    filtered = orchestrator._apply_citation_policies([hypothesis])

    assert len(filtered) == 1
    assert filtered[0].supporting_papers == []
    assert filtered[0].grounding_status == "ungrounded"
    assert any(
        warning.startswith("Removed citations not present in pulled corpus")
        for warning in filtered[0].citation_warnings
    )


def test_citation_policy_keeps_pulled_pubmed_ids() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig(strict_grounding=True))
    paper = ArxivPaper(
        arxiv_id="PMID:123456",
        title="Electrolyte interfaces in hydrogen systems",
        abstract="PubMed evidence text",
        authors=[],
        categories=[],
        source="pubmed",
    )
    orchestrator.paper_store = {paper.arxiv_id: paper}
    orchestrator.pulled_paper_keys = {orchestrator._paper_id_key(paper.arxiv_id)}

    hypothesis = _make_hypothesis(
        "h5",
        "A grounded statement without numeric claims.",
        supporting_papers=["pmid:123456"],
        evidence_trace=["[PMID:123456] PubMed evidence text"],
    )

    filtered = orchestrator._apply_citation_policies([hypothesis])

    assert len(filtered) == 1
    assert filtered[0].supporting_papers == ["PMID:123456"]
    assert filtered[0].grounding_status == "grounded"


def test_citation_policy_keeps_pulled_doi_ids() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig(strict_grounding=True))
    paper = ArxivPaper(
        arxiv_id="DOI:10.1038/s41586-023-00000-0",
        title="Cross-domain electrochemistry",
        abstract="Crossref/OpenAlex evidence",
        authors=[],
        categories=[],
        source="crossref",
    )
    orchestrator.paper_store = {paper.arxiv_id: paper}
    orchestrator.pulled_paper_keys = {orchestrator._paper_id_key(paper.arxiv_id)}

    hypothesis = _make_hypothesis(
        "h6",
        "A grounded statement without numeric claims.",
        supporting_papers=["10.1038/s41586-023-00000-0"],
        evidence_trace=["[10.1038/s41586-023-00000-0] Crossref/OpenAlex evidence"],
    )

    filtered = orchestrator._apply_citation_policies([hypothesis])

    assert len(filtered) == 1
    assert filtered[0].supporting_papers == ["DOI:10.1038/s41586-023-00000-0"]
    assert filtered[0].grounding_status == "grounded"


def test_paper_id_key_normalizes_openalex_and_pmcid() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig())
    assert orchestrator._paper_id_key("OPENALEX:W12345") == "openalex:w12345"
    assert orchestrator._paper_id_key("PMC12345") == "pmcid:pmc12345"


def test_strict_grounding_drops_when_evidence_span_missing() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig(strict_grounding=True))
    paper = ArxivPaper(
        arxiv_id="2401.11111",
        title="Grounding paper",
        abstract="Contains relevant context.",
        authors=[],
        categories=[],
    )
    orchestrator.paper_store = {paper.arxiv_id: paper}
    orchestrator.pulled_paper_bases = {
        normalize_arxiv_id(paper.arxiv_id, keep_version=False) or paper.arxiv_id
    }
    hypothesis = _make_hypothesis(
        "h7",
        "A grounded statement without numeric claims.",
        supporting_papers=["2401.11111"],
        evidence_trace=[],
    )

    filtered = orchestrator._apply_citation_policies([hypothesis])
    assert filtered == []


def test_citation_policy_populates_provenance_split_fields() -> None:
    orchestrator = RalphOrchestrator(config=RalphConfig(strict_grounding=True))
    paper = ArxivPaper(
        arxiv_id="PMID:123456",
        title="Electrolyte interfaces in hydrogen systems",
        abstract="PubMed evidence text supports this claim.",
        authors=[],
        categories=[],
        source="pubmed",
    )
    orchestrator.paper_store = {paper.arxiv_id: paper}
    orchestrator.pulled_paper_keys = {orchestrator._paper_id_key(paper.arxiv_id)}
    hypothesis = _make_hypothesis(
        "h8",
        "A grounded statement without numeric claims.",
        supporting_papers=["PMID:123456"],
        evidence_trace=["[PMID:123456] PubMed evidence text supports this claim."],
    )

    filtered = orchestrator._apply_citation_policies([hypothesis])
    assert len(filtered) == 1
    assert filtered[0].supported_facts
    assert filtered[0].novel_inference == filtered[0].hypothesis
