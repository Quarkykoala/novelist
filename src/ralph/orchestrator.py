"""Ralph Orchestrator - main loop controller for hypothesis generation sessions.

Implements the Ralph loop pattern:
- Iterative refinement until convergence
- Stop conditions (max iterations, cost, time, stagnation)
- Checkpointing and resume capability
"""

import asyncio
import inspect
import os
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from dotenv import load_dotenv

from src.contracts.schemas import (
    BDIState,
    ArxivPaper,
    ConceptMap,
    ExtractedClaim,
    GapType,
    GenerationMode,
    GroundedHypothesis,
    Hypothesis,
    IdentifiedGap,
    IterationTrace,
    RalphConfig,
    ScoreBlock,
    SimulationResult,
    SessionConstraints,
    SessionPhase,
    SessionResult,
    SoulRole,
)
from src.kb.arxiv_client import (
    ArxivClient,
    detect_categories_from_query,
    is_arxiv_id,
    normalize_arxiv_id,
)
from src.kb.citation_validator import validate_supporting_papers
from src.kb.claim_extractor import ClaimExtractor
from src.kb.concept_map import ConceptMapBuilder
from src.kb.crossref_client import CrossrefClient
from src.kb.gap_analyzer import GapAnalyzer
from src.kb.grounded_generator import GroundedHypothesisGenerator
from src.kb.openalex_client import OpenAlexClient
from src.kb.pubmed_client import PubMedClient
from src.kb.semantic_scholar_client import SemanticScholarClient
from src.kb.unpaywall_client import UnpaywallClient
from src.ralph.tree import ResearchState
from src.ralph.tree_search_orchestrator import TreeSearchOrchestrator
from src.soul.simulator import Simulator
from src.kb.paper_summarizer import PaperSummarizer
from src.soul.bdi import BDIAgent
from src.soul.collective import SoulCollective
from src.soul.memory import MemorySystem
from src.soul.prompts.visualizer import VisualizerSoul
from src.soul.prompts.persona_forge import PersonaForge
from src.soul.srsh_orchestrator import SRSHConfig, SRSHOrchestrator
from src.verify.novelty_arxiv import batch_verify_novelty
from src.verify.citation_guards import (
    validate_evidence_span_coverage,
    validate_numeric_citation_coverage,
)
from src.verify.scoring import ScoringService, heuristic_feasibility, heuristic_impact
from src.kb.semantic_search import SemanticPaperSearch

load_dotenv()


class RalphOrchestrator:
    """Main orchestrator for hypothesis generation sessions."""

    def __init__(
        self, 
        config: RalphConfig | None = None,
        callbacks: dict[str, Any] | None = None,
    ):
        self.config = config or RalphConfig()
        self.callbacks = callbacks or {}

        # Initialize components
        self.agent = BDIAgent()
        self.memory = MemorySystem()
        self.collective = SoulCollective(
            flash_model=self.config.flash_model,
            pro_model=self.config.pro_model,
        )
        self.scorer = ScoringService(model=self.config.flash_model)
        self.visualizer = VisualizerSoul(model=self.config.flash_model)
        self.persona_forge = PersonaForge(model=self.config.flash_model)

        # New literature-first pipeline components
        self.claim_extractor = ClaimExtractor(model=self.config.pro_model)
        self.gap_analyzer = GapAnalyzer(model=self.config.pro_model)
        self.grounded_generator = GroundedHypothesisGenerator(model=self.config.pro_model)
        self.simulator = Simulator(model=self.config.pro_model)
        self.srsh = SRSHOrchestrator(
            generator=self.grounded_generator,
            config=SRSHConfig(
                enabled=self.config.srsh_enabled,
                n_agents=self.config.srsh_agents,
                iterations_per_agent=self.config.srsh_iterations_per_agent,
                n_collisions=self.config.srsh_collisions,
            ),
        )
        
        # MCTS Orchestrator
        self.tree_search = TreeSearchOrchestrator(
            config=self.config,
            collective=self.collective,
            generator=self.grounded_generator,
            scorer=self.scorer,
            simulator=self.simulator
        )

        # Session state
        self.session_id = ""
        self.topic = ""
        self.start_time: datetime | None = None
        self.iteration = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.hypotheses: list[Hypothesis] = []
        self.grounded_hypotheses: list[GroundedHypothesis] = []
        self.concept_map: ConceptMap | None = None
        self.claims: list[ExtractedClaim] = []
        self.gaps: list[IdentifiedGap] = []
        
        # Knowledge Store
        self.paper_store: dict[str, ArxivPaper] = {}
        self.pulled_paper_bases: set[str] = set()
        self.pulled_paper_keys: set[str] = set()
        self.knowledge_sources: list[str] = []
        
        # Semantic Search (lazy initialization for performance)
        self._semantic_search: SemanticPaperSearch | None = None

        # User Interaction
        self.user_message_queue: asyncio.Queue[str] = asyncio.Queue()
        self.constraints: SessionConstraints | None = None
        self.current_phase: SessionPhase = SessionPhase.QUEUED
        self.persona_roster: list[dict[str, Any]] = []

    async def inject_user_message(self, message: str) -> None:
        """Receive a message from the user."""
        await self.user_message_queue.put(message)
        await self._emit_status(self.current_phase, "User message received.")
    
    async def _emit_status(self, phase: SessionPhase, detail: str | None = None) -> None:
        """Emit a status update via callback."""
        self.current_phase = phase
        payload = {
            "phase": phase.value if isinstance(phase, SessionPhase) else str(phase),
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        }

        if "on_status_change" in self.callbacks:
            try:
                if inspect.iscoroutinefunction(self.callbacks["on_status_change"]):
                    await self.callbacks["on_status_change"](payload)
                else:
                    self.callbacks["on_status_change"](payload)
            except Exception as e:
                print(f"[WARN] Error in status callback: {e}")

    async def _emit_personas(self, personas: list[dict[str, Any]]) -> None:
        """Emit persona roster updates via callback."""
        if "on_personas" in self.callbacks:
            try:
                if inspect.iscoroutinefunction(self.callbacks["on_personas"]):
                    await self.callbacks["on_personas"](personas)
                else:
                    self.callbacks["on_personas"](personas)
            except Exception as e:
                print(f"[WARN] Error in personas callback: {e}")

    async def _emit_knowledge_update(self, stats: dict[str, Any]) -> None:
        """Emit knowledge base statistics updates."""
        if "on_knowledge_update" in self.callbacks:
            try:
                if inspect.iscoroutinefunction(self.callbacks["on_knowledge_update"]):
                    await self.callbacks["on_knowledge_update"](stats)
                else:
                    self.callbacks["on_knowledge_update"](stats)
            except Exception as e:
                print(f"[WARN] Error in knowledge callback: {e}")

    async def _emit_trace(self, trace: IterationTrace) -> None:
        """Emit a trace update via callback."""
        if "on_trace" in self.callbacks:
            try:
                if inspect.iscoroutinefunction(self.callbacks["on_trace"]):
                    await self.callbacks["on_trace"](trace)
                else:
                    self.callbacks["on_trace"](trace)
            except Exception as e:
                print(f"[WARN] Error in trace callback: {e}")
    
    def _add_trace(self, msg: str) -> None:
        """Add a simple thought trace (internal helper)."""
        # This is used by some components to log non-structured thoughts
        # We can map this to a partial trace emission if needed, 
        # but for now we rely on the main iteration trace loop.
        pass

    @property
    def semantic_search(self) -> SemanticPaperSearch:
        """Get or create the semantic search instance (lazy initialization)."""
        if self._semantic_search is None:
            self._semantic_search = SemanticPaperSearch()
        return self._semantic_search

    @staticmethod
    def _paper_id_key(paper_id: str) -> str:
        """Build a canonical key for mixed-source paper identifiers."""
        raw_id = (paper_id or "").strip()
        if not raw_id:
            return ""

        lowered = raw_id.lower()
        if lowered.startswith("doi:"):
            return f"doi:{lowered.split(':', 1)[1].strip()}"
        if lowered.startswith("pmid:"):
            return f"pmid:{lowered.split(':', 1)[1].strip()}"
        if lowered.startswith("pmcid:"):
            return f"pmcid:{lowered.split(':', 1)[1].strip()}"
        if lowered.startswith("s2:"):
            return lowered
        if lowered.startswith("openalex:"):
            return lowered
        if re.match(r"^10\.\d{4,9}/\S+$", raw_id, flags=re.IGNORECASE):
            return f"doi:{lowered}"
        if re.match(r"^pmc\d+$", lowered):
            return f"pmcid:{lowered}"
        if is_arxiv_id(raw_id):
            base = normalize_arxiv_id(raw_id, keep_version=False)
            if base:
                return f"arxiv:{base.lower()}"
        return lowered

    def _build_paper_id_index(self) -> dict[str, str]:
        """Map canonical paper ID keys to stored IDs."""
        index: dict[str, str] = {}
        for stored_id in self.paper_store.keys():
            key = self._paper_id_key(stored_id)
            if key and key not in index:
                index[key] = stored_id
        return index

    def _resolve_paper_id(self, paper_id: str, index: dict[str, str] | None = None) -> str | None:
        """Resolve a possibly variant ID to a stored paper ID."""
        raw_id = (paper_id or "").strip()
        if not raw_id:
            return None

        if raw_id in self.paper_store:
            return raw_id

        lookup = index or self._build_paper_id_index()
        key = self._paper_id_key(raw_id)
        if key and key in lookup:
            return lookup[key]

        # arXiv version/base fallback
        if is_arxiv_id(raw_id):
            raw_base = normalize_arxiv_id(raw_id, keep_version=False)
            if raw_base:
                for stored in self.paper_store.keys():
                    if normalize_arxiv_id(stored, keep_version=False) == raw_base:
                        return stored
        return None

    def _score_paper_relevance(self, paper: ArxivPaper, keywords: list[str]) -> float:
        """Heuristic score used to rank mixed-source retrieval results."""
        title = (paper.title or "").lower()
        abstract = (paper.abstract or "").lower()
        score = 0.0
        for token in keywords:
            if token in title:
                score += 3.0
            if token in abstract:
                score += 1.0

        source_boost = {
            "arxiv": 1.0,
            "pubmed": 0.9,
            "openalex": 0.9,
            "semantic_scholar": 0.8,
            "crossref": 0.75,
        }
        score += source_boost.get((paper.source or "").lower(), 0.5)
        if paper.citation_count:
            score += min(1.5, paper.citation_count / 500.0)
        if paper.abstract:
            score += 0.2
        return score

    def _merge_retrieved_papers(
        self,
        *,
        arxiv_papers: list[ArxivPaper],
        pubmed_papers: list[ArxivPaper],
        semantic_papers: list[ArxivPaper],
        openalex_papers: list[ArxivPaper],
        crossref_papers: list[ArxivPaper],
        keywords: list[str],
        paper_limit: int,
    ) -> tuple[list[ArxivPaper], dict[str, int]]:
        """Merge, deduplicate, and rank papers from all free sources."""
        source_counts = {
            "arxiv": len(arxiv_papers),
            "pubmed": len(pubmed_papers),
            "semantic_scholar": len(semantic_papers),
            "openalex": len(openalex_papers),
            "crossref": len(crossref_papers),
        }
        all_papers = (
            arxiv_papers
            + pubmed_papers
            + semantic_papers
            + openalex_papers
            + crossref_papers
        )
        if not all_papers:
            return [], source_counts

        deduped: dict[str, ArxivPaper] = {}
        title_index: dict[str, str] = {}

        def _title_key(title: str) -> str:
            return re.sub(r"\s+", " ", (title or "").strip().lower())

        for paper in all_papers:
            key = self._paper_id_key(paper.arxiv_id) or paper.arxiv_id
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = paper
            else:
                # Prefer richer record (longer abstract / has citation count).
                current_quality = len(existing.abstract or "") + (existing.citation_count or 0)
                candidate_quality = len(paper.abstract or "") + (paper.citation_count or 0)
                if candidate_quality > current_quality:
                    deduped[key] = paper

            tkey = _title_key(paper.title)
            if tkey:
                existing_key = title_index.get(tkey)
                if existing_key and existing_key in deduped:
                    existing = deduped[existing_key]
                    current_quality = len(existing.abstract or "") + (existing.citation_count or 0)
                    candidate_quality = len(paper.abstract or "") + (paper.citation_count or 0)
                    if candidate_quality > current_quality:
                        deduped[existing_key] = paper
                    if key != existing_key and key in deduped:
                        deduped.pop(key, None)
                else:
                    title_index[tkey] = key

        papers = list(deduped.values())
        papers.sort(
            key=lambda p: self._score_paper_relevance(p, keywords),
            reverse=True,
        )
        return papers[:paper_limit], source_counts

    @staticmethod
    def _extract_doi(paper_id: str) -> str:
        """Extract DOI from mixed identifier strings."""
        raw_id = (paper_id or "").strip()
        if not raw_id:
            return ""
        lowered = raw_id.lower()
        if lowered.startswith("doi:"):
            return raw_id.split(":", 1)[1].strip()
        if re.match(r"^10\.\d{4,9}/\S+$", raw_id, flags=re.IGNORECASE):
            return raw_id
        return ""

    async def _enrich_with_unpaywall(self, papers: list[ArxivPaper], max_items: int = 20) -> None:
        """Best-effort OA enrichment for DOI-based papers."""
        if not papers:
            return
        doi_indices: list[tuple[int, str]] = []
        for idx, paper in enumerate(papers):
            doi = self._extract_doi(paper.arxiv_id)
            if not doi:
                continue
            doi_indices.append((idx, doi))
            if len(doi_indices) >= max_items:
                break

        if not doi_indices:
            return

        async with UnpaywallClient() as unpaywall:
            if not unpaywall.enabled:
                return
            for idx, doi in doi_indices:
                try:
                    info = await unpaywall.lookup(doi)
                except Exception as e:
                    print(f"[WARN] Unpaywall lookup failed for {doi}: {e}")
                    continue
                if not info:
                    continue
                if info.get("abs_url"):
                    papers[idx].abs_url = info["abs_url"]
                if info.get("pdf_url"):
                    papers[idx].pdf_url = info["pdf_url"]
                pmcid = info.get("pmcid")
                if pmcid and pmcid not in papers[idx].categories:
                    papers[idx].categories = (papers[idx].categories or []) + [pmcid]

    async def _retrieve_cross_domain_papers(
        self,
        *,
        query: str,
        categories: list[str],
        keywords: list[str],
        paper_limit: int,
    ) -> tuple[list[ArxivPaper], dict[str, int]]:
        """Retrieve papers from free cross-domain sources with fallback behavior."""
        arxiv_target = max(8, paper_limit)
        pubmed_target = max(6, paper_limit // 2)
        semantic_target = max(6, paper_limit // 2)
        openalex_target = max(6, paper_limit // 2)
        crossref_target = max(6, paper_limit // 2)

        arxiv_papers: list[ArxivPaper] = []
        pubmed_papers: list[ArxivPaper] = []
        semantic_papers: list[ArxivPaper] = []
        openalex_papers: list[ArxivPaper] = []
        crossref_papers: list[ArxivPaper] = []

        async with ArxivClient() as arxiv_client:
            seed_limit = min(8, arxiv_target)
            seed_papers = await arxiv_client.search(query, max_results=seed_limit)
            arxiv_papers.extend(seed_papers)

            inferred_categories = list(categories)
            if seed_papers:
                category_counts: Counter[str] = Counter()
                for paper in seed_papers:
                    for cat in paper.categories:
                        category_counts[cat] += 1
                for cat, _ in category_counts.most_common(3):
                    if cat not in inferred_categories:
                        inferred_categories.append(cat)

            remaining = max(0, arxiv_target - len(arxiv_papers))
            if inferred_categories and remaining:
                per_cat = max(4, remaining // len(inferred_categories))
                for cat in inferred_categories:
                    if len(arxiv_papers) >= arxiv_target:
                        break
                    extra = await arxiv_client.search_by_category(
                        cat,
                        query=query,
                        max_results=min(per_cat, arxiv_target - len(arxiv_papers)),
                    )
                    arxiv_papers.extend(extra)
            if len(arxiv_papers) < arxiv_target:
                arxiv_papers.extend(
                    await arxiv_client.search(
                        query,
                        max_results=arxiv_target - len(arxiv_papers),
                    )
                )

        pubmed_query = query
        if keywords:
            pubmed_query = " AND ".join(keywords[:6])

        async def _fetch_pubmed() -> list[ArxivPaper]:
            async with PubMedClient() as pubmed_client:
                return await pubmed_client.search(pubmed_query, max_results=pubmed_target)

        async def _fetch_semantic() -> list[ArxivPaper]:
            async with SemanticScholarClient() as semantic_client:
                return await semantic_client.search(query, max_results=semantic_target)

        async def _fetch_openalex() -> list[ArxivPaper]:
            async with OpenAlexClient() as openalex_client:
                return await openalex_client.search(query, max_results=openalex_target)

        async def _fetch_crossref() -> list[ArxivPaper]:
            async with CrossrefClient() as crossref_client:
                return await crossref_client.search(query, max_results=crossref_target)

        results = await asyncio.gather(
            _fetch_pubmed(),
            _fetch_semantic(),
            _fetch_openalex(),
            _fetch_crossref(),
            return_exceptions=True,
        )
        pubmed_result, semantic_result, openalex_result, crossref_result = results

        if isinstance(pubmed_result, Exception):
            print(f"[WARN] PubMed retrieval failed: {pubmed_result}")
        else:
            pubmed_papers = pubmed_result

        if isinstance(semantic_result, Exception):
            print(f"[WARN] Semantic Scholar retrieval failed: {semantic_result}")
        else:
            semantic_papers = semantic_result

        if isinstance(openalex_result, Exception):
            print(f"[WARN] OpenAlex retrieval failed: {openalex_result}")
        else:
            openalex_papers = openalex_result

        if isinstance(crossref_result, Exception):
            print(f"[WARN] Crossref retrieval failed: {crossref_result}")
        else:
            crossref_papers = crossref_result

        merged, source_counts = self._merge_retrieved_papers(
            arxiv_papers=arxiv_papers,
            pubmed_papers=pubmed_papers,
            semantic_papers=semantic_papers,
            openalex_papers=openalex_papers,
            crossref_papers=crossref_papers,
            keywords=keywords,
            paper_limit=paper_limit,
        )
        await self._enrich_with_unpaywall(merged, max_items=min(20, paper_limit))
        return merged, source_counts

    def _update_costs(self) -> None:
        """Aggregate costs from all components."""
        total_cost = 0.0
        total_tokens = 0
        
        # Collect from Collective (Souls)
        for soul in [self.collective.creative, self.collective.risk_taker, 
                     self.collective.skeptic, self.collective.methodical, 
                     self.collective.synthesizer]:
            total_cost += soul.total_cost
            total_tokens += soul.total_tokens

        # Collect from other services
        services = [
            self.scorer, 
            self.claim_extractor, 
            self.visualizer,
            self.persona_forge,
            # self.gap_analyzer, # Check if GapAnalyzer tracks usage
            # self.grounded_generator, # Check if Grounded tracks usage
        ]
        
        # For now, simplistic sum. 
        # Ideally, components share a UsageTracker, but this works for disjoint components.
        for svc in services:
             if hasattr(svc, 'total_cost'):
                 total_cost += svc.total_cost
             if hasattr(svc, 'total_tokens'):
                 total_tokens += svc.total_tokens
        
        # Concept map cost
        # NOTE: ConceptMapBuilder is inside self.concept_map creation logic usually
        # But we create it in _ingest_papers temporarily. We need to track it there.
        # See usage in _ingest_papers
        
        self.total_cost = total_cost
        self.total_tokens = total_tokens

    async def _validate_hypothesis_citations(self, hypotheses: list[Hypothesis]) -> None:
        """Validate citations against pulled corpus and enrich metadata."""
        if not hypotheses:
            return

        id_index = self._build_paper_id_index()
        async with ArxivClient() as client:
            for h in hypotheses:
                try:
                    validated, invalid, non_arxiv = await validate_supporting_papers(
                        h.supporting_papers or [],
                        self.paper_store,
                        client,
                    )
                except httpx.HTTPError as e:
                    warning = f"arXiv validation skipped: {e}"
                    if warning not in h.citation_warnings:
                        h.citation_warnings.append(warning)
                    h.grounding_status = "mixed"
                    continue

                # arXiv validator only returns arXiv IDs as "validated"; keep non-arXiv
                # citations if they exist in our pulled paper corpus.
                id_index = self._build_paper_id_index()
                resolved_ids: list[str] = []
                for pid in validated:
                    resolved = self._resolve_paper_id(pid, id_index) or pid
                    resolved_ids.append(resolved)

                unresolved_non_arxiv: list[str] = []
                for pid in non_arxiv:
                    resolved = self._resolve_paper_id(pid, id_index)
                    if resolved:
                        resolved_ids.append(resolved)
                    else:
                        unresolved_non_arxiv.append(pid)

                h.supporting_papers = list(dict.fromkeys(resolved_ids))
                h.non_arxiv_sources = unresolved_non_arxiv

                warnings = list(h.citation_warnings)
                if invalid:
                    warnings.append(f"Removed invalid arXiv IDs: {', '.join(invalid)}")
                if unresolved_non_arxiv:
                    warnings.append(
                        "Unresolved non-arXiv citations removed: "
                        + ", ".join(unresolved_non_arxiv)
                    )
                h.citation_warnings = warnings

                if h.supporting_papers:
                    h.grounding_status = "grounded"
                else:
                    h.grounding_status = "ungrounded"

    async def _backfill_citations(
        self,
        hypotheses: list[Hypothesis],
        *,
        top_k: int = 3,
        min_score: float = 0.2,
    ) -> None:
        """Backfill citations from the local paper store when missing."""
        if not hypotheses or not self.paper_store:
            return

        semantic_available = True
        # Ensure semantic index is ready
        if not self.semantic_search.is_initialized:
            try:
                await self.semantic_search.index_papers(list(self.paper_store.values()))
            except Exception as e:
                print(f"[WARN] Semantic search index failed: {e}")
                semantic_available = False

        for h in hypotheses:
            if h.supporting_papers:
                continue

            query_parts = [h.hypothesis] + (h.novelty_keywords or [])[:5]
            query = " ".join(part for part in query_parts if part)
            if not query:
                continue

            results: list[tuple[ArxivPaper, float]] = []
            if semantic_available:
                try:
                    results = await self.semantic_search.search_similar(
                        query, top_k=top_k, min_score=min_score
                    )
                except Exception as e:
                    print(f"[WARN] Citation backfill failed: {e}")
                    semantic_available = False

            if not results:
                lexical_ids = self._lexical_backfill_citations(h, top_k=top_k)
                if lexical_ids:
                    h.supporting_papers = lexical_ids
                    h.grounding_status = "grounded"
                    note = "Backfilled citations via lexical overlap with pulled paper corpus."
                    if note not in h.citation_warnings:
                        h.citation_warnings.append(note)
                    continue

                note = "No citations found in local paper store; requires external search."
                if note not in h.citation_warnings:
                    h.citation_warnings.append(note)
                continue

            h.supporting_papers = [paper.arxiv_id for paper, _ in results]
            h.grounding_status = "grounded"
            note = f"Backfilled citations via semantic search (query: {query[:120]})"
            if note not in h.citation_warnings:
                h.citation_warnings.append(note)

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set[str]:
        if not text:
            return set()
        raw_tokens = re.findall(r"[a-z0-9][a-z0-9\-]{2,}", text.lower())
        stopwords = {
            "the", "and", "for", "with", "from", "that", "this", "into", "using",
            "than", "then", "their", "have", "has", "had", "can", "could", "would",
            "should", "into", "over", "under", "between", "within", "across",
            "battery", "batteries", "hydrogen", "effect", "effects",
        }
        return {token for token in raw_tokens if token not in stopwords}

    def _lexical_backfill_citations(self, hypothesis: Hypothesis, top_k: int = 3) -> list[str]:
        """Fallback citation backfill when dense search cannot return usable matches."""
        if not self.paper_store:
            return []

        query_text = " ".join([hypothesis.hypothesis] + (hypothesis.novelty_keywords or []))
        query_tokens = self._tokenize_for_overlap(query_text)
        if not query_tokens:
            return []

        scored: list[tuple[float, str]] = []
        for paper in self.paper_store.values():
            paper_key = self._paper_id_key(paper.arxiv_id)
            if self.pulled_paper_keys and paper_key not in self.pulled_paper_keys:
                continue
            if (
                not self.pulled_paper_keys
                and self.pulled_paper_bases
                and (normalize_arxiv_id(paper.arxiv_id, keep_version=False) or paper.arxiv_id) not in self.pulled_paper_bases
            ):
                continue
            title = (paper.title or "").lower()
            abstract = (paper.abstract or "").lower()
            corpus_tokens = self._tokenize_for_overlap(f"{title} {abstract}")
            overlap = query_tokens & corpus_tokens
            if not overlap:
                continue

            title_bonus = sum(1 for token in query_tokens if token in title)
            score = float(len(overlap) + 0.5 * title_bonus)
            scored.append((score, paper.arxiv_id))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [paper_id for _, paper_id in scored[:top_k]]

    def _derive_provenance_split(self, hypothesis: Hypothesis) -> None:
        """Populate supported facts / novel inference / unsupported parts.

        Keeps explicit model-provided values when present, and fills missing pieces
        from evidence spans and citation warnings.
        """
        supported = list(hypothesis.supported_facts or [])
        if not supported:
            for span in hypothesis.evidence_spans or []:
                citation_id = getattr(span, "citation_id", "")
                quote = getattr(span, "quote", "")
                if citation_id and quote:
                    supported.append(f"[{citation_id}] {quote}")

        if not supported:
            supported = [line for line in (hypothesis.evidence_trace or []) if line.strip()]

        if not hypothesis.novel_inference.strip():
            hypothesis.novel_inference = hypothesis.hypothesis

        unsupported = list(hypothesis.unsupported_parts or [])
        if not unsupported:
            unsupported = [w for w in (hypothesis.citation_warnings or []) if w.strip()]

        hypothesis.supported_facts = list(dict.fromkeys(supported))[:8]
        hypothesis.unsupported_parts = list(dict.fromkeys(unsupported))[:8]

    def _apply_citation_policies(self, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Apply numeric-evidence checks and optional strict grounding."""
        if not hypotheses:
            return hypotheses

        kept: list[Hypothesis] = []
        dropped = 0
        id_index = self._build_paper_id_index()

        for h in hypotheses:
            # Restrict citations to papers pulled during this run.
            if (self.pulled_paper_keys or self.pulled_paper_bases) and h.supporting_papers:
                kept_ids: list[str] = []
                removed_ids: list[str] = []
                for paper_id in h.supporting_papers:
                    resolved_id = self._resolve_paper_id(paper_id, id_index)
                    key = self._paper_id_key(paper_id)
                    base_id = normalize_arxiv_id(paper_id, keep_version=False) or paper_id
                    in_pulled = False
                    if self.pulled_paper_keys:
                        in_pulled = key in self.pulled_paper_keys
                    elif self.pulled_paper_bases:
                        in_pulled = base_id in self.pulled_paper_bases

                    if in_pulled and resolved_id:
                        kept_ids.append(resolved_id)
                    elif in_pulled:
                        kept_ids.append(paper_id)
                    else:
                        removed_ids.append(paper_id)

                if removed_ids:
                    warning = (
                        "Removed citations not present in pulled corpus: "
                        + ", ".join(removed_ids[:5])
                    )
                    if warning not in h.citation_warnings:
                        h.citation_warnings.append(warning)
                h.supporting_papers = list(dict.fromkeys(kept_ids))

            if h.supporting_papers:
                h.grounding_status = "grounded"
            elif h.grounding_status == "grounded":
                h.grounding_status = "ungrounded"

            # Numeric evidence policy
            ok_evidence, evidence_warning = validate_evidence_span_coverage(h, self.paper_store)
            if evidence_warning and evidence_warning not in h.citation_warnings:
                h.citation_warnings.append(evidence_warning)
            if not ok_evidence:
                h.grounding_status = "mixed" if h.supporting_papers else "ungrounded"

            # Numeric evidence policy
            if self.config.enforce_numeric_citations:
                ok_numeric, warning = validate_numeric_citation_coverage(h, self.paper_store)
                if warning and warning not in h.citation_warnings:
                    h.citation_warnings.append(warning)
                if not ok_numeric:
                    h.grounding_status = "mixed" if h.supporting_papers else "ungrounded"

            # Strict grounding policy
            if self.config.strict_grounding and not h.supporting_papers:
                dropped += 1
                continue

            if (
                self.config.strict_grounding
                and any(w.startswith("Evidence spans") or w.startswith("Evidence span") for w in h.citation_warnings)
            ):
                dropped += 1
                continue

            if (
                self.config.strict_grounding
                and self.config.enforce_numeric_citations
                and any(
                    w.startswith("Numeric claim detected")
                    for w in h.citation_warnings
                )
            ):
                dropped += 1
                continue

            self._derive_provenance_split(h)
            kept.append(h)

        if dropped:
            print(
                f"[GroundingPolicy] Dropped {dropped} hypotheses "
                f"(strict_grounding={self.config.strict_grounding}, "
                f"enforce_numeric_citations={self.config.enforce_numeric_citations})"
            )

        return kept

    async def lock_persona(self, persona_id: str) -> None:
        """Lock a persona from regeneration."""
        for p in self.persona_roster:
            if p["id"] == persona_id:
                p["locked"] = True
        await self._emit_personas(self.persona_roster)

    async def unlock_persona(self, persona_id: str) -> None:
        """Unlock a persona for regeneration."""
        for p in self.persona_roster:
            if p["id"] == persona_id:
                p["locked"] = False
        await self._emit_personas(self.persona_roster)

    async def update_persona_weight(self, persona_id: str, weight: float) -> None:
        """Update a persona's weight in the collective."""
        for p in self.persona_roster:
            if p["id"] == persona_id:
                p["weight"] = weight
        await self._emit_personas(self.persona_roster)

    async def regenerate_persona(self, persona_id: str) -> None:
        """Regenerate a single persona if not locked."""
        persona_idx = -1
        for i, p in enumerate(self.persona_roster):
            if p["id"] == persona_id:
                if p.get("locked"):
                    return
                persona_idx = i
                break
        
        if persona_idx == -1:
            return

        new_persona = await self.persona_forge.regenerate_persona(self.topic, persona_id)
        
        # Update Roster
        self.persona_roster[persona_idx] = {
            "id": new_persona.id,
            "name": new_persona.name,
            "role": new_persona.role,
            "style": new_persona.style,
            "objective": new_persona.objective,
            "weight": new_persona.weight,
            "soul_role": new_persona.soul_role.value if new_persona.soul_role else "unknown",
            "locked": False,
        }

        # Update Collective
        if new_persona.soul_role == SoulRole.CREATIVE:
            self.collective.creative.set_persona(new_persona.name, new_persona.system_instruction)
        elif new_persona.soul_role == SoulRole.RISK_TAKER:
            self.collective.risk_taker.set_persona(new_persona.name, new_persona.system_instruction)
        elif new_persona.soul_role == SoulRole.SKEPTIC:
            self.collective.skeptic.set_persona(new_persona.name, new_persona.system_instruction)

        await self._emit_personas(self.persona_roster)
        await self._emit_status(
            self.current_phase, 
            f"Persona {persona_id} regenerated: {new_persona.name}"
        )

    async def vote_hypothesis(self, hypothesis_id: str, direction: str) -> None:
        """Vote on a hypothesis to influence future iterations."""
        # Find the hypothesis
        target = next((h for h in self.hypotheses if h.id == hypothesis_id), None)
        if not target:
            return
        
        msg = f"User voted {direction} on hypothesis: '{target.hypothesis}'"
        await self.inject_user_message(msg)
        # We could also directly adjust scores or priority in memory here

    async def investigate_hypothesis(self, hypothesis_id: str) -> None:
        """Mark a hypothesis for deeper investigation in the next loop."""
        target = next((h for h in self.hypotheses if h.id == hypothesis_id), None)
        if not target:
            return
        
        msg = f"DEEP INVESTIGATION REQUESTED for hypothesis: '{target.hypothesis}'. Focus next iteration on validating its core assumptions."
        await self.inject_user_message(msg)

    async def pin_directive(self, text: str) -> None:
        """Pin a user directive for persistent influence."""
        self.memory.working.add_pinned_directive(text)
        await self._emit_status(self.current_phase, f"Directive pinned: {text[:30]}...")

    async def unpin_directive(self, text: str) -> None:
        """Unpin a user directive."""
        self.memory.working.remove_pinned_directive(text)
        await self._emit_status(self.current_phase, f"Directive unpinned: {text[:30]}...")

    async def rerun_simulation(self, hypothesis_id: str, custom_code: str | None = None) -> None:
        """Rerun simulation for a specific hypothesis."""
        target = next((h for h in self.hypotheses if h.id == hypothesis_id), None)
        if not target:
            return

        await self._emit_status(SessionPhase.VERIFYING, f"Rerunning simulation for {hypothesis_id}...")
        
        # We need a GroundedHypothesis for the simulator
        # If it's a standard Hypothesis, we'll need to wrap it or update Simulator
        # For now, let's assume we can map it back or use a more generic verify
        
        # Find if we have a grounded version
        gh = next((g for g in self.grounded_hypotheses if g.id == hypothesis_id), None)
        if not gh:
            # Fallback: create a temporary grounded hypothesis from the hypothesis object
            from src.contracts.schemas import GroundedHypothesis, MechanismStep
            gh = GroundedHypothesis(
                id=target.id,
                claim=target.hypothesis,
                mechanism=[MechanismStep(cause="Input", effect="Effect")], # Placeholder
                prediction="Outcome predicted by model",
                null_result="No significant change observed",
                gap_addressed="Direct verification request"
            )

        if custom_code:
            # If user provided code, we just execute it
            sim_id = str(uuid.uuid4())[:8]
            result_dict = await self.simulator._execute_code(custom_code, sim_id)
            
            # Try to get visual verification if a plot was expected
            # (Requires plot path convention in custom code)
            
            new_result = SimulationResult(
                code=custom_code,
                success=result_dict["success"],
                supports_hypothesis=result_dict["supports_hypothesis"],
                output_log=result_dict["output"],
                metrics=result_dict["metrics"],
                status="complete" if result_dict["success"] else "error"
            )
        else:
            # Generate new code and run
            new_result = await self.simulator.verify_hypothesis(gh)

        # Update history
        if target.simulation_result:
            target.simulation_history.append(target.simulation_result)
        target.simulation_result = new_result
        
        await self._emit_status(SessionPhase.VERIFYING, f"Simulation rerun complete for {hypothesis_id}")

    async def run(
        self,
        topic: str,
        output_dir: Path | None = None,
        *,
        session_id: str | None = None,
        constraints: SessionConstraints | None = None,
    ) -> SessionResult:
        """Run a complete hypothesis generation session.

        Args:
            topic: Research topic/query
            output_dir: Directory to save session data

        Returns:
            SessionResult with final hypotheses and traces
        """
        # Initialize session
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.topic = topic
        self.start_time = datetime.now()
        self.iteration = 0
        self.constraints = constraints
        self.current_phase = SessionPhase.QUEUED

        # Phase 0: Dynamic Persona Generation (Gemini 3 Adaptive Team)
        if not self.persona_roster:
            await self._emit_status(SessionPhase.FORGING, "Assembling specialized research team...")
            try:
                roster = await self.persona_forge.forge_team(topic)
                specialist = roster.get("specialist")
                maverick = roster.get("maverick")
                skeptic = roster.get("skeptic")
                if not specialist or not maverick or not skeptic:
                    raise ValueError("Persona roster incomplete")

                # Inject into Collective
                if specialist:
                    self.collective.creative.set_persona(specialist.name, specialist.system_instruction)
                if maverick:
                    self.collective.risk_taker.set_persona(maverick.name, maverick.system_instruction)
                if skeptic:
                    self.collective.skeptic.set_persona(skeptic.name, skeptic.system_instruction)

                self.persona_roster = [
                    {
                        "id": "specialist",
                        "name": specialist.name,
                        "role": specialist.role,
                        "style": specialist.style,
                        "objective": specialist.objective,
                        "weight": specialist.weight,
                        "soul_role": "creative",
                        "locked": False,
                    },
                    {
                        "id": "maverick",
                        "name": maverick.name,
                        "role": maverick.role,
                        "style": maverick.style,
                        "objective": maverick.objective,
                        "weight": maverick.weight,
                        "soul_role": "risk_taker",
                        "locked": False,
                    },
                    {
                        "id": "skeptic",
                        "name": skeptic.name,
                        "role": skeptic.role,
                        "style": skeptic.style,
                        "objective": skeptic.objective,
                        "weight": skeptic.weight,
                        "soul_role": "skeptic",
                        "locked": False,
                    },
                ]
                await self._emit_personas(self.persona_roster)

                await self._emit_status(
                    SessionPhase.FORGING,
                    (
                        f"Team Assembled: {specialist.name} ({specialist.role}), "
                        f"{maverick.name} ({maverick.role}), {skeptic.name} ({skeptic.role})"
                    ),
                )
                # Also store in agent beliefs for context
                self.agent.perceive({
                    "active_personas": [p["name"] for p in self.persona_roster]
                })
                
                # Emit Team Announcement Trace
                team_trace = IterationTrace(
                    iteration=0,
                    thought=f"I have assembled a specialized team for {topic}: {specialist.name} ({specialist.role}) and {maverick.name} ({maverick.role}).",
                    action="Assemble Team",
                    observation="Team ready for debate.",
                    bdi_snapshot=self.agent.get_state(),
                )
                await self._emit_trace(team_trace)
                
            except Exception as e:
                print(f"[WARN] Failed to forge personas: {e}")
                fallback = self.persona_forge._get_fallback_personas()
                specialist = fallback["specialist"]
                maverick = fallback["maverick"]
                skeptic = fallback["skeptic"]
                self.collective.creative.set_persona(specialist.name, specialist.system_instruction)
                self.collective.risk_taker.set_persona(maverick.name, maverick.system_instruction)
                self.collective.skeptic.set_persona(skeptic.name, skeptic.system_instruction)
                self.persona_roster = [
                    {
                        "id": "specialist",
                        "name": specialist.name,
                        "role": specialist.role,
                        "style": specialist.style,
                        "objective": specialist.objective,
                        "weight": specialist.weight,
                        "soul_role": "creative",
                        "locked": False,
                    },
                    {
                        "id": "maverick",
                        "name": maverick.name,
                        "role": maverick.role,
                        "style": maverick.style,
                        "objective": maverick.objective,
                        "weight": maverick.weight,
                        "soul_role": "risk_taker",
                        "locked": False,
                    },
                    {
                        "id": "skeptic",
                        "name": skeptic.name,
                        "role": skeptic.role,
                        "style": skeptic.style,
                        "objective": skeptic.objective,
                        "weight": skeptic.weight,
                        "soul_role": "skeptic",
                        "locked": False,
                    },
                ]
                await self._emit_personas(self.persona_roster)
                await self._emit_status(SessionPhase.FORGING, "Failed to assemble team; using default agents.")

        # Set up agent
        self.agent.reset(topic)

        # Detect domain categories
        domain_tags = detect_categories_from_query(topic)
        self.agent.perceive({"topic": topic, "domain_tags": domain_tags})

        # Phase 1: Ingest papers and build concept map
        await self._ingest_papers()

        # Main Ralph loop
        while True:
            self.iteration += 1

            # Check stop conditions
            should_stop, reason = self.should_stop()
            if should_stop:
                break

            # Run one iteration
            await self._run_iteration()

        # Build result
        final_reason = reason or "Complete"
        self.hypotheses = self._apply_citation_policies(self.hypotheses)
        result = SessionResult(
            session_id=self.session_id,
            topic=topic,
            started_at=self.start_time,
            completed_at=datetime.now(),
            iterations_completed=self.iteration,
            stop_reason=final_reason,
            final_hypotheses=self.hypotheses[: self.config.max_hypotheses],
            traces=list(self.memory.episodic.episodes),
            total_tokens_used=self.total_tokens,
            total_cost_usd=self.total_cost,
            papers_ingested=self.agent.get_belief("papers_ingested") or 0,
            concept_map=self.concept_map,
            source_metadata=self.paper_store,
            constraints=self.constraints,
        )

        # Save session if output_dir provided
        if output_dir:
            await self._save_session(output_dir, result)

        await self._emit_status(SessionPhase.COMPLETE, final_reason)

        return result

    async def _ingest_papers(self) -> None:
        """Phase 1: Fetch papers from free sources, build map, extract claims."""
        def _extract_keywords(topic: str) -> list[str]:
            tokens = re.findall(r"[a-z0-9\\-]+", topic.lower())
            stop = {
                "how", "to", "build", "better", "best", "improve", "make", "create",
                "what", "why", "when", "where", "which", "who", "is", "are", "the",
                "a", "an", "of", "for", "in", "on", "with", "and", "or",
            }
            return [t for t in tokens if t not in stop]

        def _build_query(topic: str, keywords: list[str]) -> str:
            topic_lower = topic.lower()
            if any(kw in topic_lower for kw in ["battery", "batteries", "lithium", "electrolyte", "anode", "cathode", "solid-state", "solid state", "energy storage"]):
                return 'battery OR batteries OR lithium OR electrolyte OR anode OR cathode OR "solid state" OR "energy storage"'
            return " ".join(keywords) if keywords else topic

        self.paper_store = {}
        self.pulled_paper_bases = set()
        self.pulled_paper_keys = set()
        self.knowledge_sources = []
        self.claims = []
        self.gaps = []

        # Fetch papers
        keywords = _extract_keywords(self.topic)
        query = _build_query(self.topic, keywords)
        categories = detect_categories_from_query(self.topic)
        if self.config.max_runtime_seconds <= 300:
            paper_limit = 12
            claim_limit = 5
        elif self.config.max_runtime_seconds <= 600:
            paper_limit = 20
            claim_limit = 8
        else:
            paper_limit = 30
            claim_limit = 10

        papers, source_counts = await self._retrieve_cross_domain_papers(
            query=query,
            categories=categories,
            keywords=keywords,
            paper_limit=paper_limit,
        )

        if keywords and papers:
            top_score = self._score_paper_relevance(papers[0], keywords)
            if top_score <= 0:
                await self._emit_status(
                    SessionPhase.MAPPING,
                    "No relevant papers found for query; skipping literature pipeline.",
                )
                return

        if not papers:
            await self._emit_status(
                SessionPhase.MAPPING,
                "No papers retrieved from arXiv/PubMed/Semantic Scholar.",
            )
            return

        self.pulled_paper_bases = {
            normalize_arxiv_id(p.arxiv_id, keep_version=False) or p.arxiv_id
            for p in papers
        }
        self.pulled_paper_keys = {
            self._paper_id_key(p.arxiv_id)
            for p in papers
            if self._paper_id_key(p.arxiv_id)
        }
        self.knowledge_sources = sorted({(p.source or "unknown") for p in papers})

        # Populate Knowledge Store
        for p in papers:
            self.paper_store[p.arxiv_id] = p

        # Index papers for semantic search
        try:
            indexed_count = await self.semantic_search.index_papers(papers)
            print(f"[SemanticSearch] Indexed {indexed_count} papers")
        except Exception as e:
            print(f"[SemanticSearch] Warning: Failed to index papers: {e}")

        await self._emit_knowledge_update({
            "papers_indexed": len(self.paper_store),
            "sources": self.knowledge_sources,
            "source_breakdown": source_counts,
            "semantic_search_ready": self.semantic_search.is_initialized,
        })

        # Phase 1: Global Concept Mapping
        await self._emit_status(
            SessionPhase.MAPPING,
            f"Building Global Concept Map from {len(papers)} papers...",
        )
        builder = ConceptMapBuilder(model=self.config.pro_model)
        self.concept_map = await builder.build_global_map_from_abstracts(papers)
        
        await self._emit_knowledge_update({
            "papers_indexed": len(self.paper_store),
            "sources": self.knowledge_sources,
            "source_breakdown": source_counts,
            "concepts_extracted": len(self.concept_map.nodes),
            "relations_found": len(self.concept_map.edges),
            "concept_map": self.concept_map.model_dump(),
        })
        
        # Map Trans-Paper Gaps into IdentifiedGap objects
        if hasattr(self.concept_map, "gaps") and self.concept_map.gaps:
            for gap_data in self.concept_map.gaps:
                gap = IdentifiedGap(
                    gap_type=GapType.MISSING_CONNECTION,
                    description=gap_data.get("description", "Potential link discovered via global analysis"),
                    concept_a=gap_data.get("node_a", ""),
                    concept_b=gap_data.get("node_b", ""),
                    potential_value=gap_data.get("logic", "Inferred from trans-paper synthesis"),
                )
                self.gaps.append(gap)
        
        # Track concept map builder usage manually
        self.total_tokens += builder.total_tokens
        self.total_cost += builder.total_cost

        # Phase 2: Targeted Claim Extraction for Grounding
        await self._emit_status(
            SessionPhase.MAPPING,
            "Extracting quantitative baselines... (0/{})".format(claim_limit),
        )
        for idx, paper in enumerate(papers[:claim_limit], start=1):
            claims = await self.claim_extractor.extract_claims(
                paper_id=paper.arxiv_id,
                title=paper.title,
                abstract=paper.abstract,
            )
            self.claims.extend(claims)
            await self._emit_status(
                SessionPhase.MAPPING,
                f"Extracting quantitative baselines... ({idx}/{claim_limit})",
            )
        
        self._update_costs()
        await self._emit_status(
            SessionPhase.MAPPING,
            f"Global Knowledge Base Ready: {len(self.concept_map.nodes)} concepts, {len(self.gaps)} trans-paper gaps.",
        )

        # Update beliefs
        self.agent.perceive({
            "papers_ingested": len(papers),
            "concept_map": self.concept_map,
            "claims_extracted": len(self.claims),
            "gaps_identified": len(self.gaps),
        })

        # Update semantic memory
        self.memory.semantic.update(
            self.concept_map,
            len(papers),
            self.agent.get_belief("domain_tags") or [],
        )

    def _build_debate_context(self) -> dict[str, Any]:
        """Build generation context enriched with pulled-paper evidence."""
        context = self.memory.get_context_for_generation(topic=self.topic)

        source_papers: list[dict[str, str]] = []
        for paper in list(self.paper_store.values())[:20]:
            source_papers.append(
                {
                    "id": paper.arxiv_id,
                    "title": paper.title,
                    "abstract": (paper.abstract or "")[:300],
                    "source": paper.source or "unknown",
                    "url": paper.abs_url or paper.pdf_url or "",
                }
            )

        source_claims: list[dict[str, str]] = []
        for claim in self.claims[:20]:
            if claim.paper_id and claim.statement:
                source_claims.append(
                    {
                        "paper_id": claim.paper_id,
                        "statement": claim.statement,
                    }
                )

        context["source_papers"] = source_papers
        context["source_claims"] = source_claims
        context["allowed_paper_ids"] = [paper["id"] for paper in source_papers]
        context["knowledge_sources"] = self.knowledge_sources
        context["citation_policy"] = (
            "Use only paper IDs listed in source_papers (arXiv/PMID/DOI/S2). "
            "If support is weak, state uncertainty explicitly instead of inventing citations."
        )
        return context

    async def _run_iteration(self) -> None:
        """Run one iteration of the Ralph loop."""
        # Start iteration in memory
        mode = self.agent.state.current_mode
        self.memory.start_iteration(self.iteration, mode)
        
        # Process User Messages (The Research Assistant)
        while not self.user_message_queue.empty():
            msg = self.user_message_queue.get_nowait()
            self.memory.working.user_guidance.append(msg)
            self._add_trace(f"Integrating user guidance: {msg[:50]}...")
        
        if self.memory.working.user_guidance:
            await self._emit_status(
                SessionPhase.DEBATING,
                f"Iteration {self.iteration}: Pivoting based on user feedback...",
            )
        else:
            await self._emit_status(
                SessionPhase.DEBATING,
                f"Iteration {self.iteration}: Deliberating with mode {mode.value}...",
            )

        # BDI deliberation
        current_scores = self._get_average_scores()
        thought = self.agent.deliberate(current_scores, len(self.hypotheses))
        
        # Integrate user guidance into the thought trace
        if self.memory.working.user_guidance or self.memory.working.pinned_directives:
            guidance_parts = []
            if self.memory.working.user_guidance:
                guidance_parts.append("LATEST: " + " | ".join(self.memory.working.user_guidance))
            if self.memory.working.pinned_directives:
                guidance_parts.append("PINNED: " + " | ".join([d["text"] for d in self.memory.working.pinned_directives]))
            
            thought = f"USER GUIDANCE INCORPORATED: {' ; '.join(guidance_parts)}. " + thought

        # Plan next action
        new_mode = self.agent.plan(current_scores, len(self.hypotheses))
        self.agent.commit_to_plan(new_mode)

        new_hypotheses: list[Hypothesis] = []
        debate_trace: dict[str, Any] = {}
        observation = "" # Initialize observation

        # Get weights from roster for sampling
        persona_weights = {p["id"]: p.get("weight", 0.33) for p in self.persona_roster}

        # ---------------------------------------------------------------------
        # PHASE 1: GENERATION (MCTS or Linear)
        # ---------------------------------------------------------------------
        if self.iteration == 1 and self.gaps:
             await self._emit_status(SessionPhase.DEBATING, "Running Agentic Tree Search...")

             # Create initial state for MCTS
             root_state = ResearchState(
                 gaps=self.gaps,
                 claims=self.claims,
                 concept_map=self.concept_map,
                 depth=0
             )

             # Run Tree Search
             best_state = await self.tree_search.run_search(root_state)

             # Adopt hypotheses from best state
             self.grounded_hypotheses = best_state.hypotheses

             # Trace actions
             self._add_trace(f"Tree Search complete. Found {len(self.grounded_hypotheses)} hypotheses. Path depth: {best_state.depth}")
             for note in best_state.feedback:
                 self._add_trace(f"Tree Action: {note}")

             # Convert schema for compatibility
             mcts_hypotheses: list[Hypothesis] = []
             for gh in self.grounded_hypotheses:
                 # Helper function for rationale
                 def _format_rationale(grounded: GroundedHypothesis) -> str:
                     mechanism_str = " -> ".join(f"{s.cause} causes {s.effect}" for s in grounded.mechanism)
                     rationale_parts = [f"Mechanism: {mechanism_str}"]
                     if grounded.null_result:
                         rationale_parts.append(f"Null Result: {grounded.null_result}")
                     return ". ".join(rationale_parts)

                 experiments = [e.description for e in gh.suggested_experiments if e.description]
                 if not experiments:
                     experiments = ["Design a controlled experiment to test the claim."]

                 mcts_hypotheses.append(
                     Hypothesis(
                         id=gh.id or str(uuid.uuid4())[:8],
                         hypothesis=gh.claim,
                         rationale=_format_rationale(gh),
                         cross_disciplinary_connection=gh.gap_addressed or "Generated from gap",
                         experimental_design=experiments,
                         expected_impact="High - grounded in literature gaps.",
                         novelty_keywords=["Grounded", "Gap-Driven"],
                         supporting_papers=gh.supporting_papers or [],
                         evidence_trace=getattr(gh, "source_claims", []) or [],
                         iteration=self.iteration,
                         source_soul=SoulRole.SYNTHESIZER,
                         scores=gh.scores,
                     )
                 )

             srsh_hypotheses: list[Hypothesis] = []
             srsh_generated = 0
             if self.config.srsh_enabled and self.concept_map is not None:
                 await self._emit_status(SessionPhase.DEBATING, "Running SRSH parallel streams and collision synthesis...")
                 try:
                     self.srsh.status_callback = lambda msg: self._add_trace(f"SRSH: {msg}")
                     srsh_result = await self.srsh.run(
                         topic=self.topic,
                         concept_map=self.concept_map,
                         claims=self.claims,
                     )
                     srsh_hypotheses = srsh_result.hypotheses
                     srsh_generated = len(srsh_hypotheses)
                     self._add_trace(
                         "SRSH complete: "
                         f"{srsh_generated} hypotheses, "
                         f"collision ratio={srsh_result.metrics.get('collision_ratio', 0.0):.2f}, "
                         f"avg semantic distance={srsh_result.metrics.get('avg_semantic_distance', 0.0):.2f}"
                     )
                 except Exception as e:
                     self._add_trace(f"SRSH failed, continuing with tree-search outputs: {e}")

             self.hypotheses = mcts_hypotheses + srsh_hypotheses
             if self.hypotheses:
                 await self._validate_hypothesis_citations(self.hypotheses)
                 await self._backfill_citations(self.hypotheses)
                 self.hypotheses = self._apply_citation_policies(self.hypotheses)

             observation += (
                 f"\n\nAnalyzed {len(self.gaps)} research gaps via Tree Search + SRSH. "
                 f"MCTS generated {len(mcts_hypotheses)}, SRSH generated {srsh_generated}, "
                 f"kept {len(self.hypotheses)} after grounding filters."
             )
             await self._emit_status(
                 SessionPhase.DEBATING,
                 f"Initial synthesis complete. {len(self.hypotheses)} hypotheses ready.",
             )
             debate_trace = {
                 "hypotheses_generated": len(mcts_hypotheses) + srsh_generated,
                 "hypotheses_killed": 0,
                 "hypotheses_final": len(self.hypotheses),
                 "gap_based": True,
             }
             if not self.hypotheses:
                 await self._emit_status(
                     SessionPhase.DEBATING,
                     "Tree Search + SRSH produced no grounded hypotheses; falling back to debate.",
                 )
                 new_hypotheses, debate_trace = await self.collective.run_debate(
                     topic=self.topic,
                     context=self._build_debate_context(),
                     mode=GenerationMode.RANDOM_INJECTION,
                     target_hypotheses=self.config.max_hypotheses,
                     weights=persona_weights,
                 )
                 observation += f"\n\nFallback debate generated {len(new_hypotheses)} hypotheses."

        elif self.iteration == 1: # Changed 'iteration' to 'self.iteration'     
             # Fallback if no gaps (should rare)
             await self._emit_status(SessionPhase.DEBATING, "Brainstorming (Fallback)...")
             new_hypotheses, debate_trace = await self.collective.run_debate(
                 topic=self.topic,
                 context=self._build_debate_context(),
                 mode=GenerationMode.RANDOM_INJECTION,
                 target_hypotheses=self.config.max_hypotheses,
                 weights=persona_weights,
             )
             observation += f"\n\nGenerated {len(new_hypotheses)} initial hypotheses via brainstorming."
        else:
             # Subsequent iterations: Refine existing standard hypotheses
             await self._emit_status(SessionPhase.DEBATING, "Refining hypotheses via debate...")
             new_hypotheses, debate_trace = await self.collective.run_debate( # Capture debate_trace
                 topic=self.topic, # Added topic
                 context=self._build_debate_context(), # Added context
                 mode=new_mode, # Added mode
                 target_hypotheses=self.config.max_hypotheses, # Added target_hypotheses
                 existing_hypotheses=self.hypotheses, # Pass existing hypotheses for refinement
                 weights=persona_weights,
             )
             observation += "\n\nRefined hypotheses through debate."

        raw_new_count = len(new_hypotheses)

        # Validate and normalize citations for newly generated hypotheses
        if new_hypotheses:
            await self._validate_hypothesis_citations(new_hypotheses)
            await self._backfill_citations(new_hypotheses)
            new_hypotheses = self._apply_citation_policies(new_hypotheses)
        grounded_new_count = len(new_hypotheses)

        # Evolutionary Memory: Bury failed ideas (The Graveyard)
        if debate_trace and "fatal_critiques" in debate_trace:
            for fatal in debate_trace["fatal_critiques"]:
                self.memory.graveyard.bury(
                    hypothesis=fatal["hypothesis"],
                    reason=fatal["reason"],
                    topic=self.topic
                )
                self._add_trace(f"Buried failed hypothesis: {fatal['hypothesis'][:50]}...")

        self._update_costs()

        # Verify novelty
        if new_hypotheses:
            await self._emit_status(
                SessionPhase.VERIFYING,
                "Validating novelty and scoring hypotheses...",
            )
            try:
                await asyncio.wait_for(
                    batch_verify_novelty(new_hypotheses),
                    timeout=120,
                )
            except asyncio.TimeoutError:
                warning = (
                    "Novelty verification timed out; continuing with partial/local evidence."
                )
                for h in new_hypotheses:
                    if warning not in h.citation_warnings:
                        h.citation_warnings.append(warning)
                await self._emit_status(
                    SessionPhase.VERIFYING,
                    warning,
                )
            except Exception as e:
                warning = f"Novelty verification degraded due to upstream lookup failure: {e.__class__.__name__}"
                for h in new_hypotheses:
                    if warning not in h.citation_warnings:
                        h.citation_warnings.append(warning)
                await self._emit_status(
                    SessionPhase.VERIFYING,
                    f"{warning}. Continuing with local scoring.",
                )

            # Score feasibility and impact
            try:
                new_hypotheses = await asyncio.wait_for(
                    self.scorer.batch_score(new_hypotheses),
                    timeout=180,
                )
            except asyncio.TimeoutError:
                timeout_warning = (
                    "Scoring timed out; applied heuristic feasibility/impact fallback."
                )
                for h in new_hypotheses:
                    h.scores.feasibility = heuristic_feasibility(h)
                    h.scores.impact = heuristic_impact(h)
                    if timeout_warning not in h.citation_warnings:
                        h.citation_warnings.append(timeout_warning)
                await self._emit_status(
                    SessionPhase.VERIFYING,
                    timeout_warning,
                )

        # Merge with existing hypotheses
        old_count = len(self.hypotheses)
        self.hypotheses = self._merge_hypotheses(self.hypotheses, new_hypotheses)
        self.hypotheses = self._apply_citation_policies(self.hypotheses)

        # Phase 3: Visualize Top Hypotheses (Gemini 3 Visual Mechanism)
        for h in self.hypotheses[:3]:
            if not h.diagram:
                # Fire and forget / parallelize ideally, but sequential for safety now
                try:
                    h.diagram = await self.visualizer.generate_diagram(h)
                except Exception as e:
                    print(f"[WARN] Visualization failed for {h.id}: {e}")

        # Check for improvement
        new_scores = self._get_average_scores()
        improved = new_scores.aggregate > current_scores.aggregate

        # Update beliefs
        self.agent.perceive({
            "improved": improved,
            "novelty_scores": [h.scores.novelty for h in self.hypotheses],
            "feasibility_scores": [h.scores.feasibility for h in self.hypotheses],
        })

        # Build observation string
        gap_info = f" (from {len(self.gaps)} gaps)" if debate_trace.get("gap_based") else ""
        observation = (
            f"Generated {debate_trace.get('hypotheses_generated', 0)} hypotheses{gap_info}, "
            f"{debate_trace.get('hypotheses_killed', 0)} killed by Skeptic, "
            f"{len(self.hypotheses)} total. "
            f"Avg novelty: {new_scores.novelty:.2f}, feasibility: {new_scores.feasibility:.2f}"
        )
        dropped_by_grounding = max(0, raw_new_count - grounded_new_count)
        if dropped_by_grounding:
            observation += f", {dropped_by_grounding} dropped by grounding policy"
        if len(self.hypotheses) == 0 and debate_trace.get("hypotheses_generated", 0) > 0:
            observation += ". No hypotheses survived citation constraints."

        # Record trace
        trace = IterationTrace(
            iteration=self.iteration,
            thought=thought,
            action=f"Generated with mode {new_mode.value}",
            observation=observation,
            bdi_snapshot=self.agent.get_state(),
            dialogue=debate_trace.get("dialogue", []),
            hypotheses_generated=debate_trace.get("hypotheses_generated", 0),
            hypotheses_surviving=len(self.hypotheses),
            avg_novelty=new_scores.novelty,
            avg_feasibility=new_scores.feasibility,
            mode_used=new_mode,
            tokens_used=self.total_tokens,
            cost_usd=self.total_cost,
        )

        await self._emit_trace(trace)

        self.memory.end_iteration(trace)

    def _get_average_scores(self) -> ScoreBlock:
        """Calculate average scores across all hypotheses."""
        if not self.hypotheses:
            return ScoreBlock()

        n = len(self.hypotheses)
        return ScoreBlock(
            novelty=sum(h.scores.novelty for h in self.hypotheses) / n,
            feasibility=sum(h.scores.feasibility for h in self.hypotheses) / n,
            impact=sum(h.scores.impact for h in self.hypotheses) / n,
            cross_domain=sum(h.scores.cross_domain for h in self.hypotheses) / n,
        )

    def _merge_hypotheses(
        self,
        existing: list[Hypothesis],
        new: list[Hypothesis],
    ) -> list[Hypothesis]:
        """Merge new hypotheses with existing, keeping best."""
        all_hyps = existing + new

        # Sort by aggregate score
        all_hyps.sort(key=lambda h: h.scores.aggregate, reverse=True)

        # Keep top N
        return all_hyps[: self.config.max_hypotheses + 5]  # Keep a few extra

    def should_stop(self) -> tuple[bool, str]:
        """Check all stop conditions.

        Returns:
            Tuple of (should_stop, reason)
        """
        # Max iterations
        if self.iteration > self.config.max_iterations:
            return True, f"Reached max iterations ({self.config.max_iterations})"

        # Max runtime
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if elapsed >= self.config.max_runtime_seconds:
                return True, f"Reached max runtime ({self.config.max_runtime_seconds}s)"

        # Max cost
        if self.total_cost >= self.config.max_cost_usd:
            return True, f"Reached max cost (${self.config.max_cost_usd})"

        # Stagnation
        beliefs = self.agent.get_belief("stagnation_count") or 0
        if beliefs >= self.config.stagnation_threshold:
            return True, f"Stagnation detected ({beliefs} iterations without improvement)"

        # Convergence (all targets met)
        if len(self.hypotheses) >= self.config.min_hypotheses:
            scores = self._get_average_scores()
            if (
                scores.novelty >= self.config.target_novelty
                and scores.feasibility >= self.config.target_feasibility
            ):
                return True, "Convergence: all targets met"

        return False, ""

    async def _save_session(self, output_dir: Path, result: SessionResult) -> None:
        """Save session data to disk."""
        session_dir = output_dir / result.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        def _save_task():
            import json

            # Save hypotheses
            with open(session_dir / "hypotheses.json", "w") as f:
                data = [h.model_dump() for h in result.final_hypotheses]
                json.dump(data, f, indent=2, default=str)

            # Save traces
            with open(session_dir / "traces.json", "w") as f:
                data = [t.model_dump() for t in result.traces]
                json.dump(data, f, indent=2, default=str)

            # Save concept map
            if result.concept_map:
                with open(session_dir / "concept_map.json", "w") as f:
                    json.dump(result.concept_map.model_dump(), f, indent=2)

            # Save summary
            summary = {
                "session_id": result.session_id,
                "topic": result.topic,
                "status": "complete",
                "phase": SessionPhase.COMPLETE.value,
                "status_detail": result.stop_reason,
                "started_at": result.started_at.isoformat() if result.started_at else None,
                "completed_at": result.completed_at.isoformat() if result.completed_at else None,
                "iterations": result.iterations_completed,
                "stop_reason": result.stop_reason,
                "hypotheses_count": len(result.final_hypotheses),
                "papers_ingested": result.papers_ingested,
                "constraints": result.constraints.model_dump() if result.constraints else None,
                "personas": self.persona_roster,
                "config": self.config.model_dump(),
                "strict_grounding": self.config.strict_grounding,
                "enforce_numeric_citations": self.config.enforce_numeric_citations,
                "knowledge_sources": self.knowledge_sources,
                "source_breakdown": dict(
                    Counter((paper.source or "unknown") for paper in self.paper_store.values())
                ),
            }
            with open(session_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

        await asyncio.to_thread(_save_task)

