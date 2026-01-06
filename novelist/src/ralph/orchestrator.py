"""Ralph Orchestrator - main loop controller for hypothesis generation sessions.

Implements the Ralph loop pattern:
- Iterative refinement until convergence
- Stop conditions (max iterations, cost, time, stagnation)
- Checkpointing and resume capability
"""

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.contracts.schemas import (
    BDIState,
    ConceptMap,
    GenerationMode,
    Hypothesis,
    IterationTrace,
    RalphConfig,
    ScoreBlock,
    SessionResult,
)
from src.kb.arxiv_client import ArxivClient, detect_categories_from_query
from src.kb.concept_map import ConceptMapBuilder
from src.kb.paper_summarizer import PaperSummarizer
from src.soul.bdi import BDIAgent
from src.soul.collective import SoulCollective
from src.soul.memory import MemorySystem
from src.verify.novelty_arxiv import batch_verify_novelty
from src.verify.scoring import ScoringService

load_dotenv()


class RalphOrchestrator:
    """Main orchestrator for hypothesis generation sessions."""

    def __init__(self, config: RalphConfig | None = None):
        self.config = config or RalphConfig()

        # Initialize components
        self.agent = BDIAgent()
        self.memory = MemorySystem()
        self.collective = SoulCollective(
            flash_model=self.config.flash_model,
            pro_model=self.config.pro_model,
        )
        self.scorer = ScoringService(model=self.config.flash_model)

        # Session state
        self.session_id = ""
        self.topic = ""
        self.start_time: datetime | None = None
        self.iteration = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.hypotheses: list[Hypothesis] = []
        self.concept_map: ConceptMap | None = None

    async def run(
        self,
        topic: str,
        output_dir: Path | None = None,
    ) -> SessionResult:
        """Run a complete hypothesis generation session.

        Args:
            topic: Research topic/query
            output_dir: Directory to save session data

        Returns:
            SessionResult with final hypotheses and traces
        """
        # Initialize session
        self.session_id = str(uuid.uuid4())[:8]
        self.topic = topic
        self.start_time = datetime.now()
        self.iteration = 0

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
        result = SessionResult(
            session_id=self.session_id,
            topic=topic,
            started_at=self.start_time,
            completed_at=datetime.now(),
            iterations_completed=self.iteration,
            stop_reason=reason,
            final_hypotheses=self.hypotheses[: self.config.max_hypotheses],
            traces=list(self.memory.episodic.episodes),
            total_tokens_used=self.total_tokens,
            total_cost_usd=self.total_cost,
            papers_ingested=self.agent.get_belief("papers_ingested") or 0,
            concept_map=self.concept_map,
        )

        # Save session if output_dir provided
        if output_dir:
            await self._save_session(output_dir, result)

        return result

    async def _ingest_papers(self) -> None:
        """Phase 1: Fetch papers from arXiv and build concept map."""
        async with ArxivClient() as client:
            # Fetch papers
            papers = await client.search(self.topic, max_results=30)

            if not papers:
                return

            # Summarize papers
            summarizer = PaperSummarizer(model=self.config.flash_model)
            summaries = await summarizer.summarize_batch(papers, max_concurrent=5)

            # Build concept map
            builder = ConceptMapBuilder(model=self.config.pro_model)
            self.concept_map = await builder.build_from_summaries(summaries)

            # Update beliefs
            self.agent.perceive({
                "papers_ingested": len(papers),
                "concept_map": self.concept_map,
            })

            # Update semantic memory
            self.memory.semantic.update(
                self.concept_map,
                len(papers),
                self.agent.get_belief("domain_tags") or [],
            )

    async def _run_iteration(self) -> None:
        """Run one iteration of the Ralph loop."""
        # Start iteration in memory
        mode = self.agent.state.current_mode
        self.memory.start_iteration(self.iteration, mode)

        # BDI deliberation
        current_scores = self._get_average_scores()
        thought = self.agent.deliberate(current_scores, len(self.hypotheses))

        # Plan next action
        new_mode = self.agent.plan(current_scores, len(self.hypotheses))
        self.agent.commit_to_plan(new_mode)

        # Get context for generation
        context = self.memory.get_context_for_generation()

        # Run multi-soul debate
        new_hypotheses, debate_trace = await self.collective.run_debate(
            topic=self.topic,
            context=context,
            mode=new_mode,
            target_hypotheses=self.config.max_hypotheses,
        )

        # Verify novelty
        if new_hypotheses:
            await batch_verify_novelty(new_hypotheses)

            # Score feasibility and impact
            new_hypotheses = await self.scorer.batch_score(new_hypotheses)

        # Merge with existing hypotheses
        old_count = len(self.hypotheses)
        self.hypotheses = self._merge_hypotheses(self.hypotheses, new_hypotheses)

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
        observation = (
            f"Generated {debate_trace.get('hypotheses_generated', 0)} hypotheses, "
            f"{debate_trace.get('hypotheses_killed', 0)} killed by Skeptic, "
            f"{len(self.hypotheses)} total. "
            f"Avg novelty: {new_scores.novelty:.2f}, feasibility: {new_scores.feasibility:.2f}"
        )

        # Record trace
        trace = IterationTrace(
            iteration=self.iteration,
            thought=thought,
            action=f"Generated with mode {new_mode.value}",
            observation=observation,
            bdi_snapshot=self.agent.get_state(),
            hypotheses_generated=debate_trace.get("hypotheses_generated", 0),
            hypotheses_surviving=len(self.hypotheses),
            avg_novelty=new_scores.novelty,
            avg_feasibility=new_scores.feasibility,
            mode_used=new_mode,
            tokens_used=0,  # TODO: track tokens
            cost_usd=0.0,  # TODO: track cost
        )

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
            "started_at": result.started_at.isoformat() if result.started_at else None,
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
            "iterations": result.iterations_completed,
            "stop_reason": result.stop_reason,
            "hypotheses_count": len(result.final_hypotheses),
            "papers_ingested": result.papers_ingested,
        }
        with open(session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
