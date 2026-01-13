"""Ralph Orchestrator - main loop controller for hypothesis generation sessions.

Implements the Ralph loop pattern:
- Iterative refinement until convergence
- Stop conditions (max iterations, cost, time, stagnation)
- Checkpointing and resume capability
"""

import asyncio
import re
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.contracts.schemas import (
    BDIState,
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
    SessionConstraints,
    SessionPhase,
    SessionResult,
    SoulRole,
)
from src.kb.arxiv_client import ArxivClient, detect_categories_from_query
from src.kb.claim_extractor import ClaimExtractor
from src.kb.concept_map import ConceptMapBuilder
from src.kb.gap_analyzer import GapAnalyzer
from src.kb.grounded_generator import GroundedHypothesisGenerator
from src.ralph.tree import ResearchState
from src.ralph.tree_search_orchestrator import TreeSearchOrchestrator
from src.soul.simulator import Simulator
from src.kb.paper_summarizer import PaperSummarizer
from src.soul.bdi import BDIAgent
from src.soul.collective import SoulCollective
from src.soul.memory import MemorySystem
from src.soul.prompts.visualizer import VisualizerSoul
from src.soul.prompts.persona_forge import PersonaForge
from src.verify.novelty_arxiv import batch_verify_novelty
from src.verify.scoring import ScoringService

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
                if asyncio.iscoroutinefunction(self.callbacks["on_status_change"]):
                    await self.callbacks["on_status_change"](payload)
                else:
                    self.callbacks["on_status_change"](payload)
            except Exception as e:
                print(f"[WARN] Error in status callback: {e}")

    async def _emit_personas(self, personas: list[dict[str, Any]]) -> None:
        """Emit persona roster updates via callback."""
        if "on_personas" in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(self.callbacks["on_personas"]):
                    await self.callbacks["on_personas"](personas)
                else:
                    self.callbacks["on_personas"](personas)
            except Exception as e:
                print(f"[WARN] Error in personas callback: {e}")

    async def _emit_knowledge_update(self, stats: dict[str, Any]) -> None:
        """Emit knowledge base statistics updates."""
        if "on_knowledge_update" in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(self.callbacks["on_knowledge_update"]):
                    await self.callbacks["on_knowledge_update"](stats)
                else:
                    self.callbacks["on_knowledge_update"](stats)
            except Exception as e:
                print(f"[WARN] Error in knowledge callback: {e}")

    async def _emit_trace(self, trace: IterationTrace) -> None:
        """Emit a trace update via callback."""
        if "on_trace" in self.callbacks:
            try:
                if asyncio.iscoroutinefunction(self.callbacks["on_trace"]):
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
        """Phase 1: Fetch papers from arXiv, extract claims, build concept map, identify gaps."""
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

        async with ArxivClient() as client:
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

            seed_limit = min(8, paper_limit)
            seed_papers = await client.search(query, max_results=seed_limit)
            if seed_papers:
                category_counts: dict[str, int] = {}
                for paper in seed_papers:
                    for cat in paper.categories:
                        category_counts[cat] = category_counts.get(cat, 0) + 1
                top_categories = [
                    cat for cat, _ in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:3]
                ]
                for cat in top_categories:
                    if cat not in categories:
                        categories.append(cat)

            papers: list[Any] = list(seed_papers)
            remaining = max(0, paper_limit - len(papers))
            if categories and remaining:
                per_cat = max(4, remaining // len(categories))
                for cat in categories:
                    if len(papers) >= paper_limit:
                        break
                    papers.extend(
                        await client.search_by_category(
                            cat,
                            query=query,
                            max_results=min(per_cat, paper_limit - len(papers)),
                        )
                    )
            if len(papers) < paper_limit:
                papers.extend(await client.search(query, max_results=paper_limit - len(papers)))

            deduped: dict[str, Any] = {}
            for paper in papers:
                if paper.arxiv_id not in deduped:
                    deduped[paper.arxiv_id] = paper
            papers = list(deduped.values())
            if keywords:
                def _score_paper(paper: Any) -> int:
                    title = paper.title.lower()
                    abstract = paper.abstract.lower()
                    score = 0
                    for token in keywords:
                        if token in title:
                            score += 3
                        if token in abstract:
                            score += 1
                    return score

                scored = sorted((( _score_paper(p), p) for p in papers), key=lambda pair: pair[0], reverse=True)
                if scored and scored[0][0] == 0:
                    await self._emit_status(
                        SessionPhase.MAPPING,
                        "No relevant papers found for query; skipping literature pipeline.",
                    )
                    return
                papers = [p for _, p in scored]
            papers = papers[:paper_limit]

            # Populate Knowledge Store
            for p in papers:
                self.paper_store[p.arxiv_id] = p

            await self._emit_knowledge_update({
                "papers_indexed": len(self.paper_store),
                "sources": ["arXiv"],
            })

            if not papers:
                return

            # Phase 1: Global Concept Mapping (Gemini 3 Competition Feature: Long Context)
            await self._emit_status(
                SessionPhase.MAPPING,
                f"Building Global Concept Map from {len(papers)} papers...",
            )
            builder = ConceptMapBuilder(model=self.config.pro_model)
            self.concept_map = await builder.build_global_map_from_abstracts(papers)
            
            await self._emit_knowledge_update({
                "papers_indexed": len(self.paper_store),
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
        if self.iteration == 1 and self.gaps: # Changed 'iteration' to 'self.iteration'
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
             self.hypotheses = []
             for idx, gh in enumerate(self.grounded_hypotheses):
                 # Helper function for rationale
                 def _format_rationale(gh: GroundedHypothesis) -> str:
                     mechanism_str = " → ".join(f"{s.cause} causes {s.effect}" for s in gh.mechanism)
                     rationale_parts = [f"Mechanism: {mechanism_str}"]
                     if gh.null_result:
                         rationale_parts.append(f"Null Result: {gh.null_result}")
                     return ". ".join(rationale_parts)

                 experiments = [e.description for e in gh.suggested_experiments if e.description]
                 if not experiments:
                     experiments = ["Design a controlled experiment to test the claim."]

                 h = Hypothesis(
                     id=gh.id or str(uuid.uuid4())[:8],
                     hypothesis=gh.claim,
                     rationale=_format_rationale(gh),
                     cross_disciplinary_connection=gh.gap_addressed or "Generated from gap",
                     experimental_design=experiments,
                     expected_impact="High - grounded in literature gaps.",
                     novelty_keywords=["Grounded", "Gap-Driven"],
                     iteration=self.iteration, # Changed 'iteration' to 'self.iteration'
                     source_soul=SoulRole.SYNTHESIZER, # MCTS result is a synthesis
                     scores=gh.scores, # Include scores from grounded hypothesis
                 )
                 self.hypotheses.append(h)

             observation += f"\n\nAnalyzed {len(self.gaps)} research gaps using Agentic Tree Search. Generated {len(self.hypotheses)} grounded hypotheses."
             await self._emit_status(
                 SessionPhase.DEBATING,
                 f"Tree Search complete. Found {len(self.hypotheses)} hypotheses.",
             )
             debate_trace = {
                 "hypotheses_generated": len(self.hypotheses),
                 "hypotheses_killed": 0,
                 "hypotheses_final": len(self.hypotheses),
                 "gap_based": True,
             }
             if not self.hypotheses:
                 await self._emit_status(
                     SessionPhase.DEBATING,
                     "Tree search returned no hypotheses; falling back to debate.",
                 )
                 new_hypotheses, debate_trace = await self.collective.run_debate(
                     topic=self.topic,
                     context=self.memory.get_context_for_generation(topic=self.topic),
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
                 context=self.memory.get_context_for_generation(topic=self.topic),
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
                 context=self.memory.get_context_for_generation(topic=self.topic), # Added context
                 mode=new_mode, # Added mode
                 target_hypotheses=self.config.max_hypotheses, # Added target_hypotheses
                 existing_hypotheses=self.hypotheses, # Pass existing hypotheses for refinement
                 weights=persona_weights,
             )
             observation += "\n\nRefined hypotheses through debate."

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
            await batch_verify_novelty(new_hypotheses)

            # Score feasibility and impact
            new_hypotheses = await self.scorer.batch_score(new_hypotheses)

        # Merge with existing hypotheses
        old_count = len(self.hypotheses)
        self.hypotheses = self._merge_hypotheses(self.hypotheses, new_hypotheses)

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
            "constraints": result.constraints.model_dump() if result.constraints else None,
            "personas": self.persona_roster,
            "config": self.config.model_dump(),
        }
        with open(session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
