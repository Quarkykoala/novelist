"""
Tree Search Orchestrator for Novelist 2.0.

This module Bridges the abstract MCTS algorithm with the concrete domain logic
of Novelist, defining how to expand research nodes (using souls) and evaluate
them (using critics).
"""

import asyncio
from typing import Any

from src.contracts.schemas import RalphConfig, GroundedHypothesis
from src.ralph.mcts import AgenticTreeSearch
from src.ralph.tree import ResearchState, SearchNode
from src.soul.collective import SoulCollective
from src.kb.grounded_generator import GroundedHypothesisGenerator
from src.verify.scoring import ScoringService


class TreeSearchOrchestrator:
    """Manages the Agentic Tree Search process for hypothesis generation."""

    def __init__(
        self,
        config: RalphConfig,
        collective: SoulCollective,
        generator: GroundedHypothesisGenerator,
        scorer: ScoringService,
    ):
        self.config = config
        self.collective = collective
        self.generator = generator
        self.scorer = scorer
        
        # Initialize MCTS engine
        self.mcts = AgenticTreeSearch(
            config=config,
            expand_fn=self._expand_state,
            evaluate_fn=self._evaluate_state,
        )

    async def run_search(self, root_state: ResearchState) -> ResearchState:
        """Run the tree search and return the best resulting state."""
        best_node = await self.mcts.search(
            initial_state=root_state,
            iterations=self.config.max_iterations  # Use config iterations for MCTS steps
        )
        return best_node.state

    async def _expand_state(self, node: SearchNode) -> list[SearchNode]:
        """
        Generate children nodes by applying research actions.
        
        Strategy:
        - Root (Depth 0): Generate hypotheses from high-value gaps.
        - Depth 1+: Apply refinement strategies ("Add Mechanism", "Critique & Fix").
        """
        depth = node.state.depth
        children = []

        # Strategy A: Expansion from Gaps (Root)
        if depth == 0 and node.state.gaps:
            # Pick top gaps
            gaps = node.state.gaps[:3] # Top 3 gaps
            
            # Generate hypotheses for each gap (Parallel)
            tasks = []
            for gap in gaps:
                tasks.append(self.generator.generate_batch(
                    gaps=[gap],
                    claims=node.state.claims,
                    max_hypotheses=1, # One strong hypothesis per gap branch
                    iteration=1
                ))
            
            results = await asyncio.gather(*tasks)
            
            for i, hyps in enumerate(results):
                if hyps:
                    child_state = node.state.copy()
                    child_state.hypotheses = hyps
                    child_state.focus_topic = f"Gap: {gaps[i].description[:50]}..."
                    child_state.depth = depth + 1
                    child_state.feedback.append("Initial generation from gap")
                    
                    children.append(SearchNode(
                        state=child_state,
                        parent=node,
                        action_description=f"Explore Gap {i+1}"
                    ))
            return children

        # Strategy B: Refinement (Depth 1+)
        if depth >= 1 and node.state.hypotheses:
            # Create variations:
            # 1. Mechanism Deep Dive (Methodical)
            # 2. Riskier Variant (Risk Taker)
            # 3. Critique Fix (Skeptic -> Creative)
            
            # For V1, let's just do "Mechanism Deep Dive" via Creative Soul re-writing
            # We will use the Soul Collective to refine.
            
            hyp = node.state.hypotheses[0] # Focus on the primary hypothesis
            
            # Action 1: Deepen Mechanism
            refined_hyp = await self._refine_hypothesis(hyp, "Deepen Mechanism", node)
            if refined_hyp:
                s1 = node.state.copy()
                s1.hypotheses = [refined_hyp]
                s1.depth = depth + 1
                s1.feedback.append("Deepened mechanism chain")
                children.append(SearchNode(state=s1, parent=node, action_description="Deepen Mechanism"))

            # Action 2: Add Experimental Detail
            refined_hyp_Exp = await self._refine_hypothesis(hyp, "Add Experiments", node)
            if refined_hyp_Exp:
                s2 = node.state.copy()
                s2.hypotheses = [refined_hyp_Exp]
                s2.depth = depth + 1
                s2.feedback.append("Added experimental detail")
                children.append(SearchNode(state=s2, parent=node, action_description="Deepen Experiments"))

        return children

    async def _evaluate_state(self, node: SearchNode) -> float:
        """
        Evaluate the quality of a research state.
        
        Uses the ScoringService to get novelty/feasibility scores.
        """
        if not node.state.hypotheses:
            return 0.0
            
        # Score the specific hypotheses in this node
        # We need to convert GroundedHypothesis to standard Hypothesis for the Scorer
        # OR update Scorer to handle GroundedHypothesis (better)
        # For now, let's just assume we can get a quality score from the GroundedHypothesis itself if available,
        # or use the Scorer.
        
        # Use existing Scorer
        # Convert first to check
        from src.contracts.schemas import Hypothesis
        
        gh = node.state.hypotheses[0]
        h = Hypothesis(
             id=gh.id,
             hypothesis=gh.claim,
             rationale=f"Mechanism: {len(gh.mechanism)} steps", 
             cross_disciplinary_connection="",
             experimental_design=[],
             expected_impact="",
             novelty_keywords=[],
             iteration=0
        )
        
        # Scorer requires Hypothesis object
        # Since we haven't updated Scorer to take GroundedHypothesis, we do a lightweight check here or mock it
        # Actually, let's use the validator's logic if available, or just a simple heuristic for V1
        
        # Heuristic: Length of mechanism * novelty
        score = 0.5
        if gh.mechanism:
            score += min(len(gh.mechanism) * 0.1, 0.3) # Reward detailed mechanism
        if gh.suggested_experiments:
            score += 0.1
            
        return min(score, 1.0)
        
    async def _refine_hypothesis(self, hypothesis: GroundedHypothesis, instruction: str, node: SearchNode) -> GroundedHypothesis | None:
        """Helper to refine a hypothesis using the soul collective."""
        return await self.generator.refine_hypothesis(
            hypothesis=hypothesis,
            instruction=instruction,
            claims=node.state.claims
        )
