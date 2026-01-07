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
from src.soul.simulator import Simulator
from src.verify.scoring import ScoringService


class TreeSearchOrchestrator:
    """Manages the Agentic Tree Search process for hypothesis generation."""

    def __init__(
        self,
        config: RalphConfig,
        collective: SoulCollective,
        generator: GroundedHypothesisGenerator,
        scorer: ScoringService,
        simulator: Simulator,
    ):
        self.config = config
        self.collective = collective
        self.generator = generator
        self.scorer = scorer
        self.simulator = simulator
        
        # Initialize MCTS engine
        
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

        # Strategy C: Computational Verification (Depth 2+)
        if depth >= 1 and node.state.hypotheses:
            hyp = node.state.hypotheses[0]
            # If mechanism exists and NOT yet simulated
            if hyp.mechanism and not hyp.simulation_result:
                # This is a blocking call (or semi-blocking), MCTS usually fast but Sim is slow.
                # In real world, we might want to do this only for high-promise nodes.
                # For V1, we do it if depth is high enough to warrant checking.
                
                # Check if hypothesis is stable enough to test
                if len(hyp.mechanism) >= 3 and hyp.prediction:
                     try:
                         # Run Simulation
                         result = await self.simulator.verify_hypothesis(hyp)
                         
                         s3 = node.state.copy()
                         simul_hyp = hyp.model_copy()
                         simul_hyp.simulation_result = result
                         
                         # Update feedback
                         outcome = "SUCCESS" if result.success and result.supports_hypothesis else "FAILURE"
                         s3.feedback.append(f"Ran In-Silico Verification: {outcome}")
                         s3.hypotheses = [simul_hyp]
                         
                         # Increase depth? Or same depth but "Verified" state?
                         s3.depth = depth + 1 
                         
                         children.append(SearchNode(
                             state=s3, 
                             parent=node, 
                             action_description=f"Simulate ({outcome})"
                         ))
                     except Exception as e:
                         print(f"Simulation failed to run: {e}")

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
        
        gh = node.state.hypotheses[0]
        # We score grounded hypotheses directly with a heuristic to avoid
        # invalid intermediate Hypothesis objects.
        
        # Scorer requires Hypothesis object
        # Since we haven't updated Scorer to take GroundedHypothesis, we do a lightweight check here or mock it
        # Actually, let's use the validator's logic if available, or just a simple heuristic for V1
        
        # Heuristic: Length of mechanism * novelty
        score = 0.5
        if gh.mechanism:
            score += min(len(gh.mechanism) * 0.1, 0.3) # Reward detailed mechanism
        if gh.suggested_experiments:
            score += 0.1
        
        # Big reward for successful simulation
        if gh.simulation_result:
             if gh.simulation_result.success and gh.simulation_result.supports_hypothesis:
                 score += 0.5  # Huge boost
             elif gh.simulation_result.success:
                 score += 0.1  # Small boost for running code even if result negative (scientific rigor)
             else:
                 score -= 0.1 # Penalty for broken code
            
        return min(score, 1.0)
        
    async def _refine_hypothesis(self, hypothesis: GroundedHypothesis, instruction: str, node: SearchNode) -> GroundedHypothesis | None:
        """Helper to refine a hypothesis using the soul collective."""
        return await self.generator.refine_hypothesis(
            hypothesis=hypothesis,
            instruction=instruction,
            claims=node.state.claims
        )
