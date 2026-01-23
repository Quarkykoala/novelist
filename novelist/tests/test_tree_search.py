from unittest.mock import AsyncMock, MagicMock
import asyncio
import unittest

from src.contracts.schemas import (
    RalphConfig, GroundedHypothesis, IdentifiedGap, GapType, 
    ExtractedClaim, SoulRole, MechanismStep, SimulationResult
)
from src.ralph.tree import ResearchState
from src.ralph.tree_search_orchestrator import TreeSearchOrchestrator


class TestTreeSearch(unittest.IsolatedAsyncioTestCase):
    
    async def test_tree_search_flow(self):
        # 1. Setup Mocks
        config = RalphConfig(max_iterations=10)
        
        collective = MagicMock()
        
        generator = MagicMock()
        # Mock generate_batch
        generator.generate_batch = AsyncMock(side_effect=[
            [GroundedHypothesis(
                id="h1", claim="Hypothesis 1 is a valid hypothesis statement that is long enough.",
                mechanism=[
                    MechanismStep(cause="A", effect="B", evidence_paper=""),
                    MechanismStep(cause="B", effect="C", evidence_paper=""),
                    MechanismStep(cause="C", effect="D", evidence_paper="")
                ], 
                prediction="Outcome D increases",
                null_result="The null result would show no effect.", source_soul=SoulRole.CREATIVE,
                gap_addressed="Gap 1 description", supporting_papers=[], contradicting_papers=[], suggested_experiments=[], scores={}
            )],
            [], # 2nd gap yields nothing
            []  # 3rd gap yields nothing
        ])
        
        # Mock refine_hypothesis
        generator.refine_hypothesis = AsyncMock(side_effect=lambda hypothesis, instruction, claims: 
            GroundedHypothesis(
                id=f"{hypothesis.id}_refined", 
                claim=f"{hypothesis.claim} Refined", 
                mechanism=hypothesis.mechanism, # Keep mechanism
                prediction=hypothesis.prediction,
                null_result="The null result would show no effect.",
                source_soul=SoulRole.CREATIVE,
                gap_addressed=hypothesis.gap_addressed, supporting_papers=[], contradicting_papers=[], suggested_experiments=[], scores={}
            )
        )
        
        scorer = MagicMock()
        
        simulator = MagicMock()
        simulator.verify_hypothesis = AsyncMock(return_value=SimulationResult(
            code="print('SUCCESS')",
            success=True,
            supports_hypothesis=True,
            output_log="SIMULATION_RESULT: SUCCESS",
            metrics={"error": 0.01}
        ))
        
        # 2. Initialize Orchestrator
        orchestrator = TreeSearchOrchestrator(config, collective, generator, scorer, simulator)
        
        # 3. Create Root State with Gaps
        root_state = ResearchState(
            gaps=[
                IdentifiedGap(id="g1", description="Gap 1 description must be long enough too.", gap_type=GapType.MISSING_CONNECTION, concept_a="A", concept_b="B", potential_value="High Value"),
            ],
            depth=0
        )
        
        # 4. Run Search
        print("Starting Search...")
        best_state = await orchestrator.run_search(root_state)
        
        # 5. Assertions
        print(f"Best State Depth: {best_state.depth}")
        if best_state.hypotheses:
            print(f"Best Hypothesis: {best_state.hypotheses[0].claim}")
            if best_state.hypotheses[0].simulation_result:
                print("Simulation Result Present!")
            
        self.assertGreater(best_state.depth, 0, "Should have expanded beyond root")
        self.assertTrue(best_state.hypotheses, "Should have hypotheses")
        
        # Check if simulation was attempted
        # Either the best state has it, or valid simulation call was made
        simulator.verify_hypothesis.assert_awaited()
        
        # Check MCTS stats
        print(f"Nodes Expanded: {orchestrator.mcts.nodes_expanded}")
        self.assertGreater(orchestrator.mcts.nodes_expanded, 0)

if __name__ == "__main__":
    unittest.main()
