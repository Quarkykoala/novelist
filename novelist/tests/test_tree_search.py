
import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.contracts.schemas import RalphConfig, GroundedHypothesis, IdentifiedGap, GapType, ExtractedClaim, SoulRole
from src.ralph.tree import ResearchState
from src.ralph.tree_search_orchestrator import TreeSearchOrchestrator


class TestTreeSearch(unittest.IsolatedAsyncioTestCase):
    
    async def test_tree_search_flow(self):
        # 1. Setup Mocks
        config = RalphConfig(max_iterations=10)
        
        collective = MagicMock()
        
        generator = MagicMock()
        # Mock generate_batch
        generator.generate_batch = AsyncMock(return_value=[
            [GroundedHypothesis(
                id="h1", claim="Hypothesis 1", mechanism=[], prediction="", 
                null_result="", source_gap_id="g1", source_soul=SoulRole.CREATIVE,
                gap_addressed="Gap 1", supporting_papers=[], contradicting_papers=[], suggested_experiments=[], scores={}
            )],
            [GroundedHypothesis(
                id="h2", claim="Hypothesis 2", mechanism=[], prediction="", 
                null_result="", source_gap_id="g2", source_soul=SoulRole.CREATIVE,
                gap_addressed="Gap 2", supporting_papers=[], contradicting_papers=[], suggested_experiments=[], scores={}
            )],
            [] # 3rd gap yields nothing
        ])
        
        # Mock refine_hypothesis
        generator.refine_hypothesis = AsyncMock(side_effect=lambda hypothesis, instruction, claims: 
            GroundedHypothesis(
                id=f"{hypothesis.id}_refined", 
                claim=f"{hypothesis.claim} Refined", 
                mechanism=[], prediction="", null_result="", 
                source_gap_id=hypothesis.source_gap_id, source_soul=SoulRole.CREATIVE,
                gap_addressed=hypothesis.gap_addressed, supporting_papers=[], contradicting_papers=[], suggested_experiments=[], scores={}
            )
        )

        scorer = MagicMock()
        
        # 2. Initialize Orchestrator
        orchestrator = TreeSearchOrchestrator(config, collective, generator, scorer)
        
        # 3. Create Root State with Gaps
        root_state = ResearchState(
            gaps=[
                IdentifiedGap(id="g1", description="Gap 1", type=GapType.MISSING_CONNECTION, concept_a="A", concept_b="B", potential_value=0.8),
                IdentifiedGap(id="g2", description="Gap 2", type=GapType.CONTRADICTION, concept_a="C", concept_b="D", potential_value=0.9),
                IdentifiedGap(id="g3", description="Gap 3", type=GapType.UNEXPLORED_RANGE, concept_a="E", concept_b="F", potential_value=0.7),
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
            
        self.assertGreater(best_state.depth, 0, "Should have expanded beyond root")
        self.assertTrue(best_state.hypotheses, "Should have hypotheses")
        self.assertTrue("Refined" in best_state.hypotheses[0].claim or "Hypothesis" in best_state.hypotheses[0].claim)
        
        # Check MCTS stats
        print(f"Nodes Expanded: {orchestrator.mcts.nodes_expanded}")
        self.assertGreater(orchestrator.mcts.nodes_expanded, 0)

if __name__ == "__main__":
    unittest.main()
