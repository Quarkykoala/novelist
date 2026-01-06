
import asyncio
import unittest
from pathlib import Path
from src.soul.simulator import Simulator
from src.contracts.schemas import GroundedHypothesis, MechanismStep, PredictionBounds

class TestSimulatorReal(unittest.IsolatedAsyncioTestCase):
    
    async def test_real_simulation(self):
        print("Initializing Simulator...")
        simulator = Simulator(model="gemini-2.0-flash")
        
        # Create a simple hypothesis about exponential growth
        hyp = GroundedHypothesis(
            id="test_sim_1",
            claim="Bacteria population grows exponentially in unlimited resources.",
            mechanism=[
                MechanismStep(cause="Bacteria", effect="Reproduction", evidence_paper=""),
                MechanismStep(cause="Reproduction", effect="More Bacteria", evidence_paper="")
            ],
            prediction="Population doubles every fixed interval until resource cap",
            null_result="Population remains constant",
            gap_addressed="Test Gap",
            supporting_papers=[],
            contradicting_papers=[],
            suggested_experiments=[],
            scores={}
        )
        
        print("Verifying Hypothesis (Generating & Running Code)...")
        result = await simulator.verify_hypothesis(hyp)
        
        print("\n=== SIMULATION RESULT ===")
        print(f"Success: {result.success}")
        print(f"Supports Hypothesis: {result.supports_hypothesis}")
        print(f"Metrics: {result.metrics}")
        print(f"Plot Path: {result.plot_path}")
        print("Code Output:")
        print(result.output_log[:500] + "...")
        
        self.assertTrue(result.success, "Simulation execution failed")
        self.assertIsNotNone(result.code, "No code generated")
        if result.success:
            self.assertTrue(Path(result.plot_path).exists() if result.plot_path else True)

if __name__ == "__main__":
    unittest.main()
