
import asyncio
from unittest.mock import patch, MagicMock
from src.soul.llm_client import LLMClient
from src.kb.gap_analyzer import GapAnalyzer
from src.kb.grounded_generator import GroundedHypothesisGenerator
from src.contracts.schemas import GenerationResponse, TokenUsage, IdentifiedGap, GapType, ExtractedClaim, ClaimType

async def test_generation_pipeline():
    print("Testing Generation Pipeline...")
    
    # Mock LLMClient to return GenerationResponse
    async def mock_generate(*args, **kwargs):
        prompt = args[1] if len(args) > 1 else kwargs.get('prompt', '')
        
        if "gap" in prompt.lower() or "claims from papers" in prompt.lower():
            # Mock GapAnalyzer response
            return GenerationResponse(
                content='''```json
                [{"gap_type": "MISSING_CONNECTION", "description": "This is a sufficiently long description of a gap to pass validation (>20 chars).", "concept_a": "A", "concept_b": "B", "potential_value": "High", "difficulty": "Low"}]
                ```''',
                usage=TokenUsage(),
                model_name="mock",
                provider="mock"
            )
        elif "hypothesis" in prompt.lower():
            # Mock GroundedGenerator response
            return GenerationResponse(
                content='''```json
                {
                    "claim": "If we apply this specific intervention X, then we expect result Y because of mechanism Z.",
                    "mechanism": [{"cause": "Cause A", "effect": "Effect B", "evidence_paper": "1234.5678"}],
                    "prediction": "We specifically expect to observe a significant increase in the target metric.",
                    "null_result": "If we observe no significant change in the metric, the hypothesis is rejected.",
                    "supporting_papers": [],
                    "suggested_experiments": []
                }
                ```''',
                usage=TokenUsage(),
                model_name="mock",
                provider="mock"
            )
        return GenerationResponse(content="{}", usage=TokenUsage(), model_name="mock", provider="mock")

    with patch.object(LLMClient, 'generate_content', side_effect=mock_generate):
        # 1. Test GapAnalyzer
        analyzer = GapAnalyzer()
        claims = [ExtractedClaim(
            paper_id="1", 
            claim_type=ClaimType.RESULT, 
            statement="This is a sufficiently long statement for the extracted claim test data.", 
            evidence="E", 
            entities_mentioned=[]
        )]
        analyzer.add_claims(claims)
        gaps = await analyzer.analyze()
        
        print(f"Gaps found: {len(gaps)}")
        assert len(gaps) == 1
        assert gaps[0].description == "This is a sufficiently long description of a gap to pass validation (>20 chars)."
        
        # 2. Test GroundedGenerator
        generator = GroundedHypothesisGenerator()
        hypotheses = await generator.generate_batch(gaps, claims, max_hypotheses=1)
        
        print(f"Hypotheses generated: {len(hypotheses)}")
        print(f"Actual claim: '{hypotheses[0].claim}'")
        assert len(hypotheses) == 1
        assert hypotheses[0].claim == "If we apply this specific intervention X, then we expect result Y because of mechanism Z."
        
    print("Pipeline Test Passed!")

if __name__ == "__main__":
    asyncio.run(test_generation_pipeline())
