"""
Integration test for the literature-first hypothesis pipeline.

This script tests:
1. Claim extraction from paper abstracts
2. Gap identification from claims
3. Grounded hypothesis generation from gaps
"""

import asyncio
import json
from pathlib import Path

from src.kb.arxiv_client import ArxivClient
from src.kb.claim_extractor import ClaimExtractor
from src.kb.gap_analyzer import GapAnalyzer
from src.kb.grounded_generator import GroundedHypothesisGenerator


async def test_pipeline(topic: str = "lithium-ion battery dendrite suppression"):
    """Run the full pipeline on a topic."""
    
    print(f"\n{'='*60}")
    print(f"TESTING LITERATURE-FIRST PIPELINE")
    print(f"Topic: {topic}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch papers
    print("📚 Step 1: Fetching papers from arXiv...")
    async with ArxivClient() as arxiv:
        papers = await arxiv.search(topic, max_results=5)
    print(f"   Found {len(papers)} papers\n")
    
    if not papers:
        print("❌ No papers found. Exiting.")
        return
    
    # Step 2: Extract claims
    print("🔍 Step 2: Extracting structured claims...")
    extractor = ClaimExtractor()
    all_claims = []
    
    for paper in papers:
        print(f"   Processing: {paper.title[:60]}...")
        claims = await extractor.extract_claims(
            paper_id=paper.arxiv_id,
            title=paper.title,
            abstract=paper.abstract,
        )
        all_claims.extend(claims)
        print(f"   → Extracted {len(claims)} claims")
    
    print(f"\n   Total claims: {len(all_claims)}")
    print(f"   SOTA claims: {len(extractor.get_sota_claims())}")
    print(f"   Unique entities: {len(extractor.get_all_entities())}\n")
    
    # Show sample claims
    print("   Sample claims:")
    for claim in all_claims[:3]:
        print(f"   - [{claim.claim_type.value}] {claim.statement[:80]}...")
        if claim.quantitative_data:
            qd = claim.quantitative_data
            print(f"     → {qd.metric}: {qd.value} {qd.unit}")
    
    # Step 3: Identify gaps
    print("\n🔎 Step 3: Identifying research gaps...")
    analyzer = GapAnalyzer()
    analyzer.add_claims(all_claims)
    gaps = await analyzer.analyze()
    
    print(f"   Found {len(gaps)} gaps")
    
    # Show gaps by type
    from src.contracts.schemas import GapType
    for gap_type in GapType:
        type_gaps = analyzer.get_gaps_by_type(gap_type)
        if type_gaps:
            print(f"   - {gap_type.value}: {len(type_gaps)}")
    
    # Show top gaps
    print("\n   Top gaps:")
    for gap in analyzer.get_high_value_gaps(3):
        print(f"   - [{gap.gap_type.value}] {gap.description[:80]}...")
    
    # Step 4: Generate grounded hypotheses
    print("\n💡 Step 4: Generating grounded hypotheses...")
    generator = GroundedHypothesisGenerator()
    high_value_gaps = analyzer.get_high_value_gaps(3)
    
    hypotheses = await generator.generate_batch(
        gaps=high_value_gaps,
        claims=all_claims,
        max_hypotheses=3,
    )
    
    print(f"   Generated {len(hypotheses)} grounded hypotheses\n")
    
    # Display hypotheses
    for i, hyp in enumerate(hypotheses, 1):
        print(f"\n{'─'*60}")
        print(f"HYPOTHESIS {i}")
        print(f"{'─'*60}")
        print(f"Claim: {hyp.claim}")
        print(f"\nMechanism:")
        for step in hyp.mechanism:
            print(f"  {step.cause} → {step.effect}")
        print(f"\nPrediction: {hyp.prediction}")
        if hyp.prediction_bounds:
            pb = hyp.prediction_bounds
            print(f"  Bounds: {pb.lower_bound}-{pb.upper_bound} {pb.unit}")
            print(f"  Baseline: {pb.baseline_value} {pb.unit}")
        print(f"\nNull Result: {hyp.null_result}")
        print(f"\nGap Addressed: {hyp.gap_addressed[:100]}...")
        print(f"Supporting Papers: {hyp.supporting_papers}")
        if hyp.suggested_experiments:
            print(f"\nExperiment: {hyp.suggested_experiments[0].description[:100]}...")
        print(f"\nWell-formed: {'✅' if hyp.is_well_formed() else '❌'}")
    
    # Save results
    output_dir = Path("sessions/pipeline_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save claims
    claims_data = [
        {
            "paper_id": c.paper_id,
            "type": c.claim_type.value,
            "statement": c.statement,
            "quantitative": c.quantitative_data.model_dump() if c.quantitative_data else None,
        }
        for c in all_claims
    ]
    with open(output_dir / "claims.json", "w") as f:
        json.dump(claims_data, f, indent=2)
    
    # Save gaps
    gaps_data = [
        {
            "type": g.gap_type.value,
            "description": g.description,
            "concept_a": g.concept_a,
            "concept_b": g.concept_b,
            "value": g.potential_value,
        }
        for g in gaps
    ]
    with open(output_dir / "gaps.json", "w") as f:
        json.dump(gaps_data, f, indent=2)
    
    # Save hypotheses
    hyp_data = [
        {
            "claim": h.claim,
            "mechanism": [{"cause": s.cause, "effect": s.effect} for s in h.mechanism],
            "prediction": h.prediction,
            "null_result": h.null_result,
            "gap_addressed": h.gap_addressed,
            "experiments": [e.description for e in h.suggested_experiments],
            "well_formed": h.is_well_formed(),
        }
        for h in hypotheses
    ]
    with open(output_dir / "grounded_hypotheses.json", "w") as f:
        json.dump(hyp_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}\n")
    
    return hypotheses


if __name__ == "__main__":
    asyncio.run(test_pipeline())
