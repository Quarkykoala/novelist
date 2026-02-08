"""
SRSH Orchestrator — Stress-Responsive Semantic Hypermutation

This is the main orchestrator that ties together:
1. Stress Detection (find knowledge gaps)
2. Domain Injection (hypermutation)
3. Parallel Streams (isolated exploration)
4. Collision Engine (synthesis)
5. Semantic Distance Scoring (novelty reward)

Inspired by:
- Cellular SOS response (stress → targeted mutation)
- Neuroscience of creativity (parallel streams → collision → epiphany)
- Morphic resonance (consciousness as receiver, not generator)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.contracts.schemas import (
    ConceptMap, ExtractedClaim, GroundedHypothesis, Hypothesis, 
    ScoreBlock, SoulRole
)
from src.kb.stress_detector import StressDetector, StressType, StressZone
from src.kb.domain_injector import DomainInjector
from src.soul.parallel_streams import ParallelStreamOrchestrator, ParallelStreamResult
from src.soul.collision_engine import CollisionEngine, CollisionBatch, CollisionResult
from src.verify.semantic_distance import SemanticDistanceScorer, SemanticDistanceResult
from src.kb.grounded_generator import GroundedHypothesisGenerator


@dataclass
class SRSHConfig:
    """Configuration for SRSH mode."""
    
    enabled: bool = True
    n_agents: int = 3
    iterations_per_agent: int = 5
    n_collisions: int = 2
    semantic_distance_weight: float = 0.3  # Weight in final score
    stress_threshold: float = 0.3  # Minimum stress to trigger SRSH
    
    def __post_init__(self):
        self.n_agents = max(2, min(5, self.n_agents))
        self.iterations_per_agent = max(1, min(10, self.iterations_per_agent))


@dataclass
class SRSHResult:
    """Result from SRSH pipeline."""
    
    stress_zones: list[StressZone]
    parallel_results: list[ParallelStreamResult]
    collision_batches: list[CollisionBatch]
    hypotheses: list[Hypothesis]
    semantic_scores: dict[str, SemanticDistanceResult]
    metrics: dict[str, float] = field(default_factory=dict)


class SRSHOrchestrator:
    """
    Stress-Responsive Semantic Hypermutation Orchestrator.
    
    The cellular SOS response for AI hypothesis generation:
    1. Detect stress (knowledge gaps)
    2. Hypermutate at the gap (parallel exploration with domain injection)
    3. Force collision (synthesis)
    4. Score by semantic distance (reward novelty)
    """
    
    def __init__(
        self,
        generator: GroundedHypothesisGenerator,
        config: Optional[SRSHConfig] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.generator = generator
        self.config = config or SRSHConfig()
        self.status_callback = status_callback or (lambda x: None)
        
        # Initialize components
        self.stress_detector: Optional[StressDetector] = None
        self.domain_injector = DomainInjector()
        self.parallel_orchestrator = ParallelStreamOrchestrator(
            generator=generator,
            n_agents=self.config.n_agents,
            iterations_per_agent=self.config.iterations_per_agent,
        )
        self.collision_engine = CollisionEngine()
        self.semantic_scorer = SemanticDistanceScorer()
    
    async def run(
        self,
        topic: str,
        concept_map: ConceptMap,
        claims: list[ExtractedClaim],
    ) -> SRSHResult:
        """
        Run the full SRSH pipeline.
        
        Args:
            topic: Research topic
            concept_map: Current concept map
            claims: Extracted claims from literature
            
        Returns:
            SRSHResult with all hypotheses and metrics
        """
        self.status_callback("🧬 SRSH: Detecting knowledge stress zones...")
        
        # Phase 1: Detect stress zones
        self.stress_detector = StressDetector(concept_map, claims)
        stress_zones = self.stress_detector.get_top_stress_zones(n=3)
        
        if not stress_zones:
            self.status_callback("No stress zones detected, using fallback")
            # Create default stress zone from topic
            stress_zones = [StressZone(
                stress_type=StressType.MISSING_BRIDGE,
                concepts=[topic],
                intensity=0.5,
                description=f"Explore unexplored connections for: {topic}",
            )]
        
        self.status_callback(f"🔬 Found {len(stress_zones)} stress zones")
        
        # Phase 2-3: Parallel exploration for each stress zone
        all_parallel_results: list[ParallelStreamResult] = []
        all_collision_batches: list[CollisionBatch] = []
        
        for i, zone in enumerate(stress_zones):
            if zone.intensity < self.config.stress_threshold:
                continue
            
            self.status_callback(
                f"🌱 Stream {i+1}/{len(stress_zones)}: "
                f"Exploring '{zone.concepts[0] if zone.concepts else 'gap'}' "
                f"(intensity: {zone.intensity:.2f})"
            )
            
            # Run parallel streams
            parallel_result = await self.parallel_orchestrator.run_parallel(
                stress_zone=zone,
                claims=claims,
                topic=topic,
            )
            all_parallel_results.append(parallel_result)
            
            # Phase 4: Collision
            self.status_callback(f"💥 Collision phase for zone {i+1}...")
            
            collision_batches = await self.collision_engine.multi_collide(
                parallel_result=parallel_result,
                topic=topic,
                n_collisions=self.config.n_collisions,
            )
            all_collision_batches.extend(collision_batches)
        
        # Phase 5: Collect and score all hypotheses
        self.status_callback("📊 Scoring hypotheses by semantic distance...")
        
        hypotheses, semantic_scores = await self._score_and_convert(
            parallel_results=all_parallel_results,
            collision_batches=all_collision_batches,
            topic=topic,
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            stress_zones=stress_zones,
            hypotheses=hypotheses,
            semantic_scores=semantic_scores,
        )
        
        self.status_callback(
            f"✨ SRSH complete: {len(hypotheses)} hypotheses, "
            f"avg distance: {metrics.get('avg_semantic_distance', 0):.2f}"
        )
        
        return SRSHResult(
            stress_zones=stress_zones,
            parallel_results=all_parallel_results,
            collision_batches=all_collision_batches,
            hypotheses=hypotheses,
            semantic_scores=semantic_scores,
            metrics=metrics,
        )
    
    async def _score_and_convert(
        self,
        parallel_results: list[ParallelStreamResult],
        collision_batches: list[CollisionBatch],
        topic: str,
    ) -> tuple[list[Hypothesis], dict[str, SemanticDistanceResult]]:
        """Score all hypotheses and convert to Hypothesis format."""
        import uuid
        
        all_hypotheses = []
        semantic_scores = {}
        
        # Collect all grounded hypotheses
        grounded: list[tuple[GroundedHypothesis, str]] = []  # (hypothesis, source)
        
        # From parallel streams
        for pr in parallel_results:
            for discovery in pr.get_all_discoveries():
                grounded.append((discovery.hypothesis, f"stream:{discovery.agent_id}"))
        
        # From collisions (these are the premium outputs)
        for batch in collision_batches:
            for collision in batch.collisions:
                grounded.append((
                    collision.bridging_hypothesis, 
                    f"collision:{'-'.join(collision.bridged_domains)}"
                ))
        
        # Score and convert each
        for gh, source in grounded:
            if not gh.claim:
                continue
            
            # Get semantic distance score
            try:
                sd_result = await self.semantic_scorer.score_hypothesis(gh.claim)
                hyp_id = str(uuid.uuid4())[:8]
                semantic_scores[hyp_id] = sd_result
            except Exception as e:
                print(f"Semantic scoring failed: {e}")
                sd_result = None
                hyp_id = str(uuid.uuid4())[:8]
            
            # Calculate resonance score
            novelty_boost = sd_result.novelty_boost if sd_result else 1.0
            is_collision = source.startswith("collision:")
            
            # Collisions get bonus + semantic distance bonus
            base_score = 0.7 if is_collision else 0.5
            final_novelty = min(1.0, base_score * novelty_boost)
            
            # Build Hypothesis
            hypothesis = Hypothesis(
                id=hyp_id,
                hypothesis=gh.claim,
                rationale=self._format_rationale(gh, source, sd_result),
                cross_disciplinary_connection=source,
                experimental_design=[
                    f"{e.description} (Timeline: {e.expected_timeline})"
                    for e in gh.suggested_experiments
                ] or ["No explicit experimental design generated."],
                expected_impact="High" if is_collision else "Medium",
                novelty_keywords=self._extract_novelty_keywords(gh, sd_result) or ["Novelty"],
                supporting_papers=gh.supporting_papers or [],
                evidence_trace=getattr(gh, "source_claims", []) or [],
                iteration=0,
                source_soul=SoulRole.SYNTHESIZER,
                scores=ScoreBlock(
                    novelty=final_novelty,
                    feasibility=0.6,
                    impact=0.7 if is_collision else 0.5,
                    cross_domain=0.8 if is_collision else 0.4,
                ),
            )
            all_hypotheses.append(hypothesis)
        
        # Sort by aggregate score
        all_hypotheses.sort(key=lambda h: h.scores.aggregate if h.scores else 0, reverse=True)
        
        return all_hypotheses, semantic_scores
    
    def _format_rationale(
        self,
        gh: GroundedHypothesis,
        source: str,
        sd_result: Optional[SemanticDistanceResult],
    ) -> str:
        """Format rationale with source and semantic distance info."""
        lines = []
        
        if source.startswith("collision:"):
            domains = source.replace("collision:", "").split("-")
            lines.append(f"**Emerged from collision between**: {', '.join(domains)}")
        else:
            agent = source.replace("stream:", "")
            lines.append(f"**Discovered by**: {agent}")
        
        if gh.mechanism:
            lines.append("\n**Mechanism**:")
            for step in gh.mechanism[:3]:
                lines.append(f"  • {step.cause} -> {step.effect}")
        
        if sd_result and sd_result.bridging_pair:
            a, b = sd_result.bridging_pair
            lines.append(f"\n**Semantic bridge**: {a} ↔ {b}")
            lines.append(f"**Novelty boost**: {sd_result.novelty_boost:.2f}x")
        
        return "\n".join(lines)
    
    def _extract_novelty_keywords(
        self,
        gh: GroundedHypothesis,
        sd_result: Optional[SemanticDistanceResult],
    ) -> list[str]:
        """Extract keywords that indicate novelty."""
        keywords = ["SRSH", "Cross-Domain"]
        
        if sd_result:
            keywords.extend(sd_result.concepts[:3])
            if sd_result.bridging_pair:
                a, b = sd_result.bridging_pair
                keywords.append(f"{a}↔{b}")
        
        return keywords[:5]
    
    def _calculate_metrics(
        self,
        stress_zones: list[StressZone],
        hypotheses: list[Hypothesis],
        semantic_scores: dict[str, SemanticDistanceResult],
    ) -> dict[str, float]:
        """Calculate SRSH performance metrics."""
        metrics = {
            "n_stress_zones": len(stress_zones),
            "n_hypotheses": len(hypotheses),
            "avg_stress_intensity": sum(z.intensity for z in stress_zones) / max(1, len(stress_zones)),
        }
        
        # Semantic distance metrics
        if semantic_scores:
            distances = [sr.average_distance for sr in semantic_scores.values()]
            metrics["avg_semantic_distance"] = sum(distances) / len(distances)
            metrics["max_semantic_distance"] = max(distances)
            metrics["min_semantic_distance"] = min(distances)
        
        # Count collision hypotheses
        collision_count = sum(
            1 for h in hypotheses 
            if h.cross_disciplinary_connection and h.cross_disciplinary_connection.startswith("collision:")
        )
        metrics["n_collision_hypotheses"] = collision_count
        metrics["collision_ratio"] = collision_count / max(1, len(hypotheses))
        
        return metrics
