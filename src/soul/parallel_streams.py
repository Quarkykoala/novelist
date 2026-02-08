"""
Parallel Stream Orchestrator — Run isolated agents that cannot see each other.

This is the core of the "multiple independent thought streams" idea.
Each agent:
1. Gets the SAME stress zone / research topic
2. Gets DIFFERENT injected domains (unique perspective)
3. Runs for N iterations in complete ISOLATION
4. Produces a list of "discoveries" (hypotheses)

The isolation is critical — conceptual lineages must develop independently
before they can collide and produce epiphanies.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from src.kb.stress_detector import StressZone
from src.kb.domain_injector import DomainInjector, InjectedDomain, DOMAINS
from src.kb.grounded_generator import GroundedHypothesisGenerator
from src.contracts.schemas import ExtractedClaim, GroundedHypothesis


class AgentPersonality(str, Enum):
    """Different thinking styles for agents."""
    SYNTHESIZER = "synthesizer"  # Connects disparate ideas
    CONTRARIAN = "contrarian"    # Questions assumptions
    OPTIMIZER = "optimizer"       # Maximizes efficiency


@dataclass
class AgentDiscovery:
    """A hypothesis discovered by an isolated agent."""
    
    hypothesis: GroundedHypothesis
    agent_id: str
    iteration: int
    injected_domain: str
    key_insight: str  # One-sentence summary of the discovery


@dataclass
class AgentStream:
    """An isolated agent stream with its own context."""
    
    agent_id: str
    injected_domain: InjectedDomain
    personality: AgentPersonality
    discoveries: list[AgentDiscovery] = field(default_factory=list)
    iterations_completed: int = 0


@dataclass 
class ParallelStreamResult:
    """Result of parallel stream execution."""
    
    streams: list[AgentStream]
    stress_zone: StressZone
    total_discoveries: int
    
    def get_all_discoveries(self) -> list[AgentDiscovery]:
        """Get all discoveries from all streams."""
        return [d for stream in self.streams for d in stream.discoveries]
    
    def get_stream_summaries(self) -> dict[str, str]:
        """Get one-sentence summary of each stream's work."""
        summaries = {}
        for stream in self.streams:
            if stream.discoveries:
                insights = [d.key_insight for d in stream.discoveries[-3:]]
                summaries[stream.agent_id] = "; ".join(insights)
            else:
                summaries[stream.agent_id] = "No discoveries"
        return summaries


class ParallelStreamOrchestrator:
    """
    Run N isolated agents in parallel, each exploring with a different lens.
    
    The key insight: isolation prevents cross-contamination of ideas.
    When streams collide later, the collision is genuine.
    """
    
    def __init__(
        self,
        generator: GroundedHypothesisGenerator,
        n_agents: int = 3,
        iterations_per_agent: int = 5,
    ):
        self.generator = generator
        self.n_agents = n_agents
        self.iterations_per_agent = iterations_per_agent
        self.injector = DomainInjector()
    
    async def run_parallel(
        self,
        stress_zone: StressZone,
        claims: list[ExtractedClaim],
        topic: str = "",
    ) -> ParallelStreamResult:
        """
        Run parallel isolated streams on a stress zone.
        
        Args:
            stress_zone: The knowledge gap to explore
            claims: Supporting claims for context
            topic: Overall research topic
            
        Returns:
            ParallelStreamResult with all stream discoveries
        """
        # 1. Create agent streams with different domains
        streams = self._create_streams(stress_zone)
        
        # 2. Run all streams in parallel
        tasks = [
            self._run_stream(stream, stress_zone, claims, topic)
            for stream in streams
        ]
        completed_streams = await asyncio.gather(*tasks)
        
        # 3. Compile results
        total = sum(len(s.discoveries) for s in completed_streams)
        
        return ParallelStreamResult(
            streams=completed_streams,
            stress_zone=stress_zone,
            total_discoveries=total,
        )
    
    def _create_streams(self, stress_zone: StressZone) -> list[AgentStream]:
        """Create agent streams with unique domains."""
        streams = []
        used_domains: set[str] = set()
        personalities = list(AgentPersonality)
        
        for i in range(self.n_agents):
            # Select domain not yet used
            available_domains = [d for d in DOMAINS if d.name not in used_domains]
            if not available_domains:
                available_domains = DOMAINS
            
            # Weighted selection - prefer diverse domains
            import random
            domain = random.choice(available_domains)
            used_domains.add(domain.name)
            
            # Cycle through personalities
            personality = personalities[i % len(personalities)]
            
            stream = AgentStream(
                agent_id=f"agent_{i}_{domain.name[:3]}",
                injected_domain=domain,
                personality=personality,
            )
            streams.append(stream)
        
        return streams
    
    async def _run_stream(
        self,
        stream: AgentStream,
        stress_zone: StressZone,
        claims: list[ExtractedClaim],
        topic: str,
    ) -> AgentStream:
        """
        Run a single stream in isolation.
        
        Each iteration builds on the stream's own discoveries,
        NOT on discoveries from other streams.
        """
        # Build isolated context for this agent
        domain_context = self._build_domain_context(stream.injected_domain)
        personality_context = self._build_personality_context(stream.personality)
        
        for iteration in range(self.iterations_per_agent):
            try:
                # Generate hypothesis with domain injection
                hypothesis = await self._generate_with_injection(
                    stream=stream,
                    stress_zone=stress_zone,
                    claims=claims,
                    topic=topic,
                    domain_context=domain_context,
                    personality_context=personality_context,
                    iteration=iteration,
                )
                
                if hypothesis:
                    discovery = AgentDiscovery(
                        hypothesis=hypothesis,
                        agent_id=stream.agent_id,
                        iteration=iteration,
                        injected_domain=stream.injected_domain.name,
                        key_insight=self._extract_key_insight(hypothesis),
                    )
                    stream.discoveries.append(discovery)
                
                stream.iterations_completed = iteration + 1
                
            except Exception as e:
                # Log but continue
                print(f"[{stream.agent_id}] Iteration {iteration} failed: {e}")
        
        return stream
    
    async def _generate_with_injection(
        self,
        stream: AgentStream,
        stress_zone: StressZone,
        claims: list[ExtractedClaim],
        topic: str,
        domain_context: str,
        personality_context: str,
        iteration: int,
    ) -> Optional[GroundedHypothesis]:
        """Generate a hypothesis with domain injection."""
        # Build custom gap description with injected domain
        gap_description = self._build_injected_gap(
            stress_zone=stress_zone,
            domain=stream.injected_domain,
            personality=stream.personality,
            topic=topic,
            previous_discoveries=stream.discoveries,
        )
        
        # Use the grounded generator with our enriched gap description
        try:
            hypothesis = await self.generator.generate_from_description(
                gap_description=gap_description,
                claims=claims,
                iteration=iteration,
            )
            return hypothesis
        except Exception as e:
            print(f"[{stream.agent_id}] Generation failed: {e}")
            return None
    
    def _build_injected_gap(
        self,
        stress_zone: StressZone,
        domain: InjectedDomain,
        personality: AgentPersonality,
        topic: str,
        previous_discoveries: list[AgentDiscovery],
    ) -> str:
        """Build a gap description with domain injection."""
        lines = [
            f"RESEARCH TOPIC: {topic}",
            "",
            f"KNOWLEDGE GAP ({stress_zone.stress_type.value}):",
            stress_zone.description,
            f"Concepts involved: {', '.join(stress_zone.concepts)}",
            f"Stress intensity: {stress_zone.intensity:.2f}",
            "",
            "=" * 50,
            f"MANDATORY CROSS-DOMAIN LENS: {domain.name.upper()}",
            "=" * 50,
            domain.description,
            "",
            f"Key concepts you MUST consider:",
        ]
        
        for concept in domain.concepts[:4]:
            lines.append(f"  • {concept}")
        
        lines.append("")
        lines.append(f"Guiding question: {domain.metaphor_prompt}")
        lines.append("")
        
        # Add personality twist
        if personality == AgentPersonality.CONTRARIAN:
            lines.append("THINKING STYLE: Question common assumptions. What if the opposite were true?")
        elif personality == AgentPersonality.OPTIMIZER:
            lines.append("THINKING STYLE: Focus on efficiency and optimization. What's the minimal solution?")
        else:
            lines.append("THINKING STYLE: Synthesize ideas. What unexpected combinations emerge?")
        
        # Add previous discoveries from this stream only (isolation!)
        if previous_discoveries:
            lines.append("")
            lines.append("YOUR PREVIOUS DISCOVERIES (build on these):")
            for d in previous_discoveries[-2:]:  # Last 2 only
                lines.append(f"  → {d.key_insight}")
        
        return "\n".join(lines)
    
    def _build_domain_context(self, domain: InjectedDomain) -> str:
        """Build context string for a domain."""
        return f"{domain.name}: {domain.description}"
    
    def _build_personality_context(self, personality: AgentPersonality) -> str:
        """Build context string for a personality."""
        mapping = {
            AgentPersonality.SYNTHESIZER: "Connect disparate ideas into unified wholes",
            AgentPersonality.CONTRARIAN: "Challenge assumptions and consider opposites",
            AgentPersonality.OPTIMIZER: "Find efficient, minimal solutions",
        }
        return mapping.get(personality, "")
    
    def _extract_key_insight(self, hypothesis: GroundedHypothesis) -> str:
        """Extract a one-sentence key insight from a hypothesis."""
        # Use claim as key insight, truncated
        insight = hypothesis.claim
        if len(insight) > 150:
            insight = insight[:147] + "..."
        return insight
