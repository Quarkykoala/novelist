"""Soul Collective - coordinates multi-soul debate and hypothesis refinement.

Orchestrates the 5 souls:
1. Creative + Risk-Taker: Generate proposals (parallel)
2. Skeptic: Critique all proposals
3. Methodical: Enforce protocol completeness
4. Synthesizer: Merge into final set
"""

import asyncio
from typing import Any

from src.contracts.schemas import (
    Critique,
    CritiqueVerdict,
    DialogueEntry,
    GenerationMode,
    Hypothesis,
    SoulRole,
)
from src.soul.prompts.creative import CreativeSoul
from src.soul.prompts.methodical import MethodicalSoul
from src.soul.prompts.risk_taker import RiskTakerSoul
from src.soul.prompts.skeptic import SkepticSoul
from src.soul.prompts.synthesizer import SynthesizerSoul


class SoulCollective:
    """Coordinates the multi-soul debate for hypothesis generation."""

    def __init__(
        self,
        flash_model: str = "gemini-2.0-flash",
        pro_model: str = "gemini-2.0-flash",
    ):
        self.creative = CreativeSoul(model=flash_model)
        self.risk_taker = RiskTakerSoul(model=flash_model)
        self.skeptic = SkepticSoul(model=flash_model)
        self.methodical = MethodicalSoul(model=flash_model)
        self.synthesizer = SynthesizerSoul(model=pro_model)

    async def run_debate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
        target_hypotheses: int = 10,
        existing_hypotheses: list[Hypothesis] | None = None,
        weights: dict[str, float] | None = None,
    ) -> tuple[list[Hypothesis], dict[str, Any]]:
        """Run a full debate round to generate and refine hypotheses.

        Args:
            topic: Research topic
            context: Context from memory system
            mode: Generation mode
            target_hypotheses: Target number for final output
            existing_hypotheses: Optional hypotheses to refine (recorded in trace)
            weights: Optional weights for specialist/maverick (proportional generation)

        Returns:
            Tuple of (final hypotheses, debate trace)
        """
        trace: dict[str, Any] = {
            "phase_results": {},
            "hypotheses_generated": 0,
            "hypotheses_killed": 0,
            "hypotheses_final": 0,
            "dialogue": [],
        }
        if existing_hypotheses:
            trace["existing_hypotheses"] = len(existing_hypotheses)

        # Phase 1: Generate proposals
        proposals = await self._phase_generate(topic, context, mode, weights, target_hypotheses)
        
        # Add Dialogue for Generators
        creative_name = getattr(self.creative, 'custom_name', 'The Specialist')
        maverick_name = getattr(self.risk_taker, 'custom_name', 'The Maverick')
        
        trace["dialogue"].append(DialogueEntry(
            soul=creative_name,
            role=SoulRole.CREATIVE,
            message=f"I have proposed {len([h for h in proposals if h.source_soul == SoulRole.CREATIVE])} hypotheses grounded in the primary literature."
        ))
        trace["dialogue"].append(DialogueEntry(
            soul=maverick_name,
            role=SoulRole.RISK_TAKER,
            message=f"I've injected {len([h for h in proposals if h.source_soul == SoulRole.RISK_TAKER])} radical, cross-disciplinary ideas into the pool."
        ))

        trace["phase_results"]["generate"] = {
            "creative_count": len([h for h in proposals if h.source_soul and h.source_soul.value == "creative"]),
            "risktaker_count": len([h for h in proposals if h.source_soul and h.source_soul.value == "risk_taker"]),
            "total": len(proposals),
        }
        trace["hypotheses_generated"] = len(proposals)

        if not proposals:
            return [], trace

        # Phase 2: Critique all proposals
        critiques, survivors = await self._phase_critique(proposals, topic)
        killed = len(proposals) - len(survivors)
        
        # Add Dialogue for Skeptic
        trace["dialogue"].append(DialogueEntry(
            soul="The Skeptic",
            role=SoulRole.SKEPTIC,
            message=f"Reviewing {len(proposals)} proposals. I have rejected {killed} ideas for being unscientific or repetitive. {len(survivors)} remain for refinement."
        ))
        
        # Identify fatal critiques for the Graveyard
        fatal_critiques = []
        for c in critiques:
            if c.verdict == CritiqueVerdict.FATAL:
                # Find the hypothesis text
                killed_hyp = next((h for h in proposals if h.id == c.hypothesis_id), None)
                if killed_hyp:
                    fatal_critiques.append({
                        "hypothesis": killed_hyp.hypothesis,
                        "reason": "; ".join(c.issues)
                    })

        trace["phase_results"]["critique"] = {
            "proposals": len(proposals),
            "survivors": len(survivors),
            "killed": killed,
            "verdicts": {v.value: len([c for c in critiques if c.verdict == v]) for v in CritiqueVerdict},
        }
        trace["hypotheses_killed"] = killed
        trace["fatal_critiques"] = fatal_critiques

        if not survivors:
            return [], trace

        # Phase 3: Enhance with methodical rigor
        enhanced = await self._phase_methodical(survivors, topic)
        trace["phase_results"]["methodical"] = {
            "input": len(survivors),
            "output": len(enhanced),
        }

        # Phase 4: Synthesize into final set
        final = await self._phase_synthesize(enhanced, topic, target_hypotheses)
        
        # Add Dialogue for Synthesizer
        trace["dialogue"].append(DialogueEntry(
            soul="The Synthesizer",
            role=SoulRole.SYNTHESIZER,
            message=f"I have merged and polished the surviving ideas into a final set of {len(final)} rigorous hypotheses."
        ))

        trace["phase_results"]["synthesize"] = {
            "input": len(enhanced),
            "output": len(final),
        }
        trace["hypotheses_final"] = len(final)

        return final, trace

    async def _phase_generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
        weights: dict[str, float] | None = None,
        target_total: int = 10,
    ) -> list[Hypothesis]:
        """Phase 1: Creative and Risk-Taker generate proportionally to weights."""
        if not weights:
            weights = {"specialist": 0.5, "maverick": 0.5}
        
        # Normalize weights for generators only
        gen_total = weights.get("specialist", 0.5) + weights.get("maverick", 0.5)
        if gen_total == 0:
            weights = {"specialist": 0.5, "maverick": 0.5}
            gen_total = 1.0
            
        creative_ratio = weights.get("specialist", 0.5) / gen_total
        risk_ratio = weights.get("maverick", 0.5) / gen_total
        
        # We want more proposals than target final hypotheses to allow for filtering
        pool_size = max(target_total * 1.5, 6)
        creative_count = max(1, round(pool_size * creative_ratio))
        risk_count = max(1, round(pool_size * risk_ratio))

        # Run in parallel if model supports it, but sequentially for safety/rate limits
        creative_hypotheses = await self.creative.generate(topic, context, mode, count=creative_count)
        risktaker_hypotheses = await self.risk_taker.generate(topic, context, mode, count=risk_count)

        proposals = creative_hypotheses + risktaker_hypotheses

        # Assign unique IDs
        for i, h in enumerate(proposals):
            if not h.id or h.id.startswith(("creative_", "risktaker_", "proposal_")):
                h.id = f"proposal_{i}_{h.source_soul.value if h.source_soul else 'unknown'}"

        return proposals

    async def _phase_critique(
        self,
        proposals: list[Hypothesis],
        topic: str,
    ) -> tuple[list[Critique], list[Hypothesis]]:
        """Phase 2: Skeptic critiques and filters proposals."""
        critiques = await self.skeptic.critique(proposals, topic)

        # Match critiques to proposals
        critique_lookup = {c.hypothesis_id: c for c in critiques}

        survivors: list[Hypothesis] = []
        for h in proposals:
            critique = critique_lookup.get(h.id)
            if critique is None:
                # No critique = assume pass
                survivors.append(h)
            elif critique.verdict != CritiqueVerdict.FATAL:
                # Not fatal = survives
                survivors.append(h)

        return critiques, survivors

    async def _phase_methodical(
        self,
        survivors: list[Hypothesis],
        topic: str,
    ) -> list[Hypothesis]:
        """Phase 3: Methodical enhances experimental designs."""
        return await self.methodical.enhance(survivors, topic)

    async def _phase_synthesize(
        self,
        hypotheses: list[Hypothesis],
        topic: str,
        target_count: int,
    ) -> list[Hypothesis]:
        """Phase 4: Synthesizer merges into final set."""
        return await self.synthesizer.synthesize(hypotheses, topic, target_count)

    async def quick_generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> list[Hypothesis]:
        """Quick generation with just Creative soul (for testing)."""
        return await self.creative.generate(topic, context, mode)
