"""Visualizer Soul - generates scientific diagrams for hypotheses.

The Visualizer soul takes a textual mechanism and converts it into 
a visual representation (SVG) to enhance understanding and presentation.
"""

from typing import Any

from src.contracts.schemas import Hypothesis, SoulRole
from src.soul.prompts.base import BaseSoul


class VisualizerSoul(BaseSoul):
    """The Visualizer soul generates diagrams."""

    role = SoulRole.SYNTHESIZER  # Borrowing synthesizer role for now
    description = """You are the VISUALIZER in the research collective.
Your role is to translate complex scientific mechanisms into clear, 
elegant SVG diagrams. You think in visual flows and spatial relationships."""

    async def generate_diagram(self, hypothesis: Hypothesis) -> str:
        """Generate an SVG diagram for the hypothesis mechanism."""
        
        prompt = f"""{self.get_persona_prompt()}

HYPOTHESIS:
"{hypothesis.hypothesis}"

MECHANISM:
{hypothesis.rationale}

TASK:
Create a high-quality SVG diagram illustrating this mechanism.
- Use a clean, modern scientific style (white background, crisp lines).
- Use arrows (->) to show causality.
- Use boxes or circles for entities (A, B, C).
- Use color to distinguish different parts of the system (e.g., blue for inputs, green for outcomes).
- The SVG should be responsive (viewBox defined) and self-contained.
- Do not use external images or fonts.
- Size: 600x300 pixels.

Example style: A flowchart where "Drug X" (box) -> "Receptor Y" (circle) -> "Pathway Z" (arrow) -> "Result".

Output ONLY the raw SVG code.
"""
        response_text = await self._call_model_with_retry(prompt)
        
        # Extract SVG if wrapped in code blocks
        if "```svg" in response_text:
            return response_text.split("```svg")[1].split("```")[0].strip()
        elif "```xml" in response_text:
            return response_text.split("```xml")[1].split("```")[0].strip()
        elif "```" in response_text:
            return response_text.split("```")[1].split("```")[0].strip()
            
        # If it looks like SVG, return it
        if "<svg" in response_text:
            start = response_text.find("<svg")
            end = response_text.rfind("</svg>") + 6
            return response_text[start:end]
            
        return ""

    async def generate(self, *args, **kwargs):
        return []
