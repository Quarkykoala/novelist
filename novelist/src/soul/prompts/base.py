"""Base soul class and common utilities for all souls."""

from abc import ABC, abstractmethod
from typing import Any

from src.contracts.schemas import GenerationMode, Hypothesis, SoulRole


class BaseSoul(ABC):
    """Base class for all research souls.

    Each soul has a distinct personality and role in the debate.
    """

    role: SoulRole
    description: str

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        from src.soul.llm_client import LLMClient
        self.client = LLMClient(model=model)
        
        # Usage tracking
        self.total_tokens = 0
        self.total_cost = 0.0

    async def _call_model_with_retry(self, prompt: str) -> str | None:
        """Call the model via the unified client."""
        try:
             response = await self.client.generate_content(prompt)
             
             if hasattr(response, 'usage'):
                 self.total_tokens += response.usage.total_tokens
                 self.total_cost += response.usage.cost_usd
                 return response.content
             elif isinstance(response, str):
                 return response
             
             return None
        except Exception:
             return None

    @abstractmethod
    async def generate(
        self,
        topic: str,
        context: dict[str, Any],
        mode: GenerationMode,
    ) -> list[Hypothesis]:
        """Generate hypotheses based on the soul's perspective.

        Args:
            topic: Research topic
            context: Context from memory system
            mode: Generation mode (gap_hunt, contradiction, etc.)

        Returns:
            List of generated hypotheses
        """
        pass

    def get_persona_prompt(self) -> str:
        """Get the persona description for this soul."""
        return f"""You are the {self.role.value.upper()} soul in a research collective.
{self.description}

Your role is to bring your unique perspective to scientific hypothesis generation.
"""


def format_context_for_prompt(context: dict[str, Any]) -> str:
    """Format context dictionary into a prompt-friendly string."""
    parts = []

    if context.get("gaps"):
        gaps = context["gaps"][:3]
        parts.append("Known research gaps:")
        for gap in gaps:
            parts.append(f"  - {gap.get('node1', '?')} ↔ {gap.get('node2', '?')}")

    if context.get("contradictions"):
        contras = context["contradictions"][:2]
        parts.append("Potential contradictions:")
        for c in contras:
            parts.append(f"  - {c}")

    if context.get("high_freq_entities"):
        entities = context["high_freq_entities"][:10]
        parts.append(f"Key entities: {', '.join(entities)}")

    if context.get("lessons"):
        parts.append("Lessons from previous iterations:")
        for lesson in context["lessons"]:
            parts.append(f"  - {lesson}")

    return "\n".join(parts) if parts else "No prior context available."


HYPOTHESIS_OUTPUT_SCHEMA = """Respond with valid JSON array of hypotheses:
```json
[
  {
    "hypothesis": "One clear, testable sentence stating the hypothesis",
    "rationale": "Why this could be true and why it's scientifically interesting",
    "cross_disciplinary_connection": "What fields are connected and how",
    "experimental_design": ["Step 1: ...", "Step 2: ...", "Step 3: ...", "Step 4: ...", "Step 5: ..."],
    "expected_impact": "What changes if this hypothesis is validated",
    "novelty_keywords": ["keyword1", "keyword2", "keyword3"]
  }
]
```

Requirements:
- Each hypothesis must be falsifiable
- Experimental design must have at least 5 concrete steps
- Novelty keywords should be specific terms for literature search
- Avoid vague terms like "improves" or "enhances" without specifics
"""
