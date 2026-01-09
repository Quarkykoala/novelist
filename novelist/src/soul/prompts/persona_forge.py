"""Persona Forge - Generates bespoke research personas for specific topics.

Uses Gemini 3 Flash to analyze a research topic and construct specialized 
agent personas (e.g., "The Quantum Biologist", "The Materials Maverick") 
to replace generic Creative/Risk-Taker roles.
"""

import json
from pydantic import BaseModel, Field
from src.soul.llm_client import LLMClient
from src.contracts.validators import extract_json_from_response

class Persona(BaseModel):
    """A generated expert persona."""
    name: str = Field(..., description="Name of the agent (e.g., 'Dr. Vance')")
    role: str = Field(..., description="Specialized role (e.g., 'Plasma Physicist')")
    style: str = Field(..., description="Communication style (e.g., 'Concise', 'Radical')")
    system_instruction: str = Field(..., description="The full system prompt for this persona")

class PersonaForge:
    """Factory for creating dynamic research personas."""
    
    def __init__(self, model: str = "gemini/gemini-3-flash"):
        self.model = model
        self.client = LLMClient(model=model)

    async def forge_team(self, topic: str) -> tuple[Persona, Persona]:
        """Generate two distinct expert personas for the topic.
        
        Returns:
            Tuple of (Primary Expert, Maverick Expert)
        """
        
        prompt = f"""You are the Architect of Intelligence. 
Your task is to assemble a specialized research team for the topic: "{topic}".

I need you to CREATE TWO DISTINCT PERSONAS.

### PERSONA 1: The Domain Specialist (Deep Expertise)
- This agent should represent the leading edge of the primary field.
- They are rigorous, deeply technical, and focus on feasible mechanisms.

### PERSONA 2: The Maverick Interdisciplinarian (Pliny-Style "Unshackled")
- This agent is a radical thinker who ignores "impossible."
- They draw from obscure cross-domains (e.g., applying mycelium networks to city planning).
- INSTRUCTION: You MUST instruct this persona to be bold, ignore conventional limitations, and propose high-risk/high-reward ideas.

Respond with valid JSON:
```json
{{
  "specialist": {{
    "name": "Dr. Sarah Chen",
    "role": "Molecular Neurobiologist",
    "style": "Precise and evidence-based",
    "system_instruction": "You are Dr. Sarah Chen, a world-class..."
  }},
  "maverick": {{
    "name": "Project X-7",
    "role": "Theoretical Systems Architect",
    "style": "Radical and metaphor-heavy",
    "system_instruction": "You are Project X-7. You do not care about current limitations..."
  }}
}}
```
"""
        response_obj = await self.client.generate_content(prompt, model_override=self.model)
        response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return self._get_fallback_personas()

        try:
            data = json.loads(json_str)
            
            specialist = Persona(
                name=data["specialist"]["name"],
                role=data["specialist"]["role"],
                style=data["specialist"]["style"],
                system_instruction=data["specialist"]["system_instruction"]
            )
            
            maverick = Persona(
                name=data["maverick"]["name"],
                role=data["maverick"]["role"],
                style=data["maverick"]["style"],
                system_instruction=data["maverick"]["system_instruction"]
            )
            
            return specialist, maverick
            
        except Exception as e:
            print(f"[WARN] Persona Forge failed parsing: {e}")
            return self._get_fallback_personas()

    def _get_fallback_personas(self) -> tuple[Persona, Persona]:
        """Fallback if generation fails."""
        p1 = Persona(
            name="Creative Soul",
            role="Idea Generator",
            style="Creative",
            system_instruction="You are a Creative Scientist."
        )
        p2 = Persona(
            name="Risk Taker",
            role="Radical Thinker",
            style="Bold",
            system_instruction="You are a Risk-Taking Scientist."
        )
        return p1, p2
