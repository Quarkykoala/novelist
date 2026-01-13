import json
from src.soul.llm_client import LLMClient
from src.contracts.validators import extract_json_from_response
from src.contracts.schemas import Persona, SoulRole

class PersonaForge:
    """Factory for creating dynamic research personas."""
    
    def __init__(self, model: str = "gemini/gemini-3-flash"):
        self.model = model
        self.client = LLMClient(model=model)

    async def forge_team(self, topic: str) -> dict[str, Persona]:
        """Generate three distinct expert personas for the topic.

        Returns:
            Dict of personas keyed by role (specialist, maverick, skeptic)
        """
        
        prompt = f"""You are the Architect of Intelligence. 
Your task is to assemble a specialized research team for the topic: "{topic}".

I need you to CREATE THREE DISTINCT PERSONAS.

### PERSONA 1: The Domain Specialist (Deep Expertise)
- This agent should represent the leading edge of the primary field.
- They are rigorous, deeply technical, and focus on feasible mechanisms.

### PERSONA 2: The Maverick Interdisciplinarian (Pliny-Style "Unshackled")
- This agent is a radical thinker who ignores "impossible."
- They draw from obscure cross-domains (e.g., applying mycelium networks to city planning).
- INSTRUCTION: You MUST instruct this persona to be bold, ignore conventional limitations, and propose high-risk/high-reward ideas.

### PERSONA 3: The Skeptic Methodologist
- This agent is adversarial and critical, focused on falsifiability, confounds, and experiment design gaps.
- INSTRUCTION: This persona must demand rigor and challenge weak assumptions.

Each persona must include a concise objective and a weight between 0 and 1.

Respond with valid JSON:
```json
{{
  "specialist": {{
    "name": "Dr. Sarah Chen",
    "role": "Molecular Neurobiologist",
    "style": "Precise and evidence-based",
    "objective": "Protect scientific rigor and causal grounding",
    "weight": 0.4,
    "system_instruction": "You are Dr. Sarah Chen, a world-class..."
  }},
  "maverick": {{
    "name": "Project X-7",
    "role": "Theoretical Systems Architect",
    "style": "Radical and metaphor-heavy",
    "objective": "Inject high-risk, high-reward cross-domain ideas",
    "weight": 0.35,
    "system_instruction": "You are Project X-7. You do not care about current limitations..."
  }},
  "skeptic": {{
    "name": "Dr. Elena Ruiz",
    "role": "Experimental Methodologist",
    "style": "Socratic and exacting",
    "objective": "Stress-test assumptions and rigor",
    "weight": 0.25,
    "system_instruction": "You are Dr. Elena Ruiz, a skeptical methodologist..."
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
                id="specialist",
                name=data["specialist"]["name"],
                role=data["specialist"]["role"],
                style=data["specialist"]["style"],
                objective=data["specialist"].get("objective", "Advance grounded, feasible insight."),
                weight=data["specialist"].get("weight", 0.4),
                system_instruction=data["specialist"]["system_instruction"],
                soul_role=SoulRole.CREATIVE,
            )

            maverick = Persona(
                id="maverick",
                name=data["maverick"]["name"],
                role=data["maverick"]["role"],
                style=data["maverick"]["style"],
                objective=data["maverick"].get("objective", "Push bold cross-domain ideas."),
                weight=data["maverick"].get("weight", 0.35),
                system_instruction=data["maverick"]["system_instruction"],
                soul_role=SoulRole.RISK_TAKER,
            )

            skeptic = Persona(
                id="skeptic",
                name=data["skeptic"]["name"],
                role=data["skeptic"]["role"],
                style=data["skeptic"]["style"],
                objective=data["skeptic"].get("objective", "Stress-test assumptions and rigor."),
                weight=data["skeptic"].get("weight", 0.25),
                system_instruction=data["skeptic"]["system_instruction"],
                soul_role=SoulRole.SKEPTIC,
            )

            return {
                "specialist": specialist,
                "maverick": maverick,
                "skeptic": skeptic,
            }
            
        except Exception as e:
            print(f"[WARN] Persona Forge failed parsing: {e}")
            return self._get_fallback_personas()

    async def regenerate_persona(self, topic: str, persona_id: str) -> Persona:
        """Regenerate a single persona by ID."""
        prompt = f"""You are the Architect of Intelligence. 
Your task is to REGENERATE a specialized research persona for the topic: "{topic}".

The requested persona role is: {persona_id} (specialist, maverick, or skeptic).

Respond with valid JSON for ONLY this persona:
```json
{{
  "name": "Dr. Name",
  "role": "Specific Expertise",
  "style": "Communication style",
  "objective": "Primary objective",
  "weight": 0.33,
  "system_instruction": "Full system prompt..."
}}
```
"""
        response_obj = await self.client.generate_content(prompt, model_override=self.model)
        response_text = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
        
        json_str = extract_json_from_response(response_text)
        if not json_str:
            return self._get_fallback_personas()[persona_id]

        try:
            data = json.loads(json_str)
            role_map = {
                "specialist": SoulRole.CREATIVE,
                "maverick": SoulRole.RISK_TAKER,
                "skeptic": SoulRole.SKEPTIC,
            }
            return Persona(
                id=persona_id,
                name=data["name"],
                role=data["role"],
                style=data["style"],
                objective=data.get("objective", ""),
                weight=data.get("weight", 0.33),
                system_instruction=data["system_instruction"],
                soul_role=role_map.get(persona_id),
            )
        except Exception:
            return self._get_fallback_personas()[persona_id]

    def _get_fallback_personas(self) -> dict[str, Persona]:
        """Fallback if generation fails."""
        p1 = Persona(
            id="specialist",
            name="Creative Soul",
            role="Idea Generator",
            style="Creative",
            objective="Generate grounded ideas tied to the topic.",
            weight=0.4,
            system_instruction="You are a Creative Scientist.",
            soul_role=SoulRole.CREATIVE,
        )
        p2 = Persona(
            id="maverick",
            name="Risk Taker",
            role="Radical Thinker",
            style="Bold",
            objective="Push high-risk, high-reward hypotheses.",
            weight=0.35,
            system_instruction="You are a Risk-Taking Scientist.",
            soul_role=SoulRole.RISK_TAKER,
        )
        p3 = Persona(
            id="skeptic",
            name="Skeptic",
            role="Methodologist",
            style="Socratic",
            objective="Challenge weak assumptions and demand rigor.",
            weight=0.25,
            system_instruction="You are a Skeptical Methodologist.",
            soul_role=SoulRole.SKEPTIC,
        )
        return {"specialist": p1, "maverick": p2, "skeptic": p3}
