"""Protocol Generator for creating structured experiment protocols.

Generates detailed, executable experiment protocols from hypotheses,
including materials lists, step-by-step procedures, safety warnings,
and cost estimates.
"""

import json
import uuid
from datetime import datetime
from typing import Any

from src.contracts.schemas import (
    Equipment,
    ExperimentProtocol,
    GroundedHypothesis,
    HazardLevel,
    Hypothesis,
    ProtocolStep,
    Reagent,
    SafetyWarning,
)
from src.soul.llm_client import LLMClient


# Default cost estimates for common reagents (USD per unit)
REAGENT_COSTS = {
    "buffer": 20,
    "enzyme": 150,
    "antibody": 300,
    "primer": 25,
    "plasmid": 100,
    "cell line": 500,
    "media": 30,
    "default": 50,
}

# Default equipment costs
EQUIPMENT_COSTS = {
    "centrifuge": 5000,
    "pcr": 3000,
    "microscope": 10000,
    "spectrophotometer": 8000,
    "pipette": 200,
    "incubator": 4000,
    "default": 100,
}


class ProtocolGenerator:
    """Generates experiment protocols from hypotheses using LLM."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.client = llm_client or LLMClient()

    async def generate_protocol(
        self,
        hypothesis: Hypothesis | GroundedHypothesis,
        detail_level: str = "standard",  # basic, standard, detailed
    ) -> ExperimentProtocol:
        """Generate a complete experiment protocol for a hypothesis.
        
        Args:
            hypothesis: The hypothesis to create a protocol for
            detail_level: Level of detail (basic/standard/detailed)
            
        Returns:
            Complete ExperimentProtocol ready for execution
        """
        # Extract hypothesis text
        if isinstance(hypothesis, GroundedHypothesis):
            hyp_text = hypothesis.hypothesis
            exp_design = hypothesis.experimental_design
            rationale = hypothesis.rationale
            references = hypothesis.supporting_evidence
        else:
            hyp_text = hypothesis.hypothesis
            exp_design = hypothesis.experimental_design
            rationale = hypothesis.rationale
            references = hypothesis.supporting_papers

        # Build prompt for protocol generation
        prompt = self._build_prompt(hyp_text, exp_design, rationale, detail_level)
        
        try:
            response = await self.client.generate_content(prompt)
            protocol_data = self._parse_response(response.content)
            
            # Build structured protocol
            protocol = self._build_protocol(
                hypothesis_id=getattr(hypothesis, 'id', ''),
                hypothesis_text=hyp_text,
                data=protocol_data,
                references=references if isinstance(references, list) else [],
            )
            
            return protocol
            
        except Exception as e:
            print(f"[ERROR] Protocol generation failed: {e}")
            # Return minimal protocol on failure
            return ExperimentProtocol(
                id=str(uuid.uuid4())[:8],
                title=f"Protocol for: {hyp_text[:50]}...",
                hypothesis_id=getattr(hypothesis, 'id', ''),
                objective=hyp_text,
                steps=[
                    ProtocolStep(
                        step_number=i + 1,
                        action=step,
                    )
                    for i, step in enumerate(exp_design[:5] if exp_design else ["To be determined"])
                ],
            )

    def _build_prompt(
        self,
        hypothesis: str,
        exp_design: list[str],
        rationale: str,
        detail_level: str,
    ) -> str:
        """Build the LLM prompt for protocol generation."""
        detail_instructions = {
            "basic": "Provide essential steps only. Keep it concise.",
            "standard": "Include materials, steps, and basic safety info.",
            "detailed": "Provide comprehensive protocol with all materials, detailed steps, timing, temperatures, safety warnings, and troubleshooting tips.",
        }
        
        exp_steps = "\n".join(f"- {step}" for step in (exp_design or ["Not specified"]))
        
        return f"""You are a senior laboratory scientist creating an experiment protocol.

HYPOTHESIS TO TEST:
{hypothesis}

PROPOSED EXPERIMENTAL APPROACH:
{exp_steps}

RATIONALE:
{rationale}

INSTRUCTIONS:
{detail_instructions.get(detail_level, detail_instructions['standard'])}

Generate a structured JSON protocol with these fields:
{{
    "title": "descriptive protocol title",
    "objective": "what this protocol tests",
    "expected_duration": "total time estimate (e.g., '4 hours', '2 days')",
    "difficulty": "beginner/intermediate/advanced",
    "reagents": [
        {{"name": "reagent name", "quantity": "amount with units", "hazard_level": "none/low/moderate/high", "notes": "optional notes"}}
    ],
    "equipment": [
        {{"name": "equipment name", "specifications": "optional specs"}}
    ],
    "steps": [
        {{"action": "what to do", "duration": "time", "temperature": "if relevant", "expected_result": "what should happen", "is_critical": true/false}}
    ],
    "safety_summary": "overall safety considerations",
    "overall_hazard_level": "none/low/moderate/high",
    "success_criteria": ["criterion 1", "criterion 2"],
    "data_collection": ["what data to collect"]
}}

Return ONLY valid JSON, no explanation."""

    def _parse_response(self, content: str) -> dict[str, Any]:
        """Parse LLM response to extract protocol data."""
        # Try to extract JSON from response
        content = content.strip()
        
        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in content
            import re
            match = re.search(r'\{[\s\S]*\}', content)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {}

    def _build_protocol(
        self,
        hypothesis_id: str,
        hypothesis_text: str,
        data: dict[str, Any],
        references: list[str],
    ) -> ExperimentProtocol:
        """Build ExperimentProtocol from parsed data."""
        protocol_id = str(uuid.uuid4())[:8]
        
        # Parse reagents
        reagents = []
        materials_cost = 0.0
        for r in data.get("reagents", []):
            hazard = r.get("hazard_level", "low")
            try:
                hazard_level = HazardLevel(hazard)
            except:
                hazard_level = HazardLevel.LOW
            
            # Estimate cost
            name_lower = r.get("name", "").lower()
            cost = next(
                (v for k, v in REAGENT_COSTS.items() if k in name_lower),
                REAGENT_COSTS["default"]
            )
            
            reagent = Reagent(
                name=r.get("name", "Unknown"),
                quantity=r.get("quantity", "as needed"),
                concentration=r.get("concentration", ""),
                hazard_level=hazard_level,
                estimated_cost_usd=cost,
                notes=r.get("notes", ""),
            )
            reagents.append(reagent)
            materials_cost += cost
        
        # Parse equipment
        equipment = []
        equipment_cost = 0.0
        for e in data.get("equipment", []):
            name_lower = e.get("name", "").lower()
            cost = next(
                (v for k, v in EQUIPMENT_COSTS.items() if k in name_lower),
                0  # Don't add cost for common equipment
            )
            
            equip = Equipment(
                name=e.get("name", "Unknown"),
                specifications=e.get("specifications", ""),
                is_common=cost == 0,
                estimated_cost_usd=cost,
            )
            equipment.append(equip)
            equipment_cost += cost
        
        # Parse steps
        steps = []
        for i, s in enumerate(data.get("steps", [])):
            step = ProtocolStep(
                step_number=i + 1,
                action=s.get("action", ""),
                duration=s.get("duration", ""),
                temperature=s.get("temperature", ""),
                expected_result=s.get("expected_result", ""),
                is_critical=s.get("is_critical", False),
            )
            steps.append(step)
        
        # Overall hazard level
        try:
            overall_hazard = HazardLevel(data.get("overall_hazard_level", "moderate"))
        except:
            overall_hazard = HazardLevel.MODERATE
        
        return ExperimentProtocol(
            id=protocol_id,
            title=data.get("title", f"Protocol for hypothesis {hypothesis_id}"),
            hypothesis_id=hypothesis_id,
            objective=data.get("objective", hypothesis_text),
            background=data.get("background", ""),
            expected_duration=data.get("expected_duration", ""),
            difficulty=data.get("difficulty", "intermediate"),
            reagents=reagents,
            equipment=equipment,
            steps=steps,
            overall_hazard_level=overall_hazard,
            safety_summary=data.get("safety_summary", ""),
            institutional_approval_required=overall_hazard in (HazardLevel.HIGH, HazardLevel.EXTREME),
            success_criteria=data.get("success_criteria", []),
            data_collection=data.get("data_collection", []),
            analysis_methods=data.get("analysis_methods", []),
            estimated_materials_cost_usd=materials_cost,
            estimated_equipment_cost_usd=equipment_cost,
            created_at=datetime.now(),
            references=references,
        )

    def export_to_markdown(self, protocol: ExperimentProtocol) -> str:
        """Export protocol to readable Markdown format."""
        md = f"# {protocol.title}\n\n"
        md += f"**Protocol ID:** {protocol.id}  \n"
        md += f"**Difficulty:** {protocol.difficulty}  \n"
        md += f"**Duration:** {protocol.expected_duration}  \n"
        md += f"**Hazard Level:** {protocol.overall_hazard_level.value}  \n\n"
        
        md += f"## Objective\n{protocol.objective}\n\n"
        
        if protocol.reagents:
            md += "## Materials\n\n### Reagents\n"
            md += "| Reagent | Quantity | Hazard | Est. Cost |\n"
            md += "|---------|----------|--------|----------|\n"
            for r in protocol.reagents:
                md += f"| {r.name} | {r.quantity} | {r.hazard_level.value} | ${r.estimated_cost_usd:.0f} |\n"
            md += "\n"
        
        if protocol.equipment:
            md += "### Equipment\n"
            for e in protocol.equipment:
                md += f"- {e.name}"
                if e.specifications:
                    md += f" ({e.specifications})"
                md += "\n"
            md += "\n"
        
        md += "## Procedure\n\n"
        for step in protocol.steps:
            critical = " ⚠️ CRITICAL" if step.is_critical else ""
            md += f"**Step {step.step_number}:**{critical} {step.action}\n"
            if step.duration:
                md += f"  - Duration: {step.duration}\n"
            if step.temperature:
                md += f"  - Temperature: {step.temperature}\n"
            if step.expected_result:
                md += f"  - Expected: {step.expected_result}\n"
            md += "\n"
        
        if protocol.safety_summary:
            md += f"## Safety\n{protocol.safety_summary}\n\n"
        
        if protocol.success_criteria:
            md += "## Success Criteria\n"
            for c in protocol.success_criteria:
                md += f"- {c}\n"
            md += "\n"
        
        md += f"## Estimated Costs\n"
        md += f"- Materials: ${protocol.estimated_materials_cost_usd:.0f}\n"
        md += f"- Equipment: ${protocol.estimated_equipment_cost_usd:.0f}\n"
        md += f"- **Total:** ${protocol.estimated_materials_cost_usd + protocol.estimated_equipment_cost_usd:.0f}\n"
        
        return md

    def export_to_json_ld(self, protocol: ExperimentProtocol) -> dict[str, Any]:
        """Export protocol in JSON-LD format for machine readability."""
        return {
            "@context": "https://schema.org/",
            "@type": "HowTo",
            "identifier": protocol.id,
            "name": protocol.title,
            "description": protocol.objective,
            "totalTime": protocol.expected_duration,
            "supply": [
                {"@type": "HowToSupply", "name": r.name, "requiredQuantity": r.quantity}
                for r in protocol.reagents
            ],
            "tool": [
                {"@type": "HowToTool", "name": e.name}
                for e in protocol.equipment
            ],
            "step": [
                {
                    "@type": "HowToStep",
                    "position": s.step_number,
                    "text": s.action,
                    "duration": s.duration,
                }
                for s in protocol.steps
            ],
            "estimatedCost": {
                "@type": "MonetaryAmount",
                "currency": "USD",
                "value": protocol.estimated_materials_cost_usd + protocol.estimated_equipment_cost_usd,
            },
        }
