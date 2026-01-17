"""
Simulator Soul — In-Silico Verification of Hypotheses.

This module provides the Simulator class, which translates textual mechanism chains
into executable Python code (ODEs, Agent-Based Models) to verify internal consistency
and plausibility.
"""

import asyncio
import os
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from src.contracts.schemas import GroundedHypothesis, SimulationResult
from src.soul.llm_client import LLMClient

SIMULATION_PROMPT = """You are a Computational Biologist and Python Expert.
Your task is to write a Python script to SIMULATE the proposed mechanism and verify the prediction.

HYPOTHESIS:
"{claim}"

MECHANISM CHAIN:
{mechanism}

PREDICTED OUTCOME:
"{prediction}"

TASK:
1. Model this system quantitatively (e.g., using `scipy.integrate.odeint` for ODEs, or a simple Agent-Based Model).
2. Define reasonable parameters (guesses are fine, but keep them biologically plausible).
3. Run the simulation for a sufficient time duration.
4. Check if the final state matches the PREDICTED OUTCOME.
5. Generate a plot relative to the simulation (e.g. time series of key variables).
6. Save the plot to `{plot_filename}`.
7. Print 'SIMULATION_RESULT: SUCCESS' if the outcome matches the prediction, or 'SIMULATION_RESULT: FAILURE' if not.
8. Print metrics in format 'METRIC: name=value'.

CRITICAL RULES:
- Use ONLY: numpy, scipy, matplotlib, pandas.
- Do NOT assume any external data files exist.
- The script must be self-contained.
- Handle errors gracefully.

Generate the Python code:
"""

class Simulator:
    """Generates and executes simulations for hypotheses."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = LLMClient(model=model)
        self.temp_dir = Path(tempfile.gettempdir()) / "novelist_sims"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def verify_hypothesis(self, hypothesis: GroundedHypothesis) -> SimulationResult:
        """Run an in-silico verification of the hypothesis."""
        
        sim_id = str(uuid.uuid4())[:8]
        plot_filename = f"sim_plot_{sim_id}.png"
        plot_path = self.temp_dir / plot_filename
        
        # 1. Generate Code
        mechanism_text = "\n".join(
            [f"{i+1}. {s.cause} -> {s.effect}" for i, s in enumerate(hypothesis.mechanism)]
        )
        
        prompt = SIMULATION_PROMPT.format(
            claim=hypothesis.claim,
            mechanism=mechanism_text,
            prediction=hypothesis.prediction,
            plot_filename=str(plot_path).replace("\\", "/") # Fix windows paths for python string
        )
        
        response = await self.client.generate_content(prompt)
        
        # Extract code block
        # Handle GenerationResponse object or string
        content = response.content if hasattr(response, "content") else str(response)
        code = self._extract_code(content)
        
        # 2. Execute Code
        result = await self._execute_code(code, sim_id)
        
        # 3. Visual Verification (Gemini 3 Competition Feature: Multimodality)
        final_plot_path = str(plot_path) if plot_path.exists() else None
        visual_reasoning = ""
        
        if final_plot_path:
            visual_reasoning = await self._visual_verify(hypothesis, final_plot_path)
            result["output"] += f"\n\n=== VISUAL VERIFICATION ===\n{visual_reasoning}"
            
            # If visual reasoning is very confident it's a failure, we might override
            if "VERDICT: FAILURE" in visual_reasoning.upper() and "VERDICT: SUCCESS" not in visual_reasoning.upper():
                 result["supports_hypothesis"] = False
            elif "VERDICT: SUCCESS" in visual_reasoning.upper():
                 result["supports_hypothesis"] = True

        return SimulationResult(
            code=code,
            success=result["success"],
            supports_hypothesis=result["supports_hypothesis"],
            output_log=result["output"],
            plot_path=final_plot_path,
            metrics=result["metrics"],
            vision_commentary=visual_reasoning if final_plot_path else None,
            status="complete" if result["success"] else "error"
        )

    async def _visual_verify(self, hypothesis: GroundedHypothesis, plot_path: str) -> str:
        """Use multimodal Gemini to verify the simulation plot."""
        prompt = f"""You are a Senior Research Scientist verifying a simulation result.

HYPOTHESIS:
"{hypothesis.claim}"

PREDICTED OUTCOME:
"{hypothesis.prediction}"

TASK:
1. Analyze the attached simulation plot.
2. Does the visual data support the hypothesis and prediction?
3. Describe the key trends you see in the chart.
4. Provide a final verdict in the format: 'VERDICT: SUCCESS' or 'VERDICT: FAILURE'.

Visual Analysis:"""

        response = await self.client.generate_content(
            prompt, 
            image_paths=[plot_path],
            model_override="gemini-2.0-flash"
        )
        
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    def _extract_code(self, response: str) -> str:
        """Extract python code from markdown."""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        elif "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()

    async def _execute_code(self, code: str, sim_id: str) -> dict[str, Any]:
        """Execute the python script in a separate process."""
        script_path = self.temp_dir / f"sim_script_{sim_id}.py"
        
        try:
            # Use aiofiles for non-blocking I/O
            import aiofiles
            async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
                await f.write(code)
            
            # Run in subprocess
            proc = await asyncio.create_subprocess_exec(
                "python",
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )
            
            stdout, stderr = await proc.communicate()
            output = stdout.decode() + "\n" + stderr.decode()
            
            metrics = {}
            supports_hypothesis = False
            
            # Parse output
            for line in output.splitlines():
                if "SIMULATION_RESULT: SUCCESS" in line:
                    supports_hypothesis = True
                if "METRIC:" in line:
                    try:
                        # Format METRIC: name=value
                        parts = line.split("METRIC:")[1].strip().split("=")
                        if len(parts) == 2:
                            metrics[parts[0].strip()] = float(parts[1].strip())
                    except:
                        pass
            
            return {
                "success": proc.returncode == 0,
                "supports_hypothesis": supports_hypothesis,
                "output": output,
                "metrics": metrics
            }
            
        except Exception as e:
            return {
                "success": False,
                "supports_hypothesis": False,
                "output": f"Execution Error: {str(e)}",
                "metrics": {}
            }
