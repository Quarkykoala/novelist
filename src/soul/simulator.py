"""
Simulator Soul — In-Silico Verification of Hypotheses.

This module provides the Simulator class, which translates textual mechanism chains
into executable Python code (ODEs, Agent-Based Models) to verify internal consistency
and plausibility.

Enhanced with retry support and code validation for improved reliability.
"""

import asyncio
import sys
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
    """Generates and executes simulations for hypotheses.
    
    Features:
    - Code syntax validation before execution
    - Automatic retry with error feedback (up to 3 attempts)
    - Visual verification using multimodal Gemini
    """

    def __init__(self, model: str = "gemini-2.0-flash"):
        self.client = LLMClient(model=model)
        self.temp_dir = Path(tempfile.gettempdir()) / "novelist_sims"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def verify_hypothesis(
        self, hypothesis: GroundedHypothesis, max_retries: int = 3
    ) -> SimulationResult:
        """Run an in-silico verification of the hypothesis with retry support.
        
        Args:
            hypothesis: The hypothesis to verify
            max_retries: Maximum number of attempts if code fails (default 3)
        
        Returns:
            SimulationResult with retry_count and validation_errors populated
        """
        
        sim_id = str(uuid.uuid4())[:8]
        plot_filename = f"sim_plot_{sim_id}.png"
        plot_path = self.temp_dir / plot_filename
        
        mechanism_text = "\n".join(
            [f"{i+1}. {s.cause} -> {s.effect}" for i, s in enumerate(hypothesis.mechanism)]
        )
        
        previous_error = ""
        validation_errors: list[str] = []
        code = ""
        result: dict[str, Any] = {"success": False, "supports_hypothesis": False, "output": "", "metrics": {}}
        attempt = 0
        
        for attempt in range(max_retries):
            # Generate code (with error feedback on retries)
            code = await self._generate_code(
                hypothesis=hypothesis,
                mechanism_text=mechanism_text,
                plot_path=str(plot_path).replace("\\", "/"),
                previous_error=previous_error if attempt > 0 else None,
            )
            
            # Validate syntax before execution
            validation = self._validate_code(code)
            if validation["error"]:
                validation_errors.append(f"Attempt {attempt + 1}: {validation['error']}")
                previous_error = f"Syntax Error: {validation['error']}"
                continue
            
            # Execute the validated code
            result = await self._execute_code(code, sim_id)
            
            if result["success"]:
                # Success! Break out of retry loop
                break
            else:
                # Prepare error for next attempt
                error_log = result["output"][-500:] if len(result["output"]) > 500 else result["output"]
                previous_error = f"Execution failed:\n{error_log}"
        
        # Visual Verification (if plot was generated)
        final_plot_path = str(plot_path) if plot_path.exists() else None
        visual_reasoning = ""
        
        if final_plot_path:
            visual_reasoning = await self._visual_verify(hypothesis, final_plot_path)
            result["output"] += f"\n\n=== VISUAL VERIFICATION ===\n{visual_reasoning}"
            
            # Visual reasoning can override verdict
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
            status="complete" if result["success"] else "error",
            retry_count=attempt,
            validation_errors=validation_errors,
        )

    async def _generate_code(
        self,
        hypothesis: GroundedHypothesis,
        mechanism_text: str,
        plot_path: str,
        previous_error: str | None = None,
    ) -> str:
        """Generate simulation code, optionally with error feedback for retries."""
        
        base_prompt = SIMULATION_PROMPT.format(
            claim=hypothesis.claim,
            mechanism=mechanism_text,
            prediction=hypothesis.prediction,
            plot_filename=plot_path,
        )
        
        if previous_error:
            base_prompt += f"\n\n=== PREVIOUS ATTEMPT FAILED ===\n{previous_error}\n\nFix the error and generate corrected code:"
        
        response = await self.client.generate_content(base_prompt)
        if hasattr(response, "content"):
            response = response.content
        else:
            response = str(response)
        
        return self._extract_code(response)

    def _validate_code(self, code: str) -> dict[str, str | None]:
        """Validate Python code syntax before execution.
        
        Returns dict with 'valid' bool and 'error' message if invalid.
        """
        try:
            compile(code, "<simulation>", "exec")
            return {"valid": True, "error": None}
        except SyntaxError as e:
            return {"valid": False, "error": f"Line {e.lineno}: {e.msg}"}
        except Exception as e:
            return {"valid": False, "error": str(e)}

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
            def write_script():
                with open(script_path, "w", encoding="utf-8") as f:
                    f.write(code)

            await asyncio.to_thread(write_script)
            
            # Run in subprocess
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.temp_dir)
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return {
                    "success": False,
                    "supports_hypothesis": False,
                    "output": "Execution Timeout: simulation exceeded 60 seconds",
                    "metrics": {},
                }
            output = stdout.decode() + "\n" + stderr.decode()
            
            metrics: dict[str, float] = {}
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
