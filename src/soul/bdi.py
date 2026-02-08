"""BDI (Beliefs, Desires, Intentions) Agent.

Implements the cognitive architecture for rational agency:
- Beliefs: What the agent knows about the world
- Desires: What the agent wants to achieve
- Intentions: What the agent has committed to doing

Based on Rao & Georgeff (1995) and Bratman (1987).
"""

from typing import Any

from src.contracts.schemas import (
    BDIState,
    BeliefStore,
    ConceptMap,
    DesireSet,
    GenerationMode,
    ScoreBlock,
)


class BDIAgent:
    """Agent with Beliefs, Desires, and Intentions.

    The BDI architecture provides:
    - Rational decision making based on beliefs and desires
    - Plan stability through intentions
    - Adaptive behavior when beliefs change
    """

    def __init__(self, state: BDIState | None = None):
        self.state = state or BDIState()

    # =========================================================================
    # Belief Management
    # =========================================================================

    def perceive(self, observation: dict[str, Any]) -> None:
        """Update beliefs based on new observations.

        This is called after each action to update the agent's
        understanding of the world state.
        """
        beliefs = self.state.beliefs

        # Update topic if provided
        if "topic" in observation:
            beliefs.topic = observation["topic"]

        # Update domain tags
        if "domain_tags" in observation:
            beliefs.domain_tags = observation["domain_tags"]

        # Update paper/concept counts
        if "papers_ingested" in observation:
            beliefs.papers_ingested = observation["papers_ingested"]

        if "concept_map" in observation:
            cm: ConceptMap = observation["concept_map"]
            beliefs.concept_map_nodes = len(cm.nodes)
            beliefs.concept_map_edges = len(cm.edges)
            beliefs.identified_gaps = len(cm.gaps)

        # Update score history
        if "novelty_scores" in observation:
            beliefs.last_novelty_scores = observation["novelty_scores"][-5:]

        if "feasibility_scores" in observation:
            beliefs.last_feasibility_scores = observation["feasibility_scores"][-5:]

        # Track stagnation
        if "improved" in observation:
            if observation["improved"]:
                beliefs.stagnation_count = 0
            else:
                beliefs.stagnation_count += 1

        # Record failure modes
        if "failure_mode" in observation:
            mode = observation["failure_mode"]
            if mode and mode not in beliefs.known_failure_modes:
                beliefs.known_failure_modes.append(mode)
                # Keep only recent failure modes
                beliefs.known_failure_modes = beliefs.known_failure_modes[-5:]

    def get_belief(self, key: str) -> Any:
        """Get a specific belief value."""
        return getattr(self.state.beliefs, key, None)

    # =========================================================================
    # Desire Management
    # =========================================================================

    def check_desires(self, current_scores: ScoreBlock, hypothesis_count: int) -> dict[str, bool]:
        """Check which desires are currently satisfied.

        Returns:
            Dict mapping desire names to their satisfaction status
        """
        return self.state.desires.check_met(current_scores, hypothesis_count)

    def get_failing_desire(self, current_scores: ScoreBlock, hypothesis_count: int) -> str | None:
        """Determine which desire is failing the most.

        This is used during deliberation to decide what to focus on.

        Returns:
            Name of the most critical failing desire, or None if all met
        """
        status = self.check_desires(current_scores, hypothesis_count)

        # Priority order for desires
        priority = ["novelty", "feasibility", "impact", "hypothesis_count"]

        for desire in priority:
            if not status.get(desire, True):
                return desire

        return None

    def update_targets(self, **kwargs: float) -> None:
        """Update desire thresholds.

        Example: agent.update_targets(target_novelty=0.8)
        """
        desires = self.state.desires
        for key, value in kwargs.items():
            if hasattr(desires, key):
                setattr(desires, key, value)

    # =========================================================================
    # Intention Management
    # =========================================================================

    def deliberate(self, current_scores: ScoreBlock, hypothesis_count: int) -> str:
        """Deliberate on the current situation and decide what to focus on.

        Returns:
            A reasoning string explaining the deliberation
        """
        failing = self.get_failing_desire(current_scores, hypothesis_count)
        beliefs = self.state.beliefs

        if failing is None:
            return "All desires met. Ready to converge."

        # Build reasoning based on belief state
        reasoning_parts = [f"Desire '{failing}' is not met."]

        if beliefs.stagnation_count > 0:
            reasoning_parts.append(
                f"Stagnation detected: {beliefs.stagnation_count} iterations without improvement."
            )

        if beliefs.known_failure_modes:
            reasoning_parts.append(
                f"Known failure modes to avoid: {', '.join(beliefs.known_failure_modes)}"
            )

        return " ".join(reasoning_parts)

    def plan(self, current_scores: ScoreBlock, hypothesis_count: int) -> GenerationMode:
        """Select a generation mode based on deliberation.

        Uses means-ends reasoning to pick an appropriate strategy.
        """
        failing = self.get_failing_desire(current_scores, hypothesis_count)
        beliefs = self.state.beliefs
        current_mode = self.state.current_mode

        # If stagnating, switch modes
        if beliefs.stagnation_count >= 2:
            return self._select_alternative_mode(current_mode)

        # Check if graph is valid for graph-based modes
        graph_valid = beliefs.concept_map_edges > 0

        # Mode selection based on failing desire
        if failing == "novelty":
            # Low novelty → try gap hunting or contradictions
            if not graph_valid:
                # Without graph, force non-graph modes
                if current_mode == GenerationMode.RANDOM_INJECTION:
                    return GenerationMode.ANALOGY_TRANSFER
                return GenerationMode.RANDOM_INJECTION

            if current_mode == GenerationMode.GAP_HUNT:
                return GenerationMode.CONTRADICTION_HUNT
            elif current_mode == GenerationMode.CONTRADICTION_HUNT:
                return GenerationMode.ANALOGY_TRANSFER
            else:
                return GenerationMode.GAP_HUNT

        elif failing == "feasibility":
            # Low feasibility → be more methodical
            return GenerationMode.CONSTRAINT_RELAX

        elif failing == "cross_domain":
            # Low cross-domain → inject random domains
            return GenerationMode.RANDOM_INJECTION

        # Default: continue with current mode
        return current_mode

    def _select_alternative_mode(self, current: GenerationMode) -> GenerationMode:
        """Select an alternative mode when stuck.

        Avoids exhausted modes and tries something different.
        """
        all_modes = list(GenerationMode)
        exhausted = self.state.exhausted_modes
        graph_valid = self.state.beliefs.concept_map_edges > 0

        # Filter out graph-dependent modes if graph is invalid
        valid_modes = all_modes
        if not graph_valid:
            valid_modes = [
                m for m in all_modes 
                if m not in [GenerationMode.GAP_HUNT, GenerationMode.CONTRADICTION_HUNT]
            ]

        # Find available modes
        available = [m for m in valid_modes if m != current and m not in exhausted]

        if not available:
            # Reset exhausted modes and try again
            self.state.exhausted_modes = []
            available = [m for m in valid_modes if m != current]

        if available:
            # Pick the first available (could randomize)
            chosen = available[0]
            self.state.exhausted_modes.append(current)
            return chosen

        return GenerationMode.RANDOM_INJECTION

    def commit_to_plan(self, mode: GenerationMode, steps: list[str] | None = None) -> None:
        """Commit to a plan by updating intentions.

        Args:
            mode: The generation mode to use
            steps: Optional list of intention steps (uses default if None)
        """
        self.state.current_mode = mode

        if steps:
            self.state.intentions = steps
        else:
            # Default intention sequence
            self.state.intentions = [
                "generate_hypotheses",
                "critique_proposals",
                "revise_hypotheses",
                "verify_novelty",
                "score_final",
            ]

    def pop_intention(self) -> str | None:
        """Pop the next intention from the stack.

        Returns:
            Next intention step, or None if stack is empty
        """
        if self.state.intentions:
            return self.state.intentions.pop(0)
        return None

    def has_intentions(self) -> bool:
        """Check if there are remaining intentions."""
        return len(self.state.intentions) > 0

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self) -> BDIState:
        """Get the current BDI state."""
        return self.state

    def reset(self, topic: str = "") -> None:
        """Reset the agent to initial state.

        Args:
            topic: New research topic
        """
        self.state = BDIState()
        if topic:
            self.state.beliefs.topic = topic

    def snapshot(self) -> dict[str, Any]:
        """Create a snapshot of the current state for logging."""
        return {
            "beliefs": self.state.beliefs.model_dump(),
            "desires": self.state.desires.model_dump(),
            "intentions": self.state.intentions.copy(),
            "current_mode": self.state.current_mode.value,
            "exhausted_modes": [m.value for m in self.state.exhausted_modes],
        }
