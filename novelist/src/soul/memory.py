"""Memory systems for the research soul.

Implements three types of memory:
- Episodic: Iteration-by-iteration summaries (append-only log)
- Semantic: Concept map and paper knowledge (reference to KB)
- Working: Current iteration context (transient)
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.contracts.schemas import (
    ConceptMap,
    GenerationMode,
    Hypothesis,
    IterationTrace,
)


@dataclass
class EpisodicMemory:
    """Append-only log of iteration episodes.

    Each episode records what happened in an iteration,
    enabling the agent to learn from past experience.
    """

    episodes: list[IterationTrace] = field(default_factory=list)
    max_episodes: int = 50  # Keep last N episodes

    def record(self, trace: IterationTrace) -> None:
        """Record a new episode."""
        self.episodes.append(trace)
        # Trim if too many
        if len(self.episodes) > self.max_episodes:
            self.episodes = self.episodes[-self.max_episodes:]

    def get_recent(self, n: int = 5) -> list[IterationTrace]:
        """Get the N most recent episodes."""
        return self.episodes[-n:]

    def get_by_iteration(self, iteration: int) -> IterationTrace | None:
        """Get episode for a specific iteration."""
        for ep in self.episodes:
            if ep.iteration == iteration:
                return ep
        return None

    def summarize_lessons(self) -> list[str]:
        """Extract lessons learned from recent episodes.

        Returns patterns of what worked and what didn't.
        """
        lessons: list[str] = []
        recent = self.get_recent(5)

        if not recent:
            return lessons

        # Check for modes that improved novelty
        for ep in recent:
            if ep.avg_novelty > 0.6:
                lessons.append(f"Mode {ep.mode_used.value} achieved high novelty ({ep.avg_novelty:.2f})")

        # Check for stagnation patterns
        novelties = [ep.avg_novelty for ep in recent]
        if len(novelties) >= 3 and novelties[-1] <= novelties[-3]:
            lessons.append("Novelty not improving - consider mode switch")

        return lessons

    def save(self, path: Path) -> None:
        """Save episodes to file."""
        data = [ep.model_dump() for ep in self.episodes]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "EpisodicMemory":
        """Load episodes from file."""
        memory = cls()
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                memory.episodes = [IterationTrace.model_validate(ep) for ep in data]
        return memory


@dataclass
class WorkingMemory:
    """Transient memory for the current iteration.

    Holds context that's relevant for the current step
    but doesn't need to persist across iterations.
    """

    # Current iteration number
    iteration: int = 0

    # Current generation mode
    mode: GenerationMode = GenerationMode.GAP_HUNT

    # Hypotheses being worked on this iteration
    active_hypotheses: list[Hypothesis] = field(default_factory=list)

    # Pending critiques
    pending_critiques: list[dict[str, Any]] = field(default_factory=list)

    # Current focus area (from concept map gaps)
    focus_gap: dict[str, Any] | None = None

    # Tokens used this iteration
    tokens_used: int = 0

    # Errors encountered
    errors: list[str] = field(default_factory=list)

    def clear(self) -> None:
        """Clear working memory for new iteration."""
        self.active_hypotheses = []
        self.pending_critiques = []
        self.focus_gap = None
        self.tokens_used = 0
        self.errors = []

    def add_hypothesis(self, hypothesis: Hypothesis) -> None:
        """Add a hypothesis to working memory."""
        self.active_hypotheses.append(hypothesis)

    def add_critique(self, critique: dict[str, Any]) -> None:
        """Add a critique to pending list."""
        self.pending_critiques.append(critique)

    def record_error(self, error: str) -> None:
        """Record an error for debugging."""
        self.errors.append(error)


@dataclass
class SemanticMemory:
    """Reference to long-term knowledge.

    Points to the concept map and paper summaries
    without duplicating them in memory.
    """

    concept_map: ConceptMap | None = None
    paper_count: int = 0
    domain_tags: list[str] = field(default_factory=list)

    def update(self, concept_map: ConceptMap, paper_count: int, domains: list[str]) -> None:
        """Update semantic memory with new knowledge."""
        self.concept_map = concept_map
        self.paper_count = paper_count
        self.domain_tags = domains

    def get_gaps(self) -> list[dict[str, Any]]:
        """Get identified gaps from concept map."""
        if self.concept_map:
            return self.concept_map.gaps
        return []

    def get_contradictions(self) -> list[dict[str, Any]]:
        """Get identified contradictions from concept map."""
        if self.concept_map:
            return self.concept_map.contradictions
        return []

    def get_high_frequency_entities(self, min_freq: int = 3) -> list[str]:
        """Get entities that appear frequently in papers."""
        if not self.concept_map:
            return []
        return [
            node.name
            for node in self.concept_map.nodes
            if node.frequency >= min_freq
        ]


class MemorySystem:
    """Combined memory system for the research soul."""

    def __init__(self):
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()
        self.working = WorkingMemory()

    def start_iteration(self, iteration: int, mode: GenerationMode) -> None:
        """Initialize memory for a new iteration."""
        self.working.clear()
        self.working.iteration = iteration
        self.working.mode = mode

    def end_iteration(self, trace: IterationTrace) -> None:
        """Finalize iteration and record to episodic memory."""
        self.episodic.record(trace)

    def get_context_for_generation(self) -> dict[str, Any]:
        """Get relevant context for hypothesis generation.

        Compiles information from all memory types.
        """
        return {
            "iteration": self.working.iteration,
            "mode": self.working.mode.value,
            "gaps": self.semantic.get_gaps()[:3],  # Top 3 gaps
            "contradictions": self.semantic.get_contradictions()[:2],
            "high_freq_entities": self.semantic.get_high_frequency_entities()[:10],
            "lessons": self.episodic.summarize_lessons(),
            "recent_avg_novelty": self._get_recent_avg("avg_novelty"),
            "recent_avg_feasibility": self._get_recent_avg("avg_feasibility"),
        }

    def _get_recent_avg(self, field: str) -> float:
        """Get average of a field from recent episodes."""
        recent = self.episodic.get_recent(3)
        if not recent:
            return 0.0
        values = [getattr(ep, field, 0.0) for ep in recent]
        return sum(values) / len(values)

    def save(self, path: Path) -> None:
        """Save memory state to directory."""
        path.mkdir(parents=True, exist_ok=True)
        self.episodic.save(path / "episodes.json")

    def load(self, path: Path) -> None:
        """Load memory state from directory."""
        episodes_path = path / "episodes.json"
        if episodes_path.exists():
            self.episodic = EpisodicMemory.load(episodes_path)
