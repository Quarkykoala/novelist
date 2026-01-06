"""
Tree Search Data Structures for Novelist 2.0.

This module defines the Node and State classes for the Agentic Tree Search (MCTS)
orchestrator, allowing for non-linear exploration of research hypothesis.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import BaseModel, Field

from src.contracts.schemas import (
    ConceptMap,
    ExtractedClaim,
    GroundedHypothesis,
    IdentifiedGap,
)


class ResearchState(BaseModel):
    """Snapshot of the research process at a specific node."""
    
    hypotheses: list[GroundedHypothesis] = Field(default_factory=list)
    concept_map: Optional[ConceptMap] = None
    claims: list[ExtractedClaim] = Field(default_factory=list)
    gaps: list[IdentifiedGap] = Field(default_factory=list)
    
    # Context for this specific branch
    focus_topic: str = ""
    depth: int = 0
    feedback: list[str] = Field(default_factory=list)
    
    def copy(self) -> ResearchState:
        """Create a deep copy of the state."""
        return ResearchState(
            hypotheses=[h.model_copy() for h in self.hypotheses],
            concept_map=self.concept_map.model_copy() if self.concept_map else None,
            claims=[c.model_copy() for c in self.claims],
            gaps=[g.model_copy() for g in self.gaps],
            focus_topic=self.focus_topic,
            depth=self.depth,
            feedback=self.feedback.copy(),
        )


@dataclass
class SearchNode:
    """A node in the MCTS tree representing a research state."""
    
    state: ResearchState
    parent: Optional[SearchNode] = None
    children: list[SearchNode] = field(default_factory=list)
    
    # MCTS Statistics
    visits: int = 0
    value: float = 0.0  # Cumulative value (Q)
    
    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_description: str = "Root"  # Description of action that led here
    
    @property
    def uct_score(self, exploration_weight: float = 1.414) -> float:
        """Calculate Upper Confidence Bound for Trees (UCT) score."""
        if self.visits == 0:
            return float('inf')
        
        if not self.parent:
            return 0.0
            
        exploitation = self.value / self.visits
        exploration = exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
        
        return exploitation + exploration
    
    def add_child(self, child_state: ResearchState, action: str) -> SearchNode:
        """Create and add a child node."""
        child = SearchNode(
            state=child_state,
            parent=self,
            action_description=action
        )
        self.children.append(child)
        return child
    
    def update(self, reward: float):
        """Update node stats with new reward."""
        self.visits += 1
        self.value += reward
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (no children)."""
        return len(self.children) == 0
    
    def best_child(self, exploration_weight: float = 1.414) -> SearchNode:
        """Select child with highest UCT score."""
        if not self.children:
            raise ValueError("Node has no children")
            
        return max(self.children, key=lambda c: c.uct_score(exploration_weight))
    
    def get_path(self) -> list[SearchNode]:
        """Get path from root to this node."""
        path = []
        current = self
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))
