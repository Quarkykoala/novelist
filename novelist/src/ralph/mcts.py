"""
Agentic Tree Search (MCTS) Implementation for Novelist 2.0.

This module implements the Monte Carlo Tree Search algorithm adapted for
scientific discovery, managing the selection, expansion, simulation, and
backpropagation phases of the research process.
"""

import asyncio
import random
from typing import Any, Callable, Coroutine

from src.contracts.schemas import RalphConfig
from src.ralph.tree import ResearchState, SearchNode


class AgenticTreeSearch:
    """Orchestrates the MCTS process for research hypothesis generation."""

    def __init__(
        self,
        config: RalphConfig,
        expand_fn: Callable[[SearchNode], Coroutine[Any, Any, list[SearchNode]]],
        evaluate_fn: Callable[[SearchNode], Coroutine[Any, Any, float]],
    ):
        """
        Initialize the Tree Search.

        Args:
            config: Configuration settings.
            expand_fn: Async function to expand a node (generate children).
            evaluate_fn: Async function to evaluate a node (return score).
        """
        self.config = config
        self._expand_fn = expand_fn
        self._evaluate_fn = evaluate_fn
        self.root: SearchNode | None = None
        self.nodes_expanded = 0

    async def search(self, initial_state: ResearchState, iterations: int = 10) -> SearchNode:
        """
        Run the MCTS algorithm.

        Args:
            initial_state: The starting research state (e.g., with gaps identified).
            iterations: Number of search iterations to run.

        Returns:
            The best child node of the root after search.
        """
        # 1. Initialize Root
        self.root = SearchNode(state=initial_state, action_description="Root")
        
        # Initial expansion of root
        print(f"🌱 MCTS: Initializing search from root (Depth {initial_state.depth})")
        await self._expand_node(self.root)
        
        for i in range(iterations):
            print(f"🌳 MCTS Iteration {i+1}/{iterations}...")
            
            # 2. Select
            node = self._select(self.root)
            print(f"   Selected node: {node.id[:8]} ({node.action_description})")

            # 3. Expand (if not terminal and not already expanded)
            if not node.is_leaf() and node.visits > 0:
                # Already expanded, so we selected a child.
                # If we reached a leaf that has been visited, we need to expand it.
                # In standard MCTS, we expand a leaf node once it's visited.
                pass 
            elif node.visits > 0:
                 # It's a visited leaf, expand it
                 await self._expand_node(node)
                 # Select a child from the newly expanded node to simulate
                 if node.children:
                     node = self._select(node) # Greedy or random pick? Standard is pick first/random
                     node = node # Actually _select works on UCB, so calling it is fine
                     # But _select is recursive. Let's just pick one child.
                     # Simply picking the first child for simulation is common if just expanded.
                     # However, logic below handles "Simulate" on values.
            
            # 4. Simulate (Evaluate)
            # In our case, "Simulation" is running the evaluation function (Debate).
            # If the node was just expanded, we might evaluate the *children*.
            # For simplicity V1: We evaluate the node 'node' itself.
            reward = await self._evaluate_fn(node)
            print(f"   Reward: {reward:.3f}")

            # 5. Backpropagate
            self._backpropagate(node, reward)

        # Return best child of root (most visited or highest value)
        return self.root.best_child(exploration_weight=0.0)

    def _select(self, node: SearchNode) -> SearchNode:
        """Select a promising node to explore using UCT."""
        current = node
        
        # While node is fully expanded and non-terminal
        while not current.is_leaf():
            # If any child has never been visited, standard MCTS says expand/visit it? 
            # Or assume all children created at expansion.
            # Here, we assume expand() creates all children.
            # So we check UCT.
            current = current.best_child()
            
        return current

    async def _expand_node(self, node: SearchNode) -> None:
        """Generate children for the node using the expansion function."""
        # Only expand if depth limit not reached
        if node.state.depth >= 3:  # Max depth hardcoded for now
            return

        new_nodes = await self._expand_fn(node)
        for child in new_nodes:
            node.children.append(child)
        
        self.nodes_expanded += len(new_nodes)
        print(f"   Expanded {len(new_nodes)} children")

    def _backpropagate(self, node: SearchNode, reward: float) -> None:
        """Propagate reward up the tree."""
        current = node
        while current:
            current.update(reward)
            current = current.parent
