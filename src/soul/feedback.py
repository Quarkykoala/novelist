"""Feedback learning system for user preference tracking.

Tracks user interactions with hypotheses to learn preferences:
- accept: User explicitly accepts a hypothesis
- reject: User dismisses or removes a hypothesis
- star: User marks as important/interesting
- edit: User modifies the hypothesis

Over time, this builds a preference model that influences scoring.
"""

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class FeedbackEntry:
    """A single feedback action."""
    hypothesis_id: str
    action: str  # accept, reject, star, edit
    timestamp: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackStore:
    """Tracks and learns from user feedback on hypotheses.
    
    Learns preference weights that can be applied to future scoring:
    - Tracks which types of hypotheses users prefer
    - Identifies patterns in accepted vs rejected hypotheses
    - Builds scoring adjustments based on feedback history
    """
    
    feedback_log: list[FeedbackEntry] = field(default_factory=list)
    db_path: Path = field(default_factory=lambda: Path("sessions/feedback.db"))
    
    # Cached preference weights
    _preference_cache: dict[str, float] = field(default_factory=dict)
    _cache_valid: bool = False
    
    def __post_init__(self):
        """Initialize database if needed."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self) -> None:
        """Create feedback tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT,
                action TEXT,
                timestamp TEXT,
                context TEXT,
                session_id TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS preference_weights (
                key TEXT PRIMARY KEY,
                weight REAL,
                updated_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def record(
        self,
        hypothesis_id: str,
        action: str,
        context: dict[str, Any] | None = None,
        session_id: str = "",
    ) -> None:
        """Record a feedback action.
        
        Args:
            hypothesis_id: ID of the hypothesis being rated
            action: One of 'accept', 'reject', 'star', 'edit'
            context: Optional context (e.g., hypothesis attributes)
            session_id: Session where feedback was given
        """
        if action not in ("accept", "reject", "star", "edit"):
            raise ValueError(f"Invalid action: {action}")
        
        entry = FeedbackEntry(
            hypothesis_id=hypothesis_id,
            action=action,
            timestamp=datetime.now().isoformat(),
            context=context or {},
        )
        self.feedback_log.append(entry)
        self._cache_valid = False
        
        # Persist immediately
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO feedback (hypothesis_id, action, timestamp, context, session_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            hypothesis_id,
            action,
            entry.timestamp,
            json.dumps(context or {}),
            session_id,
        ))
        conn.commit()
        conn.close()
    
    def get_feedback_stats(self) -> dict[str, int]:
        """Get counts of each feedback type."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT action, COUNT(*) FROM feedback GROUP BY action
        """)
        
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        
        return {
            "accept": stats.get("accept", 0),
            "reject": stats.get("reject", 0),
            "star": stats.get("star", 0),
            "edit": stats.get("edit", 0),
            "total": sum(stats.values()),
        }
    
    def get_preference_weights(self) -> dict[str, float]:
        """Compute preference weights from feedback history.
        
        Returns scoring adjustments based on patterns in user feedback.
        These can be used to bias future hypothesis scoring.
        
        Returns:
            Dict of preference categories to weight adjustments (-1 to +1)
        """
        if self._cache_valid and self._preference_cache:
            return self._preference_cache
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load all feedback with context
        cursor.execute("""
            SELECT action, context FROM feedback
        """)
        
        all_feedback = cursor.fetchall()
        conn.close()
        
        if not all_feedback:
            return {}
        
        # Analyze patterns
        weights: dict[str, float] = {}
        
        # Count positive vs negative for different attributes
        attribute_scores: dict[str, dict[str, int]] = {}
        
        for action, context_json in all_feedback:
            try:
                context = json.loads(context_json) if context_json else {}
            except:
                context = {}
            
            # Weight by action type
            action_weight = {
                "accept": 1.0,
                "star": 1.5,
                "edit": 0.5,  # Neutral - could go either way
                "reject": -1.0,
            }.get(action, 0)
            
            # Track attributes that correlate with positive/negative feedback
            for attr, value in context.items():
                if attr not in attribute_scores:
                    attribute_scores[attr] = {"positive": 0, "negative": 0}
                
                if action_weight > 0:
                    attribute_scores[attr]["positive"] += 1
                elif action_weight < 0:
                    attribute_scores[attr]["negative"] += 1
        
        # Convert to weights (-1 to +1 scale)
        for attr, scores in attribute_scores.items():
            total = scores["positive"] + scores["negative"]
            if total > 0:
                # Bias toward what users prefer
                weight = (scores["positive"] - scores["negative"]) / total
                weights[attr] = round(weight, 3)
        
        # Also compute general preference metrics
        stats = self.get_feedback_stats()
        total_votes = stats["accept"] + stats["reject"]
        if total_votes > 0:
            weights["acceptance_rate"] = round(stats["accept"] / total_votes, 3)
        
        # Star rate indicates high-quality hypotheses
        if stats["total"] > 0:
            weights["star_rate"] = round(stats["star"] / stats["total"], 3)
        
        self._preference_cache = weights
        self._cache_valid = True
        
        return weights
    
    def get_recent_feedback(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most recent feedback entries."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT hypothesis_id, action, timestamp, context, session_id
            FROM feedback
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "hypothesis_id": row[0],
                "action": row[1],
                "timestamp": row[2],
                "context": json.loads(row[3]) if row[3] else {},
                "session_id": row[4],
            })
        
        conn.close()
        return results
    
    def clear(self) -> None:
        """Clear all feedback (for testing)."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("DELETE FROM feedback")
        conn.commit()
        conn.close()
        self.feedback_log.clear()
        self._cache_valid = False


# Global singleton for easy access
_feedback_store: FeedbackStore | None = None


def get_feedback_store() -> FeedbackStore:
    """Get the global feedback store instance."""
    global _feedback_store
    if _feedback_store is None:
        _feedback_store = FeedbackStore()
    return _feedback_store
