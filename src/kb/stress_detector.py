"""
Stress Detector — Identify knowledge gaps with "gravity" scores.

Inspired by the cellular SOS response: under stress, cells increase mutation rates
specifically at genes that could help survival. This module detects WHERE in the
concept space the topic is "starving" for knowledge.

Stress signals:
1. Weak connections in concept graph (low edge count)
2. Contradictions between claims
3. Concepts mentioned but never explained
4. High-centrality concepts with missing links
"""

from dataclasses import dataclass, field
from enum import Enum

from src.contracts.schemas import ConceptMap, ExtractedClaim, IdentifiedGap, GapType


class StressType(str, Enum):
    """Types of knowledge stress."""
    WEAK_CONNECTION = "weak_connection"      # Concepts barely linked
    CONTRADICTION = "contradiction"          # Claims that conflict
    UNEXPLAINED = "unexplained"              # Mentioned but not defined
    MISSING_BRIDGE = "missing_bridge"        # High-centrality gap
    ORPHAN_CONCEPT = "orphan_concept"        # Isolated node


@dataclass
class StressZone:
    """A region of the concept space under 'stress' (knowledge gap)."""
    
    stress_type: StressType
    concepts: list[str]  # Concepts involved in this stress zone
    intensity: float  # 0.0 = low stress, 1.0 = high stress (starvation)
    description: str
    related_claims: list[str] = field(default_factory=list)  # Claim IDs
    
    def __post_init__(self):
        self.intensity = max(0.0, min(1.0, self.intensity))


class StressDetector:
    """
    Detect knowledge gaps in concept maps with intensity scores.
    
    Like the bacterial SOS response, we identify WHERE mutations (exploration)
    should be concentrated to address survival pressure (knowledge gaps).
    """
    
    def __init__(self, concept_map: ConceptMap, claims: list[ExtractedClaim]):
        self.concept_map = concept_map
        self.claims = claims
        self._build_indices()
    
    def _build_indices(self) -> None:
        """Build lookup indices for efficient analysis."""
        # Node degree (number of connections)
        self.node_degrees: dict[str, int] = {}
        for edge in self.concept_map.edges:
            self.node_degrees[edge.source] = self.node_degrees.get(edge.source, 0) + 1
            self.node_degrees[edge.target] = self.node_degrees.get(edge.target, 0) + 1
        
        # Claims by concept
        self.claims_by_concept: dict[str, list[ExtractedClaim]] = {}
        for claim in self.claims:
            # Extract concepts mentioned in claim
            for node in self.concept_map.nodes:
                if node.name.lower() in claim.statement.lower():
                    if node.name not in self.claims_by_concept:
                        self.claims_by_concept[node.name] = []
                    self.claims_by_concept[node.name].append(claim)
        
        # Edge lookup
        self.edge_set: set[tuple[str, str]] = set()
        for edge in self.concept_map.edges:
            self.edge_set.add((edge.source, edge.target))
            self.edge_set.add((edge.target, edge.source))  # Undirected
    
    def detect_stress_zones(self) -> list[StressZone]:
        """
        Detect all stress zones in the concept space.
        
        Returns zones sorted by intensity (highest first).
        """
        zones: list[StressZone] = []
        
        # 1. Find weakly-connected concepts (orphans and near-orphans)
        zones.extend(self._detect_weak_connections())
        
        # 2. Find contradictions between claims
        zones.extend(self._detect_contradictions())
        
        # 3. Find unexplained concepts (mentioned but no claims)
        zones.extend(self._detect_unexplained())
        
        # 4. Find missing bridges (high-potential gaps)
        zones.extend(self._detect_missing_bridges())
        
        # Sort by intensity (highest stress first)
        zones.sort(key=lambda z: z.intensity, reverse=True)
        
        return zones
    
    def _detect_weak_connections(self) -> list[StressZone]:
        """Find concepts with very few connections."""
        zones = []
        
        if not self.concept_map.nodes:
            return zones
        
        # Calculate average degree
        avg_degree = sum(self.node_degrees.values()) / max(1, len(self.node_degrees))
        
        for node in self.concept_map.nodes:
            degree = self.node_degrees.get(node.name, 0)
            
            if degree == 0:
                # Orphan concept — highest stress
                zones.append(StressZone(
                    stress_type=StressType.ORPHAN_CONCEPT,
                    concepts=[node.name],
                    intensity=1.0,
                    description=f"'{node.name}' is completely isolated with no connections",
                ))
            elif degree < avg_degree * 0.5:
                # Weakly connected
                intensity = 0.5 + (0.5 * (1 - degree / max(1, avg_degree * 0.5)))
                zones.append(StressZone(
                    stress_type=StressType.WEAK_CONNECTION,
                    concepts=[node.name],
                    intensity=intensity,
                    description=f"'{node.name}' has only {degree} connection(s), below average",
                ))
        
        return zones
    
    def _detect_contradictions(self) -> list[StressZone]:
        """Find contradicting claims about the same concept."""
        zones = []
        
        # Simple heuristic: claims with opposing sentiment about same concept
        # (A more sophisticated version would use NLI or LLM)
        
        negative_markers = ["not", "cannot", "fails", "unlikely", "impossible", "decreases", "reduces"]
        positive_markers = ["can", "enables", "increases", "improves", "achieves", "succeeds"]
        
        for concept, concept_claims in self.claims_by_concept.items():
            if len(concept_claims) < 2:
                continue
            
            positive_claims = []
            negative_claims = []
            
            for claim in concept_claims:
                statement_lower = claim.statement.lower()
                has_negative = any(marker in statement_lower for marker in negative_markers)
                has_positive = any(marker in statement_lower for marker in positive_markers)
                
                if has_negative and not has_positive:
                    negative_claims.append(claim)
                elif has_positive and not has_negative:
                    positive_claims.append(claim)
            
            if positive_claims and negative_claims:
                zones.append(StressZone(
                    stress_type=StressType.CONTRADICTION,
                    concepts=[concept],
                    intensity=0.9,  # Contradictions are high-stress
                    description=f"Contradicting claims found about '{concept}'",
                    related_claims=[c.paper_id for c in positive_claims + negative_claims],
                ))
        
        return zones
    
    def _detect_unexplained(self) -> list[StressZone]:
        """Find concepts that appear in edges but have no claims."""
        zones = []
        
        for node in self.concept_map.nodes:
            claims = self.claims_by_concept.get(node.name, [])
            degree = self.node_degrees.get(node.name, 0)
            
            if degree > 0 and len(claims) == 0:
                # Connected but no claims — unexplained
                intensity = min(0.8, 0.4 + (degree * 0.1))  # More connections = more stress
                zones.append(StressZone(
                    stress_type=StressType.UNEXPLAINED,
                    concepts=[node.name],
                    intensity=intensity,
                    description=f"'{node.name}' has {degree} connections but no supporting claims",
                ))
        
        return zones
    
    def _detect_missing_bridges(self) -> list[StressZone]:
        """Find concept pairs that SHOULD be connected but aren't."""
        zones = []
        
        # Find concepts that share claims but have no edge
        concept_names = [n.name for n in self.concept_map.nodes]
        
        for i, concept_a in enumerate(concept_names):
            for concept_b in concept_names[i+1:]:
                if (concept_a, concept_b) in self.edge_set:
                    continue  # Already connected
                
                # Check if they share any claims
                claims_a = set(c.paper_id for c in self.claims_by_concept.get(concept_a, []))
                claims_b = set(c.paper_id for c in self.claims_by_concept.get(concept_b, []))
                shared = claims_a & claims_b
                
                if shared:
                    # Same papers mention both but no edge — potential bridge
                    intensity = min(0.85, 0.5 + (len(shared) * 0.1))
                    zones.append(StressZone(
                        stress_type=StressType.MISSING_BRIDGE,
                        concepts=[concept_a, concept_b],
                        intensity=intensity,
                        description=f"'{concept_a}' and '{concept_b}' co-occur in {len(shared)} paper(s) but have no direct link",
                        related_claims=list(shared),
                    ))
        
        return zones
    
    def get_top_stress_zones(self, n: int = 5) -> list[StressZone]:
        """Get the N highest-stress zones."""
        return self.detect_stress_zones()[:n]
    
    def get_stress_for_concepts(self, concepts: list[str]) -> float:
        """Get aggregate stress intensity for a set of concepts."""
        zones = self.detect_stress_zones()
        relevant = [z for z in zones if any(c in z.concepts for c in concepts)]
        
        if not relevant:
            return 0.0
        
        return max(z.intensity for z in relevant)


def convert_stress_to_gaps(stress_zones: list[StressZone]) -> list[IdentifiedGap]:
    """Convert StressZones to IdentifiedGaps for compatibility."""
    gaps = []
    
    for zone in stress_zones:
        gap_type = {
            StressType.WEAK_CONNECTION: GapType.MISSING_CONNECTION,
            StressType.CONTRADICTION: GapType.CONTRADICTION,
            StressType.UNEXPLAINED: GapType.MECHANISM_UNKNOWN,
            StressType.MISSING_BRIDGE: GapType.MISSING_CONNECTION,
            StressType.ORPHAN_CONCEPT: GapType.MISSING_CONNECTION,
        }.get(zone.stress_type, GapType.MISSING_CONNECTION)
        
        concept_a = zone.concepts[0] if zone.concepts else ""
        concept_b = zone.concepts[1] if len(zone.concepts) > 1 else ""
        
        gaps.append(IdentifiedGap(
            gap_type=gap_type,
            description=zone.description,
            concept_a=concept_a,
            concept_b=concept_b,
            potential_value=f"Stress intensity: {zone.intensity:.2f}",
            supporting_evidence=zone.related_claims,
        ))
    
    return gaps
