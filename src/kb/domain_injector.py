"""
Domain Injector — Inject unrelated domain concepts to induce cross-pollination.

Inspired by the cellular SOS response: under stress, cells increase mutation rates.
This module provides the "mutagen" — unexpected concepts from unrelated fields
that force the hypothesis generator to make unexpected connections.

The higher the stress intensity, the more aggressive the injection.
"""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class InjectedDomain:
    """A domain injected into hypothesis generation."""
    
    name: str
    concepts: list[str]
    description: str
    metaphor_prompt: str  # Prompt hint for applying this domain


# Curated domains with rich cross-pollination potential
DOMAINS: list[InjectedDomain] = [
    InjectedDomain(
        name="mycology",
        concepts=["mycelium networks", "spore dispersal", "symbiotic relationships", 
                  "decomposition cycles", "fungal communication", "nutrient transport"],
        description="The study of fungi and their networks",
        metaphor_prompt="Think of how fungal networks distribute resources and information underground",
    ),
    InjectedDomain(
        name="game_theory",
        concepts=["Nash equilibrium", "zero-sum games", "cooperative strategies",
                  "prisoner's dilemma", "evolutionary stable strategies", "auction theory"],
        description="Strategic decision-making under competition",
        metaphor_prompt="Consider what strategies emerge when agents optimize against each other",
    ),
    InjectedDomain(
        name="fluid_dynamics",
        concepts=["laminar flow", "turbulence", "viscosity", "Bernoulli principle",
                  "Reynolds number", "vortex formation", "diffusion gradients"],
        description="The physics of flowing materials",
        metaphor_prompt="How do fluids find efficient paths and what happens at interfaces?",
    ),
    InjectedDomain(
        name="swarm_intelligence",
        concepts=["ant colony optimization", "flocking behavior", "stigmergy",
                  "decentralized control", "emergence", "pheromone trails"],
        description="Collective behavior of simple agents",
        metaphor_prompt="What behaviors emerge when many simple agents follow local rules?",
    ),
    InjectedDomain(
        name="crystallography",
        concepts=["lattice structures", "nucleation", "crystal defects", "polymorphism",
                  "epitaxial growth", "self-assembly", "annealing"],
        description="The science of crystal formation and structure",
        metaphor_prompt="How do ordered structures spontaneously emerge from disorder?",
    ),
    InjectedDomain(
        name="immunology",
        concepts=["adaptive immunity", "antigen recognition", "T-cell selection",
                  "immune memory", "autoimmunity", "cytokine signaling"],
        description="The body's defense systems",
        metaphor_prompt="How does the immune system recognize novel threats and remember solutions?",
    ),
    InjectedDomain(
        name="music_theory",
        concepts=["harmonic resonance", "polyrhythms", "tension and resolution",
                  "counterpoint", "timbre", "overtone series"],
        description="The structure and patterns in music",
        metaphor_prompt="What patterns create pleasing combinations and how do dissonances resolve?",
    ),
    InjectedDomain(
        name="ecology",
        concepts=["trophic cascades", "keystone species", "carrying capacity",
                  "succession", "niche partitioning", "edge effects"],
        description="Interactions between organisms and environment",
        metaphor_prompt="How do complex systems achieve stability through interdependence?",
    ),
    InjectedDomain(
        name="thermodynamics",
        concepts=["entropy", "Carnot efficiency", "phase transitions", "free energy",
                  "irreversibility", "heat engines", "statistical mechanics"],
        description="Energy, heat, and work",
        metaphor_prompt="What are the fundamental limits on efficiency and what drives change?",
    ),
    InjectedDomain(
        name="linguistics",
        concepts=["syntax trees", "semantic fields", "morphological derivation",
                  "pragmatics", "pidgin languages", "grammaticalization"],
        description="The structure and evolution of language",
        metaphor_prompt="How do communication systems encode meaning and evolve over time?",
    ),
    InjectedDomain(
        name="origami_engineering",
        concepts=["fold patterns", "flat-foldability", "rigid origami", "auxetic structures",
                  "deployment mechanisms", "tessellations"],
        description="Folding-based design and engineering",
        metaphor_prompt="How can complex structures emerge from simple folding rules?",
    ),
    InjectedDomain(
        name="behavioral_economics",
        concepts=["loss aversion", "hyperbolic discounting", "anchoring effects",
                  "sunk cost fallacy", "nudges", "bounded rationality"],
        description="Psychology of economic decisions",
        metaphor_prompt="What cognitive biases shape decision-making and how can systems account for them?",
    ),
    InjectedDomain(
        name="circadian_biology",
        concepts=["oscillators", "entrainment", "zeitgebers", "feedback loops",
                  "phase shifts", "ultradian rhythms"],
        description="Biological timing mechanisms",
        metaphor_prompt="How do systems synchronize to cycles and what happens when rhythms are disrupted?",
    ),
    InjectedDomain(
        name="materials_science",
        concepts=["grain boundaries", "dislocations", "superalloys", "amorphous metals",
                  "shape memory", "metamaterials", "piezoelectricity"],
        description="Properties and design of materials",
        metaphor_prompt="How does microstructure determine macro properties?",
    ),
    InjectedDomain(
        name="network_science",
        concepts=["scale-free networks", "small world effect", "clustering coefficient",
                  "preferential attachment", "network robustness", "cascade failures"],
        description="Structure and dynamics of networks",
        metaphor_prompt="What network topologies are resilient and how do effects propagate?",
    ),
]


class DomainInjector:
    """
    Inject unrelated domain concepts based on stress intensity.
    
    Low stress → no injection (stay focused)
    Medium stress → 1 domain (broaden slightly)
    High stress → 2+ domains (force wild connections)
    """
    
    def __init__(self, domains: list[InjectedDomain] | None = None):
        self.domains = domains or DOMAINS
        self._used_recently: set[str] = set()
    
    def inject_for_stress(
        self, 
        stress_intensity: float,
        topic: str = "",
        exclude_domains: list[str] | None = None,
    ) -> list[InjectedDomain]:
        """
        Select domains to inject based on stress intensity.
        
        Args:
            stress_intensity: 0.0-1.0, higher = more stressed = more injection
            topic: Optional topic for smarter selection
            exclude_domains: Domain names to exclude
            
        Returns:
            List of domains to inject (0-3 based on intensity)
        """
        exclude = set(exclude_domains or [])
        available = [d for d in self.domains if d.name not in exclude]
        
        # Avoid recent repeats
        available = [d for d in available if d.name not in self._used_recently]
        if not available:
            self._used_recently.clear()
            available = [d for d in self.domains if d.name not in exclude]
        
        if not available:
            return []
        
        # Determine injection count based on stress
        if stress_intensity < 0.3:
            count = 0
        elif stress_intensity < 0.6:
            count = 1
        elif stress_intensity < 0.8:
            count = 2
        else:
            count = 3
        
        if count == 0:
            return []
        
        # Random selection (could be smarter based on topic)
        selected = random.sample(available, min(count, len(available)))
        
        # Track recently used
        for domain in selected:
            self._used_recently.add(domain.name)
            if len(self._used_recently) > len(self.domains) // 2:
                self._used_recently.clear()
        
        return selected
    
    def get_injection_prompt(
        self, 
        domains: list[InjectedDomain],
        stress_description: str = "",
    ) -> str:
        """
        Generate a prompt section that injects the domain concepts.
        
        This is added to the hypothesis generation prompt to force
        cross-domain thinking.
        """
        if not domains:
            return ""
        
        lines = [
            "\n",
            "=" * 60,
            "CROSS-DOMAIN INJECTION (Mandatory consideration)",
            "=" * 60,
            "",
            "You MUST consider connections to these unrelated fields:",
            "",
        ]
        
        for domain in domains:
            lines.append(f"### {domain.name.upper()}")
            lines.append(f"*{domain.description}*")
            lines.append(f"Key concepts: {', '.join(domain.concepts[:4])}")
            lines.append(f"→ {domain.metaphor_prompt}")
            lines.append("")
        
        lines.append(
            "At least ONE of your hypotheses must explicitly connect "
            "the research topic to concepts from these domains."
        )
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_random_concept(self, domain_name: Optional[str] = None) -> str:
        """Get a random concept, optionally from a specific domain."""
        if domain_name:
            domain = next((d for d in self.domains if d.name == domain_name), None)
            if domain:
                return random.choice(domain.concepts)
        
        # Random from any domain
        domain = random.choice(self.domains)
        return random.choice(domain.concepts)
