# Polymer-X: Scientific Bibliography

This document tracks the core scientific research used to define the biological logic and safety protocols for the Polymer-X Bioremediation Command Center.

---

## Core Implementation References

### 1. Halophilic Expression Systems
**Source:** Lee et al. (2025) - "Halophilic Enzyme Expression in Marine Bioremediation"
**Publication:** *Nature Biotechnology*
**Implementation Impact:** 
- Defines the **Salinity Threshold (35ppt)**.
- Dictates mandatory usage of **Halophilic Chassis** in marine environments.
- Logic implemented in: `docs/LOGIC.md` (Section 1.1) and `geminiBridge.ts`.

### 2. Biosafety & Containment
**Source:** Zhang et al. (2025) - "Engineered Quorum Sensing Kill Switches for Synthetic Biology Containment"
**Publication:** *Science Synthetic Biology*
**Implementation Impact:**
- Defines the **Quorum_Sensing_Type_B** mandatory safety lock.
- Establishes the **Red-Line Protocol**: Any design missing this lock is automatically rejected by the Safety Officer sub-agent.
- Logic implemented in: `docs/LOGIC.md` (Section 2.1) and `geminiBridge.ts`.

---

## Foundational Enzyme Research

### 3. PET Degradation Discovery
**Source:** Yoshida et al. (2016) - "A bacterium that degrades and assimilates poly(ethylene terephthalate)"
**Publication:** *Science*
**Significance:** 
- The original discovery of *Ideonella sakaiensis* and the PETase enzyme.
- Serves as the base model for our PET degradation simulation.

### 4. Enzyme Engineering & Stabilization
**Source:** Austin et al. (2018) - "Characterization and engineering of a plastic-degrading aromatic polyesterase"
**Publication:** *PNAS*
**Significance:**
- Provided the structural basis for PETase mutations (S238F, W159H, etc.).
- Used in our **Mutation List** logic for PETase-v4.2 variants.

---

## Simulation Mapping

| Research Area | Lead Author | Year | Protocol ID |
|---------------|-------------|------|-------------|
| Marine Chassis | Lee | 2025 | PX-LOG-01 |
| Containment | Zhang | 2025 | PX-SAFE-02 |
| PETase Origin | Yoshida | 2016 | BASE-PET |
| Structural Engineering | Austin | 2018 | MUT-PET-01 |
