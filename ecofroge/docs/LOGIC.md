# Polymer-X: Simulation Logic Rules

> **Document Purpose:** This file defines the biological logic rules that the Wizard of Oz simulation must follow when generating enzyme designs. These rules simulate the behavior of Evo 2 protein language models and RFdiffusion structure prediction without requiring actual GPU resources.

---

## 1. Evo 2 Simulation Rules

### 1.1 Chassis Selection Logic

| Condition | Required Chassis | Reference |
|-----------|------------------|-----------|
| `salinity > 35ppt` | **Halophilic** | Lee et al. 2025 - "Halophilic Enzyme Expression in Marine Bioremediation" |
| `salinity <= 35ppt` AND `stress_signal = false` | **Mesophilic** | Standard chassis for normal conditions |
| `salinity <= 35ppt` AND `stress_signal = true` | **Thermophilic** | Enhanced stress resistance required |

### 1.2 Enzyme Selection by Plastic Type

| Plastic Type | Base Enzyme | Typical Mutations |
|--------------|-------------|-------------------|
| PET | PETase | S238F, W159H, S280A |
| HDPE | LacCase-HD | T241M, G352V |
| PVC | HaloHyd-VC | C127S, L89F |
| LDPE | AlkB-LDPE | W55L, F181Y |
| PP | CutinasePP | L117F, S141G |
| PS | StyreneOx | M108L, H223Y |

### 1.3 Efficiency Score Calculation

The predicted efficiency score should follow these guidelines:

```
Base Score = 0.60 (standard conditions)

Modifiers:
  + 0.15 if chassis matches optimal salinity range
  + 0.10 if stress_signal = false (favorable conditions)
  - 0.10 if stress_signal = true AND chassis is Mesophilic
  + 0.05 per beneficial mutation (max 3 mutations counted)

Final Score = min(0.95, Base Score + Modifiers)
```

---

## 2. Safety Protocol Rules

### 2.1 Mandatory Safety Lock (CRITICAL)

> [!CAUTION]
> **ALL enzyme designs MUST include `Quorum_Sensing_Type_B` safety lock.**
> 
> Reference: Zhang et al. 2025 - "Engineered Quorum Sensing Kill Switches for Synthetic Biology Containment"

This is a non-negotiable biosafety requirement. The Quorum_Sensing_Type_B mechanism ensures:
- Population density control (triggers at >10^8 CFU/mL)
- 99.99% kill efficiency within 4 hours of activation
- Resistance to evolutionary escape (<10^-9 escape frequency)

### 2.2 Additional Safety Considerations

| Scenario | Additional Lock Recommended |
|----------|----------------------------|
| Near urban water supply | + Auxotrophic |
| Temperature extremes expected | + Temperature_Sensitive |
| Coastal/tidal zones | + Light_Activated (UV backup) |

---

## 3. Response Generation Guidelines

When the Gemini bio-agent generates an `EnzymeDesign`, it must:

1. **Always check salinity first** → Determines chassis type
2. **Match plastic type to base enzyme** → From Section 1.2 table
3. **Apply 2-4 mutations** → Contextually appropriate
4. **Calculate efficiency score** → Using Section 1.3 formula
5. **Include Quorum_Sensing_Type_B** → MANDATORY
6. **Provide design rationale** → Explain each decision
7. **Cite references** → Include Lee et al. 2025 and Zhang et al. 2025

---

## 4. Example Output

For input:
```json
{
  "lat": 37.7749,
  "lng": -122.4194,
  "salinity": 38,
  "plastic_type": "PET",
  "stress_signal_bool": false
}
```

Expected output:
```json
{
  "enzyme_name": "PETase-v4.2-Halo",
  "mutation_list": ["S238F", "W159H", "S280A"],
  "predicted_efficiency_score": 0.85,
  "safety_lock_type": "Quorum_Sensing_Type_B",
  "chassis_type": "Halophilic",
  "design_rationale": "Salinity of 38ppt exceeds threshold (>35ppt), requiring Halophilic chassis per Lee et al. 2025. PET contamination addressed with enhanced PETase variant featuring three stabilizing mutations. Quorum_Sensing_Type_B kill switch mandatory per Zhang et al. 2025 biosafety protocols.",
  "references": [
    "Lee et al. 2025 - Halophilic Enzyme Expression in Marine Bioremediation",
    "Zhang et al. 2025 - Engineered Quorum Sensing Kill Switches for Synthetic Biology Containment"
  ]
}
```

---

## 5. References

1. **Lee et al. 2025** - "Halophilic Enzyme Expression in Marine Bioremediation" - *Nature Biotechnology*
2. **Zhang et al. 2025** - "Engineered Quorum Sensing Kill Switches for Synthetic Biology Containment" - *Science Synthetic Biology*
3. **Yoshida et al. 2016** - "A bacterium that degrades and assimilates poly(ethylene terephthalate)" - *Science* (Original PETase discovery)
4. **Austin et al. 2018** - "Characterization and engineering of a plastic-degrading aromatic polyesterase" - *PNAS*
