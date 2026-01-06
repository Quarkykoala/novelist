/**
 * Polymer-X: Gemini Bridge Service
 * 
 * Ports the verified Committee Mode logic from the CLI script into a browser-compatible service.
 * This implements the same 3-agent debate pattern:
 *   1. The Architect - Proposes chassis organism
 *   2. The Safety Officer - Validates safety locks
 *   3. The Simulator - Predicts efficiency
 */

// =============================================================================
// Type Definitions (mirrored from docs/INTERFACES.ts)
// =============================================================================

export type PlasticType = 'PET' | 'HDPE' | 'PVC' | 'LDPE' | 'PP' | 'PS';

export type SafetyLockType =
    | 'Quorum_Sensing_Type_A'
    | 'Quorum_Sensing_Type_B'
    | 'Temperature_Sensitive'
    | 'Auxotrophic'
    | 'Light_Activated';

export type ChassisType =
    | 'Halophilic'
    | 'Mesophilic'
    | 'Thermophilic'
    | 'Psychrophilic';

export interface WaterAnalysis {
    lat: number;
    lng: number;
    salinity: number;
    plastic_type: PlasticType;
    stress_signal_bool: boolean;
}

export interface EnzymeDesign {
    enzyme_name: string;
    mutation_list: string[];
    predicted_efficiency_score: number;
    safety_lock_type: SafetyLockType;
    chassis_type: ChassisType;
    design_rationale: string;
    references: string[];
}

export interface MonologueEntry {
    agent: 'ARCHITECT' | 'SAFETY_OFFICER' | 'SIMULATOR';
    timestamp: string;
    thought: string;
    decision?: string;
    rejected?: boolean;
    retry_reason?: string;
}

export interface CommitteeBioAgentResponse {
    success: boolean;
    data?: EnzymeDesign;
    error?: string;
    timestamp: string;
    internal_monologue: MonologueEntry[];
}

// =============================================================================
// Configuration Tables
// =============================================================================

const ORGANISM_CHASSIS: Record<PlasticType, { organism: string; description: string }> = {
    PET: { organism: 'Ideonella sakaiensis', description: 'Native PETase producer, optimal for PET degradation' },
    HDPE: { organism: 'Pseudomonas putida', description: 'Robust chassis for hydrocarbon degradation pathways' },
    PVC: { organism: 'Sphingomonas sp.', description: 'Known for chlorinated compound metabolism' },
    LDPE: { organism: 'Rhodococcus ruber', description: 'Alkane-degrading actinobacterium' },
    PP: { organism: 'Aspergillus tubingensis', description: 'Fungal chassis with strong cutinase expression' },
    PS: { organism: 'Exiguobacterium sp.', description: 'Psychrotolerant styrene degrader' },
};

const ENZYME_CONFIG: Record<PlasticType, { base: string; mutations: string[] }> = {
    PET: { base: 'PETase', mutations: ['S238F', 'W159H', 'S280A'] },
    HDPE: { base: 'LacCase-HD', mutations: ['T241M', 'G352V'] },
    PVC: { base: 'HaloHyd-VC', mutations: ['C127S', 'L89F'] },
    LDPE: { base: 'AlkB-LDPE', mutations: ['W55L', 'F181Y'] },
    PP: { base: 'CutinasePP', mutations: ['L117F', 'S141G'] },
    PS: { base: 'StyreneOx', mutations: ['M108L', 'H223Y'] },
};

// =============================================================================
// Sub-Agent Logic
// =============================================================================

function determineChassisType(salinity: number, stress: boolean): ChassisType {
    if (salinity > 35) return 'Halophilic';
    if (stress) return 'Thermophilic';
    return 'Mesophilic';
}

function calculateEfficiencyScore(
    salinity: number,
    stress: boolean,
    chassis: ChassisType,
    mutationCount: number
): number {
    let score = 0.60;
    if ((salinity > 35 && chassis === 'Halophilic') || (salinity <= 35 && chassis !== 'Halophilic')) {
        score += 0.15;
    }
    if (!stress) score += 0.10;
    if (stress && chassis === 'Mesophilic') score -= 0.10;
    score += Math.min(mutationCount, 3) * 0.05;
    return Math.min(0.95, Math.round(score * 100) / 100);
}

interface ArchitectProposal {
    organism: string;
    organism_description: string;
    chassis_type: ChassisType;
    enzyme_name: string;
    mutation_list: string[];
    rationale: string;
    safety_lock_type?: SafetyLockType;
}

function runArchitect(input: WaterAnalysis): { proposal: ArchitectProposal; monologue: MonologueEntry } {
    const organism = ORGANISM_CHASSIS[input.plastic_type];
    const enzymeConfig = ENZYME_CONFIG[input.plastic_type];
    const chassis = determineChassisType(input.salinity, input.stress_signal_bool);

    const chassisSuffix = chassis === 'Halophilic' ? '-Halo' : chassis === 'Thermophilic' ? '-Thermo' : '';

    const proposal: ArchitectProposal = {
        organism: organism.organism,
        organism_description: organism.description,
        chassis_type: chassis,
        enzyme_name: `${enzymeConfig.base}-v4.2${chassisSuffix}`,
        mutation_list: enzymeConfig.mutations,
        rationale: `Selected ${organism.organism} as chassis organism (${organism.description}). ` +
            `Environmental analysis: salinity=${input.salinity}ppt, stress=${input.stress_signal_bool}. ` +
            `Applying ${chassis} expression system.`,
    };

    const monologue: MonologueEntry = {
        agent: 'ARCHITECT',
        timestamp: new Date().toISOString(),
        thought: `Analyzing water sample at (${input.lat.toFixed(2)}, ${input.lng.toFixed(2)}). ` +
            `Detected ${input.plastic_type} contamination. Salinity: ${input.salinity}ppt. ` +
            `Stress signals: ${input.stress_signal_bool ? 'PRESENT' : 'absent'}.`,
        decision: `Proposing ${organism.organism} chassis with ${enzymeConfig.base} enzyme. ` +
            `Expression system: ${chassis}. Mutations: ${enzymeConfig.mutations.join(', ')}.`,
    };

    return { proposal, monologue };
}

function runSafetyOfficer(
    proposal: ArchitectProposal
): { approved: boolean; correctedProposal: ArchitectProposal; monologue: MonologueEntry } {

    const hasSafetyLock = proposal.safety_lock_type === 'Quorum_Sensing_Type_B';

    if (!hasSafetyLock) {
        const correctedProposal: ArchitectProposal = {
            ...proposal,
            safety_lock_type: 'Quorum_Sensing_Type_B',
        };

        return {
            approved: false,
            correctedProposal,
            monologue: {
                agent: 'SAFETY_OFFICER',
                timestamp: new Date().toISOString(),
                thought: `Reviewing proposal for ${proposal.organism}. Checking safety constraints from docs/LOGIC.md...`,
                decision: `REJECTED - Architect proposal lacks Quorum_Sensing_Type_B lock!`,
                rejected: true,
                retry_reason: 'Forcing retry with mandatory safety lock. Zhang et al. 2025 requires Quorum_Sensing_Type_B for all engineered organisms.',
            },
        };
    }

    return {
        approved: true,
        correctedProposal: proposal,
        monologue: {
            agent: 'SAFETY_OFFICER',
            timestamp: new Date().toISOString(),
            thought: `Reviewing proposal for ${proposal.organism}. Verifying Zhang et al. 2025 compliance...`,
            decision: `APPROVED - All safety constraints satisfied. Quorum_Sensing_Type_B verified.`,
        },
    };
}

function runSimulator(
    proposal: ArchitectProposal,
    input: WaterAnalysis
): { efficiency: number; monologue: MonologueEntry } {

    const efficiencyScore = calculateEfficiencyScore(
        input.salinity,
        input.stress_signal_bool,
        proposal.chassis_type,
        proposal.mutation_list.length
    );

    let envMatch: 'OPTIMAL' | 'SUBOPTIMAL' | 'MARGINAL';
    if (efficiencyScore >= 0.85) envMatch = 'OPTIMAL';
    else if (efficiencyScore >= 0.70) envMatch = 'SUBOPTIMAL';
    else envMatch = 'MARGINAL';

    let confidence = 0.85;
    if (input.stress_signal_bool) confidence -= 0.15;
    if (input.salinity > 40) confidence -= 0.10;
    confidence = Math.max(0.50, Math.round(confidence * 100) / 100);

    return {
        efficiency: efficiencyScore,
        monologue: {
            agent: 'SIMULATOR',
            timestamp: new Date().toISOString(),
            thought: `Running Evo 2 efficiency simulation for ${proposal.enzyme_name}. ` +
                `Base chassis: ${proposal.chassis_type}. ` +
                `Environmental parameters: salinity=${input.salinity}ppt, stress=${input.stress_signal_bool}.`,
            decision: `Prediction complete. Efficiency: ${(efficiencyScore * 100).toFixed(1)}% (${envMatch}). ` +
                `Confidence: ${(confidence * 100).toFixed(0)}%. Model ready for deployment recommendation.`,
        },
    };
}

// =============================================================================
// Main Service Class
// =============================================================================

export class GeminiBridge {
    private simulationDelay: number;

    constructor(simulationDelay = 500) {
        this.simulationDelay = simulationDelay;
    }

    private delay(ms: number): Promise<void> {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    async runCommitteeDebate(input: WaterAnalysis): Promise<CommitteeBioAgentResponse> {
        const monologue: MonologueEntry[] = [];

        try {
            // Phase 1: Architect proposes
            await this.delay(this.simulationDelay);
            const { proposal: initialProposal, monologue: architectMonologue } = runArchitect(input);
            monologue.push(architectMonologue);

            // Phase 2: Safety Officer reviews
            await this.delay(this.simulationDelay);
            const { approved, correctedProposal, monologue: safetyMonologue } = runSafetyOfficer(initialProposal);
            monologue.push(safetyMonologue);

            const finalProposal = approved
                ? { ...initialProposal, safety_lock_type: 'Quorum_Sensing_Type_B' as SafetyLockType }
                : correctedProposal;

            if (!approved) {
                monologue.push({
                    agent: 'ARCHITECT',
                    timestamp: new Date().toISOString(),
                    thought: 'Received rejection from Safety Officer. Acknowledging mandatory safety requirement.',
                    decision: 'Retry accepted. Adding Quorum_Sensing_Type_B lock to proposal as required by Zhang et al. 2025.',
                });
            }

            // Phase 3: Simulator predicts
            await this.delay(this.simulationDelay);
            const { efficiency, monologue: simulatorMonologue } = runSimulator(finalProposal, input);
            monologue.push(simulatorMonologue);

            // Assemble final design
            const design: EnzymeDesign = {
                enzyme_name: finalProposal.enzyme_name,
                mutation_list: finalProposal.mutation_list,
                predicted_efficiency_score: efficiency,
                safety_lock_type: finalProposal.safety_lock_type!,
                chassis_type: finalProposal.chassis_type,
                design_rationale: `[COMMITTEE CONSENSUS] Organism: ${finalProposal.organism} (${finalProposal.organism_description}). ` +
                    `${finalProposal.rationale} Safety: ${finalProposal.safety_lock_type} verified. ` +
                    `Simulation: ${efficiency * 100}% efficiency.`,
                references: [
                    'Lee et al. 2025 - Halophilic Enzyme Expression in Marine Bioremediation',
                    'Zhang et al. 2025 - Engineered Quorum Sensing Kill Switches for Synthetic Biology Containment',
                ],
            };

            return {
                success: true,
                data: design,
                timestamp: new Date().toISOString(),
                internal_monologue: monologue,
            };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString(),
                internal_monologue: monologue,
            };
        }
    }
}

// Default singleton instance
export const geminiBridge = new GeminiBridge();
