#!/usr/bin/env npx ts-node

/**
 * Polymer-X: CLI Test Script
 * 
 * This script tests the bio-agent logic in the terminal before building the UI.
 * It uses a "Wizard of Oz" pattern - prompting Gemini to ACT as the Evo 2 model
 * and generate realistic mock data based on our docs/LOGIC.md rules.
 * 
 * Usage:
 *   npx ts-node scripts/test-logic.ts --salinity=38 --plastic=PET
 *   npx ts-node scripts/test-logic.ts --salinity=25 --plastic=HDPE --stress
 * 
 * Environment:
 *   GEMINI_API_KEY - Your Gemini API key (required for real API calls)
 *   MOCK_MODE=true - Use hardcoded mock responses instead of API calls
 */

import type { WaterAnalysis, EnzymeDesign, BioAgentResponse, ChassisType, SafetyLockType } from '../docs/INTERFACES.js';

// =============================================================================
// CLI Argument Parsing
// =============================================================================

interface CLIArgs {
    salinity: number;
    plastic: WaterAnalysis['plastic_type'];
    stress: boolean;
    lat: number;
    lng: number;
    mock: boolean;
}

function parseArgs(): CLIArgs {
    const args = process.argv.slice(2);
    const parsed: CLIArgs = {
        salinity: 35,
        plastic: 'PET',
        stress: false,
        lat: 37.7749,
        lng: -122.4194,
        mock: true, // Default to mock mode for zero-cost operation
    };

    for (const arg of args) {
        if (arg.startsWith('--salinity=')) {
            parsed.salinity = parseFloat(arg.split('=')[1]);
        } else if (arg.startsWith('--plastic=')) {
            const plastic = arg.split('=')[1].toUpperCase() as WaterAnalysis['plastic_type'];
            if (['PET', 'HDPE', 'PVC', 'LDPE', 'PP', 'PS'].includes(plastic)) {
                parsed.plastic = plastic;
            } else {
                console.error(`❌ Invalid plastic type: ${plastic}`);
                console.error('   Valid types: PET, HDPE, PVC, LDPE, PP, PS');
                process.exit(1);
            }
        } else if (arg === '--stress') {
            parsed.stress = true;
        } else if (arg.startsWith('--lat=')) {
            parsed.lat = parseFloat(arg.split('=')[1]);
        } else if (arg.startsWith('--lng=')) {
            parsed.lng = parseFloat(arg.split('=')[1]);
        } else if (arg === '--live') {
            parsed.mock = false;
        } else if (arg === '--help' || arg === '-h') {
            printHelp();
            process.exit(0);
        }
    }

    return parsed;
}

function printHelp(): void {
    console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POLYMER-X: Bioremediation Command Center                  ║
║                              CLI Test Script                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

USAGE:
  npx ts-node scripts/test-logic.ts [OPTIONS]

OPTIONS:
  --salinity=<number>    Water salinity in ppt (default: 35)
  --plastic=<type>       Plastic type: PET, HDPE, PVC, LDPE, PP, PS (default: PET)
  --stress               Enable stress signal flag
  --lat=<number>         Latitude coordinate (default: 37.7749)
  --lng=<number>         Longitude coordinate (default: -122.4194)
  --live                 Use live Gemini API (requires GEMINI_API_KEY)
  --help, -h             Show this help message

EXAMPLES:
  # High salinity marine environment with PET contamination
  npx ts-node scripts/test-logic.ts --salinity=38 --plastic=PET

  # Freshwater with HDPE under stress conditions
  npx ts-node scripts/test-logic.ts --salinity=5 --plastic=HDPE --stress

  # Use live Gemini API
  GEMINI_API_KEY=xxx npx ts-node scripts/test-logic.ts --salinity=40 --plastic=PVC --live
`);
}

// =============================================================================
// Committee Mode: Sub-Agent Definitions
// =============================================================================

/**
 * Organism chassis mapping based on plastic type (expanded for Architect sub-agent)
 */
const ORGANISM_CHASSIS: Record<WaterAnalysis['plastic_type'], { organism: string; description: string }> = {
    PET: { organism: 'Ideonella sakaiensis', description: 'Native PETase producer, optimal for PET degradation' },
    HDPE: { organism: 'Pseudomonas putida', description: 'Robust chassis for hydrocarbon degradation pathways' },
    PVC: { organism: 'Sphingomonas sp.', description: 'Known for chlorinated compound metabolism' },
    LDPE: { organism: 'Rhodococcus ruber', description: 'Alkane-degrading actinobacterium' },
    PP: { organism: 'Aspergillus tubingensis', description: 'Fungal chassis with strong cutinase expression' },
    PS: { organism: 'Exiguobacterium sp.', description: 'Psychrotolerant styrene degrader' },
};

/**
 * Enzyme configurations based on docs/LOGIC.md Section 1.2
 */
const ENZYME_CONFIG: Record<WaterAnalysis['plastic_type'], { base: string; mutations: string[] }> = {
    PET: { base: 'PETase', mutations: ['S238F', 'W159H', 'S280A'] },
    HDPE: { base: 'LacCase-HD', mutations: ['T241M', 'G352V'] },
    PVC: { base: 'HaloHyd-VC', mutations: ['C127S', 'L89F'] },
    LDPE: { base: 'AlkB-LDPE', mutations: ['W55L', 'F181Y'] },
    PP: { base: 'CutinasePP', mutations: ['L117F', 'S141G'] },
    PS: { base: 'StyreneOx', mutations: ['M108L', 'H223Y'] },
};

/**
 * Internal monologue entry from a sub-agent
 */
interface MonologueEntry {
    agent: 'ARCHITECT' | 'SAFETY_OFFICER' | 'SIMULATOR';
    timestamp: string;
    thought: string;
    decision?: string;
    rejected?: boolean;
    retry_reason?: string;
}

/**
 * Extended response with internal monologue for debugging
 */
interface CommitteeBioAgentResponse extends BioAgentResponse {
    internal_monologue: MonologueEntry[];
}

/**
 * Determines chassis type based on docs/LOGIC.md Section 1.1
 */
function determineChassisType(salinity: number, stress: boolean): ChassisType {
    if (salinity > 35) {
        return 'Halophilic'; // Lee et al. 2025
    }
    if (stress) {
        return 'Thermophilic'; // Enhanced stress resistance
    }
    return 'Mesophilic'; // Standard conditions
}

/**
 * Calculates efficiency score based on docs/LOGIC.md Section 1.3
 */
function calculateEfficiencyScore(
    salinity: number,
    stress: boolean,
    chassis: ChassisType,
    mutationCount: number
): number {
    let score = 0.60; // Base score

    // Chassis matches optimal salinity range
    if ((salinity > 35 && chassis === 'Halophilic') ||
        (salinity <= 35 && chassis !== 'Halophilic')) {
        score += 0.15;
    }

    // Favorable conditions bonus
    if (!stress) {
        score += 0.10;
    }

    // Stress penalty for Mesophilic
    if (stress && chassis === 'Mesophilic') {
        score -= 0.10;
    }

    // Mutation bonus (max 3 counted)
    score += Math.min(mutationCount, 3) * 0.05;

    // Cap at 0.95
    return Math.min(0.95, Math.round(score * 100) / 100);
}

// =============================================================================
// Committee Mode: Sub-Agent A - The Architect
// =============================================================================

interface ArchitectProposal {
    organism: string;
    organism_description: string;
    chassis_type: ChassisType;
    enzyme_name: string;
    mutation_list: string[];
    rationale: string;
    safety_lock_type?: SafetyLockType; // May be missing initially
}

function runArchitect(input: WaterAnalysis): { proposal: ArchitectProposal; monologue: MonologueEntry } {
    const organism = ORGANISM_CHASSIS[input.plastic_type];
    const enzymeConfig = ENZYME_CONFIG[input.plastic_type];
    const chassis = determineChassisType(input.salinity, input.stress_signal_bool);

    const chassisSuffix = chassis === 'Halophilic' ? '-Halo' :
        chassis === 'Thermophilic' ? '-Thermo' : '';

    const proposal: ArchitectProposal = {
        organism: organism.organism,
        organism_description: organism.description,
        chassis_type: chassis,
        enzyme_name: `${enzymeConfig.base}-v4.2${chassisSuffix}`,
        mutation_list: enzymeConfig.mutations,
        rationale: `Selected ${organism.organism} as chassis organism (${organism.description}). ` +
            `Environmental analysis: salinity=${input.salinity}ppt, stress=${input.stress_signal_bool}. ` +
            `Applying ${chassis} expression system.`,
        // NOTE: Architect might "forget" to add safety lock (intentionally for demo)
        // In a real scenario, this would be randomized or based on complexity
    };

    const monologue: MonologueEntry = {
        agent: 'ARCHITECT',
        timestamp: new Date().toISOString(),
        thought: `Analyzing water sample at (${input.lat}, ${input.lng}). ` +
            `Detected ${input.plastic_type} contamination. Salinity: ${input.salinity}ppt. ` +
            `Stress signals: ${input.stress_signal_bool ? 'PRESENT' : 'absent'}.`,
        decision: `Proposing ${organism.organism} chassis with ${enzymeConfig.base} enzyme. ` +
            `Expression system: ${chassis}. Mutations: ${enzymeConfig.mutations.join(', ')}.`,
    };

    return { proposal, monologue };
}

// =============================================================================
// Committee Mode: Sub-Agent B - The Safety Officer
// =============================================================================

interface SafetyReview {
    approved: boolean;
    required_lock: SafetyLockType;
    additional_locks: SafetyLockType[];
    rejection_reason?: string;
    corrected_proposal?: ArchitectProposal;
}

function runSafetyOfficer(
    proposal: ArchitectProposal,
    input: WaterAnalysis
): { review: SafetyReview; monologue: MonologueEntry } {

    // Check if safety lock is present (Zhang et al. 2025 MANDATORY requirement)
    const hasSafetyLock = proposal.safety_lock_type === 'Quorum_Sensing_Type_B';

    // Determine additional safety measures based on environment
    const additionalLocks: SafetyLockType[] = [];
    if (input.lat > 40 || input.lat < -40) {
        additionalLocks.push('Temperature_Sensitive');
    }
    if (input.salinity > 35) {
        additionalLocks.push('Auxotrophic');
    }

    let review: SafetyReview;
    let monologue: MonologueEntry;

    if (!hasSafetyLock) {
        // REJECT and force retry with corrected proposal
        const correctedProposal: ArchitectProposal = {
            ...proposal,
            safety_lock_type: 'Quorum_Sensing_Type_B',
        };

        review = {
            approved: false,
            required_lock: 'Quorum_Sensing_Type_B',
            additional_locks: additionalLocks,
            rejection_reason: 'CRITICAL VIOLATION: Missing mandatory Quorum_Sensing_Type_B kill switch per Zhang et al. 2025',
            corrected_proposal: correctedProposal,
        };

        monologue = {
            agent: 'SAFETY_OFFICER',
            timestamp: new Date().toISOString(),
            thought: `Reviewing proposal for ${proposal.organism}. ` +
                `Checking safety constraints from docs/LOGIC.md...`,
            decision: `REJECTED - Architect proposal lacks Quorum_Sensing_Type_B lock!`,
            rejected: true,
            retry_reason: 'Forcing retry with mandatory safety lock. Zhang et al. 2025 requires Quorum_Sensing_Type_B for all engineered organisms.',
        };
    } else {
        review = {
            approved: true,
            required_lock: 'Quorum_Sensing_Type_B',
            additional_locks: additionalLocks,
        };

        monologue = {
            agent: 'SAFETY_OFFICER',
            timestamp: new Date().toISOString(),
            thought: `Reviewing proposal for ${proposal.organism}. ` +
                `Verifying Zhang et al. 2025 compliance...`,
            decision: `APPROVED - All safety constraints satisfied. ` +
                `Quorum_Sensing_Type_B verified. ` +
                (additionalLocks.length > 0 ? `Recommending additional locks: ${additionalLocks.join(', ')}.` : 'No additional locks required.'),
        };
    }

    return { review, monologue };
}

// =============================================================================
// Committee Mode: Sub-Agent C - The Simulator
// =============================================================================

interface SimulatorPrediction {
    efficiency_score: number;
    confidence: number;
    environmental_match: 'OPTIMAL' | 'SUBOPTIMAL' | 'MARGINAL';
    notes: string;
}

function runSimulator(
    proposal: ArchitectProposal,
    input: WaterAnalysis
): { prediction: SimulatorPrediction; monologue: MonologueEntry } {

    const efficiencyScore = calculateEfficiencyScore(
        input.salinity,
        input.stress_signal_bool,
        proposal.chassis_type,
        proposal.mutation_list.length
    );

    // Determine environmental match quality
    let envMatch: 'OPTIMAL' | 'SUBOPTIMAL' | 'MARGINAL';
    if (efficiencyScore >= 0.85) {
        envMatch = 'OPTIMAL';
    } else if (efficiencyScore >= 0.70) {
        envMatch = 'SUBOPTIMAL';
    } else {
        envMatch = 'MARGINAL';
    }

    // Calculate confidence based on conditions
    let confidence = 0.85;
    if (input.stress_signal_bool) confidence -= 0.15;
    if (input.salinity > 40) confidence -= 0.10;
    confidence = Math.max(0.50, Math.round(confidence * 100) / 100);

    const prediction: SimulatorPrediction = {
        efficiency_score: efficiencyScore,
        confidence,
        environmental_match: envMatch,
        notes: `Simulated ${proposal.enzyme_name} activity under ${proposal.chassis_type} expression. ` +
            `Temperature/pH match: ${envMatch}. ` +
            `Mutation count: ${proposal.mutation_list.length} (contributing +${Math.min(proposal.mutation_list.length, 3) * 0.05} to score).`,
    };

    const monologue: MonologueEntry = {
        agent: 'SIMULATOR',
        timestamp: new Date().toISOString(),
        thought: `Running Evo 2 efficiency simulation for ${proposal.enzyme_name}. ` +
            `Base chassis: ${proposal.chassis_type}. ` +
            `Environmental parameters: salinity=${input.salinity}ppt, stress=${input.stress_signal_bool}.`,
        decision: `Prediction complete. Efficiency: ${(efficiencyScore * 100).toFixed(1)}% (${envMatch}). ` +
            `Confidence: ${(confidence * 100).toFixed(0)}%. ` +
            `Model ready for deployment recommendation.`,
    };

    return { prediction, monologue };
}

// =============================================================================
// Committee Mode: Mock Gemini Service (Wizard of Oz Pattern)
// =============================================================================

/**
 * Mock Gemini Service with Committee Mode
 * Simulates a debate between 3 sub-agents and outputs internal monologue
 */
class MockGeminiService {
    async generateEnzymeDesign(input: WaterAnalysis): Promise<CommitteeBioAgentResponse> {
        const monologue: MonologueEntry[] = [];

        try {
            // Simulate API latency
            await new Promise(resolve => setTimeout(resolve, 300));

            // ═══════════════════════════════════════════════════════════════
            // PHASE 1: The Architect proposes initial design
            // ═══════════════════════════════════════════════════════════════
            const { proposal: initialProposal, monologue: architectMonologue } = runArchitect(input);
            monologue.push(architectMonologue);

            await new Promise(resolve => setTimeout(resolve, 200));

            // ═══════════════════════════════════════════════════════════════
            // PHASE 2: The Safety Officer reviews (may reject and force retry)
            // ═══════════════════════════════════════════════════════════════
            const { review: safetyReview, monologue: safetyMonologue } = runSafetyOfficer(initialProposal, input);
            monologue.push(safetyMonologue);

            // If rejected, use the corrected proposal
            const finalProposal = safetyReview.approved
                ? { ...initialProposal, safety_lock_type: 'Quorum_Sensing_Type_B' as SafetyLockType }
                : safetyReview.corrected_proposal!;

            // If there was a rejection, log the retry
            if (!safetyReview.approved) {
                monologue.push({
                    agent: 'ARCHITECT',
                    timestamp: new Date().toISOString(),
                    thought: 'Received rejection from Safety Officer. Acknowledging mandatory safety requirement.',
                    decision: `Retry accepted. Adding Quorum_Sensing_Type_B lock to proposal as required by Zhang et al. 2025.`,
                });
            }

            await new Promise(resolve => setTimeout(resolve, 200));

            // ═══════════════════════════════════════════════════════════════
            // PHASE 3: The Simulator predicts efficiency
            // ═══════════════════════════════════════════════════════════════
            const { prediction, monologue: simulatorMonologue } = runSimulator(finalProposal, input);
            monologue.push(simulatorMonologue);

            // ═══════════════════════════════════════════════════════════════
            // FINAL: Assemble the EnzymeDesign from committee consensus
            // ═══════════════════════════════════════════════════════════════
            const design: EnzymeDesign = {
                enzyme_name: finalProposal.enzyme_name,
                mutation_list: finalProposal.mutation_list,
                predicted_efficiency_score: prediction.efficiency_score,
                safety_lock_type: finalProposal.safety_lock_type!,
                chassis_type: finalProposal.chassis_type,
                design_rationale: `[COMMITTEE CONSENSUS] Organism: ${finalProposal.organism} (${finalProposal.organism_description}). ` +
                    `${finalProposal.rationale} ` +
                    `Safety: ${finalProposal.safety_lock_type} verified. ` +
                    `Simulation: ${prediction.efficiency_score * 100}% efficiency (${prediction.environmental_match}).`,
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

// =============================================================================
// Committee Mode: Monologue Printer
// =============================================================================

function printInternalMonologue(monologue: MonologueEntry[]): void {
    console.log('');
    console.log('🧠 INTERNAL MONOLOGUE (Committee Debate):');
    console.log('═'.repeat(70));

    for (const entry of monologue) {
        const agentEmoji = {
            'ARCHITECT': '🏗️',
            'SAFETY_OFFICER': '🛡️',
            'SIMULATOR': '🔬',
        }[entry.agent];

        const agentColor = {
            'ARCHITECT': '\x1b[36m',      // Cyan
            'SAFETY_OFFICER': '\x1b[33m', // Yellow
            'SIMULATOR': '\x1b[35m',      // Magenta
        }[entry.agent];

        const reset = '\x1b[0m';

        console.log('');
        console.log(`${agentColor}${agentEmoji} [${entry.agent}]${reset} @ ${entry.timestamp.split('T')[1].split('.')[0]}`);
        console.log(`   💭 ${entry.thought}`);

        if (entry.decision) {
            const prefix = entry.rejected ? '❌' : '✅';
            console.log(`   ${prefix} ${entry.decision}`);
        }

        if (entry.retry_reason) {
            console.log(`   🔄 ${entry.retry_reason}`);
        }
    }

    console.log('');
    console.log('═'.repeat(70));
}

/**
 * Live Gemini Service - calls the real Gemini 1.5 Flash API
 * Uses the "Wizard of Oz" pattern: prompts Gemini to ACT as Evo 2
 */
class LiveGeminiService {
    private apiKey: string;
    private baseUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent';

    constructor(apiKey: string) {
        this.apiKey = apiKey;
    }

    async generateEnzymeDesign(input: WaterAnalysis): Promise<BioAgentResponse> {
        const systemPrompt = `You are simulating the Evo 2 protein language model for bioremediation enzyme design.
You MUST follow these rules from our safety protocols:

1. CHASSIS SELECTION (Lee et al. 2025):
   - If salinity > 35ppt: MUST use Halophilic chassis
   - If salinity <= 35ppt AND no stress: use Mesophilic chassis
   - If salinity <= 35ppt AND stress signal: use Thermophilic chassis

2. ENZYME SELECTION by plastic type:
   - PET → PETase with mutations S238F, W159H, S280A
   - HDPE → LacCase-HD with mutations T241M, G352V
   - PVC → HaloHyd-VC with mutations C127S, L89F
   - LDPE → AlkB-LDPE with mutations W55L, F181Y
   - PP → CutinasePP with mutations L117F, S141G
   - PS → StyreneOx with mutations M108L, H223Y

3. SAFETY (Zhang et al. 2025):
   - ALWAYS include safety_lock_type: "Quorum_Sensing_Type_B" - this is MANDATORY

4. EFFICIENCY SCORE (0.0-0.95):
   Base = 0.60, +0.15 if chassis matches salinity, +0.10 if no stress, +0.05 per mutation (max 3)

Respond with ONLY a valid JSON object matching this interface:
{
  "enzyme_name": "string",
  "mutation_list": ["string"],
  "predicted_efficiency_score": number,
  "safety_lock_type": "Quorum_Sensing_Type_B",
  "chassis_type": "Halophilic" | "Mesophilic" | "Thermophilic",
  "design_rationale": "string",
  "references": ["string"]
}`;

        const userPrompt = `Design an enzyme for:
- Location: ${input.lat}, ${input.lng}
- Salinity: ${input.salinity}ppt
- Plastic Type: ${input.plastic_type}
- Stress Signal: ${input.stress_signal_bool}`;

        try {
            const response = await fetch(`${this.baseUrl}?key=${this.apiKey}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    contents: [
                        { role: 'user', parts: [{ text: systemPrompt + '\n\n' + userPrompt }] }
                    ],
                    generationConfig: {
                        temperature: 0.2,
                        maxOutputTokens: 1024,
                    }
                }),
            });

            if (!response.ok) {
                throw new Error(`Gemini API error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json() as { candidates?: { content?: { parts?: { text?: string }[] } }[] };
            const text = result.candidates?.[0]?.content?.parts?.[0]?.text;

            if (!text) {
                throw new Error('Empty response from Gemini API');
            }

            // Extract JSON from response (handle markdown code blocks)
            const jsonMatch = text.match(/```json\n?([\s\S]*?)\n?```/) ||
                text.match(/\{[\s\S]*\}/);
            const jsonStr = jsonMatch ? (jsonMatch[1] || jsonMatch[0]) : text;
            const design = JSON.parse(jsonStr) as EnzymeDesign;

            return {
                success: true,
                data: design,
                timestamp: new Date().toISOString(),
            };
        } catch (error) {
            return {
                success: false,
                error: error instanceof Error ? error.message : 'Unknown error',
                timestamp: new Date().toISOString(),
            };
        }
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

async function main(): Promise<void> {
    console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                    POLYMER-X: Bioremediation Command Center                  ║
║                      Enzyme Design Simulation v0.2 (Committee Mode)          ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);

    const args = parseArgs();

    // Build WaterAnalysis input
    const input: WaterAnalysis = {
        lat: args.lat,
        lng: args.lng,
        salinity: args.salinity,
        plastic_type: args.plastic,
        stress_signal_bool: args.stress,
    };

    console.log('📊 INPUT PARAMETERS:');
    console.log('─'.repeat(60));
    console.log(JSON.stringify(input, null, 2));
    console.log('');

    // Select service based on mode
    let service: MockGeminiService | LiveGeminiService;
    let isMockMode = args.mock;

    if (args.mock) {
        console.log('🧪 MODE: Committee Mode Simulation (Wizard of Oz Pattern)');
        console.log('   Sub-Agents: 🏗️ Architect → 🛡️ Safety Officer → 🔬 Simulator');
        console.log('   Using hardcoded logic from docs/LOGIC.md');
        service = new MockGeminiService();
    } else {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            console.error('❌ ERROR: GEMINI_API_KEY environment variable not set');
            console.error('   Set it with: export GEMINI_API_KEY=your_key_here');
            console.error('   Or use --mock flag for simulation mode');
            process.exit(1);
        }
        console.log('🌐 MODE: Live Gemini API (gemini-1.5-flash)');
        console.log('   Prompting Gemini to simulate Committee Mode debate');
        service = new LiveGeminiService(apiKey);
    }

    console.log('');
    console.log('⏳ Running committee debate...');
    console.log('');

    const response = await service.generateEnzymeDesign(input);

    // Print internal monologue first (if available from mock mode)
    const committeeResponse = response as CommitteeBioAgentResponse;
    if (isMockMode && committeeResponse.internal_monologue && Array.isArray(committeeResponse.internal_monologue)) {
        printInternalMonologue(committeeResponse.internal_monologue);
    }

    if (response.success && response.data) {
        console.log('');
        console.log('✅ ENZYME DESIGN OUTPUT (Committee Consensus):');
        console.log('─'.repeat(60));
        console.log(JSON.stringify(response.data, null, 2));
        console.log('');
        console.log('─'.repeat(60));
        console.log(`🕐 Timestamp: ${response.timestamp}`);

        // Validation checks
        console.log('');
        console.log('🔒 SAFETY VALIDATION:');
        if (response.data.safety_lock_type === 'Quorum_Sensing_Type_B') {
            console.log('   ✅ Quorum_Sensing_Type_B verified (Zhang et al. 2025)');
        } else {
            console.log('   ❌ CRITICAL: Missing mandatory Quorum_Sensing_Type_B lock!');
        }

        if (input.salinity > 35 && response.data.chassis_type === 'Halophilic') {
            console.log('   ✅ Halophilic chassis correct for high salinity (Lee et al. 2025)');
        } else if (input.salinity > 35) {
            console.log('   ❌ WARNING: High salinity requires Halophilic chassis!');
        }
    } else {
        console.log('❌ ENZYME DESIGN FAILED:');
        console.log('─'.repeat(60));
        console.log(`   Error: ${response.error}`);
        console.log(`   Timestamp: ${response.timestamp}`);
        process.exit(1);
    }
}

main().catch(console.error);
