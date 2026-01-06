import { useRef, useEffect, useState, useCallback } from 'react';
import { geminiBridge, type WaterAnalysis, type CommitteeBioAgentResponse } from '../services/geminiBridge';
import PhysarumCanvas from './PhysarumCanvas';

// Great Pacific Garbage Patch coordinates
const GARBAGE_PATCH = {
    lat: 32.0,
    lng: -145.0,
};

interface OceanMapProps {
    apiKey?: string;
}

export default function OceanMap({ apiKey }: OceanMapProps) {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const [isDeploying, setIsDeploying] = useState(false);
    const [deploymentResult, setDeploymentResult] = useState<CommitteeBioAgentResponse | null>(null);
    const [showMonologue, setShowMonologue] = useState(false);

    // Load Google Maps 3D
    useEffect(() => {
        if (!apiKey) return;

        const script = document.createElement('script');
        script.src = `https://maps.googleapis.com/maps/api/js?key=${apiKey}&v=alpha&libraries=maps3d`;
        script.async = true;
        script.defer = true;

        script.onload = () => {
            if (mapContainerRef.current) {
                // Create 3D Map element
                const map3d = document.createElement('gmp-map-3d');
                map3d.setAttribute('center', `${GARBAGE_PATCH.lat},${GARBAGE_PATCH.lng}`);
                map3d.setAttribute('altitude', '500000');
                map3d.setAttribute('tilt', '45');
                map3d.setAttribute('heading', '0');
                map3d.setAttribute('range', '2000000');
                map3d.style.width = '100%';
                map3d.style.height = '100%';

                mapContainerRef.current.appendChild(map3d);
            }
        };

        document.head.appendChild(script);

        return () => {
            script.remove();
        };
    }, [apiKey]);

    // Handle Deploy button click
    const handleDeploy = useCallback(async () => {
        setIsDeploying(true);
        setDeploymentResult(null);

        // Create water analysis for Great Pacific Garbage Patch
        const analysis: WaterAnalysis = {
            lat: GARBAGE_PATCH.lat,
            lng: GARBAGE_PATCH.lng,
            salinity: 35.5, // Typical ocean salinity
            plastic_type: 'PET', // Most common ocean plastic
            stress_signal_bool: true, // Oceanic stress conditions
        };

        console.log('🌊 POLYMER-X: Initiating deployment at Great Pacific Garbage Patch...');
        console.log('📊 Water Analysis:', analysis);

        try {
            const result = await geminiBridge.runCommitteeDebate(analysis);
            setDeploymentResult(result);

            // Log the full result
            console.log('');
            console.log('═'.repeat(70));
            console.log('🧠 COMMITTEE DEBATE COMPLETE');
            console.log('═'.repeat(70));

            result.internal_monologue.forEach((entry) => {
                const emoji = { ARCHITECT: '🏗️', SAFETY_OFFICER: '🛡️', SIMULATOR: '🔬' }[entry.agent];
                console.log('');
                console.log(`${emoji} [${entry.agent}]`);
                console.log(`   💭 ${entry.thought}`);
                if (entry.decision) {
                    console.log(`   ${entry.rejected ? '❌' : '✅'} ${entry.decision}`);
                }
                if (entry.retry_reason) {
                    console.log(`   🔄 ${entry.retry_reason}`);
                }
            });

            console.log('');
            console.log('═'.repeat(70));
            console.log('✅ ENZYME DESIGN OUTPUT:');
            console.log(JSON.stringify(result.data, null, 2));
            console.log('═'.repeat(70));

        } catch (error) {
            console.error('❌ Deployment failed:', error);
        } finally {
            setIsDeploying(false);
        }
    }, []);

    return (
        <div className="relative w-full h-full">
            {/* Google Maps 3D Container - or fallback ocean gradient */}
            <div
                ref={mapContainerRef}
                className="absolute inset-0"
                style={{
                    background: apiKey
                        ? '#0a0a1a'
                        : 'linear-gradient(180deg, #0a1628 0%, #0d2847 30%, #0a3d62 60%, #0b4f6c 100%)',
                }}
            >
                {!apiKey && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center opacity-50">
                            <div className="text-6xl mb-4">🌊</div>
                            <div className="text-sm text-cyan-400">
                                Great Pacific Garbage Patch<br />
                                32°N, 145°W
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Physarum Slime Mold Overlay */}
            <PhysarumCanvas
                isActive={!!deploymentResult?.success}
                safetyLockEnabled={deploymentResult?.data?.safety_lock_type === 'Quorum_Sensing_Type_B'}
                centerX={typeof window !== 'undefined' ? window.innerWidth / 2 : 500}
                centerY={typeof window !== 'undefined' ? window.innerHeight / 2 : 400}
            />

            {/* Deploy Button Overlay */}
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                <button
                    onClick={handleDeploy}
                    disabled={isDeploying}
                    className={`
            pointer-events-auto
            px-8 py-4 rounded-xl
            font-bold text-lg uppercase tracking-wider
            transition-all duration-300
            ${isDeploying
                            ? 'bg-gray-700 text-gray-400 cursor-wait'
                            : 'bg-gradient-to-r from-cyan-500 to-blue-600 text-white hover:from-cyan-400 hover:to-blue-500 pulse-glow cursor-pointer'
                        }
          `}
                >
                    {isDeploying ? (
                        <span className="flex items-center gap-3">
                            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                            </svg>
                            Committee Debating...
                        </span>
                    ) : (
                        '🧬 DEPLOY POLYMER-X'
                    )}
                </button>
            </div>

            {/* Result Panel */}
            {deploymentResult && (
                <div className="absolute bottom-4 left-4 right-4 md:left-auto md:right-4 md:w-96">
                    <div className="glass rounded-xl p-4 shadow-2xl">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-cyan-400 font-bold flex items-center gap-2">
                                {deploymentResult.success ? '✅' : '❌'}
                                {deploymentResult.success ? 'Deployment Ready' : 'Deployment Failed'}
                            </h3>
                            <button
                                onClick={() => setShowMonologue(!showMonologue)}
                                className="text-xs text-gray-400 hover:text-white transition-colors"
                            >
                                {showMonologue ? 'Hide' : 'Show'} Debate
                            </button>
                        </div>

                        {deploymentResult.data && (
                            <div className="space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Enzyme:</span>
                                    <span className="text-white font-mono">{deploymentResult.data.enzyme_name}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Efficiency:</span>
                                    <span className={`font-bold ${deploymentResult.data.predicted_efficiency_score >= 0.8 ? 'text-green-400' :
                                            deploymentResult.data.predicted_efficiency_score >= 0.6 ? 'text-yellow-400' : 'text-red-400'
                                        }`}>
                                        {(deploymentResult.data.predicted_efficiency_score * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Safety Lock:</span>
                                    <span className={`${deploymentResult.data.safety_lock_type === 'Quorum_Sensing_Type_B'
                                            ? 'text-green-400' : 'text-red-400'
                                        }`}>
                                        {deploymentResult.data.safety_lock_type === 'Quorum_Sensing_Type_B' ? '🔒 Active' : '⚠️ Missing'}
                                    </span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-gray-400">Chassis:</span>
                                    <span className="text-purple-400">{deploymentResult.data.chassis_type}</span>
                                </div>
                            </div>
                        )}

                        {/* Internal Monologue (expandable) */}
                        {showMonologue && deploymentResult.internal_monologue && (
                            <div className="mt-4 pt-4 border-t border-gray-700 max-h-48 overflow-y-auto text-xs">
                                {deploymentResult.internal_monologue.map((entry, i) => (
                                    <div key={i} className="mb-2 opacity-80">
                                        <span className={`
                      ${entry.agent === 'ARCHITECT' ? 'text-cyan-400' : ''}
                      ${entry.agent === 'SAFETY_OFFICER' ? 'text-yellow-400' : ''}
                      ${entry.agent === 'SIMULATOR' ? 'text-purple-400' : ''}
                    `}>
                                            {entry.agent === 'ARCHITECT' && '🏗️'}
                                            {entry.agent === 'SAFETY_OFFICER' && '🛡️'}
                                            {entry.agent === 'SIMULATOR' && '🔬'}
                                            {' '}{entry.agent}
                                        </span>
                                        <div className="text-gray-400 ml-5">{entry.decision}</div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Location Badge */}
            <div className="absolute top-4 left-4 glass rounded-lg px-4 py-2">
                <div className="text-xs text-gray-400">TARGET ZONE</div>
                <div className="text-cyan-400 font-mono text-sm">
                    Great Pacific Garbage Patch
                </div>
                <div className="text-xs text-gray-500">
                    {GARBAGE_PATCH.lat}°N, {Math.abs(GARBAGE_PATCH.lng)}°W
                </div>
            </div>
        </div>
    );
}
