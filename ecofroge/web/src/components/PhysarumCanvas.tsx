import { useRef, useEffect, useCallback } from 'react';

/**
 * Physarum Polycephalum (Slime Mold) Simulation
 * 
 * A biologically-inspired algorithm where agents:
 * 1. Spawn at the center (deployment zone)
 * 2. Sense plastic zones (red areas) with 45° sensors
 * 3. Leave pheromone trails (green) that diffuse and decay
 * 4. If safety lock is disabled, all particles die (turn grey)
 */

interface PhysarumCanvasProps {
    isActive: boolean;
    safetyLockEnabled: boolean;
    centerX: number;
    centerY: number;
}

// Particle agent struct
interface PhysarumAgent {
    x: number;
    y: number;
    angle: number;
    speed: number;
    alive: boolean;
}

// Configuration
const CONFIG = {
    AGENT_COUNT: 1000,
    SENSOR_ANGLE: Math.PI / 4, // 45 degrees
    SENSOR_DISTANCE: 10,
    MOVE_SPEED: 1.5,
    TURN_SPEED: 0.3,
    PHEROMONE_DEPOSIT: 5,
    PHEROMONE_DECAY: 0.98,
    PHEROMONE_DIFFUSE: 0.1,
    TRAIL_COLOR: { r: 0, g: 255, b: 100 },
    DEAD_COLOR: { r: 100, g: 100, b: 100 },
    PLASTIC_ZONES: [
        // Simulated plastic concentration zones (relative to center)
        { dx: -150, dy: -80, radius: 60 },
        { dx: 200, dy: 50, radius: 80 },
        { dx: -50, dy: 180, radius: 50 },
        { dx: 100, dy: -150, radius: 70 },
        { dx: -200, dy: 100, radius: 55 },
    ],
};

export default function PhysarumCanvas({
    isActive,
    safetyLockEnabled,
    centerX,
    centerY,
}: PhysarumCanvasProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const agentsRef = useRef<PhysarumAgent[]>([]);
    const pheromoneMapRef = useRef<Float32Array | null>(null);
    const animationFrameRef = useRef<number>(0);
    const dimensionsRef = useRef({ width: 0, height: 0 });

    // Initialize agents
    const initializeAgents = useCallback(() => {
        const agents: PhysarumAgent[] = [];
        for (let i = 0; i < CONFIG.AGENT_COUNT; i++) {
            // Spawn at center with random angle
            const angle = Math.random() * Math.PI * 2;
            // Small random offset from center
            const offsetRadius = Math.random() * 20;
            agents.push({
                x: centerX + Math.cos(angle) * offsetRadius,
                y: centerY + Math.sin(angle) * offsetRadius,
                angle: angle,
                speed: CONFIG.MOVE_SPEED * (0.8 + Math.random() * 0.4),
                alive: true,
            });
        }
        agentsRef.current = agents;
    }, [centerX, centerY]);

    // Check if position is in a plastic zone
    const senseChemical = useCallback((x: number, y: number, currentCenterX: number, currentCenterY: number): number => {
        let totalSignal = 0;

        for (const zone of CONFIG.PLASTIC_ZONES) {
            const zoneX = currentCenterX + zone.dx;
            const zoneY = currentCenterY + zone.dy;
            const dist = Math.sqrt((x - zoneX) ** 2 + (y - zoneY) ** 2);

            if (dist < zone.radius) {
                // Stronger signal closer to center
                totalSignal += (zone.radius - dist) / zone.radius * 10;
            }
        }

        // Also sense pheromone trail
        const map = pheromoneMapRef.current;
        const { width, height } = dimensionsRef.current;
        if (map && x >= 0 && x < width && y >= 0 && y < height) {
            const idx = Math.floor(y) * width + Math.floor(x);
            if (idx >= 0 && idx < map.length) {
                totalSignal += map[idx] * 0.5;
            }
        }

        return totalSignal;
    }, []);

    // Main simulation step
    const simulationStep = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        const { width, height } = canvas;
        dimensionsRef.current = { width, height };

        // Initialize pheromone map if needed
        if (!pheromoneMapRef.current || pheromoneMapRef.current.length !== width * height) {
            pheromoneMapRef.current = new Float32Array(width * height);
        }

        const map = pheromoneMapRef.current;
        const agents = agentsRef.current;
        const currentCenterX = width / 2;
        const currentCenterY = height / 2;

        // Clear canvas with fade effect (trail persistence)
        ctx.fillStyle = 'rgba(10, 10, 26, 0.05)';
        ctx.fillRect(0, 0, width, height);

        // Draw plastic zones (red tint)
        ctx.globalAlpha = 0.15;
        for (const zone of CONFIG.PLASTIC_ZONES) {
            const gradient = ctx.createRadialGradient(
                currentCenterX + zone.dx, currentCenterY + zone.dy, 0,
                currentCenterX + zone.dx, currentCenterY + zone.dy, zone.radius
            );
            gradient.addColorStop(0, 'rgba(255, 50, 50, 0.8)');
            gradient.addColorStop(1, 'rgba(255, 50, 50, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(currentCenterX + zone.dx, currentCenterY + zone.dy, zone.radius, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.globalAlpha = 1;

        // Safety check - kill all agents if safety lock missing
        if (!safetyLockEnabled && isActive) {
            // Cell death visualization
            for (const agent of agents) {
                agent.alive = false;
                // Draw dead particles as grey
                ctx.fillStyle = `rgba(${CONFIG.DEAD_COLOR.r}, ${CONFIG.DEAD_COLOR.g}, ${CONFIG.DEAD_COLOR.b}, 0.5)`;
                ctx.beginPath();
                ctx.arc(agent.x, agent.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }
            return;
        }

        // Process each agent
        for (const agent of agents) {
            if (!agent.alive || !isActive) continue;

            // Sensor positions (left, center, right)
            const sensorL = {
                x: agent.x + Math.cos(agent.angle - CONFIG.SENSOR_ANGLE) * CONFIG.SENSOR_DISTANCE,
                y: agent.y + Math.sin(agent.angle - CONFIG.SENSOR_ANGLE) * CONFIG.SENSOR_DISTANCE,
            };
            const sensorC = {
                x: agent.x + Math.cos(agent.angle) * CONFIG.SENSOR_DISTANCE,
                y: agent.y + Math.sin(agent.angle) * CONFIG.SENSOR_DISTANCE,
            };
            const sensorR = {
                x: agent.x + Math.cos(agent.angle + CONFIG.SENSOR_ANGLE) * CONFIG.SENSOR_DISTANCE,
                y: agent.y + Math.sin(agent.angle + CONFIG.SENSOR_ANGLE) * CONFIG.SENSOR_DISTANCE,
            };

            // Sense at each position
            const signalL = senseChemical(sensorL.x, sensorL.y, currentCenterX, currentCenterY);
            const signalC = senseChemical(sensorC.x, sensorC.y, currentCenterX, currentCenterY);
            const signalR = senseChemical(sensorR.x, sensorR.y, currentCenterX, currentCenterY);

            // Turn towards strongest signal
            if (signalC >= signalL && signalC >= signalR) {
                // Keep going straight
            } else if (signalL > signalR) {
                agent.angle -= CONFIG.TURN_SPEED * (1 + Math.random() * 0.5);
            } else if (signalR > signalL) {
                agent.angle += CONFIG.TURN_SPEED * (1 + Math.random() * 0.5);
            } else {
                // Random turn
                agent.angle += (Math.random() - 0.5) * CONFIG.TURN_SPEED;
            }

            // Move forward
            agent.x += Math.cos(agent.angle) * agent.speed;
            agent.y += Math.sin(agent.angle) * agent.speed;

            // Wrap around edges
            if (agent.x < 0) agent.x = width;
            if (agent.x > width) agent.x = 0;
            if (agent.y < 0) agent.y = height;
            if (agent.y > height) agent.y = 0;

            // Deposit pheromone
            const idx = Math.floor(agent.y) * width + Math.floor(agent.x);
            if (idx >= 0 && idx < map.length) {
                map[idx] = Math.min(255, map[idx] + CONFIG.PHEROMONE_DEPOSIT);
            }

            // Draw agent
            const intensity = Math.min(1, map[idx] / 100);
            ctx.fillStyle = `rgba(${CONFIG.TRAIL_COLOR.r}, ${CONFIG.TRAIL_COLOR.g}, ${CONFIG.TRAIL_COLOR.b}, ${0.3 + intensity * 0.7})`;
            ctx.beginPath();
            ctx.arc(agent.x, agent.y, 1.5, 0, Math.PI * 2);
            ctx.fill();
        }

        // Pheromone decay and diffusion
        for (let i = 0; i < map.length; i++) {
            map[i] *= CONFIG.PHEROMONE_DECAY;
        }

    }, [isActive, safetyLockEnabled, senseChemical]);

    // Animation loop
    useEffect(() => {
        if (!isActive) {
            // Clear canvas when inactive
            const canvas = canvasRef.current;
            if (canvas) {
                const ctx = canvas.getContext('2d');
                if (ctx) {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                }
            }
            return;
        }

        // Initialize on activation
        initializeAgents();

        const animate = () => {
            simulationStep();
            animationFrameRef.current = requestAnimationFrame(animate);
        };

        animationFrameRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationFrameRef.current) {
                cancelAnimationFrame(animationFrameRef.current);
            }
        };
    }, [isActive, initializeAgents, simulationStep]);

    // Handle resize
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const handleResize = () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
            if (isActive) {
                initializeAgents();
            }
        };

        handleResize();
        window.addEventListener('resize', handleResize);

        return () => {
            window.removeEventListener('resize', handleResize);
        };
    }, [isActive, initializeAgents]);

    return (
        <canvas
            ref={canvasRef}
            className="absolute inset-0 pointer-events-none"
            style={{ mixBlendMode: 'screen' }}
        />
    );
}
