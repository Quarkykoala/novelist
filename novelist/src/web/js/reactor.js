/**
 * Synthesis Reactor — Orbital Animation Controller
 */

export class Reactor {
    constructor(element) {
        this.element = element;
        this.isRunning = false;
        this.statusLabel = element?.querySelector('.reactor__label');
        this.core = element?.querySelector('.reactor__core');
        this.outerRing = element?.querySelector('.reactor__ring--outer');
        this.innerRing = element?.querySelector('.reactor__ring--inner');
        this.glow = element?.querySelector('.reactor__glow');

        this.phases = [
            'Initializing...',
            'Fetching papers...',
            'Building concept map...',
            'Generating hypotheses...',
            'Critiquing proposals...',
            'Synthesizing results...',
            'Verifying novelty...',
            'Scoring hypotheses...',
        ];
        this.currentPhase = 0;
        this.phaseInterval = null;
    }

    start() {
        if (this.isRunning) return;
        this.isRunning = true;
        this.currentPhase = 0;

        // Increase animation speeds
        if (this.outerRing) {
            this.outerRing.style.animationDuration = '8s';
            this.outerRing.style.borderColor = 'rgba(255, 255, 255, 0.2)';
        }
        if (this.innerRing) {
            this.innerRing.style.animationDuration = '10s';
            this.innerRing.style.borderColor = 'rgba(255, 255, 255, 0.25)';
        }
        if (this.glow) {
            this.glow.style.opacity = '0.8';
            this.glow.style.animationDuration = '2s';
        }
        if (this.core) {
            this.core.style.boxShadow = '0 0 40px rgba(185, 28, 28, 0.5)';
        }

        // Cycle through phases
        this.updatePhase();
        this.phaseInterval = setInterval(() => this.updatePhase(), 3000);
    }

    stop() {
        this.isRunning = false;

        // Reset animation speeds
        if (this.outerRing) {
            this.outerRing.style.animationDuration = '20s';
            this.outerRing.style.borderColor = 'rgba(255, 255, 255, 0.1)';
        }
        if (this.innerRing) {
            this.innerRing.style.animationDuration = '25s';
            this.innerRing.style.borderColor = 'rgba(255, 255, 255, 0.15)';
        }
        if (this.glow) {
            this.glow.style.opacity = '0.5';
            this.glow.style.animationDuration = '4s';
        }
        if (this.core) {
            this.core.style.boxShadow = '0 0 20px rgba(185, 28, 28, 0.2)';
        }

        // Clear phase cycling
        if (this.phaseInterval) {
            clearInterval(this.phaseInterval);
            this.phaseInterval = null;
        }

        this.setStatus('Complete');
    }

    updatePhase() {
        this.setStatus(this.phases[this.currentPhase]);
        this.currentPhase = (this.currentPhase + 1) % this.phases.length;
    }

    setStatus(text) {
        if (this.statusLabel) {
            this.statusLabel.textContent = text;
        }
    }

    setPhase(phaseName) {
        this.setStatus(phaseName);
    }

    pulse() {
        if (!this.core) return;

        this.core.classList.add('pulse-once');
        setTimeout(() => this.core.classList.remove('pulse-once'), 500);
    }
}
