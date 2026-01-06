/**
 * API Client — Backend Communication
 */

export class API {
    constructor(baseUrl = '') {
        // Default to same origin, or localhost for development
        this.baseUrl = baseUrl || (window.location.hostname === 'localhost'
            ? 'http://localhost:8000'
            : '');
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;

        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers,
            },
            ...options,
        };

        if (options.body && typeof options.body === 'object') {
            config.body = JSON.stringify(options.body);
        }

        const response = await fetch(url, config);

        if (!response.ok) {
            const error = await response.json().catch(() => ({ message: response.statusText }));
            throw new Error(error.message || `API Error: ${response.status}`);
        }

        return response.json();
    }

    /**
     * Check API connectivity
     */
    async checkStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/api/health`, {
                method: 'GET',
                timeout: 5000
            });
            return response.ok;
        } catch {
            return false;
        }
    }

    /**
     * Start a new hypothesis generation session
     */
    async startSession(topic, options = {}) {
        return this.request('/api/sessions', {
            method: 'POST',
            body: {
                topic,
                max_iterations: options.maxIterations || 4,
                max_time: options.maxTime || 300,
                superprompt: options.superprompt ?? true,
            },
        });
    }

    /**
     * Get session status and results
     */
    async getSessionStatus(sessionId) {
        return this.request(`/api/sessions/${sessionId}`);
    }

    /**
     * Stop a running session
     */
    async stopSession(sessionId) {
        return this.request(`/api/sessions/${sessionId}/stop`, {
            method: 'POST',
        });
    }

    /**
     * Get all sessions (history)
     */
    async getSessions(limit = 20) {
        return this.request(`/api/sessions?limit=${limit}`);
    }

    /**
     * Get session hypotheses
     */
    async getHypotheses(sessionId) {
        return this.request(`/api/sessions/${sessionId}/hypotheses`);
    }

    /**
     * Get knowledge base stats
     */
    async getKnowledgeStats() {
        return this.request('/api/knowledge/stats');
    }

    /**
     * Search papers in knowledge base
     */
    async searchPapers(query, limit = 10) {
        return this.request(`/api/knowledge/search?q=${encodeURIComponent(query)}&limit=${limit}`);
    }
}

// Singleton instance for convenience
export const api = new API();
