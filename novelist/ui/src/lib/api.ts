const BASE_URL = "http://localhost:8000";

export interface SessionConstraintsPayload {
  domains: string[];
  modalities: string[];
  timeline?: string;
  dataset_links: string[];
}

interface StartSessionOptions {
  maxIterations?: number;
  maxTime?: number;
  superprompt?: boolean;
  constraints?: SessionConstraintsPayload;
}

export class API {
  baseUrl: string;

  constructor(baseUrl: string = BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async request(endpoint: string, options: Omit<RequestInit, 'body'> & { body?: any } = {}) {
    const url = `${this.baseUrl}${endpoint}`;
    
    const config: RequestInit = {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    };

    if (options.body && typeof options.body === "object") {
      config.body = JSON.stringify(options.body);
    }

    try {
      const response = await fetch(url, config);

      if (!response.ok) {
         const errorText = await response.text();
         let errorJson;
         try {
             errorJson = JSON.parse(errorText);
         } catch {
             // ignore
         }
        throw new Error((errorJson && errorJson.detail) || errorText || `API Error: ${response.status}`);
      }

      return response.json();
    } catch (err) {
      console.error("API Request Failed:", err);
      throw err;
    }
  }

  async checkStatus() {
    try {
      const response = await fetch(`${this.baseUrl}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  }

  async startSession(topic: string, options: StartSessionOptions = {}) {
    return this.request("/api/sessions", {
      method: "POST",
      body: {
        topic,
        max_iterations: options.maxIterations || 4,
        max_time: options.maxTime || 300,
        superprompt: options.superprompt ?? true,
        constraints: options.constraints,
      },
    });
  }

  async getSessionStatus(sessionId: string) {
    return this.request(`/api/sessions/${sessionId}`);
  }

  async sendChatMessage(sessionId: string, message: string) {
    return this.request(`/api/sessions/${sessionId}/chat`, {
      method: "POST",
      body: { message },
    });
  }

  async pinDirective(sessionId: string, message: string) {
    return this.request(`/api/sessions/${sessionId}/chat/pin`, {
      method: "POST",
      body: { message },
    });
  }

  async unpinDirective(sessionId: string, message: string) {
    return this.request(`/api/sessions/${sessionId}/chat/unpin`, {
      method: "POST",
      body: { message },
    });
  }

  async lockPersona(sessionId: string, personaId: string) {
    return this.request(`/api/sessions/${sessionId}/personas/${personaId}/lock`, {
      method: "POST",
    });
  }

  async unlockPersona(sessionId: string, personaId: string) {
    return this.request(`/api/sessions/${sessionId}/personas/${personaId}/unlock`, {
      method: "POST",
    });
  }

  async updatePersonaWeight(sessionId: string, personaId: string, weight: number) {
    return this.request(`/api/sessions/${sessionId}/personas/${personaId}/weight`, {
      method: "POST",
      body: { weight },
    });
  }

  async regeneratePersona(sessionId: string, personaId: string) {
    return this.request(`/api/sessions/${sessionId}/personas/${personaId}/regenerate`, {
      method: "POST",
    });
  }

  async voteHypothesis(sessionId: string, hypothesisId: string, direction: "up" | "down") {
    return this.request(`/api/sessions/${sessionId}/hypotheses/${hypothesisId}/vote`, {
      method: "POST",
      body: { direction },
    });
  }

  async investigateHypothesis(sessionId: string, hypothesisId: string) {
    return this.request(`/api/sessions/${sessionId}/hypotheses/${hypothesisId}/investigate`, {
      method: "POST",
    });
  }

  async rerunSimulation(sessionId: string, hypothesisId: string, code?: string) {
    return this.request(`/api/sessions/${sessionId}/hypotheses/${hypothesisId}/rerun`, {
      method: "POST",
      body: code ? { code } : undefined,
    });
  }

  async exportSession(sessionId: string, format: "json" | "markdown") {
    return this.request(`/api/sessions/${sessionId}/export?format=${format}`);
  }

  async buryHypothesis(sessionId: string, hypothesisId: string, reason: string) {
    return this.request(`/api/sessions/${sessionId}/hypotheses/${hypothesisId}/graveyard`, {
      method: "POST",
      body: { reason },
    });
  }

  async getSessions() {
    return this.request("/api/sessions");
  }

  async resumeSession(sessionId: string) {
    return this.request(`/api/sessions/${sessionId}/resume`, {
      method: "POST",
    });
  }

  async getKnowledgeStats() {
    return this.request("/api/knowledge/stats");
  }
}

export const api = new API();
