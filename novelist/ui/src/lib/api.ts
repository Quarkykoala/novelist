
const BASE_URL = "http://localhost:8000";

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

  async startSession(topic: string, options: any = {}) {
    return this.request("/api/sessions", {
      method: "POST",
      body: {
        topic,
        max_iterations: options.maxIterations || 4,
        max_time: options.maxTime || 300,
        superprompt: options.superprompt ?? true,
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

  async stopSession(sessionId: string) {
    return this.request(`/api/sessions/${sessionId}/stop`, {
      method: "POST",
    });
  }

  async getSessions(limit: number = 20) {
    return this.request(`/api/sessions?limit=${limit}`);
  }

  async getKnowledgeStats() {
    return this.request("/api/knowledge/stats");
  }
}

export const api = new API();
