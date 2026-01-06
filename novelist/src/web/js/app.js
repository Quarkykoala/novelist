/**
 * Scientific Hypothesis Synthesizer — Main Application
 */

import { API } from './api.js';
import { Reactor } from './reactor.js';
import { SoulFeed } from './souls.js';

class App {
  constructor() {
    this.api = new API();
    this.reactor = new Reactor(document.getElementById('reactor'));
    this.soulFeed = new SoulFeed(document.getElementById('soul-feed'));

    this.state = {
      topic: '',
      isRunning: false,
      currentLoop: 0,
      maxLoops: 4,
      hypotheses: [],
      sessionId: null,
      logs: [],
      lastSoulCount: 0,
    };

    this.init();
  }

  init() {
    this.bindEvents();
    this.checkApiStatus();
    this.loadTheme();
  }

  bindEvents() {
    // Generate button
    const generateBtn = document.getElementById('generate-btn');
    generateBtn?.addEventListener('click', () => this.startGeneration());

    // Topic input
    const topicInput = document.getElementById('topic-input');
    topicInput?.addEventListener('input', (e) => {
      this.state.topic = e.target.value;
    });
    this.state.topic = topicInput?.value || '';

    // Theme toggle
    const themeToggle = document.getElementById('theme-toggle');
    themeToggle?.addEventListener('click', () => this.toggleTheme());

    // SuperPrompt toggle
    const superPromptToggle = document.getElementById('superprompt-toggle');
    superPromptToggle?.addEventListener('click', () => {
      superPromptToggle.classList.toggle('active');
    });

    // Sidebar navigation
    document.querySelectorAll('.sidebar__link').forEach(link => {
      link.addEventListener('click', (e) => {
        e.preventDefault();
        document.querySelectorAll('.sidebar__link').forEach(l => l.classList.remove('active'));
        link.classList.add('active');
      });
    });
  }

  async checkApiStatus() {
    const statusDot = document.querySelector('.sidebar__status-dot');
    const statusText = document.querySelector('.sidebar__status');

    try {
      const isConnected = await this.api.checkStatus();
      if (isConnected) {
        statusDot.style.background = 'var(--success)';
        statusText.innerHTML = '<span class="sidebar__status-dot"></span>API Connected';
      } else {
        statusDot.style.background = 'var(--warning)';
        statusText.innerHTML = '<span class="sidebar__status-dot" style="background:var(--warning)"></span>API Offline';
      }
    } catch {
      statusDot.style.background = 'var(--error)';
      statusText.innerHTML = '<span class="sidebar__status-dot" style="background:var(--error)"></span>API Error';
    }
  }

  async startGeneration() {
    if (this.state.isRunning) return;
    if (!this.state.topic.trim()) {
      alert('Please enter a research topic');
      return;
    }

    this.state.isRunning = true;
    this.state.currentLoop = 0;
    this.reactor.start();
    this.updateDockStatus('Starting generation...');
    this.clearHypotheses();
    this.soulFeed.clear();
    this.resetLog();

    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = true;
    generateBtn.innerHTML = '<span class="material-symbols-outlined animate-spin-slow">sync</span> Generating...';

    try {
      // Start session
      const session = await this.api.startSession(this.state.topic);
      this.state.sessionId = session.id;

      // Poll for updates
      this.pollForUpdates();

    } catch (error) {
      console.error('Generation failed:', error);
      this.handleGenerationError(error);
    }
  }

  async pollForUpdates() {
    if (!this.state.sessionId || !this.state.isRunning) return;

    try {
      const status = await this.api.getSessionStatus(this.state.sessionId);

      // Update loop counter
      this.state.currentLoop = status.iteration || 0;
      this.updateDockStatus(`Loop ${this.state.currentLoop} / ${this.state.maxLoops} — ${status.phase || 'Processing'}...`);
      this.addLog({
        label: 'status',
        text: `Loop ${this.state.currentLoop} — ${status.phase || 'Processing'}`,
      });

      // Update hypotheses
      if (status.hypotheses?.length) {
        this.updateHypotheses(status.hypotheses);
      }

      // Update gaps
      if (status.gaps?.length) {
        this.updateGaps(status.gaps);
      }

      // Update soul messages
      if (status.soulMessages?.length) {
        this.appendSoulMessages(status.soulMessages);
      }

      // Update gauge
      if (status.relevanceScore) {
        this.updateGauge(status.relevanceScore);
      }

      // Check if complete
      if (status.complete) {
        this.handleGenerationComplete(status);
      } else {
        // Continue polling
        setTimeout(() => this.pollForUpdates(), 2000);
      }

    } catch (error) {
      console.error('Polling error:', error);
      setTimeout(() => this.pollForUpdates(), 3000);
    }
  }

  handleGenerationComplete(status) {
    this.state.isRunning = false;
    this.reactor.stop();

    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = false;
    generateBtn.innerHTML = '<span class="material-symbols-outlined">play_arrow</span> Generate Hypotheses';

    this.updateDockStatus(`Complete — ${status.hypotheses?.length || 0} hypotheses generated`);

    // Final update
    if (status.hypotheses?.length) {
      this.updateHypotheses(status.hypotheses);
    }
  }

  handleGenerationError(error) {
    this.state.isRunning = false;
    this.reactor.stop();

    const generateBtn = document.getElementById('generate-btn');
    generateBtn.disabled = false;
    generateBtn.innerHTML = '<span class="material-symbols-outlined">play_arrow</span> Generate Hypotheses';

    this.updateDockStatus(`Error: ${error.message}`);
  }

  updateHypotheses(hypotheses) {
    const container = document.getElementById('hypothesis-list');
    container.innerHTML = '';

    hypotheses.forEach((h, i) => {
      const card = document.createElement('div');
      card.className = 'hypothesis-card animate-fade-in-up';
      card.style.animationDelay = `${i * 50}ms`;

      // Check for mechanism in rationale
      const hasMechanism = h.rationale && h.rationale.includes('→');
      const mechanismHtml = hasMechanism
        ? '<div style="margin-top:8px;padding:8px;background:rgba(139,92,246,0.1);border-radius:6px;font-size:12px;color:var(--text-dim);"><span style="color:var(--accent);font-weight:600;">⛓️</span> ' + h.rationale.split('.')[0] + '</div>'
        : '';

      const text = h.hypothesis || h.statement || h.text || h;

      card.innerHTML = '<div class="hypothesis-card__number" style="opacity:' + (1 - i * 0.1) + '">' + (i + 1) + '</div>' +
        '<div style="flex:1;"><p class="hypothesis-card__text">' + text + '</p>' + mechanismHtml + '</div>';
      container.appendChild(card);
    });
  }

  clearHypotheses() {
    const container = document.getElementById('hypothesis-list');
    container.innerHTML = `
      <div class="skeleton skeleton--text"></div>
      <div class="skeleton skeleton--text"></div>
      <div class="skeleton skeleton--text"></div>
    `;
  }

  updateGauge(score) {
    const progress = document.getElementById('gauge-progress');
    const value = document.getElementById('gauge-value');

    // Calculate stroke offset (188.5 is the circumference / 2 for a semicircle)
    const offset = 188.5 * (1 - score / 10);
    progress.style.strokeDashoffset = offset;
    value.textContent = score.toFixed(1);
  }

  updateDockStatus(text) {
    const statusText = document.getElementById('dock-status-text');
    if (statusText) statusText.textContent = text;
  }

  appendSoulMessages(messages) {
    const newMessages = messages.slice(this.state.lastSoulCount);
    newMessages.forEach(msg => {
      this.soulFeed.addMessage(msg);
      this.addLog({ label: msg.soul || 'soul', text: msg.text });
    });
    this.state.lastSoulCount = messages.length;
  }

  resetLog() {
    this.state.logs = [];
    this.state.lastSoulCount = 0;
    const log = document.getElementById('activity-log');
    if (log) {
      log.innerHTML = `
        <div class="skeleton skeleton--text"></div>
        <div class="skeleton skeleton--text"></div>
        <div class="skeleton skeleton--text"></div>
      `;
    }
  }

  addLog(entry) {
    const log = document.getElementById('activity-log');
    if (!log) return;

    this.state.logs.push({
      ...entry,
      ts: new Date(),
    });

    // Keep last 20 entries
    this.state.logs = this.state.logs.slice(-20);

    log.innerHTML = '';
    this.state.logs.forEach((item, idx) => {
      const el = document.createElement('div');
      el.className = 'activity-log__item';
      el.style.display = 'flex';
      el.style.alignItems = 'center';
      el.style.justifyContent = 'space-between';
      el.style.padding = '8px 12px';
      el.style.borderRadius = '10px';
      el.style.background = 'rgba(255,255,255,0.04)';
      el.style.border = '1px solid rgba(255,255,255,0.06)';

      el.innerHTML = `
        <div style="display:flex; align-items:center; gap:8px; max-width: 75%;">
          <span style="width:8px;height:8px;border-radius:50%; background: var(--accent); opacity:${0.9 - idx * 0.03};"></span>
          <div style="font-weight:600;">${item.label || 'event'}</div>
          <div style="color: var(--text-dim); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${item.text}</div>
        </div>
        <div style="color: var(--text-muted); font-size: 12px;">${item.ts.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</div>
      `;
      log.appendChild(el);
    });
  }

  updateGaps(gaps) {
    const container = document.getElementById('gaps-list');
    const countBadge = document.getElementById('gaps-count');
    if (!container) return;
    countBadge.textContent = gaps.length;
    container.innerHTML = '';
    const icons = { missing_connection: '🔗', contradiction: '⚡', unexplored_range: '📊', cross_domain: '🌐', mechanism_unknown: '❓' };
    gaps.slice(0, 5).forEach(gap => {
      const el = document.createElement('div');
      el.style.cssText = 'padding:10px 12px;background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);border-radius:8px;font-size:13px;margin-bottom:8px;';
      const icon = icons[gap.type] || '🔍';
      el.innerHTML = '<div style="display:flex;align-items:flex-start;gap:8px;"><span style="font-size:16px;">' + icon + '</span><div style="flex:1;"><div style="font-weight:500;margin-bottom:4px;">' + (gap.concept_a || '') + ' ↔ ' + (gap.concept_b || '') + '</div><div style="color:var(--text-dim);line-height:1.4;">' + (gap.description || '').substring(0, 100) + '...</div></div></div>';
      container.appendChild(el);
    });
  }

  loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') {
      document.body.classList.add('dark');
    }
  }

  toggleTheme() {
    document.body.classList.toggle('dark');
    const isDark = document.body.classList.contains('dark');
    localStorage.setItem('theme', isDark ? 'dark' : 'light');

    const icon = document.querySelector('#theme-toggle .material-symbols-outlined');
    icon.textContent = isDark ? 'light_mode' : 'dark_mode';
  }
}

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
  window.app = new App();
});
