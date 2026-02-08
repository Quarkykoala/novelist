/**
 * Soul Feed — Multi-Soul Message Manager
 */

const SOUL_CONFIG = {
    creative: {
        name: 'The Creative',
        emoji: '🎨',
        gradient: 'linear-gradient(135deg, #fef3c7, #fcd34d)',
        textColor: 'var(--soul-creative)',
    },
    skeptic: {
        name: 'The Skeptic',
        emoji: '🔍',
        gradient: 'linear-gradient(135deg, #f3f4f6, #d1d5db)',
        textColor: 'var(--soul-skeptic)',
    },
    methodical: {
        name: 'The Methodical',
        emoji: '📐',
        gradient: 'linear-gradient(135deg, #dbeafe, #93c5fd)',
        textColor: 'var(--soul-methodical)',
    },
    risktaker: {
        name: 'The Risk Taker',
        emoji: '🚀',
        gradient: 'linear-gradient(135deg, #ede9fe, #c4b5fd)',
        textColor: 'var(--soul-risktaker)',
    },
    synthesizer: {
        name: 'The Synthesizer',
        emoji: '⚗️',
        gradient: 'linear-gradient(135deg, #fee2e2, #fca5a5)',
        textColor: 'var(--soul-synthesizer)',
    },
};

export class SoulFeed {
    constructor(container) {
        this.container = container;
        this.messages = [];
    }

    addMessage(message) {
        const { soul, text, timestamp, highlighted } = message;
        const config = SOUL_CONFIG[soul] || SOUL_CONFIG.creative;

        const card = document.createElement('div');
        card.className = `soul-card soul-card--${soul} animate-slide-in-right`;
        if (highlighted) card.classList.add('soul-card--highlighted');

        const time = timestamp ? this.formatTime(timestamp) : this.getCurrentTime();

        card.innerHTML = `
      <div class="soul-card__avatar">
        <div class="soul-card__avatar-img" style="background: ${config.gradient};">${config.emoji}</div>
        ${soul === 'synthesizer' ? '<div class="soul-card__avatar-status"></div>' : ''}
      </div>
      <div class="soul-card__content">
        <div class="soul-card__header">
          <span class="soul-card__name" style="color: ${config.textColor};">${config.name}</span>
          <span class="soul-card__time">${time}</span>
        </div>
        <p class="soul-card__message">${this.escapeHtml(text)}</p>
      </div>
    `;

        this.container.appendChild(card);
        this.messages.push(message);

        // Scroll to bottom
        this.container.scrollTop = this.container.scrollHeight;

        // Limit visible messages
        if (this.container.children.length > 20) {
            this.container.removeChild(this.container.firstChild);
        }
    }

    addTypingIndicator(soul) {
        const config = SOUL_CONFIG[soul] || SOUL_CONFIG.creative;

        const indicator = document.createElement('div');
        indicator.className = `soul-card soul-card--${soul} soul-card--typing`;
        indicator.id = `typing-${soul}`;

        indicator.innerHTML = `
      <div class="soul-card__avatar">
        <div class="soul-card__avatar-img" style="background: ${config.gradient};">${config.emoji}</div>
      </div>
      <div class="soul-card__content">
        <div class="soul-card__header">
          <span class="soul-card__name" style="color: ${config.textColor};">${config.name}</span>
        </div>
        <div class="typing-dots">
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
          <span class="typing-dot"></span>
        </div>
      </div>
    `;

        this.container.appendChild(indicator);
        this.container.scrollTop = this.container.scrollHeight;
    }

    removeTypingIndicator(soul) {
        const indicator = document.getElementById(`typing-${soul}`);
        if (indicator) indicator.remove();
    }

    clear() {
        this.container.innerHTML = '';
        this.messages = [];
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
    }

    getCurrentTime() {
        return new Date().toLocaleTimeString('en-US', {
            hour: 'numeric',
            minute: '2-digit',
            hour12: true
        });
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}
