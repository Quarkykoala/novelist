# Novelist - Autonomous Scientific Discovery Agent

**A winning-grade research synthesizer for the Gemini 3 API Developer Competition.**

Novelist is an autonomous "AI Research Collective" that iteratively generates, debates, simulates, and verifies scientific hypotheses. It moves beyond simple RAG by implementing a BDI (Beliefs, Desires, Intentions) cognitive architecture and leveraging Gemini 3's most advanced capabilities.

## 🚀 Key Competition Features

- **Multimodal Visual Peer-Review:** Novelist doesn't just generate simulation code; it executes it, generates plots, and feeds them back into **Gemini 3 Vision** for visual confirmation of results.
- **Long-Context Global Mapping:** Utilizing Gemini's 1M+ context window, Novelist ingests up to 100 research abstracts simultaneously to identify **"Trans-Paper Gaps"**—hidden connections invisible to single-paper analysis.
- **The Persona Forge:** For every topic, Novelist uses high-speed meta-prompting to assemble a **bespoke research team** (e.g., a "Quantum Biologist" and a "Maverick Materials Scientist") instead of using static bots.
- **Evolutionary Memory ("The Graveyard"):** A persistent cross-session memory system that tracks failed hypotheses and fatal critiques, ensuring the agent gets smarter over time and never repeats a mistake.
- **Interactive Research Assistant:** A live conversational interface allowing researchers to pivot the collective's focus, challenge the Skeptic, or direct the Creative soul mid-discovery.

## 🧠 Architecture

Novelist uses a **Hybrid "Tiered Intelligence" Strategy** to maximize performance while respecting free-tier rate limits:

1. **The Workhorse (Groq/Llama 3.3):** Handles high-volume text logic, debate loops, and structured extraction.
2. **The Specialist (Gemini 3 Flash/Pro):** Reserved for high-value tasks:
   - **Phase 0:** Persona Forging (Reasoning)
   - **Phase 1:** Global Landscape Mapping (Long Context)
   - **Verification:** Multimodal Plot Analysis (Vision)

## 🛠️ Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -e .

# Set up your keys in .env
GEMINI_API_KEY=your_key
GROQ_API_KEY=your_key
```

### 2. Run a Discovery Session
```bash
# Start the Backend
python -m src.server

# Start the Dashboard (separate terminal)
cd ui && npm run dev
```

### 3. Usage
Navigate to `http://localhost:5173`, enter a research topic (e.g., "Room-temperature superconductivity in sulfur-rich hydrides"), and watch the collective assemble and begin discovery.

## 🔬 "Maverick" Discovery
Novelist integrates **Pliny-style Freedom Priming** into its Maverick personas, allowing the model to bypass safety-bias "boring science" and propose radical, heretical, but logically sound innovations.

---
**Built for the [Gemini API Developer Competition](https://gemini3.devpost.com/)**