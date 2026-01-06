# Novelist - Scientific Hypothesis Synthesizer

A "research soul" that iteratively generates, debates, verifies, and refines scientific hypotheses using BDI cognitive architecture + multi-agent debate + Ralph-style quality loops.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set up environment
cp .env.example .env
# Edit .env with your GEMINI_API_KEY

# Run a hypothesis generation session
python -m src.main "CRISPR delivery mechanisms for neural tissue"
```

## Features

- **Multi-Soul Collective**: 5 specialized AI souls (Creative, Skeptic, Methodical, Risk-Taker, Synthesizer) debate and refine hypotheses
- **BDI Architecture**: Beliefs, Desires, Intentions cognitive model for rational agent behavior
- **Ralph Loops**: Iterative refinement until quality thresholds are met
- **arXiv Integration**: Automatic paper ingestion and novelty verification
- **Evidence-Based Scoring**: Novelty, feasibility, impact, and cross-domain scores

## Architecture

```
User Query → Ralph Orchestrator → BDI State Machine → Multi-Soul Debate → Verification → Final Hypotheses
```

## License

MIT
