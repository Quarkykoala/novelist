# Novelist Autonomous Discovery Dashboard – Product Requirements Document

## 1. Product Overview
Novelist needs a web dashboard that orchestrates a full autonomous scientific discovery session. A researcher inputs a topic, Novelist forges bespoke expert personas, maps relevant literature, runs multi-agent debates, executes simulations/verification loops with visualization, and surfaces a ranked list of novel hypotheses. Users can steer the session live via chat while monitoring evidence, critiques, and plots. This PRD defines the product experience; it does not cover implementation.

## 2. Goals & Success Criteria
- **Accelerate discovery:** Reduce the time from topic entry to a vetted hypothesis shortlist (<10 minutes for default session).
- **Improve hypothesis novelty & rigor:** Each hypothesis links to literature coverage, debate verdicts, and verification plots so researchers trust recommendations.
- **Support interactive steering:** Live chat feedback changes persona behavior, iteration focus, or hypothesis parameters mid-run.
- **Competition readiness:** Demo-ready UX for the Gemini 3 API Developer Competition, highlighting multimodal reasoning and persona forging.
- **Metrics:** Session completion rate, average time per phase, number of hypotheses promoted, user steering interactions per session, subjective usefulness (≥4/5), plot verification coverage (% hypotheses with simulations).

## 3. Target Users & Personas
1. **Principal Investigator (PI):** Needs rapid landscape scans before committing lab time.
2. **Computational Scientist:** Wants automated hypothesis generation plus simulation traces to validate plausibility.
3. **Innovation Strategist / VC Analyst:** Evaluates frontier opportunities and values ranked novelty with citations.

Accessibility considerations: keyboard navigation, high-contrast themes for plots, realtime status cues for visually-impaired researchers.

## 4. User Journey & Experience Flow
1. **Topic Intake:** User enters research question plus optional constraints (domains, desired modalities, timeline). Dashboard validates and seeds session metadata.
2. **Persona Forge:** System displays forged persona roster (names, expertise, stances) with editable toggles so users can lock or reshuffle before launch.
3. **Literature Mapping Stage:** Progress view showing ingestion sources, cluster visualizations, and “trans-paper gap” annotations.
4. **Hypothesis Studio:** Debate timeline reveals each hypothesis, persona critiques, strength scores, and supporting citations.
5. **Simulation & Verification:** Executed code status, generated plots, Gemini Vision feedback, and pass/fail verdicts per hypothesis.
6. **Ranked Output:** Final list with novelty rank, confidence, key citations, next-step suggestions.
7. **Live Steering Chat:** Persistent panel allowing prompts (“challenge the skeptic”, “expand to bioelectronics”) influencing ongoing loops.
8. **Session Wrap:** Export (PDF/JSON), save to “Graveyard/Memory,” and option to re-run with adjustments.

## Routing Policy
- Commit URLs are invalid.
- Unknown GitHub subpaths canonicalize to repo root.

## 5. Functional Requirements
### [x] US-101: Session Creation & Management
- Form captures topic, constraints, and optional dataset links; validates tokens/key availability.
- Persistent session ID with resume/replay support; show status (queued, forging, mapping, debating, verifying, complete, error).
- Ability to run multiple sessions sequentially with history list.

### [x] US-102: Persona Forge Module
- Generate ≥3 personas per topic with roles (Creative, Skeptic, Specialist) and display expertise, stance, and key objectives.
- Allow user edits: lock persona, request “regenerate persona,” or adjust weightings (e.g., aggressiveness slider).
- Visual indicator of persona influence on later outputs.

### [x] US-103: Literature Mapping & Evidence Board
- Show ingestion status (paper counts, sources, coverage).
- Visualization (graph/map) of clusters, citation overlaps, and trans-paper gaps with hover details.
- Evidence board linking each hypothesis to supporting papers/snippets (capped at 5 top citations per hypothesis).

### [x] US-104: Hypothesis Debate & Evolution
- Timeline/table for hypotheses showing stage, owners, opposing critiques, decision (advance, merge, drop).
- Debate transcripts with persona badges; highlight key critiques and resolutions.
- Buttons for user to up/down-rank hypotheses or request deeper investigation.

### [x] US-105: Simulation & Verification Pipeline
- Display auto-generated code snippets, execution status, and logs.
- Surface plots inline (support multiple types: line, scatter, heatmap) with Gemini Vision commentary (pass/fail, anomalies).
- Allow user-triggered reruns with tweaked parameters.

### [x] US-106: Ranked Output & Export
- Final list sorted by novelty x confidence; each entry includes summary, supporting evidence, simulation verdict, and recommended next experiments.
- Provide export options (PDF, Markdown, JSON) and push to memory (“Graveyard”) with tags.

### [x] US-107: Live Steering Chat
- Chat panel with streaming responses and system prompts showing applied instructions.
- Controls to pin directives (e.g., “prioritize bio-safe materials”) and view effect history.
- Safety guardrails to confirm destructive commands before execution.

### [x] US-108: Notifications & Error Handling
- Toasts/status banners for phase completions, API rate issues, or missing keys.
- Retry actions for failed steps, plus “contact support/log download” option.

## 6. Non-Functional Requirements
- **Performance:** Default session completes <10 minutes; UI updates within 300ms after receiving backend events.
- **Reliability:** Automatic retries for API calls; persistent logs for postmortem.
- **Security:** Secrets stored server-side; enforce auth for dashboard access; log user steering instructions.
- **Scalability:** Architecture supports simultaneous sessions by at least three users without degradation.
- **Compliance:** Cite external data sources; display disclaimers for non-validated hypotheses.

## 7. Data, Integrations, & Dependencies
- Gemini 3 Pro/Flash APIs (reasoning, long-context ingestion, vision analysis).
- Groq/Llama 3.3 for debate loops and structured extraction.
- Literature ingestion sources: arXiv, PubMed, custom uploads; need metadata normalization pipeline.
- Plotting stack (e.g., Plotly) feeding generated images back to Gemini Vision.
- Session storage + “Graveyard” memory database.
- Frontend-backend websocket channel for realtime updates and steering.

## 8. Analytics & Telemetry
- Track per-phase durations, persona regenerate counts, chat interventions, hypothesis conversions, and simulation reruns.
- Instrument errors with correlation IDs; aggregate success metrics for dashboards.
- Privacy controls for anonymizing uploaded literature.

## 9. Risks & Open Questions
- **Compute & Rate Limits:** Need guardrails when simultaneous simulations exceed quotas.
- **Data Licensing:** Confirm rights for scraping or uploading proprietary papers.
- **Simulation Fidelity:** How to validate auto-generated code before execution?
- **User Trust:** Provide transparency on how persona instructions alter hypotheses.
- **Live Steering Scope:** Define boundaries for instructions that could derail experiments.
- **Plot Validation Latency:** Mitigate delays when uploading plots to Gemini Vision.

## 10. Release Strategy
- **Alpha (internal):** Persona forge + literature mapping visualizations, manual trigger for debates.
- **Beta (competition demo):** End-to-end automation with steering chat, limited export options.
- **GA:** Multi-session management, full export suite, granular analytics, enterprise auth.
