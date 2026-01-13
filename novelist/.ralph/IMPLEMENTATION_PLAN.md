# Implementation Plan

## Summary
- Session intake currently only collects a topic and the backend discards all metadata, so capturing constraints, resume data, and deterministic phase status is the most urgent blocker to making the dashboard useful for researchers.
- Persona forging already happens server-side, but there is no API or UI exposure for the roster, no ability to lock/regenerate personas, and no way to visualize how they influence debates.
- Literature mapping, hypothesis debate timelines, simulation reruns, exports, interactive chat steering, and notification flows are all partially implemented in the core orchestration code but have no productized surfaces, leaving most PRD stories unsatisfied.

## Tasks
### US-101: Session Creation & Management
- [x] Capture constraints and dataset metadata during session creation
  - Scope: Extend `SessionCreateRequest` plus `RalphConfig` in `src/contracts/schemas.py`, wire the new fields through `src/server.py` into `RalphOrchestrator`, and augment `ui/src/pages/Dashboard.tsx` + `ui/src/lib/api.ts` to render/submit topic constraints (domains, modalities, timeline, dataset URL) with client-side validation.
  - Acceptance: Researchers can enter optional constraints/dataset links, invalid URLs are rejected client-side, and the session payload persisted on disk includes the supplied metadata for later replay.
  - Verification: `pytest tests/test_api_contract_final.py::test_create_session_success` and `cd ui && npm run build`
- [x] Persist session metadata and surface a resumeable history list
  - Scope: Record summaries (topic, constraints, phase history, timestamps) inside `sessions/<id>/summary.json`, expand `/api/sessions` and a new `/api/sessions/{id}/resume` in `src/server.py`, and build a History view (hook up `Sidebar` navigation plus a new history panel component) that lets users reload or resume a past run.
  - Acceptance: The History view lists recent sessions from disk with status badges, selecting one loads its hypotheses without re-running, and hitting "Resume" starts a run with the original config.
  - Verification: `pytest tests/test_api_contract_final.py -k session` and `cd ui && npm run build`
- [x] Implement deterministic session lifecycle states
  - Scope: Define explicit phase enums on `SessionStatusResponse` (`src/server.py`) fed by orchestrator callbacks (`src/ralph/orchestrator.py`), update `_emit_status` calls to use the enum, and render a progress timeline + status badges in `ui/src/pages/Dashboard.tsx`/`ui/src/components/Reactor.tsx`.
  - Acceptance: API responses always contain one of the documented phases (queued, forging, mapping, debating, verifying, complete, error) and the dashboard timeline highlights the current phase rather than a free-form string.
  - Verification: `pytest tests/test_orchestrator_mock.py` and `cd ui && npm run build`

### US-102: Persona Forge Module
- [x] Extend Persona Forge to emit a full roster and expose it via the API
  - Scope: Update `src/soul/prompts/persona_forge.py` to generate at least three personas (creative, skeptic, specialist) with weights/objectives, persist them on `RalphOrchestrator`, and include a `personas` field on session status responses from `src/server.py`.
  - Acceptance: After topic intake the API returns persona metadata (name, role, stance, objectives, instruction) for ≥3 personas and those personas are applied to `SoulCollective` roles.
  - Verification: `pytest tests/test_orchestrator_mock.py::test_orchestrator_flow`
- [x] Add persona locking, regeneration, and weighting endpoints
  - Scope: Introduce routes such as `POST /api/sessions/{id}/personas/{persona_id}/lock` and `POST /api/sessions/{id}/personas/{persona_id}/regenerate` inside `src/server.py`, persist lock state on the orchestrator, and allow weight sliders to influence debate sampling inside `src/soul/collective.py`.
  - Acceptance: Locked personas persist across iterations/resumes, regeneration yields new instructions unless locked, and weight sliders alter the proportion of hypotheses attributed to each persona.
  - Verification: `pytest tests/test_api_contract_final.py -k persona`
- [x] Build Persona Forge UI controls
  - Scope: Create a dedicated persona panel in `ui/src/pages/Dashboard.tsx` backed by reusable card/slider components that display persona attributes, lock toggles, and "Regenerate" buttons that call the new endpoints through `ui/src/lib/api.ts`.
  - Acceptance: Users see persona cards before launch, can lock/unlock/regenerate them, and adjustments immediately reflect in the session payload preview.
  - Verification: `cd ui && npm run build`

### US-103: Literature Mapping & Evidence Board
- [x] Stream ingestion metrics and concept map data
  - Scope: Ensure `_ingest_papers` updates contain normalized counters (papers fetched, categories covered, coverage %) and embed `concept_map` (nodes, edges, trans_paper_gaps) plus top evidence snippets in the session state stored by `src/server.py`.
  - Acceptance: `GET /api/sessions/{id}` returns live ingestion stats and the serialized concept map so the frontend can render coverage and gaps without extra calls.
  - Verification: `pytest tests/test_pipeline.py -k concept_map`
- [x] Implement concept map visualization and knowledge board UI
  - Scope: Add a middle-column section in `ui/src/pages/Dashboard.tsx` (or a dedicated component) that renders the concept map using a lightweight force-graph, shows ingestion gauges, and highlights trans-paper gaps with hover tooltips that link to relevant papers.
  - Acceptance: Researchers can inspect the global map, see coverage metrics update in real time, and hover any gap to view the supporting nodes/papers.
  - Verification: `cd ui && npm run build`
- [x] Build an evidence board linking hypotheses to citations
  - Scope: Limit `Hypothesis.supporting_papers` to the top five items in `src/ralph/orchestrator.py`, attach paper titles/authors from `paper_store`, and add a right-rail evidence board component that shows each hypothesis with its supporting snippets/citations.
  - Acceptance: Every hypothesis card lists ≤5 clickable citations with tooltips from `source_metadata`, and the new evidence board component groups hypotheses by citation clusters per the PRD.
  - Verification: `pytest tests/test_pipeline_fix.py` and `cd ui && npm run build`

### US-104: Hypothesis Debate & Evolution
- [ ] Normalize debate traces for the API
  - Scope: Fix `sessions[session_id]["soulMessages"]` in `src/server.py` so traces are flattened instead of nested lists, embed per-iteration metadata (mode, owners, verdicts) coming from `src/soul/collective.py`, and ensure `_serialize_trace` preserves Skeptic critiques and Synthesizer outcomes.
  - Acceptance: The session status payload includes a chronological array of debate entries with persona badges plus kill/merge decisions, enabling the frontend to reconstruct the debate timeline.
  - Verification: `pytest tests/test_orchestrator_mock.py`
- [ ] Expose hypothesis voting/deeper-investigation controls
  - Scope: Add endpoints such as `POST /api/sessions/{id}/hypotheses/{hid}/vote` and `POST /api/sessions/{id}/hypotheses/{hid}/investigate` that update orchestrator memory/user guidance, and log those interventions to the trace for audit.
  - Acceptance: Votes adjust hypothesis ordering or mark items for deeper study, and the next iteration acknowledges the intervention in both status text and stored trace.
  - Verification: `pytest tests/test_api_contract_final.py`
- [ ] Build a debate timeline UI with ranking buttons
  - Scope: Introduce a timeline/stacked table component in `ui/src/pages/Dashboard.tsx` (or a dedicated component) that visualizes each iteration (phase, persona owner, action, vote controls) and wires the up/down-rank buttons plus “Investigate” CTA to the new APIs.
  - Acceptance: Users can see a scrollable timeline of iterations, apply up/down votes, and observe immediate UI feedback showing the effect on hypothesis ordering.
  - Verification: `cd ui && npm run build`

### US-105: Simulation & Verification Pipeline
- [ ] Expand simulator outputs and rerun hooks
  - Scope: Modify `src/soul/simulator.py` to persist structured status (`queued`, `running`, `vision_pass/fail`), store Gemini Vision commentary alongside code/log/plot paths, and add a backend endpoint that queues a rerun with user-supplied parameter overrides.
  - Acceptance: API responses expose simulation status + commentary per hypothesis, and a rerun request enqueues a new simulation whose outcome replaces or appends to prior runs.
  - Verification: `pytest tests/test_simulator_real.py -k simulator`
- [ ] Build Simulation Lab UI
  - Scope: Refactor `ui/src/components/HypothesisList.tsx` into smaller subcomponents that render simulation status badges, logs, Gemini Vision verdicts, and a rerun form (parameter inputs + rerun button) surfaced inline for each hypothesis.
  - Acceptance: Researchers can inspect code/log/plots plus Vision commentary without leaving the dashboard and trigger reruns with custom parameters.
  - Verification: `cd ui && npm run build`
- [ ] Persist and display rerun history
  - Scope: Store simulation attempts (timestamp, params, verdict) on each hypothesis object inside `src/ralph/orchestrator.py`, expose them via the API, and visualize them as a collapsible history list in the Hypothesis detail view.
  - Acceptance: Each hypothesis shows a rerun history with the ability to switch between attempts, and new runs append entries rather than overwrite silently.
  - Verification: `pytest tests/test_pipeline.py`

### US-106: Ranked Output & Export
- [ ] Align ranking logic with novelty × confidence metric
  - Scope: Update `_merge_hypotheses` and scoring logic in `src/ralph/orchestrator.py` so final ordering uses a documented novelty × feasibility product (or weighted composite), and expose the rank + composite score in the API.
  - Acceptance: Hypotheses are sorted deterministically by the composite metric, and the frontend labels each card with its rank and score per PRD.
  - Verification: `pytest tests/test_tree_search.py`
- [ ] Provide export + Graveyard integration endpoints
  - Scope: Implement `/api/sessions/{id}/export` that produces PDF/Markdown/JSON artifacts, ensure exports include evidence/simulation summaries, and add an action to push hypotheses into the Graveyard (`src/soul/memory.py`) with tags/notes.
  - Acceptance: Users can download all three export formats and send selected hypotheses to the memory store with descriptive tags.
  - Verification: `pytest tests/test_api_contract_final.py -k export`
- [ ] Surface export controls and next-step suggestions in the UI
  - Scope: Add an “Output” panel on the dashboard with rank-sorted cards, recommended next experiments, export buttons, and Graveyard toggles that call the new backend endpoints.
  - Acceptance: Final hypotheses display summaries, recommended next experiments, and provide one-click export/push-to-graveyard controls per the PRD.
  - Verification: `cd ui && npm run build`

### US-107: Live Steering Chat
- [ ] Stream chat acknowledgements with directive history
  - Scope: Enhance `session_chat` in `src/server.py` to stream acknowledgements plus the updated directive backlog, persist directives in `MemorySystem.working.user_guidance`, and include a `directives` array in the status payload.
  - Acceptance: Users see their pinned directives echoed immediately, and each subsequent iteration references the active directives in the Soul Feed.
  - Verification: `pytest tests/test_api_contract_final.py -k chat`
- [ ] Add pinning + destructive-command guardrails
  - Scope: Introduce UI controls to pin/unpin directives, require modal confirmation for instructions flagged as destructive, and enforce backend validation that blocks dangerous commands unless confirmed.
  - Acceptance: Pinned directives remain visible in a side panel, destructive instructions prompt a confirmation modal, and the backend refuses execution without confirmation metadata.
  - Verification: `cd ui && npm run build`
- [ ] Visualize chat impact history
  - Scope: Store directive application logs per iteration in `src/soul/memory.py`, expose them via the API, and render a small history list showing which persona reacted to each directive and when.
  - Acceptance: Researchers can audit how their instructions changed debate behavior via a timestamped history panel.
  - Verification: `pytest tests/test_orchestrator_mock.py`

### US-108: Notifications & Error Handling
- [ ] Categorize backend errors and add retry endpoints
  - Scope: Standardize error codes/messages from `src/server.py` and orchestrator callbacks, write the error log path into the session response, and implement retry endpoints for failed phases (e.g., `/api/sessions/{id}/retry?phase=simulation`).
  - Acceptance: Each failure surfaces a structured error with a retry option and link to downloadable logs, aligning with the PRD’s support guidance.
  - Verification: `pytest tests/test_api_contract_final.py::test_create_session_missing_api_key`
- [ ] Implement toast notifications and log download hooks in the UI
  - Scope: Use a lightweight toast system inside `ui/src/pages/Dashboard.tsx` to announce phase completions, rate-limit warnings, and retries while adding controls to download `sessions/<id>/error.log` via the API.
  - Acceptance: Users receive real-time toasts for each phase transition and can download logs directly from the dashboard when something fails.
  - Verification: `cd ui && npm run build`
- [ ] Surface rate-limit/missing-key warnings proactively
  - Scope: Enhance `src/soul/llm_client.py` to emit structured warnings when keys are missing or rate limits hit, bubble them to FastAPI responses, and render banner notifications in the UI explaining the remediation steps.
  - Acceptance: Attempts to start a session without required keys show actionable messages instead of silent failures, and rate-limit hits display a toast/banner rather than only logging to the console.
  - Verification: `pytest tests/test_llm_simple.py`

## Notes
- The existing `tests/` directory contains several scripts that hit live APIs; when adding automated tests for the new endpoints/components we should prefer to mock LLM responses to avoid flaky runs.
- `sessions[session_id]["soulMessages"]` currently becomes a list of lists when a session completes; fixing this in US-104 is critical because the React `SoulFeed` expects a flat array.
- Any new UI modules should follow the Tailwind/React component conventions already used in `ui/src/components` to keep styling consistent.
