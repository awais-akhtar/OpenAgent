# OpenAgent

OpenAgent is a privacy-first, self-hosted AI operating system created and maintained by Awais Akhtar. It unifies conversation, memory, knowledge, agents, automation, communication, local models, and action in a user-owned platform.

The repository contains a runnable local-first implementation: a native desktop shell, local core service, SQLite-backed memory store, context engine, shared adaptive model gateway, persistent chat, model-backed agents with deterministic local safety fallbacks, workflows, semantic memory, media and voice runtimes, production controls, audit logs, and a full workspace UI.

## Principles

- 100% self-hosted by default
- User-owned data
- Local-first and offline-capable
- No telemetry
- No external network calls unless explicitly configured
- Modular architecture
- Vendor-independent model routing
- Open standards and inspectable behavior

## What Works Today

- Local HTTP server bound to `127.0.0.1` by default
- Native `openagent-desktop` command with Windows installer and Start Menu/Desktop shortcuts
- Native desktop workbench tabs for workspace, knowledge/research, assistant/comms, models/local AI, voice/media, automation/production, security/developer, chat, agents, memory, and audit
- SQLite event and memory store
- Unified context snapshot from local events
- Search over local context
- Persistent local conversations and chat memory
- Stage 1 model gateway with offline local, Ollama, llama.cpp, and OpenAI-compatible providers
- Model routing, fallback, streaming, provider privacy warnings, and local embeddings
- Built-in agents:
  - Executive Agent
  - Research Agent
  - Coding Agent
  - Writing Agent
  - Planning Agent
  - Automation Agent
  - Knowledge Agent
  - Voice Agent
  - Communication Agent
  - Security Agent
  - Operations Agent
  - Scientific Agent
- Basic local workflow engine
- Audit log for chat, agent, and workflow actions
- Stage 2 knowledge engine with document ingestion, chunking, FTS search, vector search, source attribution, entity extraction, graph relationships, compression, and replay
- Stage 3 multi-agent runtime with permissions, delegation, parallel execution, verification, approvals, and marketplace manifests
- Stage 4 workflow automation with scheduled/event triggers, conditions, file actions, browser/API actions, rollback hooks, background worker, and visual builder model
- Stage 5 communication hub with connectors, contacts, inbox, summaries, follow-ups, drafts, approval-gated sending, calendar, and analytics
- Stage 6 local AI infrastructure with model registry/downloads, smooth 8 GB runtime planning, sub-2-bit CPU survival planning, auto quantization plan records, benchmarks, hardware detection, acceleration adapters, cluster planning, and marketplace
- Stage 7 enterprise/team deployment with RBAC, multi-user workspaces, organization memory, audit exports, SSO config records, local HS256 SSO validation, backups, disaster recovery plans, policies, and Kubernetes manifests
- Stage 8 developer platform with GraphQL-style queries, webhooks, API key records, plugin/tool registries, MCP manifests, extension packages, and generated SDK artifacts
- Stage 9 security/privacy hardening with encrypted local secret vault implementation, retention controls, egress approvals, sharing grants, audit-chain sealing, privacy reports, and sandbox policy profiles
- Stage 10 workspace and personal assistant layer with folders, docs, wiki pages, dashboards, canvases, whiteboards, mind maps, templates, chat branching/search/import/export/replay, tasks, goals, meeting briefs, travel plans, CRM records, and productivity coaching
- Stage 11 semantic, media, voice, and UI runtime layer with typed semantic memories, local vector collections, reranking, graph analytics, media ingestion, voice sessions, connector sync workers, command palette, shortcuts, and drag/drop workflow layout saves
- Stage 12 production runtime layer with policy gates, queue jobs, append-only audit anchoring, production vector indexes, semantic science metrics, production reranking, connector polling adapters, OIDC/JWKS validation, encrypted envelopes, desktop/browser automation records, package manifests, agent marketplace records, GPU/fine-tune plans, proactive jobs, federated aggregation, encrypted sync bundles, and sandboxed plugin run records
- Advanced local RAG with SQLite lineage, FAISS acceleration, a persistent ChromaDB mirror, reciprocal-rank fusion, source-grounded answers, conflict flags, and generation-aware caching
- Adaptive CPU inference with prompt compaction, intent-based token budgets, warm local model sessions, exact-response caching, and measured first-token telemetry
- Ultra-scale sharded GGUF orchestration with honest 200B feasibility math, resident/out-of-core/private-RPC plans, safe loopback `llama-server` lifecycle, warm prompt/KV caches, speculative decoding, and grounded output rejection
- Adaptive CPU/GPU/NPU compiler with GGUF, SafeTensors, ONNX, and OpenVINO frontends; graph/tensor/kernel IR; transfer-aware prefill/decode planning; mixed-precision budgeting; paged KV metadata; deadline batching; vendor compilation; exact-model differential calibration; and a persistent network-free OpenVINO hardware worker
- Advanced Engine workspace with persistent model roles, per-role identity prompts, live provider/model assignment, native direct token streaming, custom model sequences, and an audited reasoning-code-supervisor revision loop
- Scientific AI workspace with a finite-difference heat solver, Fourier ridge surrogate, real DeepONet branch/trunk model, constrained engineering optimisation, native local vision routing, structured verification, and materials discovery workflows
- Scientific AI tooling with validated local-model planning and deterministic fallback, evidence-bound result analysis, project-scoped memory, traceback-driven self-correction, AST execution policy, SymPy and Pint checks, an optional FEniCS heat adapter, experiment planning, runtime profiles, and a collapsible workbench navigation; FNO and PINN remain clearly labeled baselines
- CodeQ IDE directly below Chat with a native folder chooser, protected copied workspaces, Explorer, closable multi-file tabs, terminal and Git panels, a Copilot-style live run feed, repository intelligence, permission-aware tools, repair loops, independent review, snapshots, resumable state, and evidence-based completion gates
- Private PhD Application Studio with reusable academic data, local CV/SOP evidence uploads, position tracking, approval-gated professor and lab research, evidence-grounded document generation, revision history, local quality review, and DOCX/PDF/ZIP exports
- Stage 13 full local product runtime with submitted-audio STT, local voiceprint clone records, generated WAV voice audio, browser/form automation, approved desktop file automation, and real PWA/mobile/desktop package directories
- Stage 14 native Windows desktop installability with a standard-library desktop shell, local status command, Start Menu/Desktop launcher scripts, and hard-audit proof that no browser is required
- Stage 15 native desktop workbench with mission-control tabs for the full local OpenAgent feature set
- Knowledge upgrades with automatic tags, timeline generation, hybrid search, and source-attributed local synthesis
- Workflow data actions for SQLite reads, CSV spreadsheet read/write, report generation, monitoring checks, and alerts
- Static web workspace for adding notes, searching memory, and running agents
- Standard-library test suite
- Docker deployment skeleton

## Quick Start

### Windows Desktop App

Requires Python 3.11 or newer with `tkinter` support. Open PowerShell in the repository root and run:

```powershell
.\installers\windows\install.ps1
```

The default installer includes local voice/STT extras. Use `-SkipVoiceExtras` only when you want the smallest install and do not need Vosk/faster-whisper transcription.

Launch OpenAgent from the Start Menu, Desktop shortcut, or:

```powershell
%LOCALAPPDATA%\OpenAgent\OpenAgent.cmd
```

This opens the native OpenAgent desktop shell. It does not open a browser. The desktop app can show local status, save and search memory, run local chat, run built-in agents, view audit records, and execute the hard audit.

The desktop workbench includes tabs for:

- Chat, PhD Studio, Agents, Memory, Workspace, Knowledge, Assistant, Models, Voice + Media, Automation, Security + Dev, and Audit.

For a no-GUI install check:

```powershell
openagent-desktop --status-json
```

Show the full desktop feature manifest:

```powershell
openagent-desktop --feature-manifest-json
```

Run the no-GUI desktop feature smoke:

```powershell
openagent-desktop --feature-smoke-json --project core
```

Run the full executable hard check:

```powershell
openagent-hard-check --pretty
```

Run the strict project hard check:

```powershell
openagent-hard-check --project hard-check
```

### Local Web/API Server

Requires Python 3.11 or newer.

```bash
python -m openagent --port 8765
```

Then open:

```text
http://127.0.0.1:8765
```

The local server serves the modern OpenAgent Workbench UI from `web/`. It is a
CSS3/Sass-backed, component-style interface for chat, agents, workspace,
knowledge, PhD applications, automation, assistant, communications, models, security, and
developer tools. The UI talks only to the local OpenAgent API at the same
localhost origin unless a specific action asks for an approved external
boundary.

Install the optional isolated Intel CPU/GPU/NPU runtime on Windows:

```powershell
.\installers\windows\setup-accelerator.ps1
```

To convert a local Transformers checkpoint into an INT4 OpenVINO GenAI model:

```powershell
.\installers\windows\setup-accelerator.ps1 -WithConversionTools
.\installers\windows\convert-openvino-model.ps1 -Model D:\models\checkpoint -OutputDirectory D:\models\checkpoint-ov
```

See [Adaptive Execution Engine](docs/ADAPTIVE_EXECUTION_ENGINE.md) for model
formats, state meanings, hardware evidence, privacy boundaries, and APIs.

By default, data is stored in `~/.openagent`. To keep data inside a project folder:

```bash
python -m openagent --data-dir ./data --port 8765
```

### Modern Desktop Workbench

`openagent-desktop` now opens the same modern workbench UI in a localhost app
window by default. It owns the stable `127.0.0.1:8765` runtime, uses a
user-level single-instance lock, and closes the server when the app window
closes. The older Tk shell is still available for fallback testing:

```bash
openagent-desktop
openagent-desktop --legacy-tk
openagent-workbench --status-json
```

During development, run `.\.venv\Scripts\python.exe -m openagent.workbench`
to see source changes without reinstalling. Rerun the Windows installer only
when you want to refresh the installed private environment; both launch paths
share `%LOCALAPPDATA%\OpenAgent\data` by default.

## Run Tests

```bash
python -m unittest discover -s tests
```

## API Preview

```bash
curl http://127.0.0.1:8765/api/health
curl http://127.0.0.1:8765/api/context
curl http://127.0.0.1:8765/api/capabilities
curl http://127.0.0.1:8765/api/models
curl http://127.0.0.1:8765/api/model-policy
curl http://127.0.0.1:8765/api/semantic/status
curl http://127.0.0.1:8765/api/media/status
curl http://127.0.0.1:8765/api/voice/status
curl http://127.0.0.1:8765/api/production/status
curl http://127.0.0.1:8765/api/phd/status
```

Run the executable hard audit:

```bash
curl -X POST http://127.0.0.1:8765/api/capabilities/hard-audit \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"hard-audit\"}"
```

Create a local memory event:

```bash
curl -X POST http://127.0.0.1:8765/api/events \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Project goal\",\"content\":\"Build a privacy-first AI OS\",\"source\":\"manual\"}"
```

Run an agent:

```bash
curl -X POST http://127.0.0.1:8765/api/agents/run \
  -H "Content-Type: application/json" \
  -d "{\"agent\":\"executive\",\"goal\":\"Plan the next OpenAgent milestone\"}"
```

Send a local chat message:

```bash
curl -X POST http://127.0.0.1:8765/api/chat \
  -H "Content-Type: application/json" \
  -d "{\"message\":\"Plan a local-first model gateway\",\"provider\":\"local\"}"
```

Create a workflow:

```bash
curl -X POST http://127.0.0.1:8765/api/workflows \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"Local review\",\"steps\":[{\"action\":\"search_context\",\"query\":\"OpenAgent\"}]}"
```

Create a semantic memory:

```bash
curl -X POST http://127.0.0.1:8765/api/semantic/memories \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"title\":\"Model preference\",\"content\":\"Use local models unless cloud is approved.\",\"memory_type\":\"preference\",\"tags\":[\"privacy\"]}"
```

## Model Providers

OpenAgent Auto routing tries local model servers first and falls back to the offline `local` provider when no local model server is available. The built-in CPU stack uses Phi-4 Mini for chat/research, Qwen2.5 Coder 3B for code, Gemma 3 2B for speed, DeepSeek Coder 1.3B for low-resource coding, and Qwen2.5 7B only for 16 GB+ research.

Ollama:

```bash
set OPENAGENT_OLLAMA_URL=http://127.0.0.1:11434
set OPENAGENT_OLLAMA_MODEL=phi4-mini
```

llama.cpp:

```bash
set OPENAGENT_LLAMACPP_URL=http://127.0.0.1:8080
set OPENAGENT_LLAMACPP_MODEL=phi4-mini
```

OpenAI-compatible:

```bash
set OPENAGENT_OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAGENT_OPENAI_API_KEY=your-key
set OPENAGENT_OPENAI_MODEL=gpt-4o-mini
```

## Repository Map

- `openagent/` - local core service, storage, context engine, and agents
- `web/` - static workspace UI served by the local core
- `docs/` - architecture, roadmap, privacy, and agent design
- `tests/` - standard-library regression tests
- `Dockerfile` and `docker-compose.yml` - self-hosted deployment skeleton

## Documentation

The README is a starting point; the design and operational details live in the
linked documents below.

- [Architecture](docs/ARCHITECTURE.md) - system layers, data flow, and module boundaries
- [Dashboard control surface](docs/DASHBOARD_CONTROL_SURFACE.md) - real-time local telemetry, model latency, readiness, and audit activity
- [Autonomous software engineering](docs/AUTONOMOUS_SOFTWARE_ENGINEERING.md) - codebase workspaces, repository intelligence, task graphs, tools, sandbox boundaries, approvals, repair loops, verification, and live events
- [Chat, memory, voice, and research](docs/CHAT_MEMORY_AND_RESEARCH.md) - conversation recall, project memory, attachments, source-backed discovery, voice mode, privacy boundaries, and history organization
- [Scientific AI](docs/SCIENTIFIC_AI.md) - scientific agents, tools, memory, verification, multimodal work, materials discovery, and experiment planning
- [Scientific Agents](docs/science/SCIENTIFIC_AGENTS.md) - planning, local evidence, bounded execution, self-correction, profiling, and audit
- [Scientific Computing](docs/science/SCIENTIFIC_COMPUTING.md) - reference solvers, operator-learning interfaces, simulation adapters, and benchmarks
- [Engineering Optimisation](docs/science/ENGINEERING_OPTIMISATION.md) - constrained beam and heat-exchanger workflows
- [Multimodal Science](docs/science/MULTIMODAL_SCIENCE.md) - local PDF, text, table, image, equation, and simulation evidence fusion
- [Scientific Verification](docs/science/SCIENTIFIC_VERIFICATION.md) - numeric, equation, unit, constraint, citation, and consistency checks
- [Materials Discovery](docs/science/MATERIALS_DISCOVERY.md) - local datasets, prediction, ranking, experiment selection, tables, and charts
- [Roadmap](docs/ROADMAP.md) - staged implementation plan
- [Vision coverage](docs/VISION_COVERAGE.md) - feature-by-feature coverage map
- [Workspace and assistant](docs/WORKSPACE_ASSISTANT.md) - workspace, chat, assistant, and knowledge workflows
- [Knowledge engine](docs/KNOWLEDGE_ENGINE.md) - ingestion, retrieval, graph, and source attribution
- [Advanced RAG and CPU engine](docs/ADVANCED_RAG_AND_CPU_ENGINE.md) - local retrieval and CPU execution controls
- [Adaptive execution engine](docs/ADAPTIVE_EXECUTION_ENGINE.md) - model compilation, hardware planning, and runtime boundaries
- [Model gateway](docs/MODEL_GATEWAY.md) - provider routing, fallback, and privacy decisions
- [Multi-agent runtime](docs/MULTI_AGENT_RUNTIME.md) - delegation, approvals, parallel work, and verification
- [Workflows](docs/WORKFLOWS.md) - triggers, actions, queues, and rollback
- [Communication hub](docs/COMMUNICATION_HUB.md) - local connectors, inbox, drafts, and approvals
- [PhD Application Studio](docs/PHD_APPLICATION_STUDIO.md) - private evidence, position research, and document generation
- [Semantic media and voice](docs/SEMANTIC_MEDIA_VOICE.md) - media ingestion, voice sessions, and local transcription
- [Production runtime](docs/PRODUCTION_RUNTIME.md) - production policies, queues, audit anchoring, and packaging
- [Lifecycle completion](docs/LIFECYCLE_COMPLETION.md) - durable research, image, document, identity, model, and proof controls
- [Privacy and security](docs/PRIVACY_SECURITY.md) - local ownership, secrets, approvals, and egress controls
- [Windows desktop](docs/WINDOWS_DESKTOP.md) - install, launch, stable port, and desktop lifecycle
- [Developer platform](docs/DEVELOPER_PLATFORM.md) - APIs, SDKs, webhooks, plugins, and MCP

## Project Status

OpenAgent has completed the first sixteen roadmap slices, from Stage 0 through Stage 15. The current implementation is a complete local-first implementation for the requested self-hosted feature set; certified enterprise appliances, public app stores, and specific external-provider credentials can still be added by deployments. It now includes runnable local core, model gateway, knowledge, multi-agent, workflow, communication, local AI infrastructure, enterprise/team deployment, developer platform, security/privacy hardening, workspace/personal-assistant, semantic memory, media, voice, UI productivity, production-runtime coverage, local app packaging, browser/form automation, desktop file automation, submitted-audio STT, local TTS audio, native Windows desktop installability, and a native desktop workbench that exposes the full feature set.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged build plan.

See [docs/RELEASES.md](docs/RELEASES.md) for release and package policy.

See [docs/STAGE_COMPLETION_AUDIT.md](docs/STAGE_COMPLETION_AUDIT.md) for the Stage 0-15 completion check.

See [docs/WORKSPACE_ASSISTANT.md](docs/WORKSPACE_ASSISTANT.md) for the workspace, chat management, assistant, knowledge synthesis, and workflow data-action APIs.

See [docs/SEMANTIC_MEDIA_VOICE.md](docs/SEMANTIC_MEDIA_VOICE.md) for semantic memory, local vectors, media ingestion, connector sync workers, voice sessions, command palette, and workflow layout APIs.

See [docs/PRODUCTION_RUNTIME.md](docs/PRODUCTION_RUNTIME.md) for policy, queue, audit anchoring, vector, reranking, connector, crypto, automation, packaging, marketplace, GPU, fine-tuning, proactive, federated, sync, and sandbox APIs.

See [docs/LIFECYCLE_COMPLETION.md](docs/LIFECYCLE_COMPLETION.md) for durable research streaming, reversible image operations, document processing, communication, task outputs, skills, identity, model certification, and production-proof boundaries.

See [docs/PHD_APPLICATION_STUDIO.md](docs/PHD_APPLICATION_STUDIO.md) for the private applicant data vault, public research boundary, grounded writing workflow, revisions, exports, and PhD API routes.

See [docs/ADVANCED_RAG_AND_CPU_ENGINE.md](docs/ADVANCED_RAG_AND_CPU_ENGINE.md) for FAISS, ChromaDB, grounded retrieval, CPU latency controls, privacy boundaries, APIs, and honest benchmark scope. See [docs/ULTRA_CPU_MODEL_ENGINE.md](docs/ULTRA_CPU_MODEL_ENGINE.md) for large sharded GGUF planning, 200B physical limits, warm lifecycle control, private distribution, and output verification.

See [docs/VISION_COVERAGE.md](docs/VISION_COVERAGE.md) for the item-by-item cross-check against the original full OpenAgent vision.

## License

OpenAgent is released under the MIT License. See [LICENSE](LICENSE).
