# OpenAgent

OpenAgent is a privacy-first, self-hosted AI operating system created and maintained by Awais Akhtar. It unifies conversation, memory, knowledge, agents, automation, communication, local models, and action in a user-owned platform.

The repository contains a runnable local-first implementation: a native desktop shell, local core service, SQLite-backed memory store, context engine, shared adaptive model gateway, persistent chat, model-backed agents with deterministic local safety fallbacks, workflows, semantic memory, media and voice runtimes, production controls, audit logs, and a full workspace UI.

The source tree includes optional and experimental integrations alongside the local runtime. See [modernization status](docs/MODERNIZATION_STATUS.md) for what has been checked locally and what still needs work before a commercial server deployment.

## Principles

- Self-hosted and local by default
- User-owned data
- Local-first operation with offline paths where the installed provider supports them
- No remote telemetry service is configured by default; local audit and runtime metrics remain on the host
- Explicit approval for shared outbound paths; review older connector paths before server deployment
- Modular architecture
- Provider-independent model routing
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
- Stage 6 local AI infrastructure with model registry/downloads, memory-fit and quantization planning, benchmarks, hardware detection, acceleration adapters, cluster planning, and marketplace records
- Stage 7 enterprise/team deployment with RBAC, multi-user workspaces, organization memory, audit exports, SSO configuration and local validation, backups, disaster recovery plans, policies, and Kubernetes manifests
- Stage 8 developer platform with GraphQL-style queries, webhooks, API key records, plugin/tool registries, MCP manifests, extension packages, and generated SDK artifacts
- Stage 9 security/privacy hardening with encrypted local secret vault implementation, retention controls, egress approvals, sharing grants, audit-chain sealing, privacy reports, and sandbox policy profiles
- Stage 10 workspace and personal assistant layer with folders, docs, wiki pages, dashboards, canvases, whiteboards, mind maps, templates, chat branching/search/import/export/replay, tasks, goals, meeting briefs, travel plans, CRM records, and productivity coaching
- Stage 11 semantic, media, voice, and UI runtime layer with typed semantic memories, local vector collections, reranking, graph analytics, media ingestion, voice sessions, connector sync workers, command palette, shortcuts, and drag/drop workflow layout saves
- Stage 12 production-control layer with policy gates, queue jobs, append-only audit anchoring, vector indexes, semantic metrics, reranking, connector polling adapters, OIDC/JWKS validation, encrypted envelopes, desktop/browser automation records, package manifests, marketplace records, GPU/fine-tune plans, proactive jobs, federated aggregation, encrypted sync bundles, and plugin sandbox records
- Advanced local RAG with SQLite lineage, FAISS acceleration, a persistent ChromaDB mirror, reciprocal-rank fusion, source-grounded answers, conflict flags, and generation-aware caching
- Adaptive CPU inference with prompt compaction, intent-based token budgets, warm local model sessions, exact-response caching, and measured first-token telemetry
- Ultra-scale sharded GGUF orchestration with honest 200B feasibility math, resident/out-of-core/private-RPC plans, safe loopback `llama-server` lifecycle, warm prompt/KV caches, speculative decoding, and grounded output rejection
- Adaptive CPU/GPU/NPU planning engine with GGUF, SafeTensors, ONNX, and OpenVINO frontends; graph/tensor/kernel IR; transfer-aware prefill/decode planning; mixed-precision budgeting; paged KV metadata; deadline batching; vendor compilation hooks; exact-model differential calibration; and an optional persistent network-free OpenVINO worker
- Advanced Engine workspace with persistent model roles, per-role identity prompts, live provider/model assignment, native direct token streaming, custom model sequences, and an audited reasoning-code-supervisor revision loop
- Scientific AI workspace with a finite-difference heat solver, Fourier ridge surrogate, real DeepONet branch/trunk model, constrained engineering optimisation, native local vision routing, structured verification, and materials discovery workflows
- Scientific AI tooling with validated local-model planning and deterministic fallback, evidence-bound result analysis, project-scoped memory, traceback-driven self-correction, AST execution policy, SymPy and Pint checks, an optional FEniCS heat adapter, experiment planning, runtime profiles, and a collapsible workbench navigation; FNO and PINN remain clearly labeled baselines
- CodeQ IDE directly below Chat with a native folder chooser, protected working copies, verified write-back to the selected folder, Explorer, closable multi-file tabs, terminal and Git panels, a live agent feed and editor, model-planned greenfield and existing-project work without a forced default framework, cross-framework host and toolchain preflight, permission-aware tools, repair loops, independent review, snapshots, resumable state, and evidence-based completion gates
- Public CodeQ generator SDK with declarative or protocol-based framework adapters, Python entry-point discovery, artifact and validation contracts, environment preparation plans, and the `openagent-codeq` catalog/inspect/preflight CLI
- Stage 13 local product runtime with submitted-audio STT, local voiceprint clone records, generated WAV voice audio, offline HTML form simulation, approved desktop file automation, and PWA/mobile/desktop package directories. Optional Playwright browser control is documented separately.
- Stage 14 native Windows desktop installability with a standard-library desktop shell, local status command, Start Menu/Desktop launcher scripts, and hard-audit proof that no browser is required
- Stage 15 native desktop workbench exposing the current local OpenAgent feature set through one stable workbench
- Unified runtime foundation with typed tool requests/results, local policy decisions, trace IDs, bounded output, an audited terminal adapter, bounded MCP invocation, and capability-aware model planning. The shared registry is intentionally small today; the existing CodeQ, browser, workflow, and provider surfaces remain separate until they are migrated with contract tests.
- Knowledge upgrades with automatic tags, timeline generation, hybrid search, and source-attributed local synthesis
- Workflow data actions for SQLite reads, CSV spreadsheet read/write, report generation, monitoring checks, and alerts
- Static web workspace for adding notes, searching memory, and running agents
- Standard-library test suite
- Authenticated Docker and Kubernetes deployment baselines; shared-state server deployment remains unfinished

## Quick Start

### Package Install

Install the base local runtime from PyPI:

```bash
pip install openagent
openagent
```

Optional capability groups remain opt-in:

```bash
pip install "openagent[browser,office]"
playwright install chromium
```

`browser` enables approval-aware Playwright automation. `office` enables local
DOCX, PPTX, and XLSX artifact export. Neither group is required for local chat,
memory, workflows, or the desktop workbench.

### Google Colab

In a Colab notebook, install the package and start the same local HTTP runtime:

```python
!pip install openagent

from openagent.colab import launch
runtime = launch()
print(runtime.url, runtime.recommended_model)
```

The launcher probes available NVIDIA memory when `nvidia-smi` is present,
chooses a conservative model recommendation, and stores data under
`/content/openagent` unless `data_dir` is provided. It deliberately does not
create a public tunnel; expose the returned local URL only through an approved
notebook proxy or tunnel.

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

- Chat, Agents, Memory, Workspace, Knowledge, Assistant, Models, Voice + Media, Automation, Security + Dev, and Audit.

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

The local server serves the React 19 OpenAgent Workbench UI from `web/`, built
with TypeScript, Vite, Arco Design and UnoCSS. Its responsive chat, workspace,
assistant, tools, settings and security screens call the existing local
OpenAgent API at the same
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

## Publish to PyPI

The distribution name is `openagent`. Publishing uses PyPI Trusted Publishing
through the GitHub `pypi` environment, not a stored upload token. Configure
that publisher for this repository and workflow in PyPI before tagging.
Ordinary pushes and pull requests run CI but do not upload a package.

After the version in `pyproject.toml` is ready, create and push the matching
tag:

```bash
git tag v0.13.0
git push origin v0.13.0
```

The `Publish to PyPI` workflow verifies the frontend bundle and version tag,
runs the test suite, builds and checks the distributions, and publishes the
same verified artifacts. PyPI must make the `openagent` project name available
to your account; the first upload is the final ownership check.

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

OpenAgent Auto routing tries local model servers first and falls back to the offline `local` provider when no local model server is available. The default role policy names lightweight model families for chat, code, speed, and research; they are recommendations, not bundled weights. Actual availability and speed depend on the installed server, model file, backend, and hardware.

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

- [Modernization status](docs/MODERNIZATION_STATUS.md) - checked changes, deployment boundaries, and remaining work

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
- [Local intelligence services](docs/LOCAL_INTELLIGENCE_SERVICES.md) - durable research, embedding lanes, document processing, image studio, email, calendar, and contacts
- [Advanced RAG and CPU engine](docs/ADVANCED_RAG_AND_CPU_ENGINE.md) - local retrieval and CPU execution controls
- [Adaptive execution engine](docs/ADAPTIVE_EXECUTION_ENGINE.md) - model compilation, hardware planning, and runtime boundaries
- [Disk-backed expert execution](docs/EXPERT_STREAMING.md) - bounded CPU expert weights, streamed SwiGLU execution, the OLMoE family adapter, setup commands, and verification boundaries
- [MoE model families](docs/ENGINE_MODEL_FAMILIES.md) - local library adapters, tensor contracts, placement and architecture differences
- [Packed expert weights](docs/ENGINE_PACKED_QUANTIZATION.md) - INT4 conversion, reference execution and compiled CPU reductions
- [Q8_0 expert integration](docs/ENGINE_Q8_0.md) - matrix/SafeTensors packing, hashed manifests and OLMoE full-model/CLI integration; leased C execution is covered separately by the native range documentation
- [Native Q8_0 CPU kernel](docs/ENGINE_Q8_0_NATIVE.md) - optional explicitly registered Numba raw-byte reductions, exhaustive binary16 decoding checks and bounded activation scratch; not a C backend or measured speedup
- [Q4_0 expert execution](docs/ENGINE_Q4_0.md) - independent block packing, direct-byte scalar/compiled CPU reductions, checkpoint conversion and model-gateway selection
- [Q4_1 expert execution](docs/ENGINE_Q4_1.md) - affine block weights, direct-byte CPU reductions, hashed conversion and format-specific validation
- [Q5_0 expert execution](docs/ENGINE_Q5_0.md) - fifth-bit block layout, direct compiled CPU reductions, checkpoint conversion and gateway selection
- [Q5_1 expert execution](docs/ENGINE_Q5_1.md) - affine fifth-bit blocks, direct compiled reductions and independent model integration tests
- [Grouped INT4 experts](docs/ENGINE_GROUPED_INT4.md) - explicit group metadata, bounded reductions and format-specific validation
- [FP8 block experts](docs/ENGINE_FP8_BLOCK.md) - E4M3FN weights with explicit FP32 block scales and reference/compiled CPU execution
- [Native range execution](docs/ENGINE_NATIVE_SEGMENT_EXECUTION.md) - callback-free C OLMoE range and edge math with transactional KV; resident FP32 and leased packed INT8-row/Q8_0 paths, composed ranges and identity-bound snapshots
- [Native OLMoE operator](docs/ENGINE_NATIVE_OPERATOR.md) - explicit compiled library selection, bounded leased generation and independent-token oracle checks
- [Independent decoders](docs/ENGINE_NATIVE_DECODERS.md) - Mixtral and Qwen3.5 text equations, compiled gated-delta transitions, exact-family oracle tests and remaining native-decoder boundaries
- [Native family execution](docs/ENGINE_NATIVE_FAMILIES.md) - compiled resident Qwen3.5 text arithmetic, transactional hybrid caches, exact-oracle checks and unsupported-family boundaries
- [Native Mixtral execution](docs/ENGINE_NATIVE_MIXTRAL.md) - bounded resident C text decoder, fused/split expert banks, transactional KV and local checkpoint loading
- [GLM-MoE-DSA execution](docs/ENGINE_NATIVE_GLM_DSA.md) - exact resident decoder, compressed latent state, sparse indexing and independent numerical checks
- [Qwen4 text execution](docs/ENGINE_NATIVE_QWEN4.md) - resident linear/QSA/PLE equations, compiled C arithmetic and cached numerical tests
- [Native text checkpoints](docs/ENGINE_NATIVE_CHECKPOINTS.md) - local sharded F32/F16/BF16 loading, exact identities and weight-memory admission
- [GLM5-next text execution](docs/ENGINE_NATIVE_GLM5_NEXT.md) - bounded compiled hybrid text execution and exact oracle scope
- [DeepSeek V4 execution](docs/ENGINE_NATIVE_DEEPSEEK_V4.md) - bounded resident C attention and expert math with token-history replay
- [Inkling text execution](docs/ENGINE_NATIVE_INKLING.md) - resident full/sliding attention, FIR state and dense/MoE layers
- [Kimi Linear execution](docs/ENGINE_NATIVE_KIMI.md) - bounded resident token-to-logit execution and checksum-pinned fused-schema oracle tests; complete published-checkpoint compatibility remains unverified
- [Kimi K3 architecture](docs/ENGINE_NATIVE_KIMI_K3.md) - exact family differences and current implementation boundaries
- [Kimi K3 hybrid execution](docs/ENGINE_NATIVE_KIMI_K3_HYBRID.md) - bounded native KDA/MLA, latent experts and whole-call transactional caches; equation fixtures are separate from published-model verification
- [Decoder execution workstreams](docs/ENGINE_EXECUTION_IMPLEMENTATION.md) - K3 and V4.1 composition, publisher loading, durable state, comparison adapters, and platform runners; this code-first pass remains unexecuted
- [V4.1 FP4 codec](docs/ENGINE_NATIVE_DEEPSEEK_V41_QUANT.md) - local E2M1/E4M3 arithmetic, explicit packing and rounding; not a published checkpoint format claim
- [DeepSeek V4.1 architecture](docs/ENGINE_NATIVE_DEEPSEEK_V41.md) - CSA2 publication ownership primitive and outstanding decoder contracts
- [Qwen3.5 vision subset](docs/ENGINE_NATIVE_QWEN35_VISION.md) - compiled depth-zero image merge fixture; pretrained vision checkpoints remain unsupported
- [Caller-defined E8 kernels](docs/ENGINE_E8_CALLER.md) - explicit lattice codebook and orthogonal transform, hashed identity and direct CPU reductions; not an upstream IQ3 codec
- [Engine implementation boundaries](docs/ENGINE_PARITY.md) - exact family, packed-format and Segment/Edge scope against the reviewed architecture
- [Segment/Edge interface](docs/ENGINE_SEGMENT_ABI.md) - C handles, bounded transfers, snapshot v2 identity and a standalone native FP32 RMSNorm primitive; not a native full-model decoder
- Hardware validation: [Linux](docs/ENGINE_LINUX_VALIDATION.md), [CUDA/HIP](docs/ENGINE_CUDA_HIP_VALIDATION.md), [Metal](docs/ENGINE_METAL_VALIDATION.md) - strict certification remains unverified; target machines were unavailable for the recorded validation assessment
- [Model gateway](docs/MODEL_GATEWAY.md) - provider routing, fallback, and privacy decisions
- [Multi-agent runtime](docs/MULTI_AGENT_RUNTIME.md) - delegation, approvals, parallel work, and verification
- [Workflows](docs/WORKFLOWS.md) - triggers, actions, queues, and rollback
- [Communication hub](docs/COMMUNICATION_HUB.md) - local connectors, inbox, drafts, and approvals
- [Semantic media and voice](docs/SEMANTIC_MEDIA_VOICE.md) - media ingestion, voice sessions, and local transcription
- [Production runtime](docs/PRODUCTION_RUNTIME.md) - production policies, queues, audit anchoring, and packaging
- [Lifecycle completion](docs/LIFECYCLE_COMPLETION.md) - durable research, image, document, identity, model, and proof controls
- [Privacy and security](docs/PRIVACY_SECURITY.md) - local ownership, secrets, approvals, and egress controls
- [Windows desktop](docs/WINDOWS_DESKTOP.md) - install, launch, stable port, and desktop lifecycle
- [Developer platform](docs/DEVELOPER_PLATFORM.md) - APIs, SDKs, webhooks, plugins, and MCP
- [Unified runtime and capability routing](docs/UNIFIED_RUNTIME_AND_CAPABILITY_ROUTING.md) - shared tool execution, MCP invocation, approvals, traces, and model/device planning

## Project Status

OpenAgent contains tested local implementations for chat and model routing, knowledge retrieval, multi-agent execution, workflows, communication records, local model operations, semantic memory, media and voice handling, browser and desktop tool boundaries, Windows installation, and the desktop workbench. Some areas also expose planning or record layers rather than a complete external service. The repository tests the local contracts at service, route, and browser levels. Provider certification, credentials, hardware-specific performance, and operating-system packaging still depend on the target deployment.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged build plan.

See [docs/RELEASES.md](docs/RELEASES.md) for release and package policy.

See [docs/WORKSPACE_ASSISTANT.md](docs/WORKSPACE_ASSISTANT.md) for the workspace, chat management, assistant, knowledge synthesis, and workflow data-action APIs.

See [docs/SEMANTIC_MEDIA_VOICE.md](docs/SEMANTIC_MEDIA_VOICE.md) for semantic memory, local vectors, media ingestion, connector sync workers, voice sessions, command palette, and workflow layout APIs.

See [docs/PRODUCTION_RUNTIME.md](docs/PRODUCTION_RUNTIME.md) for policy, queue, audit anchoring, vector, reranking, connector, crypto, automation, packaging, marketplace, GPU, fine-tuning, proactive, federated, sync, and sandbox APIs.

See [docs/LIFECYCLE_COMPLETION.md](docs/LIFECYCLE_COMPLETION.md) for durable research streaming, reversible image operations, document processing, communication, task outputs, skills, identity, model certification, and production-proof boundaries.

See [docs/ADVANCED_RAG_AND_CPU_ENGINE.md](docs/ADVANCED_RAG_AND_CPU_ENGINE.md) for FAISS, ChromaDB, grounded retrieval, CPU latency controls, privacy boundaries, APIs, and honest benchmark scope. See [docs/ULTRA_CPU_MODEL_ENGINE.md](docs/ULTRA_CPU_MODEL_ENGINE.md) for large sharded GGUF planning, 200B physical limits, warm lifecycle control, private distribution, and output verification.

See [docs/VISION_COVERAGE.md](docs/VISION_COVERAGE.md) for the item-by-item cross-check against the original full OpenAgent vision.

## License

OpenAgent is released under the MIT License. See [LICENSE](LICENSE).
