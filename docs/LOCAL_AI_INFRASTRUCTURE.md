# Local AI Infrastructure

Stage 6 is implemented as a local AI infrastructure manager.

## Implemented Features

- Local model registry
- Model download manager
- Network download approval boundary
- Quantization metadata
- Smooth runtime planning for 8 GB CPU-only systems
- Sub-2-bit survival planning for very low-memory CPU systems
- Auto quantization plan records for registered models
- Benchmark runner
- Hardware detection
- GPU acceleration adapter registry
- Multi-node inference cluster planning
- Local model marketplace
- Production GPU model-management plan records
- Fine-tuning and LoRA job records

## APIs

Status:

```bash
GET /api/local-ai/status
```

Models:

```bash
GET /api/local-ai/models
POST /api/local-ai/models/register
POST /api/local-ai/models/download
GET /api/local-ai/setup/status
POST /api/local-ai/models/setup-chat
POST /api/local-ai/ollama/pull
POST /api/local-ai/llamacpp/register
```

Chat model setup:

- `GET /api/local-ai/setup/status` checks only local executables and loopback servers. It reports whether `ollama`, `llama-server`, or `llama-cli` are available and whether their local servers are reachable.
- `POST /api/local-ai/models/setup-chat` chooses the best local setup path. When Ollama is installed or its local server is reachable, it pulls the model only if `allow_network=true`, using the CLI when available or Ollama's local HTTP pull API when the Windows app is running without `ollama.exe` on PATH. It then registers the pulled model as a local provider record. If only llama.cpp is installed, it tells the UI to register a local `.gguf` file because llama.cpp does not provide a built-in model marketplace pull command.
- `POST /api/local-ai/llamacpp/register` registers an existing local `.gguf` file and returns a local command like `llama-server -m "model.gguf" --host 127.0.0.1 --port 8080`. Start that command, then keep Chat provider on Auto or llama.cpp.

Quantization:

```bash
POST /api/local-ai/models/quantize
POST /api/local-ai/models/auto-quantize
GET /api/local-ai/smooth-runtime?task=chat&target_ram_gb=8
GET /api/core/runtime?task=research
```

## Smooth Runtime Mode

Smooth runtime mode chooses conservative local settings from local hardware only. On 8 GB machines it prefers small quantized profiles, single-worker execution, compact context windows, and shorter model outputs so chat, code, voice, and research stay responsive without a GPU.

Default CPU model stack:

- General chat, agents, and reasoning: Phi-4 Mini 3.8B Q4.
- Coding and debugging: Qwen2.5 Coder 3B Q4.
- Speed-critical chat: Gemma 3 2B Q4.
- Lowest-resource coding: DeepSeek Coder 1.3B Q4.
- RAG extraction and fast summarization: Llama 3.2 3B Q4.
- 16 GB+ deep research: Qwen2.5 7B Q4.
- Voice: faster-whisper `base.en` in int8 CPU mode when the optional voice extra is installed.

Runtime tier behavior:

- 3-4 GB survival mode: sub-2-bit `Q1.58` planning, Gemma speed profiles, DeepSeek Coder, single-thread execution, and short context windows.
- 5-6 GB micro mode: 2-bit planning, Gemma 3 2B for chat, DeepSeek Coder for code, and compact staged prompts.
- 8 GB mode: Gemma 3 2B for real-time chat, Qwen2.5 Coder 3B for code, and Llama 3.2 3B for extraction-heavy research.
- 12 GB+ mode: Phi-4 Mini becomes the default chat/agent/research model.
- 16 GB+ mode: Qwen2.5 7B is allowed for high-context research after local benchmark evidence.

The model gateway passes bounded request options to Ollama and llama.cpp: context tokens, output tokens, worker count, temperature, and top-p. `OPENAGENT_MODEL_TIMEOUT_SECONDS` defaults to `35` seconds and is clamped between 5 and 120 seconds to avoid long local hangs.

Auto-quantize records a safe local quantization plan for a registered model. It can plan `quality`, `maximum-compression`, or `below-2bit` modes and estimates runtime memory from parameter count, effective bits, context size, and CPU strategy. It does not upload the model and does not claim a converted artifact exists until a local converter or engine produces one.

The Settings screen exposes runtime controls for chat, code, research, and voice models, Ollama and llama.cpp local server URLs, quantization mode, target RAM, local model timeout, and a one-click CPU model stack restore. Developer tools expose local runner status, smart chat-model setup, GGUF registration, model registration, local/file download, quantization planning, fine-tune navigation, and model record deletion.

Benchmark:

```bash
POST /api/local-ai/benchmark
```

Hardware and adapters:

```bash
GET /api/local-ai/hardware
GET /api/local-ai/adapters
```

Marketplace:

```bash
GET /api/local-ai/marketplace
```

Cluster planning:

```bash
POST /api/local-ai/cluster-plan
```

## Safety

Local file model downloads are allowed. HTTP and HTTPS model downloads require explicit `allow_network=true`. Ollama pulls also require `allow_network=true` and are performed by the local Ollama runtime into its local model store. llama.cpp registration requires a local `.gguf` path and stores only a model record plus the local start command. The infrastructure manager records audit events for downloads, Ollama pulls, and GGUF registrations.

Stage 12 extends this layer with production GPU planning and fine-tuning/LoRA job records. The records are local and auditable; deployments can attach concrete GPU schedulers, training services, quantization converters, or hardware-specific runners without changing the local privacy boundary.

## Adaptive CPU, GPU, And NPU Compiler

The operational heterogeneous runtime is documented in
`docs/ADAPTIVE_EXECUTION_ENGINE.md`. Unlike the earlier adapter registry, this
path discovers actual OpenVINO devices, safely inspects local model containers,
builds separate prefill/decode placements, compiles compatible ONNX/OpenVINO
models on the selected vendor device, and stores measured correctness evidence.

On Windows, install it into an isolated Python 3.12 environment:

```powershell
.\installers\windows\setup-accelerator.ps1
```

The direct runtime keeps native OpenVINO models resident in an isolated worker
and communicates only over inherited local pipes. Use
`setup-accelerator.ps1 -WithConversionTools` plus
`convert-openvino-model.ps1` to prepare a local INT4 GenAI checkpoint. Exact
ONNX/OpenVINO calibration compares CPU, GPU, and NPU outputs before persisting a
measured device preference.

The main OpenAgent environment does not import or upload model weights during
hardware discovery. GGUF remains on the llama.cpp execution path; NPU placement
is used only for model formats and operators accepted by OpenVINO.
