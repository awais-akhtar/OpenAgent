# Model Gateway

The model gateway gives OpenAgent one interface for local, cloud, and hybrid model providers.

## Current Providers

### Local

Provider id: `local`

The offline fallback provider. It is deterministic, dependency-free, and makes no network calls. It exists so chat, agents, workflows, storage, and audit can work before a real model backend is configured.

### Ollama

Provider id: `ollama`

Uses Ollama's local HTTP API. Configure with:

```bash
OPENAGENT_OLLAMA_URL=http://127.0.0.1:11434
OPENAGENT_OLLAMA_MODEL=phi4-mini
```

### llama.cpp

Provider id: `llamacpp`

Uses the OpenAI-compatible chat endpoint exposed by `llama-server`.

```bash
OPENAGENT_LLAMACPP_URL=http://127.0.0.1:8080
OPENAGENT_LLAMACPP_MODEL=local-gguf
```

Register an existing GGUF file from Developer tools or the API, then start the returned local command:

```bash
POST /api/local-ai/llamacpp/register
```

### OpenAI-Compatible

Provider id: `openai-compatible`

Uses a `/chat/completions` compatible API. Configure with:

```bash
OPENAGENT_OPENAI_BASE_URL=https://api.openai.com/v1
OPENAGENT_OPENAI_API_KEY=...
OPENAGENT_OPENAI_MODEL=gpt-4o-mini
```

This provider crosses the local data boundary. Users must explicitly select and configure it.

## Routing and Fallback

Default routing policy:

```text
OPENAGENT_MODEL_ROUTING=local-first
```

The default `local-first` route tries local model servers first: Ollama, then llama.cpp, then the offline fallback. Other supported policies are:

- `offline` - only use the deterministic local provider
- `local-server-first` - prefer Ollama, then llama.cpp, then local
- `cloud-allowed` - allow OpenAI-compatible routing only when the request explicitly sets `allow_external=true`

Default fallback policy:

```text
OPENAGENT_MODEL_FALLBACK=local
```

If a selected provider fails, OpenAgent falls back to the offline local provider. Set `OPENAGENT_MODEL_FALLBACK=none` to disable fallback.

To make Auto route to a real local model, use:

```bash
GET /api/local-ai/setup/status
POST /api/local-ai/models/setup-chat
```

`setup-chat` pulls the selected CPU chat model with Ollama when Ollama is installed and `allow_network=true`. If the installed runner is llama.cpp, register a local `.gguf` path instead; OpenAgent will return the `llama-server` command to start on loopback.

## Streaming

Streaming chat is available at:

```bash
POST /api/chat/stream
```

The endpoint returns server-sent events. The local provider streams deterministic chunks, and network providers can stream through their compatible APIs.

## Smooth Local Runtime

The gateway applies local smooth-mode limits before sending requests to local model servers. It detects available RAM and CPU count, then caps context size, output tokens, worker count, and timeout so OpenAgent can run on CPU-only 8 GB machines without locking the system.

Useful settings:

```text
OPENAGENT_TOTAL_RAM_GB=8
OPENAGENT_MODEL_TIMEOUT_SECONDS=35
OPENAGENT_SMOOTH_CHAT_MODEL=phi4-mini
OPENAGENT_SMOOTH_CODE_MODEL=qwen2.5-coder:3b
OPENAGENT_SMOOTH_RESEARCH_MODEL=phi4-mini
OPENAGENT_SMOOTH_VOICE_ENGINE=faster-whisper:base.en
```

OpenAgent's built-in CPU stack is opinionated:

- Phi-4 Mini 3.8B Q4 for chat, agents, and reasoning-heavy research.
- Qwen2.5 Coder 3B Q4 for coding and debugging.
- Gemma 3 2B Q4 for speed-critical chat on smaller CPUs.
- DeepSeek Coder 1.3B Q4 for lowest-resource coding.
- Llama 3.2 3B Q4 for RAG extraction and fast summarization.
- Qwen2.5 7B Q4 for high-context research on 16 GB+ systems.

Runtime profile APIs:

```bash
GET /api/core/runtime?task=chat
GET /api/core/runtime?task=research
GET /api/local-ai/smooth-runtime?task=chat&target_ram_gb=8
```

The profile is included in provider specs, model policy responses, chat metadata, research output, and local AI status.

## Embeddings

Local deterministic embeddings are available without network calls:

```bash
POST /api/embeddings
{
  "texts": ["OpenAgent local memory"],
  "provider": "local"
}
```

The local embedding implementation is hash-based and intended as a private bootstrap interface. Production semantic embeddings can be added behind the same API.

## API

List providers:

```bash
GET /api/models
```

Send chat:

```bash
POST /api/chat
{
  "message": "Plan the next milestone",
  "provider": "local",
  "project": "default"
}
```

## Extension Points

- vLLM adapter
- ONNX embedding adapter
- Provider health checks
- Token usage accounting
- Local GGUF conversion runners for auto-quantize plans
