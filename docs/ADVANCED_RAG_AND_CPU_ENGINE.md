# Advanced Local RAG And CPU Inference

OpenAgent includes two connected local performance systems:

- LatticeRAG combines FTS5, FAISS, ChromaDB, reciprocal-rank fusion, evidence diversity, source citations, conflict flags, and generation-aware caching.
- The adaptive CPU inference engine reduces prompt prefill and repeated work with bounded context, intent-based output budgets, physical-core thread planning, warm model sessions, prompt-cache hints, streaming, and an exact-response cache.

## Install

The base package still runs without heavy dependencies. Install the local RAG extra to activate both vector accelerators:

```powershell
python -m pip install -e ".[rag]"
```

Install voice and RAG dependencies together:

```powershell
python -m pip install -e ".[full]"
```

## Storage Architecture

SQLite is the source of truth for collection metadata, chunk lineage, content hashes, text, embeddings, and audit records.

FAISS builds a normalized inner-product index lazily from committed SQLite vectors. It is an in-memory accelerator and can always be reconstructed.

ChromaDB stores a persistent local mirror under `<data-dir>/vectors/chroma`. OpenAgent supplies embeddings directly and disables anonymized telemetry. Chroma does not download an embedding model or receive permission to call an external endpoint.

An upsert is committed to SQLite first. Chroma mirroring occurs afterward and any mirror error is returned and audited. A failed accelerator therefore cannot lose the authoritative record.

## LatticeRAG Pipeline

1. Compute a knowledge-generation fingerprint from local chunk count, timestamps, and document hashes.
2. Create bounded lexical query variants without a model call.
3. Retrieve FTS5/BM25 candidates and semantic candidates.
4. Search normalized vectors through both FAISS and Chroma when installed.
5. Fuse lane ranks with reciprocal-rank fusion and backend-consensus scoring.
6. Rerank by query-term coverage and exact phrase evidence.
7. Apply maximal-marginal-relevance selection to reduce duplicate passages.
8. Pack evidence into a deterministic token budget while preserving source lineage.
9. Flag likely negation conflicts and lower confidence.
10. Return an extractive answer or, when requested, ask the selected model for a strictly evidence-grounded answer with `[S#]` citations.

Cache keys include the knowledge-generation fingerprint. Ingesting or changing source material therefore invalidates old answers without a global cache flush.

## CPU Fast Path

The inference planner estimates prompt size before dispatch. It keeps system instructions and recent turns intact, then creates a lossy digest only when history exceeds the hardware context budget. Short chat prompts receive smaller output budgets, while code and research retain larger limits.

For Ollama, OpenAgent uses a configurable warm-session keepalive and bounded runtime options. For llama.cpp, it sends a prompt-cache hint. Repeated identical requests can be served from an in-process LRU cache, which does not write prompt content to a separate telemetry or cache service.

Useful settings:

```text
OPENAGENT_VECTOR_ACCELERATORS=faiss,chroma
OPENAGENT_CHROMA_KEEP_OPEN=0
OPENAGENT_RAG_CACHE_TTL_SECONDS=900
OPENAGENT_INFERENCE_CACHE_ENTRIES=128
OPENAGENT_INFERENCE_CACHE_TTL_SECONDS=900
OPENAGENT_FIRST_TOKEN_TARGET_MS=1000
OPENAGENT_CPU_THREADS=<physical-core budget>
OPENAGENT_OLLAMA_KEEP_ALIVE=30m
```

## API

```text
GET  /api/vector-store/status
POST /api/vector-store/upsert
POST /api/vector-store/search
POST /api/vector-store/benchmark

GET  /api/rag/status
POST /api/rag/query
POST /api/rag/sync
POST /api/rag/benchmark

GET  /api/inference/status
POST /api/inference/warmup
POST /api/inference/benchmark
POST /api/inference/cache/clear
```

## Performance Claims

OpenAgent targets sub-second warm retrieval and first-token delivery. Exact cache hits should be much faster than one second. Cold model load and token generation cannot be guaranteed at one second on every CPU because model size, quantization, RAM bandwidth, context length, and provider implementation remain physical constraints.

Use the built-in benchmarks on the deployment machine. Engine-overhead results and model-generation results must be reported separately.

### Local Validation Snapshot

Measured on the Windows development machine on 2026-07-11 with FAISS CPU 1.14.3, ChromaDB 1.5.9, and local Ollama `phi4-mini:latest` Q4_K_M:

- Cold dual-backend RAG retrieval: 64.808 ms.
- Warm RAG retrieval: 1.571 ms minimum and 5.513 ms mean over ten runs.
- CPU request planning: 0.096 ms mean over 500 runs.
- Exact response-cache lookup: 0.000221 ms mean over 500 runs.
- Warm model first streamed token: 557 ms in the first live probe.
- The same live probe completed a short generated answer in 1.84 seconds.
- A separate exact-answer probe completed in 815.222 ms on its first local model call and 0.366 ms on the cached repeat.

These are local measurements, not universal guarantees. Cold model warmup in that probe took 3.75 seconds.

## Privacy And Audit Boundary

- FAISS, Chroma, embeddings, RAG caches, and inference caches are local.
- Chroma telemetry is disabled by OpenAgent configuration.
- No external model is selected unless the caller explicitly permits the boundary crossing.
- RAG queries, vector writes, model completions, warmup, cache clearing, and benchmark actions have audit surfaces.
- Native Chroma handles are closed when the HTTP runtime shuts down so Windows data files are not left locked.
