# Ultra-Scale CPU Model Engine

OpenAgent includes a local orchestration layer for very large sharded GGUF
models. It uses `llama-server` for tensor execution and adds resource planning,
safe process lifecycle control, warm sessions, cache policy, speculative
decoding configuration, and grounded output verification.

The engine identifier is `openagent-tiered-inference-v1`.

## What It Solves

The engine makes four different states explicit:

- **resident-mmap**: weights, KV cache, and runtime buffers fit usable RAM.
- **out-of-core-mmap**: all GGUF shards exist, but some weights must be paged
  from local storage during inference.
- **trusted-rpc-sharded**: explicitly trusted private nodes supply enough
  aggregate memory for llama.cpp RPC distribution.
- **planning-only**: the hardware calculation is available, but model files are
  not registered and the configuration cannot be launched.

`launchable` and `interactive_predicted` are separate fields. A 200B model can
be launchable with memory mapping and still be far too slow for interactive
chat.

## Why 200B Is Hard On A CPU

A dense 200B model at two bits needs about 50 GB of raw decimal weight storage
(46.6 GiB) before GGUF metadata, quantization tables, KV cache, and runtime
buffers. At four bits the raw weights are about 93.1 GiB. Dense generation
touches essentially all active weights for every token. Splitting a model into
files changes storage and loading, but does not remove that per-token work.

OpenAgent therefore estimates throughput from:

1. active weight bytes per token;
2. bytes resident in local RAM;
3. bytes assigned to trusted RPC memory;
4. bytes that must be read from storage;
5. configured or measured memory, disk, and private-network bandwidth; and
6. a conservative execution-efficiency factor.

For Mixture-of-Experts models, throughput uses active expert parameters while
capacity planning still accounts for the complete model file set.

## Runtime Techniques

- automatic discovery and validation of standard split GGUF names;
- loopback-only `llama-server` binding;
- memory mapping with an optional safe resident `mlock` plan;
- Q8 or configured quantized KV cache;
- prompt prefix caching and cache reuse;
- built-in llama.cpp warmup plus an OpenAgent keep-warm heartbeat;
- compatible draft-model speculation when supplied;
- model-free n-gram speculation otherwise;
- one-slot execution for bandwidth-constrained oversized models;
- short-request routing to the existing small local model;
- escalation of complex and high-assurance requests to the target model; and
- source support, citation, contradiction, and structured-output checks.

These are an original OpenAgent orchestration policy built from established
inference techniques. They are not a claim that software can bypass RAM,
storage, or CPU bandwidth limits.

## Local Setup

Install a current llama.cpp build and make `llama-server` available on `PATH`,
or set:

```powershell
$env:OPENAGENT_LLAMA_SERVER = "D:\llama.cpp\build\bin\Release\llama-server.exe"
```

Open **Developer > Ultra-scale inference**, enter the first GGUF shard path,
register the shards, and run **Plan Feasibility**. Standard names such as
`model-00001-of-00004.gguf` cause the remaining contiguous shards to be found
and validated automatically.

The UI will not start a process until **Approve local model process** is
enabled. The server always binds to loopback.

## API

Read operations:

```text
GET /api/ultra-model/status
GET /api/ultra-model/manifests
GET /api/ultra-model/plans
GET /api/ultra-model/sessions
```

Actions:

```text
POST /api/ultra-model/manifests
POST /api/ultra-model/plan
POST /api/ultra-model/route
POST /api/ultra-model/calibrate
POST /api/ultra-model/launch
POST /api/ultra-model/warmup
POST /api/ultra-model/verify
POST /api/ultra-model/sessions/{id}/stop
```

Example planning request without model files:

```json
{
  "project": "default",
  "specification": {
    "name": "200B Q2 planning profile",
    "architecture": "dense",
    "parameter_count_b": 200,
    "quantization_bits": 2
  },
  "context_tokens": 2048
}
```

## Distributed Safety

llama.cpp describes its RPC backend as experimental and insecure on open
networks. OpenAgent only accepts loopback or private IP endpoints marked
`trusted`. Launch additionally requires `allow_distributed=true` and the local
operator opt-in:

```powershell
$env:OPENAGENT_ENABLE_LLAMA_RPC = "1"
```

Use an authenticated private network or operating-system tunnel. OpenAgent does
not expose the RPC service or claim that the transport is encrypted.

## Correctness Boundary

No local or cloud model can guarantee that every generated answer is correct.
OpenAgent does not hide this limitation. The verification endpoint can mark a
response `verified`, `review`, `rejected`, or `abstain` by checking source
support, citations, simple contradiction signals, and required JSON validity.
High-assurance applications must reject anything other than their configured
acceptance state.

## Privacy And Audit

Model paths, prompts, caches, logs, plans, and verification reports stay in the
configured OpenAgent data directory. Manifest registration, planning,
calibration, launch, warmup, verification, and stop operations write local
audit events. No network call occurs unless a user starts a trusted distributed
deployment or separately enables an external model provider.

## Verification

```powershell
python -m unittest tests.test_ultra_model_engine -v
python -m unittest discover -s tests
openagent-hard-check --pretty
```

The command flags are aligned with the current official llama.cpp server,
speculative decoding, split-GGUF, and RPC documentation:

- https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/gguf-split/README.md
- https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md
