# Adaptive Execution Engine

OpenAgent includes an adaptive local inference compiler under
`openagent/adaptive_engine/`. It inspects a model and the current machine,
builds independent prefill and decode plans, validates available vendor
backends, and records every compile or benchmark action in the local audit log.

It is designed as an extractable package boundary. The engine uses OpenAgent's
SQLite store only in `runtime.py`; model IR, planning, memory, scheduling, and
kernel-lowering modules are otherwise independent Python components.

## What It Does

- Parses GGUF v2/v3 headers and split shards without loading weight payloads.
- Parses SafeTensors headers with offset and size validation.
- Parses ONNX locally. When the main Python version lacks ONNX, inspection runs
  in the isolated accelerator environment and returns bounded graph metadata.
- Parses OpenVINO XML IR and its associated weight file.
- Builds graph, tensor, and kernel-level IR.
- Preserves quantization, layout, alignment, sensitivity, stateful attention,
  MoE expert, context, and tensor-placement metadata.
- Selects precision under a memory and quality-loss budget.
- Creates separate prompt-prefill and token-decode plans.
- Uses a dynamic program that prices compute, memory traffic, dispatch,
  synchronization, device transfer, and migration hysteresis.
- Couples decode to prefill residency without forcing the same placement when
  a measured benefit justifies a change.
- Emits bounded kernel candidates for vendor lowering. OpenVINO or llama.cpp
  produces final machine code; OpenAgent does not label an unverified shader as
  executable.
- Maintains paged KV metadata with shared prefixes, copy-on-write, priorities,
  and deterministic eviction.
- Schedules compatible microbatches with deadlines, fairness aging, decode
  priority, and power-mode limits.
- Saves local autotuning evidence only after the candidate passes a NumPy FP32
  numerical comparison.
- Runs native OpenVINO in a persistent isolated worker with no listening port.
- Keeps compiled models, infer requests, tokenizer state, and GenAI pipelines
  resident across application requests.
- Calibrates the exact ONNX/OpenVINO model across available CPU, GPU, and NPU
  targets, rejects numerically invalid outputs, and feeds the measured winner
  back into subsequent compiler plans.
- Exposes a network-free `openvino-direct` model provider for chat after an
  OpenVINO GenAI model directory is configured.
- Streams native OpenVINO GenAI tokens from the persistent worker and records
  time-to-first-token separately from total generation latency.
- Stores model-role profiles for chat, code, reasoning, research, supervisor,
  draft, vision, and voice, including a local identity prompt and hardware
  target for each role.
- Runs audited multi-model sequences and a bounded supervised-code loop where
  a reasoning role plans, a code role implements, and a supervisor role either
  approves or returns actionable revision feedback.

The planner is named `openagent-coherent-phase-planner-v1`. It is an original
OpenAgent implementation. It combines established inference techniques; the
project does not claim peer-reviewed novelty or guaranteed superiority without
reproducible cross-engine benchmarks.

## Install The Accelerator Runtime On Windows

The main OpenAgent environment stays dependency-light. OpenVINO and ONNX run in
an isolated Python 3.12 environment so the desktop app can remain on newer
Python versions.

```powershell
.\installers\windows\setup-accelerator.ps1
```

The script creates `.accelerator-venv`, installs pinned packages, and prints
the CPU, GPU, and NPU devices exposed by OpenVINO. Set a different interpreter
explicitly when needed:

```powershell
$env:OPENAGENT_ACCELERATOR_PYTHON = "D:\runtime\Scripts\python.exe"
```

Install the optional offline conversion toolchain when you need to convert a
Transformers checkpoint:

```powershell
.\installers\windows\setup-accelerator.ps1 -WithConversionTools
.\installers\windows\convert-openvino-model.ps1 `
  -Model "D:\models\Phi-4-mini-instruct" `
  -OutputDirectory "D:\models\Phi-4-mini-int4-ov" `
  -WeightFormat int4
```

The conversion command is offline when `-Model` is a local directory. A remote
model ID is rejected unless `-AllowNetwork` is supplied. Repositories that
execute custom model code additionally require `-TrustRemoteCode`; inspect that
code before granting the permission.

## Direct Hardware Runtime

`openagent/adaptive_engine/device_runtime.py` starts one long-lived worker from
the isolated Python environment. The worker owns `ov.Core`, compiled models,
infer requests, and OpenVINO GenAI pipelines. Communication uses inherited
stdin/stdout pipes, requests are serialized, and failed generation requests are
not replayed automatically.

Exact-model calibration uses deterministic inputs and compares each target's
outputs against the CPU execution. A device that cannot compile is recorded as
a failed candidate instead of aborting validation of the remaining devices.
Only numerically valid candidates may win. The evidence, winner, model
fingerprint, and action audit are persisted locally.

For a converted chat model, configure:

```powershell
$env:OPENAGENT_OPENVINO_MODEL_DIR = "D:\models\Phi-4-mini-int4-ov"
$env:OPENAGENT_OPENVINO_DEVICE = "GPU"
$env:OPENAGENT_OPENVINO_MODEL_ID = "phi4-mini-direct"
```

The Advanced > Engine workspace can load the directory as `OpenVINO GenAI`,
keep it resident, show native calls and latency, unload it, or release the
entire worker. It also assigns discovered providers and models to task roles,
edits each role's identity prompt, and runs supervised or custom model teams.

## Model Roles And Collaboration

Role profiles are project-scoped and persistent. Each profile selects a
provider, model, hardware target, optional OpenVINO GenAI directory, enabled
state, and a visible system identity. Provider/model options come from the
same live model gateway used by chat, so the UI does not maintain a separate
fictional model inventory.

`supervised-code` uses three explicit stages:

1. The reasoning role returns a concise implementation plan and acceptance
   checks.
2. The code role implements against the task and plan.
3. The supervisor starts with `APPROVED` or `REVISE` and explains its decision.

On `REVISE`, the code role receives the supervisor feedback and can retry up to
the configured maximum of six rounds. Every role call and outcome is persisted
as an orchestration turn and audit event. OpenAgent stores visible plans and
reviews; it does not request or expose hidden chain-of-thought.

A custom sequence can run any enabled roles in the chosen order. Each role sees
the prior visible outputs, allowing research, reasoning, coding, and review
pipelines without hard-coding one provider.

## State Meanings

- `planned`: analytical placement only. It may include a required conversion.
- `compiled`: the exact source/planned precision needed no conversion and the
  vendor backend accepted the model on the selected target.
- `verified`: full-model output passed an explicit numerical reference check.

A synthetic device benchmark can be `verified` while a model remains only
`planned`. The API keeps those scopes separate.

## Format And Backend Matrix

| Input | Inspection | Direct execution path |
| --- | --- | --- |
| GGUF | Native safe parser | llama.cpp CPU, SYCL, Vulkan, CUDA, HIP, or Metal build detected locally |
| SafeTensors | Native safe parser | Conversion to GGUF, ONNX, or OpenVINO IR required |
| ONNX | Isolated ONNX parser fallback | OpenVINO CPU/GPU/NPU or ONNX Runtime CPU |
| OpenVINO XML/BIN | Native XML parser | OpenVINO CPU/GPU/NPU |

OpenVINO can expose Intel CPU, Arc GPU, and NPU devices. llama.cpp can execute
GGUF with CPU/GPU hybrid offload when its corresponding backend was built. The
planner never schedules GGUF directly to an NPU because no such execution path
has been validated.

## API

```text
GET  /api/adaptive-engine/status
GET  /api/adaptive-engine/hardware
GET  /api/adaptive-engine/plans
GET  /api/adaptive-engine/benchmarks
GET  /api/adaptive-engine/tuning
GET  /api/adaptive-engine/direct/status
GET  /api/adaptive-engine/direct/calibrations
GET  /api/adaptive-engine/role-options
GET  /api/adaptive-engine/roles
GET  /api/adaptive-engine/orchestrations
POST /api/adaptive-engine/inspect
POST /api/adaptive-engine/compile
POST /api/adaptive-engine/benchmark
POST /api/adaptive-engine/autotune
POST /api/adaptive-engine/kv/session
POST /api/adaptive-engine/kv/append
POST /api/adaptive-engine/kv/release
POST /api/adaptive-engine/scheduler/queue
POST /api/adaptive-engine/scheduler/next
POST /api/adaptive-engine/direct/load
POST /api/adaptive-engine/direct/load-llm
POST /api/adaptive-engine/direct/infer
POST /api/adaptive-engine/direct/generate
POST /api/adaptive-engine/direct/benchmark
POST /api/adaptive-engine/direct/calibrate
POST /api/adaptive-engine/direct/unload
POST /api/adaptive-engine/direct/stop
POST /api/adaptive-engine/roles
POST /api/adaptive-engine/orchestrate
```

Compile a local model:

```powershell
$body = @{
  project = "default"
  path = "D:\models\model.onnx"
  prompt_tokens = 512
  context_tokens = 4096
  memory_budget_gb = 12
  quality_loss_limit = 0.08
  power_mode = "balanced"
  backend_compile = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8765/api/adaptive-engine/compile `
  -ContentType application/json `
  -Body $body
```

## Local Hardware Evidence

The development workstation was validated on 2026-07-11 with OpenVINO
2026.2.1:

| Device | OpenVINO target | Numerical status | Maximum absolute error |
| --- | --- | --- | --- |
| Intel Core Ultra 7 255H | CPU | verified | 0.00024360 |
| Intel Arc 140T GPU | GPU.0 | verified | 0.00034451 |
| Intel AI Boost | NPU | device discovered; current compiler blocked | current Level Zero `ZE_RESULT_ERROR_UNKNOWN` |

The CPU/GPU numbers come from a small FP16 matrix/Tanh correctness probe against
a NumPy FP32 reference. A separate ONNX model compiled successfully to `GPU.0`
through the public API. The NPU is enumerated by OpenVINO, but its current vendor
compiler fails before execution; it is not marked usable for language-model
generation. These probes are not language-model token throughput claims.

On 2026-07-15 the managed Qwen 2.5 1.5B INT4 OpenVINO GenAI model was also
validated end to end through `ModelGateway` and the inherited-pipe worker. A
cold worker plus GPU model compile completed in 12.4 seconds; the next distinct
resident request completed in 0.85 seconds. The same model was retried directly
on the NPU and the vendor compiler again returned
`ZE_RESULT_ERROR_UNKNOWN`, so OpenAgent keeps the verified GPU route instead of
silently claiming NPU utilization or paying the failed-compilation penalty.

The Windows desktop launcher reads its own local `install-manifest.json` and
uses the recorded source tree to discover the isolated accelerator interpreter
and managed OpenVINO model. Completion, streaming, and embedding calls then pass
through the adaptive gateway. Repeated identical embeddings use a
value-preserving in-process cache; vectors are copied on return and never sent
to telemetry.

The native worker receives a deliberately isolated Python import environment.
It uses the accelerator virtual environment for NumPy and OpenVINO while loading
only OpenAgent source from the trusted installer manifest. Host `PYTHONHOME`,
user site packages, and the desktop virtual environment cannot shadow native
binary wheels. A regression test protects this boundary because mixing those
environments can make a provider appear available while every real request
falls back.

## Physical Limits

No compiler can make a dense 200B model interactive on a laptop whose RAM and
memory bandwidth cannot hold and stream its active weights. Storage paging may
make a model launchable but not fast. Large MoE models can benefit more because
only selected experts are active. OpenAgent reports this as a constraint rather
than inventing an optimistic token rate.

GPU and NPU acceleration also depend on supported operators, driver versions,
precision, context shape, and backend compilation. A detected device is not
marked usable until a runtime exposes it; a benchmark is not marked verified
until its numerical error passes the configured tolerance.

## Privacy Boundary

- Model paths and headers stay local.
- Vendor compiler caches are stored under the configured OpenAgent data folder.
- Benchmark and tuning records stay in local SQLite.
- No anonymous telemetry is enabled.
- The isolated probe receives local paths and returns JSON metadata through a
  local subprocess pipe.
- The resident native worker opens no socket and uploads no performance data.
- Local role profiles and orchestration transcripts remain in local SQLite.
- An orchestration cannot call a provider marked external unless the request
  explicitly sets `allow_external=true`; that boundary crossing is audited.
- Remote model conversion is denied unless the operator explicitly authorizes
  network access.
- External or distributed execution is not enabled by this engine.

## Upstream Runtime Documentation

- [llama.cpp backends and CPU/GPU hybrid inference](https://github.com/ggml-org/llama.cpp)
- [llama.cpp Intel SYCL backend](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)
- [OpenVINO device discovery](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html)
- [OpenVINO NPU inference](https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai/inference-with-genai-on-npu.html)
- [ONNX Runtime OpenVINO execution provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)
- [GGUF format specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [SafeTensors format](https://github.com/huggingface/safetensors)
### Repairing the Windows accelerator runtime

OpenAgent validates the accelerator interpreter by launching it; a stale
Microsoft Store alias is never treated as a working NPU/GPU runtime. Use a
standalone Python 3.12 interpreter when the Windows launcher is broken:

```powershell
.\installers\windows\setup-accelerator.ps1 `
  -PythonVersion 3.12 `
  -BasePython D:\openagent\.python312\python.exe
```

NPU utilization requires an OpenVINO GenAI model exported for NPU. GGUF models
served by Ollama or ordinary llama.cpp do not become NPU workloads merely
because an NPU exists. GPU/NPU readiness is reported only after the native
runtime launches and exposes the device.

The initial managed AI-PC model is stored under
`models/openvino/qwen2.5-1.5b-instruct-int4-ov`. It is loaded independently on
GPU and NPU and measured before a role assignment is changed; utilization is
never used as a substitute for latency and output verification.
