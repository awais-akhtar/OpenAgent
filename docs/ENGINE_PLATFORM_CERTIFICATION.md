# Engine Platform Certification

This opt-in runner requires a real Linux/liburing, Linux/CUDA, Linux/HIP,
or macOS/Metal host. Nothing runs on import. This source implementation update did not
execute builds, probes, suites, or hardware verification.

Run on a provisioned trusted host:

```text
python -m openagent.adaptive_engine.platform_certification --backend cuda --device 0 --report certification/cuda.json
```

Backends: `linux`, `cuda`, `hip`, `metal`. CUDA and HIP use separate runtime
identity requirements despite both using PyTorch's `cuda:N` device notation.
The runner sets the existing suite opt-ins and disables MPS CPU fallback before
launching fresh Python processes. Host mismatches, missing compilers, unavailable
devices, failed synchronization, numerical mismatches, skipped tests, empty
suites, command errors and timeouts are failures, never successful skips.

Native commands compile the existing liburing C bridge with `cc`/`gcc`/`clang`
and `-luring`, a device kernel with `nvcc` or `hipcc`, or a Metal compute command
with `xcrun swiftc -framework Metal`. GPU executables must dispatch, synchronize,
read back and compare their result. Linux additionally performs actual O_DIRECT
io_uring reads through the existing runtime. Existing certification suites run
through the package-owned strict `platform_certification.suite` runner;
accelerator suites audit actual engine operation placement and reference accuracy.
An installed SDK or successful compilation is not execution evidence.

## Certificate Scope

`--scope engine` is the default. Schema version 3 separates
`device_probe_evidence` from `engine_backend_evidence`. Full engine success
requires the mandatory backend-specific integration test to be recorded as
successful, not merely loaded or counted: CUDA/HIP payload, scalar, batch and
placement integration; Metal scalar and grouped-batch placement integration;
or Linux real reads and store integrity. CPU-only tests and standalone device
probes cannot satisfy this requirement. Expected failures are rejected too.

`--scope probe` runs native/runtime probes and additional commands but does not
run engine suites. It can pass only as a probe certificate, always with
`engine_certified: false` and `engine_backend_evidence.executed: false`.
Consumers must check scope and `engine_certified`, not just `status`.
Engine evidence is limited to the named implementation and tested operations,
not every engine backend or model family. CI uses the default engine scope.

`--device-name` requires a substring of the reported CUDA/HIP device name.
`--min-memory-bytes` enforces reported GPU memory; nonzero requirements fail for
backends without that property. `--timeout` bounds each subprocess. Reports
contain schema version, required backend/device, host, timestamp, CI revision,
execution evidence, argv, return codes, durations, stdout/stderr, and failure
reason. A pass requires all mandatory native/runtime/suite stages and all added
commands to succeed. Reports are local execution records, not signed attestations.

## Trusted Extensions

`--matrix path.json` loads repository/operator-trusted configuration only:

```json
{
  "cuda": {
    "suites": ["tests.test_platform_certification_runner.CertificationRunnerContractTests"],
    "build_commands": [["cmake", "--build", "build/cuda"]],
    "commands": [["build/cuda/engine-certify", "--device", "0"]]
  }
}
```

Entries append to mandatory stages; they cannot replace device evidence or core
suites. Commands are explicit argv arrays, run from the repository root with
`shell=False`. This is not a sandbox: do not accept untrusted matrices, executables,
suite imports, or pull-request code on privileged self-hosted machines. Extension
commands must themselves reject skips; their exit status does not independently
prove hardware execution. Full stdout/stderr may contain host-sensitive details.

## CI And Remaining Work

The manual-only workflow schedules four labeled self-hosted runners and uploads
JSON artifacts even after failure. Provision Python dependencies (NumPy and the
correct CUDA/ROCm/MPS PyTorch build), toolchains, liburing headers, kernel access,
and a direct-I/O-capable temporary filesystem outside this workflow. It neither
installs drivers nor substitutes hosted CPU machines. A missing runner queues
the job; it does not constitute certification.

All new code remains unexecuted during this implementation update. Native compilation compatibility,
device evidence, runner contracts and CI behavior require the next authorized
hardware execution pass. CUDA/HIP native probes certify runtime kernel execution,
not a new native engine backend; engine execution here uses the existing Torch
backend. Additional engine families, performance gates, machine identity signing,
and deployment-specific minimum device policies need explicitly configured suites
or future implementation. Metal defaults to the system device, not indexed GPUs.

## Explicit Outcomes

`passed` exits 0; `failed` exits 1; `unavailable` exits 2. An unavailable host,
compiler, Python runtime dependency, CUDA/HIP identity, device index, MPS device,
io_uring kernel or direct-I/O target is never a pass. Reports include `stage` and
structured `failure` details. Compilation, native executable errors, timeouts,
suite import errors and numerical failures remain failures; they are not
automatically relabeled as missing hardware. Native probes explicitly report
missing CUDA/HIP runtime/device or Metal device; allocation, queue, dispatch and
synchronization errors remain failures. Inspect command stderr for details.

The Linux runtime probe binds the freshly compiled bridge in its isolated process
using the existing reader's ctypes ABI. The integration suite remains a separate
process using the engine's normal lazy loader. That loader may compile its cache
copy; the enclosing suite subprocess timeout bounds that execution. No persistent
engine files or global cache configuration are modified by the probe binding.

## Inspected Backend Coverage

The mandatory suites cover Linux direct/io_uring byte ranges, EOF and store
checksum integrity; CUDA/HIP Torch scalar and grouped-batch routed SwiGLU with
dispatch placement, dimensions, leases, error handling and cleanup; and Metal
Torch scalar/grouped-batch placement, accuracy, leases and cleanup. Accelerator
payloads are float32, float16 and int8-row, computed in FP32 against the NumPy
routed-expert contract. This is source-inspected scope, not execution evidence.

Vulkan/wgpu and OpenVINO implementations exist but have no mandatory certification
mapping here. Packed/native expert variants, decoder composition, publisher
checkpoint admission, full-model logits/generation/cache parity, performance,
and deployment-scale behavior are outside these certificates. A CUDA/HIP kernel
probe does not certify native decoder kernels. Extension commands cannot expand
the named mandatory engine implementation merely by returning exit code 0.

Tests, validation imports, compilation, builds, benchmarks and hardware probes
have not been performed for this source implementation. Hardware results and
completion percentages remain unverified.
