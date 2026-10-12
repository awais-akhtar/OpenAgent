# Linux Engine Validation

## Certification Contract

Run from the repository root on real Linux with Python 3.12, a system C
compiler, liburing headers/library, and the `engine-native` and `engine-hf-moe`
extras already provisioned. The certification command does not install anything
or download model weights. CI provisions dependencies explicitly before running
with Hugging Face offline mode enabled.

```sh
export OPENAGENT_REQUIRE_URING=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Choose a directory on the filesystem that you intend to certify.
export TMPDIR=/path/to/linux-filesystem/scratch
python -m tests.test_linux_engine_certification \
  tests.test_linux_engine_certification.CertificationRunnerTests \
  tests.test_uring_io tests.test_expert_io tests.test_expert_store \
  tests.test_native_expert tests.test_expert_kernels tests.test_sparse_expert_runtime \
  tests.test_olmoe_batch tests.test_olmoe_continuous tests.test_olmoe_serve \
  tests.test_hf_moe tests.test_olmoe_validation tests.test_olmoe_transformers_reference \
  tests.test_cpu_threads tests.test_olmoe_autotune tests.test_vectorized_expert \
  tests.test_olmoe_segments tests.test_segment_snapshot tests.test_expert_cli_kernels \
  tests.test_checkpoint_recovery
```

The runner fails on non-Linux hosts, unavailable io_uring, rejected O_DIRECT,
missing dependencies, import errors, and skipped tests. Only the two explicitly
identified Windows-only tests in `test_uring_io` and `test_expert_io` may skip
on Linux. Ordinary `python -m unittest` does not enforce this skip policy.

The dedicated native test uses both real backends without mocks or buffered
fallback. It compares aligned and unaligned ranges, a multi-chunk read, partial
EOF, beyond EOF, zero-length reads, and empty files against known bytes. It also
checks missing-file errors and DiskExpertStore payloads and checksum rejection
for both `direct` and `uring` modes. Files and stores are closed and cleaned up.

`capability_report()` only proves that a ring can be created. Its `direct_io`
field is a backend declaration, not proof that the target filesystem accepts
O_DIRECT. Successful nonempty reads in the certification are required as well.
The C bridge opens with O_DIRECT and submits liburing reads; synchronous direct
I/O opens with O_DIRECT and uses aligned `preadv`. Neither falls back to buffered
reads. This suite does not certify throughput, parallel submission, every
filesystem/device, GPU execution, or production-size model accuracy.

## Environment Constraints

WSL must already have an installed Linux distribution; `wsl --list --verbose`
is a detection step, not permission to install one. Docker requires a working
Linux daemon and an existing image with the dependencies. Do not pull images
or install an OS merely to run certification. Container seccomp/kernel policy
may reject io_uring even when liburing is installed; report that as blocked,
not as successful certification. Prefer a native Linux scratch filesystem over
a Windows bind mount. No kernel/sysctl or security-policy changes are automatic.

The Ubuntu 24.04 workflow fails if its runner cannot create an io_uring ring or
perform the real reads. A failed environment probe must be investigated rather
than converted into a skip or buffered fallback.

## Local Verification Status

On 2026-10-02 the Windows workspace had `wsl.exe`, but
`wsl --list --verbose` reported that WSL was not installed. Docker was not found
on PATH or at the standard Docker Desktop CLI path. No Linux distribution,
container image, or dependencies were downloaded or installed.

Actual Linux io_uring/O_DIRECT execution is therefore **blocked locally**.
The workflow and runnable certification are provided, but no successful Linux
CI run is claimed. Windows unit/mock results do not constitute Linux evidence.

Local runner self-tests passed (2 tests); invoking certification on Windows
failed with `certification requires real Linux`, exit status 1. After engine
integration, the combined I/O and runner suite completed 18 tests with one
Linux-only skip. This is Windows regression coverage, not Linux certification.
Ordinary discovery leaves hardware certification disabled; the module entry
point enables it explicitly and enforces the strict skip policy.

### Recheck On 2026-10-02

Source HEAD: `b1b2495880631078414d3a7428580aea61e67353`, with existing
uncommitted changes retained. `wsl --list --verbose` again exited 1 with
"The Windows Subsystem for Linux is not installed." Docker was again absent
from PATH and the standard Docker Desktop CLI path; no Linux execution host
was available. No OS, image, driver, or toolchain was installed or downloaded.

Using `.accelerator-venv/Scripts/python.exe` (Python 3.12.10),
`python -m tests.test_linux_engine_certification` exited 1 with
`certification requires real Linux` in setup (one test attempted, no native
reads executed). Linux io_uring and O_DIRECT remain **unverified**.
