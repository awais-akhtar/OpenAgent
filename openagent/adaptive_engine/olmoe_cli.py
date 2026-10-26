"""Operator commands for OpenAgent's direct OLMoE runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from .expert_packing import _scan_directory
from .expert_runtime import SparseExpertManifest
from .olmoe import OlmoeConfig, OlmoeModel, olmoe_metadata_fingerprint, pack_olmoe_experts, validate_olmoe_tensors
from .device_runtime import DirectHardwareClient
from .expert_kernels import ExpertKernelRegistry
from .hardware import HardwareProfiler
from .openvino_expert import OpenVINOExpertBackend
from .cpu_threads import blas_thread_scope


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openagent-olmoe", description="Inspect, pack and verify an OLMoE checkpoint.")
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="validate checkpoint metadata and SafeTensors headers")
    inspect.add_argument("model_directory")

    pack = commands.add_parser("pack", help="preserve FP32 experts or convert routed experts to an admitted packed format")
    pack.add_argument("model_directory")
    pack.add_argument("--output", default=None)
    pack.add_argument("--manifest", default=None)
    pack.add_argument("--dtype", choices=("float32", "int8-row", "int2-row", "int3-grouped-g64", "int4-row", "int4-grouped-g64", "mxfp4-e2m1-g32-ue8m0", "q8_0", "q8_1", "q4_0", "q4_1", "q5_0", "q5_1", "fp8-e4m3-b128-f32", "e8-lattice-caller-v1", "iq3-format6-98-v1"), default="int8-row")
    pack.add_argument('--lattice-file', type=Path, help='persisted E8 contract or IQ3 grid')
    pack.add_argument('--lattice-sha256', help='explicit expected contract identity')
    pack.add_argument('--conversion-mib', type=int, default=1024)
    pack.add_argument(
        "--stripe",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="round-robin experts into an additional named container",
    )

    verify = commands.add_parser("verify", help="compare greedy token IDs with a JSON oracle")
    verify.add_argument("model_directory")
    verify.add_argument("manifest")
    verify.add_argument("oracle")
    verify.add_argument("--cache-gb", type=float, default=4.0)
    verify.add_argument("--io-mode", choices=("buffered", "mmap", "direct", "uring"), default="mmap")
    verify.add_argument("--storage-root", action="append", default=[], metavar="NAME=PATH")
    verify.add_argument("--kernel", default=None, help="registered expert kernel backend name")

    generate = commands.add_parser("generate", help="run one local greedy generation")
    generate.add_argument("model_directory")
    generate.add_argument("manifest")
    generate.add_argument("prompt")
    generate.add_argument("--max-tokens", type=int, default=32)
    generate.add_argument("--cache-gb", type=float, default=4.0)
    generate.add_argument("--io-mode", choices=("buffered", "mmap", "direct", "uring"), default="mmap")
    generate.add_argument("--storage-root", action="append", default=[], metavar="NAME=PATH")
    generate.add_argument("--kernel", default=None, help="registered expert kernel backend name")

    benchmark = commands.add_parser("benchmark", help="measure one local checkpoint with explicit cache state")
    benchmark.add_argument("model_directory")
    benchmark.add_argument("manifest")
    benchmark.add_argument("prompt")
    benchmark.add_argument("--max-tokens", type=int, default=16)
    benchmark.add_argument("--warmup", type=int, default=1)
    benchmark.add_argument("--runs", type=int, default=3)
    benchmark.add_argument("--cache-gb", type=float, default=4.0)
    benchmark.add_argument("--io-mode", choices=("buffered", "mmap", "direct", "uring"), default="mmap")
    benchmark.add_argument("--storage-root", action="append", default=[], metavar="NAME=PATH")
    benchmark.add_argument("--kernel", default=None, help="registered expert kernel backend name")
    for command in (verify, generate, benchmark):
        command.add_argument("--blas-threads", type=int, default=None,
                             help="process-wide BLAS thread limit for this operator run")
    tune = commands.add_parser("tune", help="measure output-preserving CPU thread settings")
    tune.add_argument("model_directory")
    tune.add_argument("manifest")
    tune.add_argument("prompt")
    tune.add_argument("profile", type=Path)
    tune.add_argument("--candidates", type=int, nargs='+', default=[1, 2, 4])
    tune.add_argument("--cache-gb", type=float, default=1.0)
    tune.add_argument("--max-tokens", type=int, default=2)
    tune.add_argument("--warmup", type=int, default=1)
    tune.add_argument("--runs", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with blas_thread_scope(getattr(args, "blas_threads", None)):
            return _dispatch(args)
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args) -> int:
    try:
        if args.command in ('tune','verify','generate','benchmark'):
            raise ValueError(
                'the Python OLMoE reference CLI is not an admitted P02 execution backend; '
                'use native_olmoe_cli for the source-bound native route')
        if args.command == "tune":
            from .olmoe_autotune import tune_olmoe_threads, save_profile

            if not 0 < args.cache_gb < float('inf'):
                raise ValueError("cache-gb must be finite and positive")
            with OlmoeModel(args.model_directory, SparseExpertManifest.read(args.manifest),
                            expert_cache_bytes=int(args.cache_gb * 1024**3)) as model:
                profile = tune_olmoe_threads(model, model.encode(args.prompt),
                    candidates=tuple(args.candidates), max_tokens=args.max_tokens,
                    warmup=args.warmup, runs=args.runs, no_active_external_users=True)
                save_profile(profile, args.profile)
                print(json.dumps(profile, indent=2))
            return 0
        if args.command == "inspect":
            return _inspect(Path(args.model_directory))
        if args.command == "pack":
            defaults = {
                "float32": ("openagent-experts.f32", "openagent-experts.f32.manifest.json"),
                "int8-row": ("openagent-experts.i8", "openagent-experts.manifest.json"),
                "int4-row": ("openagent-experts.i4", "openagent-experts.i4.manifest.json"),
                "q8_0": ("openagent-experts.q8", "openagent-experts.q8.manifest.json"),
                "q8_1": ("openagent-experts.q81", "openagent-experts.q81.manifest.json"),
                "int4-grouped-g64": ("openagent-experts.gi4", "openagent-experts.gi4.manifest.json"),
                "int2-row": ("openagent-experts.i2", "openagent-experts.i2.manifest.json"),
                "int3-grouped-g64": ("openagent-experts.gi3", "openagent-experts.gi3.manifest.json"),
                "mxfp4-e2m1-g32-ue8m0": ("openagent-experts.mxfp4", "openagent-experts.mxfp4.manifest.json"),
                "q4_0": ("openagent-experts.q4", "openagent-experts.q4.manifest.json"),
                "q4_1": ("openagent-experts.q41", "openagent-experts.q41.manifest.json"),
                "q5_0": ("openagent-experts.q5", "openagent-experts.q5.manifest.json"),
                "q5_1": ("openagent-experts.q51", "openagent-experts.q51.manifest.json"),
                "fp8-e4m3-b128-f32": ("openagent-experts.fp8", "openagent-experts.fp8.manifest.json"),
                "e8-lattice-caller-v1": ("openagent-experts.e8", "openagent-experts.e8.manifest.json"),
                "iq3-format6-98-v1": ("openagent-experts.iq3", "openagent-experts.iq3.manifest.json"),
            }
            output = args.output or defaults[args.dtype][0]
            manifest = args.manifest or defaults[args.dtype][1]
            return _pack(Path(args.model_directory), output, manifest, args.stripe, args.dtype,
                         lattice_file=args.lattice_file, lattice_sha256=args.lattice_sha256,
                         conversion_mib=args.conversion_mib)
        if args.command == "verify":
            return _verify(
                Path(args.model_directory), Path(args.manifest), Path(args.oracle),
                args.cache_gb, args.io_mode, args.storage_root, args.kernel,
            )
        if args.command == "generate":
            return _generate(
                Path(args.model_directory), Path(args.manifest), args.prompt,
                args.max_tokens, args.cache_gb, args.io_mode, args.storage_root, args.kernel,
            )
        if args.command == "benchmark":
            return _benchmark(
                Path(args.model_directory), Path(args.manifest), args.prompt,
                args.max_tokens, args.warmup, args.runs, args.cache_gb,
                args.io_mode, args.storage_root, args.kernel,
            )
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _inspect(model_directory: Path) -> int:
    root = model_directory.resolve(strict=True)
    config = OlmoeConfig.read(root)
    tensors = _scan_directory(root)
    validate_olmoe_tensors(config, tensors)
    payload = {
        "status": "ready",
        "model_directory": str(root),
        "fingerprint": olmoe_metadata_fingerprint(root),
        "shards": [{"name": path.name, "bytes": path.stat().st_size} for path in sorted(root.glob("*.safetensors"))],
        "tensors": len(tensors),
        "layers": config.num_hidden_layers,
        "experts_per_layer": config.num_experts,
        "experts_used": config.num_experts_per_tok,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _pack(model_directory: Path, output_name: str, manifest_name: str, stripes: list[str], dtype: str = "int8-row",
          *, lattice_file=None, lattice_sha256=None, conversion_mib=1024) -> int:
    if type(conversion_mib) is not int or conversion_mib <= 0:
        raise ValueError('positive conversion working-memory budget required')
    lattice_contract = iq3_grid = None
    if dtype in {'e8-lattice-caller-v1', 'iq3-format6-98-v1'}:
        if lattice_file is None or not lattice_sha256:
            raise ValueError('lattice packing requires --lattice-file and --lattice-sha256')
        if dtype == 'e8-lattice-caller-v1':
            from .e8_lattice_expert import LatticeContract
            lattice_contract = LatticeContract.read(lattice_file, expected_identity=lattice_sha256)
        else:
            from .iq3_lattice import IQ3Grid
            iq3_grid = IQ3Grid.read(lattice_file, expected_sha256=lattice_sha256)
    elif lattice_file is not None or lattice_sha256 is not None:
        raise ValueError('lattice options require a lattice dtype')
    root = model_directory.resolve(strict=True)
    output = _inside(root, output_name)
    manifest_path = _inside(root, manifest_name)
    started = time.perf_counter()

    def progress(completed: int, total: int, layer: str, expert: str) -> None:
        if completed == 1 or completed == total or completed % 16 == 0:
            elapsed = max(time.perf_counter() - started, 1e-6)
            print(f"packed {completed}/{total} experts ({completed / elapsed:.2f}/s) · {layer}/{expert}", flush=True)

    storage_destinations = _named_paths(stripes)
    manifest = pack_olmoe_experts(
        root, output, progress=progress, storage_destinations=storage_destinations, dtype=dtype,
        lattice_contract=lattice_contract, iq3_grid=iq3_grid,
        max_conversion_bytes=conversion_mib*1024**2,
    )
    manifest.write(manifest_path)
    print(json.dumps({
        "status": "ready",
        "container": str(output),
        "manifest": str(manifest_path),
        "bytes": output.stat().st_size,
        "striped_containers": {
            name: {"path": str(path), "bytes": path.stat().st_size}
            for name, path in storage_destinations.items()
        },
        "seconds": round(time.perf_counter() - started, 3),
    }, indent=2))
    return 0


def _verify(
    model_directory: Path,
    manifest_path: Path,
    oracle_path: Path,
    cache_gb: float,
    io_mode: str,
    storage_roots: list[str],
    kernel: str | None,
) -> int:
    if cache_gb <= 0:
        raise ValueError("cache-gb must be positive")
    oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
    prompt = _integer_list(oracle.get("prompt_ids"), "prompt_ids")
    full = _integer_list(oracle.get("full_ids"), "full_ids")
    if full[:len(prompt)] != prompt or len(full) <= len(prompt):
        raise ValueError("oracle full_ids must begin with prompt_ids and include a continuation")
    expected = full[len(prompt):]
    started = time.perf_counter()
    kernels, client = _kernel_resources(model_directory, kernel, cache_gb)
    try:
        with OlmoeModel(
            model_directory,
            SparseExpertManifest.read(manifest_path),
            expert_cache_bytes=int(cache_gb * 1024**3),
            io_mode=io_mode,
            storage_roots=_named_paths(storage_roots, directories=True),
            kernels=kernels,
            preferred_kernel=kernel,
        ) as model:
            actual = model.generate_ids(prompt, max_tokens=len(expected))
            status = model.status()
    finally:
        _close_kernel_resources(kernels, client)
    matched = actual == expected
    print(json.dumps({
        "status": "passed" if matched else "failed",
        "prompt_ids": prompt,
        "expected_ids": expected,
        "actual_ids": actual,
        "seconds": round(time.perf_counter() - started, 3),
        "runtime": status,
    }, indent=2))
    return 0 if matched else 1


def _generate(
    model_directory: Path,
    manifest_path: Path,
    prompt: str,
    max_tokens: int,
    cache_gb: float,
    io_mode: str,
    storage_roots: list[str],
    kernel: str | None,
) -> int:
    if max_tokens <= 0 or cache_gb <= 0:
        raise ValueError("max-tokens and cache-gb must be positive")
    kernels, client = _kernel_resources(model_directory, kernel, cache_gb)
    try:
        with OlmoeModel(
            model_directory,
            SparseExpertManifest.read(manifest_path),
            expert_cache_bytes=int(cache_gb * 1024**3),
            io_mode=io_mode,
            storage_roots=_named_paths(storage_roots, directories=True),
            kernels=kernels,
            preferred_kernel=kernel,
        ) as model:
            print(model.generate(prompt, max_tokens=max_tokens))
    finally:
        _close_kernel_resources(kernels, client)
    return 0


def _benchmark(
    model_directory: Path,
    manifest_path: Path,
    prompt: str,
    max_tokens: int,
    warmup: int,
    runs: int,
    cache_gb: float,
    io_mode: str,
    storage_roots: list[str],
    kernel: str | None,
) -> int:
    if max_tokens <= 0 or warmup < 0 or runs <= 0 or cache_gb <= 0:
        raise ValueError("benchmark counts and cache-gb are outside their valid range")
    manifest = SparseExpertManifest.read(manifest_path)
    kernels, client = _kernel_resources(model_directory, kernel, cache_gb)
    try:
        with OlmoeModel(
            model_directory,
            manifest,
            expert_cache_bytes=int(cache_gb * 1024**3),
            io_mode=io_mode,
            storage_roots=_named_paths(storage_roots, directories=True),
            kernels=kernels,
            preferred_kernel=kernel,
        ) as model:
            prompt_ids = model.encode(prompt)
            for _ in range(warmup):
                list(model.generate_id_stream(prompt_ids, max_tokens=max_tokens))
            measurements: list[dict[str, float | int]] = []
            for _ in range(runs):
                started = time.perf_counter()
                stream = model.generate_id_stream(prompt_ids, max_tokens=max_tokens)
                generated: list[int] = []
                try:
                    generated.append(next(stream))
                    first_token_ms = (time.perf_counter() - started) * 1000
                    generated.extend(stream)
                except StopIteration:
                    first_token_ms = (time.perf_counter() - started) * 1000
                elapsed = time.perf_counter() - started
                measurements.append({
                    "first_token_ms": round(first_token_ms, 3),
                    "elapsed_ms": round(elapsed * 1000, 3),
                    "generated_tokens": len(generated),
                    "tokens_per_second": round(len(generated) / elapsed, 3) if elapsed else 0.0,
                })
            status = model.status()
    finally:
        _close_kernel_resources(kernels, client)
    first_tokens = [float(item["first_token_ms"]) for item in measurements]
    throughputs = [float(item["tokens_per_second"]) for item in measurements]
    print(json.dumps({
        "status": "measured",
        "model_id": manifest.model_id,
        "fingerprint": manifest.fingerprint,
        "io_mode": io_mode,
        "cache_gb": cache_gb,
        "warmup_runs": warmup,
        "measured_runs": runs,
        "prompt_tokens": len(prompt_ids),
        "max_output_tokens": max_tokens,
        "median_first_token_ms": round(statistics.median(first_tokens), 3),
        "p95_first_token_ms": round(_percentile(first_tokens, 0.95), 3),
        "median_tokens_per_second": round(statistics.median(throughputs), 3),
        "runs": measurements,
        "runtime": status,
    }, indent=2))
    return 0


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _close_kernel_resources(kernels, client):
    try:
        if kernels is not None:
            kernels.close()
    finally:
        if client is not None:
            client.close()


def _kernel_resources(model_directory: Path, kernel: str | None, cache_gb: float):
    if kernel is None or kernel == "numpy-reference":
        return None, None
    for dtype, filename in (('iq3-format6-98-v1', 'openagent-iq3-grid.bin'),
                            ('e8-lattice-caller-v1', 'openagent-e8-contract.bin')):
        prefix = dtype+'-'
        if kernel.startswith(prefix) and kernel.endswith('-numba'):
            identity = kernel[len(prefix):-len('-numba')]
            if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
                raise ValueError('lattice kernel requires an exact lowercase contract SHA-256')
            path = model_directory.resolve(strict=True)/filename
            if dtype == 'iq3-format6-98-v1':
                from .iq3_lattice import IQ3Grid
                from .iq3_expert import NativeIQ3ExpertBackend
                backend = NativeIQ3ExpertBackend(IQ3Grid.read(path, expected_sha256=identity))
            else:
                from .e8_lattice_expert import LatticeContract
                from .native_e8_lattice_expert import NativeE8LatticeExpertBackend
                backend = NativeE8LatticeExpertBackend(LatticeContract.read(path, expected_identity=identity))
            if backend.NAME != kernel or not backend.info().available:
                raise RuntimeError(backend.info().unavailable_reason or 'lattice backend identity mismatch')
            return ExpertKernelRegistry([backend]), None
    if kernel in {"int2-row-reference", "int3-grouped-g64-reference", "packed-int4-reference", "grouped-int4-reference", "mxfp4-e2m1-g32-ue8m0-reference", "q8_0-numpy", "q8_1-reference", "q4_0-reference", "q4_1-reference", "q5_0-reference", "q5_1-reference"}:
        return ExpertKernelRegistry(), None
    if kernel in {"int2-row-numba", "int3-grouped-g64-numba"}:
        from .native_lowbit_expert import NativeLowBitExpertBackend

        backend = NativeLowBitExpertBackend(kernel.removesuffix("-numba"))
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "grouped-int4-numba":
        from .native_grouped_int4_expert import NativeGroupedInt4ExpertBackend

        backend = NativeGroupedInt4ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "mxfp4-e2m1-g32-ue8m0-numba":
        from .native_mxfp4_expert import NativeMXFP4ExpertBackend

        backend = NativeMXFP4ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q8_1-numba":
        from .native_q8_1_expert import NativeQ8_1ExpertBackend

        backend = NativeQ8_1ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q5_1-numba":
        from .native_q5_1_expert import NativeQ5_1ExpertBackend

        backend = NativeQ5_1ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "fp8-e4m3-b128-f32-numba":
        from .native_fp8_block_expert import NativeFP8BlockExpertBackend

        backend = NativeFP8BlockExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q5_0-numba":
        from .native_q5_0_expert import NativeQ5_0ExpertBackend

        backend = NativeQ5_0ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q4_1-numba":
        from .native_q4_1_expert import NativeQ4_1ExpertBackend

        backend = NativeQ4_1ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q4_0-numba":
        from .native_q4_0_expert import NativeQ4_0ExpertBackend

        backend = NativeQ4_0ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "q8_0-numba":
        from .native_q8_0_expert import NativeQ8_0ExpertBackend

        backend = NativeQ8_0ExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "packed-int4-numba":
        from .native_packed_expert import NativePackedExpertBackend

        backend = NativePackedExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "vectorized-cpu-numba":
        from .vectorized_expert import VectorizedExpertBackend

        backend = VectorizedExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "vulkan-wgpu":
        from .vulkan_expert import VulkanRoutedSwiGLUBackend

        backend = VulkanRoutedSwiGLUBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "torch-expert":
        from .torch_expert import TorchExpertBackend

        backend = TorchExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if kernel == "native-cpu-numba":
        from .native_expert import NativeExpertBackend

        backend = NativeExpertBackend()
        if not backend.info().available:
            raise RuntimeError(backend.info().unavailable_reason)
        return ExpertKernelRegistry([backend]), None
    if not kernel.startswith("openvino-"):
        raise ValueError(f"unknown expert kernel backend: {kernel}")
    device = kernel.removeprefix("openvino-").upper()
    client = DirectHardwareClient(HardwareProfiler(model_directory))
    backend = OpenVINOExpertBackend(
        client,
        device=device,
        capacity_bytes=int(cache_gb * 1024**3),
    )
    if not backend.info().available:
        raise RuntimeError(backend.info().unavailable_reason)
    return ExpertKernelRegistry([backend]), client


def _inside(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("output paths must stay inside the model directory")
    return path


def _integer_list(value, name: str) -> list[int]:
    if not isinstance(value, list) or not value or not all(isinstance(item, int) and item >= 0 for item in value):
        raise ValueError(f"{name} must be a non-empty array of nonnegative integers")
    return value


def _named_paths(values: list[str], *, directories: bool = False) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or "/" in name or "\\" in name or not raw_path:
            raise ValueError("named paths must use NAME=PATH with a simple name")
        path = Path(raw_path).resolve(strict=directories)
        if directories and not path.is_dir():
            raise ValueError(f"storage root is not a directory: {path}")
        if name in parsed:
            raise ValueError(f"duplicate storage name: {name}")
        parsed[name] = path
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
