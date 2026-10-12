"""Fresh-process identity and synchronized device execution evidence."""
import json
import platform
import sys


class Unavailable(RuntimeError):
    """Required runtime, device, kernel or filesystem is unavailable."""


def main():
    backend, index = sys.argv[1], int(sys.argv[2])
    evidence = {"backend": backend, "host": platform.platform(), "device_index": index}
    if backend == "linux":
        from openagent.adaptive_engine import uring_io
        import tempfile
        from pathlib import Path
        import ctypes
        # Bind the just-built bridge using the existing engine reader's ABI.
        library = ctypes.CDLL(sys.argv[3])
        library.oa_uring_probe.argtypes = []
        library.oa_uring_probe.restype = ctypes.c_int
        library.oa_uring_read.argtypes = [ctypes.c_char_p, ctypes.c_uint64,
                                        ctypes.c_uint64, ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_uint64)]
        library.oa_uring_read.restype = ctypes.c_int
        uring_io._library = library
        report = uring_io.capability_report()
        if not report["available"]:
            raise Unavailable(report["reason"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.bin"
            path.write_bytes(bytes(range(256)) * 32)
            try:
                actual = uring_io.read_region(path, 17, 5000)
            except uring_io.UringIOUnsupported as exc:
                raise Unavailable(str(exc)) from exc
            if actual != path.read_bytes()[17:5017]:
                raise RuntimeError("direct io_uring read mismatch")
        evidence["device"] = "io_uring/O_DIRECT"
    else:
        import torch
        if backend == "metal":
            if platform.system() != "Darwin" or not torch.backends.mps.is_available():
                raise Unavailable("macOS MPS device required")
            device = torch.device("mps")
            synchronize = torch.mps.synchronize
            identity = "metal"
        else:
            identity = "hip" if torch.version.hip else "cuda" if torch.version.cuda else "cpu"
            if identity != backend or not torch.cuda.is_available():
                raise Unavailable(f"required {backend}, runtime is {identity}")
            if index < 0 or index >= torch.cuda.device_count():
                raise Unavailable("required device unavailable")
            device = torch.device(f"cuda:{index}")
            synchronize = lambda: torch.cuda.synchronize(device)
            evidence.update(name=torch.cuda.get_device_name(device),
                            memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        x = torch.ones((17, 17), device=device)
        y = x @ x
        if x.device != device or y.device != device:
            raise RuntimeError("placement mismatch")
        synchronize()
        if not torch.equal(y.cpu(), torch.full((17, 17), 17.0)):
            raise RuntimeError("numerical mismatch")
        evidence.update(device=str(device), runtime=identity, torch=str(torch.__version__))
    evidence["executed"] = True
    print("OPENAGENT_RUNTIME_EVIDENCE " + json.dumps(evidence, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Unavailable, ImportError) as exc:
        print("OPENAGENT_RUNTIME_UNAVAILABLE " + json.dumps({
            "reason": str(exc), "kind": type(exc).__name__,
        }), flush=True)
        raise SystemExit(2)
