"""Stage 6 — Hardware detection public API.

Wraps and extends the existing HardwareProfiler in hardware.py with:
  - CUDA device discovery (nvidia-smi or torch.cuda when available)
  - Intel Arc / integrated GPU detection via PnP / lspci
  - NPU discovery (Windows AI Boost, Qualcomm, Apple ANE)
  - CPU capability flags (AVX-512, AMX, NEON)
  - A clean, serialisable HardwareCapabilities summary
  - Adaptive-engine placement recommendation

No network I/O.  Import is safe in constrained environments.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CudaDeviceInfo:
    index: int
    name: str
    vram_bytes: int
    compute_capability: str   # e.g. "8.9"
    driver_version: str
    is_available: bool = True


@dataclass
class IntelGpuInfo:
    name: str
    vram_bytes: int
    driver_version: str
    arc_family: bool          # True if Intel Arc (discrete)
    openvino_device_id: str   # e.g. "GPU.0"
    is_available: bool = True


@dataclass
class NpuInfo:
    name: str
    kind: str                 # "windows-npu", "qualcomm-npu", "apple-ane", "other"
    tops_estimate: Optional[float] = None  # TOPS if published
    is_available: bool = True


@dataclass
class CpuInfo:
    name: str
    physical_cores: int
    logical_cores: int
    avx512: bool = False
    amx: bool = False        # Intel Advanced Matrix Extensions
    neon: bool = False       # ARM NEON
    clock_mhz: Optional[int] = None


@dataclass
class HardwareCapabilities:
    """Complete hardware capability summary for Stage 6 placement decisions."""
    cpu: CpuInfo
    cuda_devices: List[CudaDeviceInfo] = field(default_factory=list)
    intel_gpus: List[IntelGpuInfo] = field(default_factory=list)
    npus: List[NpuInfo] = field(default_factory=list)
    openvino_available: bool = False
    openvino_devices: List[str] = field(default_factory=list)  # device ids
    platform_os: str = ""
    python_version: str = ""
    errors: List[str] = field(default_factory=list)  # non-fatal discovery errors

    @property
    def has_cuda(self) -> bool:
        return any(d.is_available for d in self.cuda_devices)

    @property
    def has_intel_gpu(self) -> bool:
        return any(g.is_available for g in self.intel_gpus)

    @property
    def has_npu(self) -> bool:
        return any(n.is_available for n in self.npus)

    @property
    def best_placement(self) -> str:
        """Recommend an adaptive-engine placement backend."""
        if self.has_cuda:
            return "device"
        if self.has_intel_gpu and self.openvino_available:
            return "device"
        if self.has_npu:
            return "native"  # NPU path still uses native executor
        return "native"

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_os,
            "python": self.python_version,
            "cpu": {
                "name": self.cpu.name,
                "physical_cores": self.cpu.physical_cores,
                "logical_cores": self.cpu.logical_cores,
                "avx512": self.cpu.avx512,
                "amx": self.cpu.amx,
                "neon": self.cpu.neon,
                "clock_mhz": self.cpu.clock_mhz,
            },
            "cuda": [
                {
                    "index": d.index, "name": d.name,
                    "vram_gb": round(d.vram_bytes / 2**30, 2),
                    "compute_capability": d.compute_capability,
                    "driver": d.driver_version,
                }
                for d in self.cuda_devices
            ],
            "intel_gpus": [
                {
                    "name": g.name, "arc": g.arc_family,
                    "vram_gb": round(g.vram_bytes / 2**30, 2),
                    "openvino_id": g.openvino_device_id,
                }
                for g in self.intel_gpus
            ],
            "npus": [
                {"name": n.name, "kind": n.kind, "tops": n.tops_estimate}
                for n in self.npus
            ],
            "openvino": {
                "available": self.openvino_available,
                "devices": self.openvino_devices,
            },
            "best_placement": self.best_placement,
            "errors": self.errors,
        }


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _run_silent(args: List[str], *, timeout: int = 10) -> Optional[str]:
    """Run a subprocess, return stdout or None on failure."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        r = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            check=False, creationflags=flags,
        )
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _detect_cuda() -> Tuple[List[CudaDeviceInfo], List[str]]:
    """Detect CUDA GPUs via nvidia-smi, then via torch.cuda as fallback."""
    devices: List[CudaDeviceInfo] = []
    errors: List[str] = []

    smi = shutil.which("nvidia-smi")
    if smi:
        fields = "index,name,memory.total,driver_version,compute_cap"
        out = _run_silent([smi, "--query-gpu=" + fields, "--format=csv,noheader,nounits"])
        if out:
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    vram = int(parts[2]) * 1024 * 1024  # MiB → bytes
                    driver = parts[3]
                    cc = parts[4]
                    devices.append(CudaDeviceInfo(
                        index=idx, name=name, vram_bytes=vram,
                        compute_capability=cc, driver_version=driver,
                    ))
                except (ValueError, IndexError):
                    errors.append(f"nvidia-smi parse error: {line!r}")
            return devices, errors

    # torch.cuda fallback
    try:
        import importlib
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                devices.append(CudaDeviceInfo(
                    index=i,
                    name=prop.name,
                    vram_bytes=prop.total_memory,
                    compute_capability=f"{prop.major}.{prop.minor}",
                    driver_version="",
                ))
    except (ImportError, Exception) as exc:
        errors.append(f"torch.cuda probe: {exc}")

    return devices, errors


def _detect_intel_gpu_windows() -> Tuple[List[IntelGpuInfo], List[str]]:
    """Detect Intel GPU on Windows via WMI/PowerShell."""
    gpus: List[IntelGpuInfo] = []
    errors: List[str] = []
    ps = shutil.which("powershell.exe") or shutil.which("powershell")
    if not ps:
        return gpus, errors
    cmd = (
        "Get-CimInstance Win32_VideoController | "
        "Where-Object { $_.Name -match 'Intel' } | "
        "Select-Object Name,AdapterRAM,DriverVersion | "
        "ConvertTo-Json -Compress"
    )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(
            [ps, "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, timeout=15, check=False,
            creationflags=flags,
        )
        if r.returncode != 0 or not r.stdout.strip():
            return gpus, errors
        import json
        data = json.loads(r.stdout)
        if isinstance(data, dict):
            data = [data]
        arc_keywords = ("arc", "a370", "a380", "a550", "a580", "a750", "a770",
                        "b580", "b770")
        for rec in data:
            name = str(rec.get("Name", ""))
            is_arc = any(k in name.lower() for k in arc_keywords)
            vram_raw = rec.get("AdapterRAM") or 0
            try:
                vram = int(vram_raw)
            except (TypeError, ValueError):
                vram = 0
            driver = str(rec.get("DriverVersion", ""))
            gpus.append(IntelGpuInfo(
                name=name, vram_bytes=vram, driver_version=driver,
                arc_family=is_arc,
                openvino_device_id=f"GPU.{len(gpus)}",
            ))
    except Exception as exc:
        errors.append(f"Intel GPU Windows probe: {exc}")
    return gpus, errors


def _detect_intel_gpu_linux() -> Tuple[List[IntelGpuInfo], List[str]]:
    """Detect Intel GPU on Linux via lspci."""
    gpus: List[IntelGpuInfo] = []
    errors: List[str] = []
    lspci = shutil.which("lspci")
    if not lspci:
        return gpus, errors
    out = _run_silent([lspci, "-nn"])
    if not out:
        return gpus, errors
    arc_keywords = ("arc", "a370", "a380", "a550", "a580", "a750", "a770",
                    "b580", "b770", "alchemist")
    for line in out.splitlines():
        if "Intel" not in line:
            continue
        if not any(k in ("VGA", "Display", "3D") for k in line[:30].split()):
            continue
        is_arc = any(k in line.lower() for k in arc_keywords)
        gpus.append(IntelGpuInfo(
            name=line.strip(), vram_bytes=0, driver_version="",
            arc_family=is_arc,
            openvino_device_id=f"GPU.{len(gpus)}",
        ))
    return gpus, errors


def _detect_npu() -> Tuple[List[NpuInfo], List[str]]:
    """Detect NPUs across platforms."""
    npus: List[NpuInfo] = []
    errors: List[str] = []

    if os.name == "nt":
        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if ps:
            cmd = (
                "Get-PnpDevice -PresentOnly | "
                "Where-Object { $_.Class -eq 'ComputeAccelerator' -or "
                "$_.FriendlyName -match '\\bNPU\\b|Neural|AI Boost|\\bVPU\\b' } | "
                "Select-Object FriendlyName,Status | ConvertTo-Json -Compress"
            )
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                r = subprocess.run(
                    [ps, "-NoProfile", "-NonInteractive", "-Command", cmd],
                    capture_output=True, text=True, timeout=15, check=False,
                    creationflags=flags,
                )
                if r.returncode == 0 and r.stdout.strip():
                    import json
                    data = json.loads(r.stdout)
                    if isinstance(data, dict):
                        data = [data]
                    for rec in data:
                        name = str(rec.get("FriendlyName", "NPU"))
                        npus.append(NpuInfo(
                            name=name, kind="windows-npu",
                            is_available=rec.get("Status") == "OK",
                        ))
            except Exception as exc:
                errors.append(f"NPU Windows probe: {exc}")

    elif sys.platform == "darwin":
        # Apple Silicon ANE — presence implied by arm64 + macOS 12+
        machine = platform.machine().lower()
        if machine == "arm64":
            npus.append(NpuInfo(name="Apple Neural Engine", kind="apple-ane",
                                tops_estimate=15.8, is_available=True))

    else:
        # Linux: look for Qualcomm or Intel VPU
        out = _run_silent(["lspci", "-nn"]) if shutil.which("lspci") else None
        if out:
            for line in out.splitlines():
                if "Qualcomm" in line and "NPU" in line:
                    npus.append(NpuInfo(name=line.strip(), kind="qualcomm-npu"))

    return npus, errors


def _cpu_flags() -> Dict[str, bool]:
    """Read CPU feature flags where available."""
    flags: Dict[str, bool] = {"avx512": False, "amx": False, "neon": False}
    if os.name == "nt":
        # Python 3.11+ has platform.processor() but not feature flags;
        # use CPUID via a best-effort check with ctypes.
        arch = platform.machine().lower()
        if arch in ("arm64", "aarch64"):
            flags["neon"] = True
        try:
            import ctypes
            # IsProcessorFeaturePresent(PF_XMMI64_INSTRUCTIONS_AVAILABLE=10)
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            # AVX-512 foundation: PF_AVX512F_INSTRUCTIONS_AVAILABLE = 40
            flags["avx512"] = bool(k32.IsProcessorFeaturePresent(40))
            # AMX tile compute = PF_XSAVE_ENABLED = 17 (approximate)
        except Exception:
            pass
    elif sys.platform == "linux":
        try:
            cpuinfo = open("/proc/cpuinfo", encoding="utf-8").read()
            fl = cpuinfo.split("flags")[1].split("\n")[0] if "flags" in cpuinfo else ""
            flags["avx512"] = "avx512f" in fl
            flags["amx"] = "amx_tile" in fl
            flags["neon"] = False
        except Exception:
            pass
        machine = platform.machine().lower()
        if machine in ("aarch64", "arm64", "armv8l"):
            flags["neon"] = True
    elif sys.platform == "darwin":
        machine = platform.machine().lower()
        if machine in ("arm64", "aarch64"):
            flags["neon"] = True
    return flags


def _detect_cpu() -> Tuple[CpuInfo, List[str]]:
    errors: List[str] = []
    name = platform.processor() or platform.machine() or "unknown"
    logical = os.cpu_count() or 1
    physical = logical  # conservative default
    clock: Optional[int] = None

    if os.name == "nt":
        # Python's platform.processor() may expose only the CPUID vendor/model
        # tuple on Windows. Read the local processor brand string first so a
        # restricted PowerShell/WMI environment still reports the real SKU.
        try:
            import winreg  # type: ignore[import-not-found]
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as cpu_key:
                registry_name, _ = winreg.QueryValueEx(cpu_key, "ProcessorNameString")
            if isinstance(registry_name, str) and registry_name.strip():
                name = registry_name.strip()
        except (ImportError, OSError, AttributeError) as exc:
            errors.append(f"CPU registry probe: {exc}")

        ps = shutil.which("powershell.exe") or shutil.which("powershell")
        if ps:
            cmd = (
                "Get-CimInstance Win32_Processor | "
                "Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
                "ConvertTo-Json -Compress"
            )
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                r = subprocess.run(
                    [ps, "-NoProfile", "-NonInteractive", "-Command", cmd],
                    capture_output=True, text=True, timeout=15, check=False,
                    creationflags=flags,
                )
                if r.returncode == 0 and r.stdout.strip():
                    import json
                    data = json.loads(r.stdout)
                    if isinstance(data, list):
                        data = data[0]
                    wmi_name = data.get("Name")
                    if isinstance(wmi_name, str) and wmi_name.strip():
                        name = wmi_name.strip()
                    physical = int(data.get("NumberOfCores", physical))
                    logical = int(data.get("NumberOfLogicalProcessors", logical))
                    clock = data.get("MaxClockSpeed")
                    if clock:
                        clock = int(clock)
            except Exception as exc:
                errors.append(f"CPU WMI probe: {exc}")

    cpu_flags = _cpu_flags()
    return CpuInfo(
        name=name, physical_cores=physical, logical_cores=logical,
        avx512=cpu_flags["avx512"], amx=cpu_flags["amx"], neon=cpu_flags["neon"],
        clock_mhz=clock,
    ), errors


def _detect_openvino() -> Tuple[bool, List[str], List[str]]:
    """Check OpenVINO availability and list devices."""
    try:
        from openvino.runtime import Core  # type: ignore[import]
        core = Core()
        devices = core.available_devices
        return True, list(devices), []
    except ImportError:
        return False, [], []
    except Exception as exc:
        return False, [], [f"OpenVINO probe: {exc}"]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_hardware(*, skip_openvino: bool = False) -> HardwareCapabilities:
    """Run full hardware detection and return a HardwareCapabilities summary.

    May take up to ~15 seconds on first call (WMI/PowerShell queries).
    Thread-safe; spawns no threads itself.
    """
    all_errors: List[str] = []

    cpu, errs = _detect_cpu()
    all_errors.extend(errs)

    cuda_devices, errs = _detect_cuda()
    all_errors.extend(errs)

    if os.name == "nt":
        intel_gpus, errs = _detect_intel_gpu_windows()
    else:
        intel_gpus, errs = _detect_intel_gpu_linux()
    all_errors.extend(errs)

    npus, errs = _detect_npu()
    all_errors.extend(errs)

    ov_available, ov_devices, errs = (False, [], [])
    if not skip_openvino:
        ov_available, ov_devices, errs = _detect_openvino()
    all_errors.extend(errs)

    return HardwareCapabilities(
        cpu=cpu,
        cuda_devices=cuda_devices,
        intel_gpus=intel_gpus,
        npus=npus,
        openvino_available=ov_available,
        openvino_devices=ov_devices,
        platform_os=platform.platform(),
        python_version=sys.version,
        errors=all_errors,
    )


# ---------------------------------------------------------------------------
# Cached singleton (process-level, refresh on explicit call)
# ---------------------------------------------------------------------------

_cached: Optional[HardwareCapabilities] = None
_cache_lock = __import__("threading").Lock()


def get_hardware_capabilities(*, refresh: bool = False,
                               skip_openvino: bool = False) -> HardwareCapabilities:
    """Return cached capabilities; call with refresh=True to re-probe."""
    global _cached
    with _cache_lock:
        if _cached is None or refresh:
            _cached = detect_hardware(skip_openvino=skip_openvino)
        return _cached
