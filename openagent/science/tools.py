"""Local scientific tool adapters and capability discovery."""

from __future__ import annotations

import importlib.util
import json
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..process_policy import ProcessPolicy, run_process


@dataclass(frozen=True)
class ToolCapability:
    """Describes one optional scientific package or simulator."""

    name: str
    kind: str
    package: str | None = None
    executable: str | None = None
    available: bool = False
    privacy_boundary: str = "Runs locally and returns data to the OpenAgent process."

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "package": self.package,
            "executable": self.executable,
            "available": self.available,
            "privacy_boundary": self.privacy_boundary,
        }


def discover_scientific_tools() -> list[ToolCapability]:
    """Report installed local Python tools without importing them."""

    packages = (
        ("NumPy", "numpy"),
        ("SciPy", "scipy"),
        ("SymPy", "sympy"),
        ("pandas", "pandas"),
        ("Matplotlib", "matplotlib"),
        ("FEniCS", "fenics"),
    )
    return [
        ToolCapability(name, "python", package, available=importlib.util.find_spec(package) is not None)
        for name, package in packages
    ] + [
        ToolCapability(name, "simulator", executable=command, available=bool(shutil.which(command)))
        for name, command in (("OpenFOAM", "foamRun"), ("COMSOL", "comsol"), ("LAMMPS", "lmp"))
    ]


class SimulationTool:
    """Small interface shared by local simulation adapters."""

    name = "simulation"

    def capability(self) -> ToolCapability:
        return ToolCapability(self.name, "simulator", executable=self.name, available=False)

    def run(self, *, inputs: Mapping[str, Any], timeout_seconds: float = 30.0) -> dict[str, Any]:
        raise NotImplementedError

    def mock_example(self, *, inputs: Mapping[str, Any]) -> dict[str, Any]:
        temperature = float(inputs.get("temperature", 300.0))
        duration = max(0.0, float(inputs.get("duration", 1.0)))
        conductivity = max(1e-9, float(inputs.get("conductivity", 1.0)))
        return {
            "status": "mock-completed",
            "tool": self.name,
            "inputs": dict(inputs),
            "outputs": {"temperature_drop": conductivity * duration * 0.1, "final_temperature": temperature - conductivity * duration * 0.1},
            "warning": "Demonstration output only; configure the local simulator for a physical result.",
        }


class CommandSimulationTool(SimulationTool):
    """Run a configured executable without invoking a shell."""

    def __init__(
        self,
        name: str,
        executable: str | Path | None = None,
        args: Sequence[str] = (),
        *,
        working_directory: str | Path | None = None,
    ) -> None:
        self.name = name
        self.executable = str(executable) if executable else ""
        self.args = tuple(args)
        self.working_directory = Path(working_directory).expanduser().resolve() if working_directory else None

    def capability(self) -> ToolCapability:
        path = Path(self.executable) if self.executable else None
        resolved = str(path.resolve()) if path and path.is_file() else (shutil.which(self.executable) if self.executable else "")
        return ToolCapability(
            self.name,
            "simulator",
            executable=resolved or self.executable or self.name,
            available=bool(resolved),
            privacy_boundary="Runs only the explicitly configured local executable; no network access is added.",
        )

    def run(
        self,
        *,
        inputs: Mapping[str, Any],
        timeout_seconds: float = 30.0,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        capability = self.capability()
        if not capability.available:
            return {"status": "unavailable", "tool": self.name, "inputs": dict(inputs), "message": "Configure a local simulator executable to run this adapter."}
        payload = json.dumps(dict(inputs), ensure_ascii=False)
        workspace = self.working_directory or Path.cwd().resolve()
        completed = run_process(
            [str(capability.executable), *self.args],
            cwd=workspace,
            input_data=payload,
            cancel_event=cancel_event,
            policy=ProcessPolicy(
                timeout_seconds=max(0.1, timeout_seconds),
                max_output_bytes=20_000,
                max_input_bytes=1_000_000,
                allowed_cwd_roots=(workspace,),
                allowed_executables=(str(capability.executable),),
            ),
        )
        status = "cancelled" if completed.cancelled else "timeout" if completed.timed_out else "completed" if completed.returncode == 0 else "failed"
        return {
            "status": status,
            "tool": self.name,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "output_truncated": completed.output_truncated,
        }


class FenicsTool(SimulationTool):
    """Use the Python FEniCS API, with a visibly labeled reference fallback."""

    name = "FEniCS"

    def __init__(self, executable: str | Path | None = None, *, allow_fallback: bool = True) -> None:
        # ``executable`` remains accepted for compatibility with older callers.
        self.executable = str(executable) if executable else ""
        self.allow_fallback = allow_fallback

    def capability(self) -> ToolCapability:
        available = importlib.util.find_spec("fenics") is not None
        return ToolCapability(
            self.name,
            "python-simulator",
            package="fenics",
            executable=None,
            available=available,
            privacy_boundary="Runs the FEniCS Python API locally; when absent, callers may explicitly use the labeled finite-difference fallback.",
        )

    def run(self, *, inputs: Mapping[str, Any], timeout_seconds: float = 30.0) -> dict[str, Any]:
        from .simulations.fenics_heat import FenicsHeatAdapter

        return FenicsHeatAdapter(allow_fallback=self.allow_fallback).run(inputs=inputs)


class OpenFOAMTool(CommandSimulationTool):
    def __init__(self, executable: str | Path | None = None) -> None:
        super().__init__("OpenFOAM", executable)


class COMSOLTool(CommandSimulationTool):
    def __init__(self, executable: str | Path | None = None) -> None:
        super().__init__("COMSOL", executable)


class LAMMPSTool(CommandSimulationTool):
    def __init__(self, executable: str | Path | None = None) -> None:
        super().__init__("LAMMPS", executable)
