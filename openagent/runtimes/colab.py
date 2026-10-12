"""Google Colab launcher for the existing local OpenAgent HTTP runtime."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import threading
from typing import Any

from ..process_policy import ProcessPolicy, run_process
from ..routes.runtime import build_http_server
from ..storage import OpenAgentStore


@dataclass(frozen=True)
class ColabLaunch:
    """Details for a running notebook-local OpenAgent server."""

    url: str
    data_dir: Path
    hardware: dict[str, Any]
    recommended_model: str
    server: Any
    thread: threading.Thread

    def stop(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


def detect_colab() -> bool:
    """Return whether this process is running inside a Google Colab kernel."""

    if "COLAB_RELEASE_TAG" in os.environ:
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def detect_hardware() -> dict[str, Any]:
    """Return a dependency-free GPU summary when notebook GPU tools exist."""

    result: dict[str, Any] = {"gpu": False, "name": "", "vram_mb": 0}
    try:
        probe = run_process(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            policy=ProcessPolicy(
                timeout_seconds=3,
                max_output_bytes=4_000,
                allowed_executables=("nvidia-smi",),
            ),
        )
    except FileNotFoundError:
        return result
    if probe.timed_out or probe.output_truncated or probe.returncode != 0 or not probe.stdout.strip():
        return result
    name, _, memory = probe.stdout.splitlines()[0].partition(",")
    try:
        vram_mb = int(memory.strip())
    except ValueError:
        vram_mb = 0
    return {"gpu": True, "name": name.strip(), "vram_mb": vram_mb}


def recommend_model(hardware: dict[str, Any]) -> str:
    """Choose a conservative model recommendation; no model is downloaded."""

    if not hardware.get("gpu"):
        return "phi4-mini"
    if int(hardware.get("vram_mb", 0) or 0) >= 14_000:
        return "qwen2.5:7b"
    return "phi4-mini"


def launch(
    *,
    data_dir: Path | str | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    warm_openvino: bool = False,
) -> ColabLaunch:
    """Start OpenAgent in a background thread and return its notebook-local URL.

    Colab users may expose the returned local port with their organization’s
    approved proxy or tunnel. The launcher deliberately does not create a
    public tunnel or send workspace data to a third party.
    """

    workspace = Path(data_dir).expanduser() if data_dir else Path("/content/openagent")
    workspace.mkdir(parents=True, exist_ok=True)
    server = build_http_server(host, port, OpenAgentStore(workspace), warm_openvino=warm_openvino)
    thread = threading.Thread(target=server.serve_forever, name="openagent-colab", daemon=True)
    thread.start()
    actual_port = int(server.server_address[1])
    hardware = detect_hardware()
    return ColabLaunch(
        url=f"http://{host}:{actual_port}/",
        data_dir=workspace,
        hardware=hardware,
        recommended_model=recommend_model(hardware),
        server=server,
        thread=thread,
    )
