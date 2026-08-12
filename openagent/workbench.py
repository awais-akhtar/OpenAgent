"""Modern local workbench desktop launcher.

This module starts the OpenAgent local HTTP server and opens the same rich
workbench UI in an app-style desktop window when the platform provides one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import webbrowser

from . import __version__
from .config import desktop_home, web_root
from .models import AuditEvent
from .storage import OpenAgentStore


WORKBENCH_PRIVACY_BOUNDARY = (
    "OpenAgent Workbench serves the UI from localhost and reads/writes the configured local data directory. "
    "It does not call external services unless an explicit local action allows that boundary crossing."
)

DEFAULT_WORKBENCH_PORT = 8765


def configured_workbench_port() -> int:
    """Return the single stable desktop port.

    Development servers may still opt into another port with the explicit
    ``--port`` CLI flag, but an inherited environment variable must never move
    an installed desktop launch to a new address.
    """

    return DEFAULT_WORKBENCH_PORT


def installation_source_root() -> Path | None:
    """Resolve the installer-recorded source tree without probing the network."""

    configured = os.environ.get("OPENAGENT_SOURCE_ROOT", "").strip()
    candidates: list[Path] = [Path(configured).expanduser()] if configured else []
    local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
    if local_app_data:
        manifest_path = Path(local_app_data) / "OpenAgent" / "install-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        source = str(manifest.get("source", "")).strip() if isinstance(manifest, dict) else ""
        if source:
            candidates.append(Path(source).expanduser())
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.is_dir() and (resolved / "openagent").is_dir():
            return resolved
    return None


def configure_managed_accelerator() -> str:
    """Select the trusted local accelerator interpreter for the installed app."""

    configured = os.environ.get("OPENAGENT_ACCELERATOR_PYTHON", "").strip()
    if configured and Path(configured).is_file():
        return str(Path(configured).resolve())
    source_root = installation_source_root()
    if source_root is None:
        return ""
    layouts = (
        (("Scripts", "python.exe"), ("bin", "python"))
        if sys.platform == "win32"
        else (("bin", "python"), ("Scripts", "python.exe"))
    )
    executable = next(
        (
            source_root / ".accelerator-venv" / scripts / executable_name
            for scripts, executable_name in layouts
            if (source_root / ".accelerator-venv" / scripts / executable_name).is_file()
        ),
        None,
    )
    if executable is None:
        return ""
    selected = str(executable.resolve())
    os.environ["OPENAGENT_SOURCE_ROOT"] = str(source_root)
    os.environ["OPENAGENT_ACCELERATOR_PYTHON"] = selected
    return selected


def ui_build_token() -> str:
    """Fingerprint UI assets so an old desktop shell cannot hide an update."""

    root = web_root()
    digest = hashlib.sha256()
    found = False
    for name in ("index.html", "app.js", "styles.css", "sw.js", "manifest.json"):
        path = root / name
        try:
            digest.update(name.encode("ascii"))
            digest.update(path.read_bytes())
            found = True
        except OSError:
            continue
    return digest.hexdigest()[:16] if found else __version__.replace(".", "-")


def workbench_runtime_dir() -> Path:
    configured = os.environ.get("OPENAGENT_RUNTIME_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "OpenAgent" / "runtime"
    return desktop_home().parent / "runtime"


class WorkbenchInstanceLock:
    """Process-owned lock preventing duplicate localhost servers per data directory."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "workbench.lock"
        self.handle: Any | None = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, PermissionError):
            handle.close()
            return False
        self.handle = handle
        return True

    def release(self) -> None:
        handle = self.handle
        self.handle = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self) -> WorkbenchInstanceLock:
        if not self.acquire():
            raise RuntimeError("OpenAgent Workbench is already running for this data directory.")
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


def _running_instance_url(data_dir: Path) -> str:
    state_path = data_dir / "workbench-state.json"
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(value.get("url", "")).strip() if isinstance(value, dict) else ""


def configure_managed_openvino_model(data_dir: Path) -> str:
    """Select a verified local OpenVINO model without crossing the network."""

    configured = os.environ.get("OPENAGENT_OPENVINO_MODEL_DIR", "").strip()
    if configured and Path(configured).is_dir():
        return str(Path(configured).resolve())
    project_root = Path(__file__).resolve().parents[1]
    candidates = [data_dir / "models" / "openvino" / "qwen2.5-1.5b-instruct-int4-ov"]
    source_root = installation_source_root()
    if source_root is not None:
        candidates.append(source_root / "models" / "openvino" / "qwen2.5-1.5b-instruct-int4-ov")
    candidates.append(project_root / "models" / "openvino" / "qwen2.5-1.5b-instruct-int4-ov")
    for candidate in candidates:
        if (candidate / "openvino_model.xml").is_file() and (candidate / "openvino_model.bin").is_file():
            os.environ["OPENAGENT_OPENVINO_MODEL_DIR"] = str(candidate.resolve())
            os.environ.setdefault("OPENAGENT_OPENVINO_DEVICE", "GPU")
            os.environ.setdefault("OPENAGENT_OPENVINO_MODEL_ID", "openagent-qwen2.5-1.5b-int4")
            return str(candidate.resolve())
    return ""


def project_venv_python() -> Path | None:
    """Return the installer-recorded source interpreter when it is available."""

    package_root = Path(__file__).resolve().parents[1]
    source_root = installation_source_root() or package_root
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    executable = "python.exe" if sys.platform == "win32" else "python"
    candidate = source_root / ".venv" / scripts_dir / executable
    return candidate.resolve() if candidate.is_file() else None


def maybe_reexec_project_venv() -> None:
    """Prefer the installer-recorded source virtualenv for development launches."""

    if os.environ.get("OPENAGENT_SKIP_PROJECT_VENV_REEXEC", "").strip():
        return
    if os.environ.get("OPENAGENT_PROJECT_VENV_REEXECED", "").strip():
        return
    venv_python = project_venv_python()
    if venv_python is None:
        return
    try:
        current = Path(sys.executable).resolve()
        target = venv_python.resolve()
    except OSError:
        return
    if current == target:
        return
    os.environ["OPENAGENT_PROJECT_VENV_REEXECED"] = "1"
    os.environ.setdefault("OPENAGENT_FASTER_WHISPER_MODEL", "base.en")
    existing_path = os.environ.get("PYTHONPATH", "")
    root_text = str(target.parents[2])
    os.environ["PYTHONPATH"] = root_text if not existing_path else f"{root_text}{os.pathsep}{existing_path}"
    command = [str(target), "-m", "openagent.workbench", *sys.argv[1:]]
    if sys.platform == "win32":
        completed = subprocess.run(command, check=False)
        raise SystemExit(completed.returncode)
    os.execv(str(target), command)


def workbench_status(data_dir: Path | str | None = None, *, project: str = "default") -> dict[str, Any]:
    data_path = Path(data_dir).expanduser() if data_dir else desktop_home()
    assets = web_root()
    return {
        "status": "ready",
        "mode": "modern-desktop-workbench",
        "project": project,
        "data_dir": str(data_path),
        "localhost_only": True,
        "single_instance": True,
        "permanent_address": f"http://127.0.0.1:{DEFAULT_WORKBENCH_PORT}/",
        "default_port": configured_workbench_port(),
        "runtime_dir": str(workbench_runtime_dir()),
        "browser_tab_required": False,
        "app_window_supported": bool(resolve_app_browser()),
        "entrypoints": {
            "package_script": "openagent-workbench",
            "desktop_script": "openagent-desktop",
            "python_module": "python -m openagent.workbench",
        },
        "assets": {
            "build": ui_build_token(),
            "web_root": str(assets),
            "index": str(assets / "index.html"),
            "css": str(assets / "styles.css"),
            "sass": str(assets / "styles.scss"),
            "js": str(assets / "app.js"),
        },
        "privacy_boundary": WORKBENCH_PRIVACY_BOUNDARY,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the modern OpenAgent local workbench.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to localhost.")
    parser.add_argument(
        "--port",
        type=int,
        default=configured_workbench_port(),
        help="Stable localhost port. Installed desktop launches always default to 8765.",
    )
    parser.add_argument("--project", default="default", help="Project scope for startup audit events.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for local OpenAgent data. Windows desktop launches share %%LOCALAPPDATA%%\\OpenAgent\\data.",
    )
    parser.add_argument("--no-open", action="store_true", help="Start the local workbench server without opening a window.")
    parser.add_argument("--status-json", action="store_true", help="Print workbench status as JSON without launching.")
    return parser


def launch_workbench(
    data_dir: Path | str | None = None,
    *,
    project: str = "default",
    host: str = "127.0.0.1",
    port: int = DEFAULT_WORKBENCH_PORT,
    open_window: bool = True,
) -> int:
    from .server import build_http_server

    data_path = Path(data_dir).expanduser() if data_dir else desktop_home()
    runtime_path = workbench_runtime_dir()
    runtime_path.mkdir(parents=True, exist_ok=True)
    instance_lock = WorkbenchInstanceLock(runtime_path)
    if not instance_lock.acquire():
        existing_url = _running_instance_url(runtime_path)
        if existing_url and open_window:
            open_app_window(existing_url, data_path, profile_dir=runtime_path / "browser-reconnect")
        print(existing_url or "OpenAgent Workbench is already running for this data directory.")
        return 0
    store = OpenAgentStore(data_path)
    managed_accelerator = configure_managed_accelerator()
    managed_openvino_model = configure_managed_openvino_model(data_path)
    state_path = runtime_path / "workbench-state.json"
    browser_profile = runtime_path / f"browser-{os.getpid()}"
    server = None
    server_thread: threading.Thread | None = None
    try:
        server = build_http_server(host, port, store, warm_openvino=bool(managed_openvino_model))
        address = f"http://{host}:{server.server_address[1]}"
        build_token = ui_build_token()
        launch_url = f"{address}/?openagent_version={__version__}&ui_build={build_token}&ui=core"
        state_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "started_at_unix": time.time(),
                    "url": launch_url,
                    "port": server.server_address[1],
                    "version": __version__,
                    "ui_build": build_token,
                    "data_dir": str(data_path.resolve()),
                    "browser_profile": str(browser_profile),
                }
            ),
            encoding="utf-8",
        )
        store.add_audit_event(
            AuditEvent(
                actor="desktop",
                action="workbench.launch",
                resource="openagent-workbench",
                project=project,
                metadata={
                    "address": address,
                    "localhost_only": True,
                    "app_window": open_window,
                    "pid": os.getpid(),
                    "accelerator_python": bool(managed_accelerator),
                    "openvino_model": bool(managed_openvino_model),
                },
            )
        )
        print(f"OpenAgent Workbench running at {launch_url}")
        print(f"Data directory: {store.data_dir}")

        process: subprocess.Popen[Any] | None = None
        if open_window:
            warm_event = getattr(server, "openvino_warm_event", None)
            if warm_event is not None and not warm_event.is_set():
                print("Warming the local GPU model before opening the desktop window...")
                warm_event.wait(timeout=120)
            process = open_app_window(launch_url, data_path, profile_dir=browser_profile)

        if process is not None:
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            while process.poll() is None:
                time.sleep(0.4)
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OpenAgent Workbench.")
    finally:
        if server is not None:
            if server_thread is not None and server_thread.is_alive():
                server.shutdown()
                server_thread.join(timeout=5)
            server.server_close()
        try:
            state_path.unlink(missing_ok=True)
        finally:
            instance_lock.release()
        shutil.rmtree(browser_profile, ignore_errors=True)
    return 0


def open_app_window(url: str, data_dir: Path, *, profile_dir: Path | None = None) -> subprocess.Popen[Any] | None:
    browser = resolve_app_browser()
    if browser:
        profile = profile_dir or (data_dir / "workbench-profile")
        profile.mkdir(parents=True, exist_ok=True)
        args = [
            browser,
            f"--app={url}",
            "--new-window",
            f"--user-data-dir={profile}",
            "--disable-background-mode",
            "--disable-extensions",
            "--disable-sync",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        try:
            return subprocess.Popen(args)
        except OSError:
            pass
    webbrowser.open(url)
    return None


def focus_openagent_window() -> bool:
    """Bring the existing app window forward without spawning another server or tab."""

    if sys.platform != "win32":
        return False
    try:
        import ctypes

        user32 = ctypes.windll.user32
        handles: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

        @callback_type
        def collect(handle, _param):
            length = user32.GetWindowTextLengthW(handle)
            if length:
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(handle, title, length + 1)
                if (
                    "openagent" in title.value.casefold()
                    and user32.IsWindowVisible(handle)
                    and _browser_window_process(user32, handle, ctypes)
                ):
                    handles.append(int(handle))
                    return False
            return True

        user32.EnumWindows(collect, 0)
        if not handles:
            return False
        handle = handles[0]
        user32.ShowWindow(handle, 9)
        user32.SetForegroundWindow(handle)
        return True
    except Exception:
        return False


def _browser_window_process(user32: Any, handle: int, ctypes_module: Any) -> bool:
    """Return true only for a visible Chromium browser process on Windows."""

    process_id = ctypes_module.c_ulong()
    user32.GetWindowThreadProcessId(handle, ctypes_module.byref(process_id))
    process = ctypes_module.windll.kernel32.OpenProcess(0x1000, False, process_id.value)
    if not process:
        return False
    try:
        path = ctypes_module.create_unicode_buffer(32768)
        size = ctypes_module.c_ulong(len(path))
        if not ctypes_module.windll.kernel32.QueryFullProcessImageNameW(process, 0, path, ctypes_module.byref(size)):
            return False
        return Path(path.value).name.casefold() in {"msedge.exe", "chrome.exe", "chromium.exe"}
    finally:
        ctypes_module.windll.kernel32.CloseHandle(process)


def resolve_app_browser() -> str:
    candidates: list[str] = []
    if sys.platform == "win32":
        candidates.extend(
            [
                str(Path.home() / "AppData" / "Local" / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ]
        )
    candidates.extend(["msedge", "chrome", "chromium"])
    for candidate in candidates:
        resolved = shutil.which(candidate) if not Path(candidate).is_file() else candidate
        if resolved:
            return str(resolved)
    return ""


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        maybe_reexec_project_venv()
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else desktop_home()
    if args.status_json:
        import json

        print(json.dumps(workbench_status(data_dir, project=args.project), indent=2, sort_keys=True))
        return 0
    return launch_workbench(
        data_dir,
        project=args.project,
        host=args.host,
        port=args.port,
        open_window=not args.no_open,
    )


if __name__ == "__main__":
    raise SystemExit(main())
