"""Command line hard-check runner for OpenAgent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import default_home
from .desktop import desktop_status, windows_installation_manifest
from .hard_audit import run_hard_audit
from .storage import OpenAgentStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the executable OpenAgent hard audit.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory for local OpenAgent data. Defaults to OPENAGENT_HOME or ~/.openagent.",
    )
    parser.add_argument("--project", default="hard-check", help="Project scope for audit evidence.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser


def run_project_hard_check(data_dir: Path | str | None = None, *, project: str = "hard-check") -> dict[str, object]:
    data_path = Path(data_dir).expanduser() if data_dir else default_home()
    store = OpenAgentStore(data_path)
    runtime_audit = run_hard_audit(store, project=project)
    desktop = desktop_status(data_path, project=project)
    installer = windows_installation_manifest()
    missing_installer_assets = [
        relative for relative, status in installer["files"].items() if not status["exists"]
    ]
    local_runtime_ok = bool(runtime_audit["ok"])
    windows_native_ok = (
        desktop["mode"] == "native-desktop"
        and desktop["windows_installable"] is True
        and desktop["browser_required"] is False
        and desktop["entrypoints"]["package_script"] == "openagent-desktop"
    )
    installer_assets_ok = not missing_installer_assets
    return {
        "ok": local_runtime_ok and windows_native_ok and installer_assets_ok,
        "project": project,
        "data_dir": str(data_path),
        "local_runtime_ok": local_runtime_ok,
        "windows_native_ok": windows_native_ok,
        "installer_assets_ok": installer_assets_ok,
        "missing_installer_assets": missing_installer_assets,
        "runtime_audit": runtime_audit,
        "desktop_status": desktop,
        "windows_installation": installer,
        "open_source_dependency_policy": {
            "allowed": True,
            "rule": "Use permissively licensed open-source libraries through declared dependencies; do not copy third-party source into OpenAgent without license review.",
            "current_runtime_dependencies": [],
            "declared_optional_dependencies": [
                {"package": "faiss-cpu", "extra": "rag", "license": "MIT", "purpose": "local vector acceleration"},
                {"package": "chromadb", "extra": "rag", "license": "Apache-2.0", "purpose": "persistent local vector mirror"},
                {"package": "faster-whisper", "extra": "voice", "license": "MIT", "purpose": "local speech recognition"},
                {"package": "vosk", "extra": "voice", "license": "Apache-2.0", "purpose": "lightweight local speech recognition"},
                {"package": "Pillow", "extra": "media", "license": "HPND", "purpose": "local reversible image operations"},
                {"package": "pypdf", "extra": "media", "license": "BSD-3-Clause", "purpose": "local PDF parsing and deterministic document processing"},
                {"package": "pypdfium2", "extra": "media", "license": "Apache-2.0 OR BSD-3-Clause", "purpose": "local PDF page rendering"},
                {"package": "reportlab", "extra": "media", "license": "BSD-3-Clause", "purpose": "deterministic local PDF generation"},
                {"package": "webauthn", "extra": "security", "license": "BSD-3-Clause", "purpose": "standards-compliant local passkey verification"},
                {"package": "openvino", "environment": ".accelerator-venv", "license": "Apache-2.0", "purpose": "local CPU, GPU, and NPU compilation"},
                {"package": "onnx", "environment": ".accelerator-venv", "license": "Apache-2.0", "purpose": "bounded local ONNX graph inspection"},
            ],
        },
        "deployment_level_remaining_work": [
            "Optional code-signed Windows installer artifact.",
            "Optional Microsoft Store, mobile store, or enterprise MDM distribution.",
            "Optional provider credentials for live external connectors and model services.",
            "Optional certified HSM/secure-enclave integrations for regulated deployments.",
        ],
        "privacy_boundary": "The hard check uses local data, local installer files, and the configured OpenAgent data directory only.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = Path(args.data_dir).expanduser() if args.data_dir else default_home()
    result = run_project_hard_check(data_dir, project=args.project)
    print(json.dumps(result, indent=2 if args.pretty else None, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
