"""Read-only, fail-closed local product packaging verification.

This does not replace backend/frontend functional test suites, Windows installation,
or visual QA. It prevents a missing or stale React bundle being called 'ready'.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REFERENCE_SHA256 = "bdcf0f5e24121a94c562b524ef49ab2f9b8582ed5f083b79045ae3c01aac6ce1"


def check_product(root: Path, ui_root: Path | None = None) -> dict:
    root = root.resolve()
    checks: list[dict[str, str]] = []

    def verify(name: str, condition: bool, detail: str) -> None:
        checks.append({"name": name, "status": "PASS" if condition else "BLOCKED", "detail": detail})

    reference = root / "frontend-second/reference/frontend-final-approved.html"
    digest = hashlib.sha256(reference.read_bytes()).hexdigest() if reference.is_file() else "missing"
    verify("approved-design-reference", digest == REFERENCE_SHA256,
           f"Expected pinned HTML SHA256, observed: {digest}")

    selected = (ui_root or root / "frontend-second/dist").resolve()
    try:
        sys.path.insert(0, str(root))
        from openagent.workbench import validate_desktop_ui
        assets = validate_desktop_ui(selected)
    except Exception as exc:
        verify("production-react-build", False, str(exc))
    else:
        verify("production-react-build", bool(assets), f"Validated {len(assets)} HTML-referenced assets in {selected}")
        try:
            html = (selected / "index.html").read_text(encoding="utf-8")
        except OSError:
            html = ""
        verify("compiled-entrypoint", "src/main.tsx" not in html and "src/main.jsx" not in html,
               "UI must reference compiled JavaScript, never Vite source entries")
        verify("no-demo-fixture-artifacts", not any(
            file.name in {"reference-fixtures.json", "approved-shell-fixtures.json"}
            or "demo" in file.relative_to(selected).parts
            for file in selected.rglob("*") if file.is_file()
        ), "Production bundle must exclude Demo-only JSON fixture files")

    installer = root / "installers/windows/install.ps1"
    body = installer.read_text(encoding="utf-8") if installer.is_file() else ""
    verify("windows-install-uses-approved-react", all(
        marker in body for marker in ("npm", "run build", "frontend-second", "--ui-root", "ui.previous")
    ), "Installer must build, stage, validate, and explicitly load approved React assets")

    workflows = root / ".github/workflows"
    expected = ("ci.yml", "security.yml", "expert-engine.yml")
    verify("automated-release-gates", all((workflows / name).is_file() for name in expected),
           "Required CI, security and native engine workflows must exist")
    blocked = [c["name"] for c in checks if c["status"] != "PASS"]
    return {
        "kind": "openagent-local-production-package-gate",
        "overallStatus": "BLOCKED" if blocked else "PACKAGE_CHECKS_PASS",
        "blockedChecks": blocked,
        "checks": checks,
        "scope": "Packaging/source integrity only. Full Python/React, actual Windows installation, visual/UI acceptance, provider behaviour and migrations are separate mandatory release gates.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--ui-root", type=Path, default=None)
    args = parser.parse_args()
    result = check_product(args.root, args.ui_root)
    print(json.dumps(result, indent=2))
    return 1 if result["blockedChecks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
