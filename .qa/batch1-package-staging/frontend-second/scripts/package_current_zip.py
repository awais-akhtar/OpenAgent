"""Create and verify the current frontend-second implementation ZIP.

Run from the repository root with:
    python frontend-second/scripts/package_current_zip.py

The archive intentionally includes only frontend source/config/docs/tests and
the current approved-HTML/React screenshot pair matrix. It excludes runtime,
build, git, cache, environment, credential and database files.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "frontend-second"
OUTPUT = ROOT / "OpenAgent_Frontend_Second_Current.zip"
EXPECTED_CANONICAL_SHA256 = "BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1"
SCREENSHOT_DIRS = (
    ".qa/desktop-acceptance/continued-milestone-a/tab-states-final-current",
    ".qa/desktop-acceptance/interactions",
    ".qa/desktop-acceptance/milestone-a-resume/agents-run-final",
    ".qa/desktop-acceptance/assistant-goal-dialog-agent-current",
    ".qa/desktop-acceptance/continued-milestone-a/science-scraper-fixed",
    ".qa/desktop-acceptance/continued-milestone-a/vault-fix-final",
    ".qa/desktop-acceptance/continued-milestone-a/vault-interaction-final",
)
CURRENT_EVIDENCE_DIRS = (
    ".qa/living-ui-desktop-evidence-20261017-current",
    ".qa/milestone-a-current-20261017/tab-states",
    ".qa/targeted-corrections-20261017",
)
REPORTS = (
    "README.md",
    "AGENTS.md",
    "APPROVED_FRONTEND_SOURCE.md",
    "BACKEND_CAPABILITY_GAPS.md",
    "FEATURE_PARITY_REPORT.md",
    "FRONTEND_BACKEND_MASTER_GAPS.md",
    "FRONTEND_CAPABILITY_GAPS.md",
    "PIXEL_FIDELITY_GAPS.md",
    "VISUAL_EVIDENCE_INDEX.md",
    "VISUAL_FIDELITY_FINAL.md",
    "INTERACTION_STATE_CATALOG.md",
    "INTERACTION_STATE_INVENTORY.md",
    "INTERACTION_STATE_COVERAGE_REPORT.md",
    "LIVING_UI_IMPLEMENTATION_REPORT.md",
    "LIVING_UI_BACKEND_MAPPING.md",
    "LIVING_UI_TEST_RESULTS.md",
    "VISUAL_CORRECTION_REPORT.md",
)
ROOT_FILES = (
    ".gitignore",
    "index.html",
    "package.json",
    "package-lock.json",
    "playwright.config.ts",
    "tsconfig.json",
    "vite.config.ts",
    "vitest.config.ts",
    "visual-baseline.config.json",
)
QA_SCRIPTS = (
    ".qa/capture-vault-interaction-final.mjs",
)
EXCLUDED_PARTS = {"node_modules", ".git", "dist", "test-results", "playwright-report", "__pycache__", ".pytest_cache", ".vite", ".turbo"}
EXCLUDED_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx", ".tsbuildinfo", ".pyc"}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def unsafe_or_unnecessary(relative: Path) -> bool:
    if any(part.casefold() in EXCLUDED_PARTS for part in relative.parts):
        return True
    name = relative.name.casefold()
    if name == ".env" or name.startswith(".env.") or "credential" in name or "secret" in name:
        return True
    return relative.suffix.casefold() in EXCLUDED_SUFFIXES


def collect_files() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for dirname in ("src", "public", "reference", "tests", "scripts"):
        directory = APP / dirname
        for source in sorted(directory.rglob("*")):
            if not source.is_file():
                continue
            rel = source.relative_to(APP)
            if unsafe_or_unnecessary(rel):
                continue
            files[(Path("frontend-second") / rel).as_posix()] = source

    for name in (*ROOT_FILES, *REPORTS):
        source = APP / name
        if source.is_file() and not unsafe_or_unnecessary(Path(name)):
            files[(Path("frontend-second") / name).as_posix()] = source

    for name in QA_SCRIPTS:
        source = APP / name
        if source.is_file() and not unsafe_or_unnecessary(Path(name)):
            files[(Path("frontend-second") / name).as_posix()] = source

    for dirname in SCREENSHOT_DIRS:
        manifest_source = APP / dirname / "manifest.json"
        files[(Path("frontend-second") / dirname / "manifest.json").as_posix()] = manifest_source
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
        for capture in manifest.get("captures", manifest.get("records", [])):
            keys = ("canonicalPath", "reactPath") if "canonicalPath" in capture else ("approved", "react")
            for key in keys:
                source = APP / Path(str(capture[key]))
                rel = source.relative_to(APP)
                # These paths are exact screenshot references from the pinned
                # canonical fixture run; e.g. the Vault route filename contains
                # "secrets" but the image only contains masked demo placeholders.
                files[(Path("frontend-second") / rel).as_posix()] = source

    for dirname in CURRENT_EVIDENCE_DIRS:
        directory = APP / dirname
        for source in sorted(directory.rglob("*")):
            # This filename is a screenshot of the approved Vault demo tab,
            # whose credentials are synthetic and fully masked. The same
            # filename filter still excludes secret-bearing source/configs.
            safe_masked_demo_capture = source.suffix.casefold() == ".png" and source.name.casefold() == "vault-01-secrets-0.png"
            if source.is_file() and (safe_masked_demo_capture or not unsafe_or_unnecessary(source.relative_to(APP))):
                rel = source.relative_to(APP)
                files[(Path("frontend-second") / rel).as_posix()] = source

    score = APP / ".qa" / "visual-match-current.json"
    if score.is_file():
        files["frontend-second/.qa/visual-match-current.json"] = score
    return files


def archive_bytes(archive_path: str, source: Path) -> bytes:
    if archive_path.endswith("/manifest.json"):
        data = json.loads(source.read_text(encoding="utf-8"))
        data["source"] = "reference/frontend-final-approved.html"
        for capture in data.get("records", []):
            filename = Path(str(capture.get("file", ""))).name
            route, theme, viewport = capture.get("route"), capture.get("theme"), capture.get("viewport")
            if route and theme and viewport:
                capture["file"] = f".qa/{'approved-html-current' if 'approved-html' in archive_path else 'react-parity-current'}/{theme}/{viewport}/{route}.png"
            elif filename:
                capture["file"] = filename
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if archive_path.endswith("visual-match-current.json"):
        data = json.loads(source.read_text(encoding="utf-8"))
        for capture in data.get("captures", []):
            for key in ("reference", "target", "canonicalPath", "reactPath"):
                if isinstance(capture.get(key), str):
                    capture[key] = capture[key].replace("\\", "/")
        return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    return source.read_bytes()


def assert_current_evidence() -> None:
    canonical = APP / "reference" / "frontend-final-approved.html"
    if sha256(canonical.read_bytes()) != EXPECTED_CANONICAL_SHA256:
        raise SystemExit("The bundled canonical HTML does not match its approved SHA-256.")
    expected_counts = (328, 140, 12, 4, 32, 12, 56)
    manifests = []
    for dirname, count in zip(SCREENSHOT_DIRS, expected_counts, strict=True):
        manifest_path = APP / dirname / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit(f"Missing current screenshot manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = manifest.get("canonicalSourceSha256", manifest.get("canonicalHtmlSha256", ""))
        if source_hash.upper() != EXPECTED_CANONICAL_SHA256:
            raise SystemExit(f"Screenshot manifest is not pinned to the approved HTML: {manifest_path}")
        captures = manifest.get("captures", manifest.get("records", []))
        if len(captures) != count:
            raise SystemExit(f"Expected {count} matched screenshot captures in {manifest_path}.")
        if manifest.get("browser") != "Chrome 155.0.8059.40" or manifest.get("deviceScaleFactor") != 1:
            raise SystemExit(f"Unpinned browser or DPR in {manifest_path}.")
        if manifest.get("demoApiCalls", 0) != 0 or manifest.get("browserErrors", 0) != 0:
            raise SystemExit(f"Capture manifest reports errors or demo API calls: {manifest_path}")
        manifests.append(manifest)
        for capture in captures:
            keys = ("canonicalPath", "reactPath") if "canonicalPath" in capture else ("approved", "react")
            for key in keys:
                image = APP / Path(str(capture[key]))
                if not image.is_file() or not image.read_bytes().startswith(PNG_SIGNATURE):
                    raise SystemExit(f"Missing or invalid paired screenshot: {image}")

    living_dir = APP / CURRENT_EVIDENCE_DIRS[0]
    living_manifest_path = living_dir / "manifest.json"
    if not living_manifest_path.is_file():
        raise SystemExit(f"Missing current Living UI manifest: {living_manifest_path}")
    living = json.loads(living_manifest_path.read_text(encoding="utf-8"))
    if living.get("canonicalSha256", "").upper() != EXPECTED_CANONICAL_SHA256 or living.get("chrome") != "155.0.8059.80":
        raise SystemExit("Current Living UI evidence is not pinned to the canonical source and Chrome version.")
    if len(living.get("rows", [])) != 144 or living.get("readinessAssertions") != 80:
        raise SystemExit("Current Living UI capture set is incomplete.")
    if living.get("demoApiCalls") != 0 or living.get("browserErrors") != 0:
        raise SystemExit("Current Living UI evidence reports API calls or browser errors.")
    for capture in living["rows"]:
        for key in ("approved", "react"):
            image = living_dir / Path(str(capture[key]))
            if not image.is_file() or not image.read_bytes().startswith(PNG_SIGNATURE):
                raise SystemExit(f"Missing or invalid Living UI screenshot: {image}")

    tabs_dir = APP / CURRENT_EVIDENCE_DIRS[1]
    tabs_manifest_path = tabs_dir / "manifest.json"
    if not tabs_manifest_path.is_file():
        raise SystemExit(f"Missing latest desktop tab-state manifest: {tabs_manifest_path}")
    tabs = json.loads(tabs_manifest_path.read_text(encoding="utf-8"))
    if tabs.get("canonicalSourceSha256", "").upper() != EXPECTED_CANONICAL_SHA256 or tabs.get("browser") != "Chrome 155.0.8059.80":
        raise SystemExit("Latest desktop tab-state evidence is not pinned to the canonical source and Chrome version.")
    if len(tabs.get("captures", [])) != 344 or tabs.get("demoApiCalls") != 0 or tabs.get("browserErrors") != 0 or tabs.get("unmatchedTabs"):
        raise SystemExit("Latest desktop tab-state capture set is incomplete or reports errors.")
    for capture in tabs["captures"]:
        for key in ("approved", "react"):
            image = APP / Path(str(capture[key]))
            if not image.is_file() or not image.read_bytes().startswith(PNG_SIGNATURE):
                raise SystemExit(f"Missing or invalid desktop tab screenshot: {image}")


def main() -> None:
    assert_current_evidence()
    files = collect_files()
    with zipfile.ZipFile(OUTPUT, "w", allowZip64=True) as archive:
        for name, source in files.items():
            data = archive_bytes(name, source)
            compression = zipfile.ZIP_STORED if name.endswith(".png") else zipfile.ZIP_DEFLATED
            archive.writestr(name, data, compress_type=compression, compresslevel=6 if compression == zipfile.ZIP_DEFLATED else None)

    with zipfile.ZipFile(OUTPUT, "r") as archive:
        broken = archive.testzip()
        if broken:
            raise SystemExit(f"ZIP CRC verification failed for {broken}.")
        names = set(archive.namelist())
        required = {
            "frontend-second/src/App.tsx",
            "frontend-second/package.json",
            "frontend-second/package-lock.json",
            "frontend-second/vite.config.ts",
            "frontend-second/FRONTEND_BACKEND_MASTER_GAPS.md",
            "frontend-second/PIXEL_FIDELITY_GAPS.md",
            "frontend-second/VISUAL_EVIDENCE_INDEX.md",
            "frontend-second/.qa/capture-vault-interaction-final.mjs",
            "frontend-second/INTERACTION_STATE_COVERAGE_REPORT.md",
            "frontend-second/.qa/living-ui-desktop-evidence-20261017-current/manifest.json",
            "frontend-second/.qa/milestone-a-current-20261017/tab-states/manifest.json",
            "frontend-second/.qa/targeted-corrections-20261017/manifest.json",
        "frontend-second/.qa/desktop-acceptance/continued-milestone-a/tab-states-final-current/manifest.json",
        "frontend-second/.qa/desktop-acceptance/continued-milestone-a/vault-interaction-final/manifest.json",
        }
        missing = required - names
        if missing:
            raise SystemExit("Required files missing from archive: " + ", ".join(sorted(missing)))
        if any(any(part.casefold() in EXCLUDED_PARTS for part in PurePosixPath(name).parts) for name in names):
            raise SystemExit("The ZIP contains an excluded runtime/build/cache directory.")
        if any(Path(name).name.casefold() == ".env" or Path(name).name.casefold().startswith(".env.") for name in names):
            raise SystemExit("The ZIP contains an environment file.")
        pngs = [name for name in names if name.endswith(".png")]
        # 1,168 tab-state + 288 Living UI + 688 retained interaction PNGs,
        # plus 21 focused correction captures (20 Demo desktop + 1 Live guide).
        expected_pngs = 1168 + 288 + 688 + 21
        if len(pngs) != expected_pngs:
            raise SystemExit(f"Expected {expected_pngs} current comparison PNGs; found {len(pngs)}.")
        for required_prefix, expected_captures in zip(SCREENSHOT_DIRS, (328, 140, 12, 4, 32, 12, 56), strict=True):
            if sum(name.startswith(f"frontend-second/{required_prefix}/") and name.endswith(".png") for name in names) != expected_captures * 2:
                raise SystemExit(f"Screenshot set is incomplete: {required_prefix}.")
        canonical_copy = archive.read("frontend-second/reference/frontend-final-approved.html")
        if sha256(canonical_copy) != EXPECTED_CANONICAL_SHA256:
            raise SystemExit("Canonical HTML changed while packaging.")

        # Exercise an actual extraction in a temporary directory and check key
        # files and the full screenshot count there.
        with tempfile.TemporaryDirectory(prefix="openagent-frontend-second-zip-check-") as temp:
            archive.extractall(temp)
            extracted = Path(temp) / "frontend-second"
            if not (extracted / "src" / "App.tsx").is_file():
                raise SystemExit("Extracted source is incomplete.")
            for dirname, expected_captures in zip(SCREENSHOT_DIRS, (328, 140, 12, 4, 32, 12, 56), strict=True):
                if len(list((extracted / dirname).rglob("*.png"))) != expected_captures * 2:
                    raise SystemExit(f"Extracted screenshot set is incomplete: {dirname}.")
            living_images = list((extracted / CURRENT_EVIDENCE_DIRS[0]).rglob("*.png"))
            tab_images = list((extracted / CURRENT_EVIDENCE_DIRS[1]).rglob("*.png"))
            if len(living_images) != 288 or len(tab_images) != 688:
                raise SystemExit("Extracted current desktop evidence sets are incomplete.")

    digest = sha256(OUTPUT.read_bytes())
    print(f"Created and extraction-verified: {OUTPUT}")
    print(f"Files: {len(files)}; PNG evidence: {expected_pngs}; size: {OUTPUT.stat().st_size:,} bytes; SHA-256: {digest}")
    print("Excluded node_modules, .git, dist, test-results, caches, .env/secrets, credentials and databases.")


if __name__ == "__main__":
    main()
