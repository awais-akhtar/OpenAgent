from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
SOURCE_ZIP = REPO / "OpenAgent_Frontend_Second_Updated.zip"
EVIDENCE_DIR = ROOT / ".qa" / "milestone-a-final-20261017" / "tab-states"
EVIDENCE_ZIP = REPO / "DESKTOP_VISUAL_EVIDENCE.zip"

EXCLUDED_DIRS = {
    ".git", ".qa", "node_modules", "dist", ".vite", "coverage", "test-results",
    "playwright-report", "__pycache__", ".pytest_cache", ".mypy_cache",
}
EXCLUDED_SUFFIXES = {
    ".tsbuildinfo", ".sqlite", ".sqlite3", ".db", ".pem", ".key", ".p12", ".pfx",
}


def safe_source_file(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS for part in rel.parts):
        return False
    if path.name.lower().startswith(".env"):
        return False
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    # The output archive can live beside old review bundles; nesting a previous
    # ZIP inside the source delivery is redundant and inflates the package.
    if path.suffix.lower() == ".zip":
        return False
    if path.name in {"id_rsa", "id_ed25519", "credentials.json", "secrets.json"}:
        return False
    return path.is_file()


def verify_zip(path: Path, required_member: str) -> int:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member:
            raise RuntimeError(f"CRC failure in {path.name}: {bad_member}")
        names = set(archive.namelist())
        if required_member not in names:
            raise RuntimeError(f"Required file missing from {path.name}: {required_member}")
        with tempfile.TemporaryDirectory(prefix="openagent-living-ui-extract-") as temp:
            archive.extractall(temp)
            extracted = Path(temp) / required_member
            if not extracted.is_file() or extracted.stat().st_size == 0:
                raise RuntimeError(f"Extraction verification failed for {required_member}")
        return len(names)


def main() -> None:
    canonical = ROOT / "reference" / "frontend-final-approved.html"
    import hashlib

    expected_hash = "BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1"
    actual_hash = hashlib.sha256(canonical.read_bytes()).hexdigest().upper()
    if actual_hash != expected_hash:
        raise RuntimeError(f"Refusing to package: canonical source hash changed ({actual_hash})")

    source_files = sorted(path for path in ROOT.rglob("*") if safe_source_file(path))
    if not source_files:
        raise RuntimeError("No source files found")
    with zipfile.ZipFile(SOURCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for source in source_files:
            archive.write(source, Path("frontend-second") / source.relative_to(ROOT))

    evidence_files = [path for path in EVIDENCE_DIR.rglob("*") if path.is_file()]
    required_evidence = {"manifest.json", "comparison-report.md"}
    if not required_evidence.issubset({path.name for path in evidence_files}):
        raise RuntimeError("Visual evidence manifest/report are missing")
    png_count = sum(path.suffix.lower() == ".png" for path in evidence_files)
    if png_count < 100:
        raise RuntimeError(f"Expected full paired desktop PNG evidence, found only {png_count} PNGs")
    with zipfile.ZipFile(EVIDENCE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for file in sorted(evidence_files):
            archive.write(file, Path("desktop-tab-comparisons") / file.relative_to(EVIDENCE_DIR))

    source_count = verify_zip(SOURCE_ZIP, "frontend-second/src/App.tsx")
    evidence_count = verify_zip(EVIDENCE_ZIP, "desktop-tab-comparisons/manifest.json")
    with zipfile.ZipFile(SOURCE_ZIP) as archive:
        names = archive.namelist()
        forbidden = [name for name in names if any(part in EXCLUDED_DIRS for part in Path(name).parts)]
        forbidden += [name for name in names if Path(name).name.lower().startswith(".env")]
        if forbidden:
            raise RuntimeError(f"Excluded content found in source archive: {forbidden[:8]}")

    print(f"Canonical SHA-256 verified: {actual_hash}")
    print(f"Source ZIP: {SOURCE_ZIP} ({SOURCE_ZIP.stat().st_size:,} bytes; {source_count} members; extracted OK)")
    print(f"Visual ZIP: {EVIDENCE_ZIP} ({EVIDENCE_ZIP.stat().st_size:,} bytes; {evidence_count} members; {png_count} PNGs; extracted OK)")


if __name__ == "__main__":
    main()
