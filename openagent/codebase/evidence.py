"""Truthful completion gate for CodeQ runs (roadmap Phase 4).

A run may end as completed only when all of these hold, computed from recorded
evidence and the current workspace, never from a model's claim:

* validation passed, and at least one executed check (lint, typecheck, test,
  build, browser or runtime) exited 0;
* that evidence was produced from the current source fingerprint, so a
  changed file invalidates earlier results;
* no independent review finding is still blocking;
* no acceptance item (audit missing task) is still unresolved.

The model's own `done` flag is never an input. The artifact scan
(`kind == "artifacts"`) inspects files only; it proves nothing about behaviour.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Iterable

EXECUTED_CHECK_KINDS = ("lint", "typecheck", "test", "build", "browser", "runtime")
SKIPPED_DIRECTORIES = {".git", "node_modules", ".venv", "__pycache__"}


def source_fingerprint(root: str | Path) -> str:
    """Deterministic digest of every workspace file's path and content.

    Changing, adding or removing any file changes the fingerprint. Dependency and
    VCS directories are skipped, matching the artifact scan's exclusions.
    """
    base = Path(root)
    digest = hashlib.sha256()
    if not base.is_dir():
        return "missing-workspace"
    base = base.resolve()

    def raise_walk_error(error: OSError) -> None:
        # os.walk otherwise silently skips unreadable directories by default.
        raise error

    for current, directories, files in os.walk(base, topdown=True, onerror=raise_walk_error, followlinks=False):
        directory = Path(current)
        retained: list[str] = []
        for name in sorted(directories):
            if name in SKIPPED_DIRECTORIES:
                continue
            candidate = directory / name
            if candidate.is_symlink():
                raise OSError(f"cannot fingerprint symlinked directory: {candidate}")
            retained.append(name)
        directories[:] = retained
        for name in sorted(files):
            path = directory / name
            relative = path.relative_to(base)
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if base not in resolved.parents:
                    raise OSError(f"cannot fingerprint file outside workspace: {relative.as_posix()}")
            if not path.is_file():
                continue
            file_digest = hashlib.sha256()
            # Stream large files. Read errors deliberately fail closed: an
            # incomplete walk must never appear to be a current fingerprint.
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    file_digest.update(chunk)
            file_hash = file_digest.hexdigest()
            digest.update(relative.as_posix().encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def executed_passing_checks(validation: dict[str, Any], fingerprint: str | None = None) -> list[str]:
    """Kinds that really ran and exited 0 (and, when given, were produced from `fingerprint`)."""
    passed: list[str] = []
    for item in validation.get("checks", []):
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", ""))
        if kind not in EXECUTED_CHECK_KINDS:
            continue
        if not str(item.get("evidence_id", "")).strip():
            continue
        if item.get("status") != "passed" or item.get("exit_code") not in (0, "0"):
            continue
        if fingerprint is not None and item.get("source_fingerprint", "") != fingerprint:
            continue
        passed.append(kind)
    return passed


def completion_gate(
    validation: dict[str, Any],
    current_fingerprint: str,
    *,
    review_blocking: int = 0,
    unresolved: Iterable[Any] = (),
    pending_approvals: int = 0,
) -> dict[str, Any]:
    """Decide whether the run may be reported completed.

    Returns {"passed", "status", "reason", "next_action", "evidence", "stale"}.
    `status` is "completed" only when passed; otherwise "blocked", with the exact
    reason and the next action recorded for the owner.
    """
    unresolved = list(unresolved)

    if pending_approvals > 0:
        # Waiting on an owner decision is not a failure: the run stays open and can resume.
        return {"passed": False, "status": "awaiting_approval",
                "reason": f"{pending_approvals} approval(s) pending",
                "next_action": "resolve the pending approval and resume",
                "evidence": [], "stale": False}

    def blocked(reason: str, next_action: str, stale: bool = False) -> dict[str, Any]:
        return {"passed": False, "status": "blocked", "reason": reason, "next_action": next_action,
                "evidence": [], "stale": stale}

    if not validation or not validation.get("passed"):
        failing = sorted({str(item.get("kind")) for item in validation.get("checks", [])
                          if isinstance(item, dict) and item.get("status") != "passed"})
        return blocked("verification failed" + (f": {', '.join(failing)}" if failing else ""),
                       "repair the failing check and run verification again")

    validated_fingerprint = str(validation.get("source_fingerprint", ""))
    if not validated_fingerprint or validated_fingerprint != current_fingerprint:
        return blocked("sources changed after verification; the recorded results are stale",
                       "re-run verification on the current sources", stale=True)

    evidence = executed_passing_checks(validation, current_fingerprint)
    if not evidence:
        return blocked("no executed verification evidence: no lint, typecheck, test, build, browser or runtime check passed",
                       "configure and run a build, test or lint check")

    if review_blocking > 0:
        return blocked(f"{review_blocking} independent review finding(s) still blocking",
                       "resolve the review findings and verify again")

    if unresolved:
        return blocked(f"{len(unresolved)} acceptance item(s) unresolved",
                       "complete the unresolved work items and verify again")

    return {"passed": True, "status": "completed", "reason": "executed verification passed on current sources",
            "next_action": "", "evidence": evidence, "stale": False}


def criterion_evidence_ids(criterion: dict[str, Any], checks: list[dict[str, Any]], current_fingerprint: str) -> list[str]:
    """A criterion is satisfied only by a passing executed check on the current sources
    that ran for one of the criterion's linked tasks.

    `criterion` carries `linked_task_ids` (from codebase_goal_criteria). Each check carries
    persisted `evidence_id`, `task_ids`, and `source_fingerprint`. Model claims,
    unidentified checks, and stale results never satisfy it.
    """
    linked = {str(task_id) for task_id in criterion.get("linked_task_ids", []) if str(task_id)}
    if not linked:
        return []
    evidence_ids: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if str(check.get("kind", "")) not in EXECUTED_CHECK_KINDS:
            continue
        if check.get("status") != "passed" or check.get("exit_code") not in (0, "0"):
            continue
        if str(check.get("source_fingerprint", "")) != current_fingerprint:
            continue
        if linked & {str(task_id) for task_id in check.get("task_ids", [])}:
            evidence_id = str(check.get("evidence_id", "")).strip()
            if evidence_id:
                evidence_ids.append(evidence_id)
    return sorted(set(evidence_ids))


def criterion_satisfied(criterion: dict[str, Any], checks: list[dict[str, Any]], current_fingerprint: str) -> bool:
    """A criterion needs a persisted, identifiable current check as evidence."""
    return bool(criterion_evidence_ids(criterion, checks, current_fingerprint))


def file_digests(root: str | Path) -> dict[str, str]:
    """Per-file content digests, with the same exclusions as source_fingerprint.

    Used to see which files actually changed across a batch, including writes that no
    tool recorded, such as a shell command's output.
    """
    base = Path(root)
    result: dict[str, str] = {}
    if not base.is_dir():
        return result
    base = base.resolve()
    for current, directories, files in os.walk(base, topdown=True, followlinks=False):
        directories[:] = [name for name in directories if name not in SKIPPED_DIRECTORIES]
        directory = Path(current)
        for name in files:
            path = directory / name
            if path.is_symlink() or not path.is_file():
                continue
            digest = hashlib.sha256()
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[path.relative_to(base).as_posix()] = digest.hexdigest()
    return result
