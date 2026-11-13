"""Phase 3: validate a run's task dependency graph before execution.

Reports unknown dependencies and cycles explicitly, so a deadlock is named rather than
reported as a vague "cannot continue". Pure: operates on task dicts, no I/O.
"""

from __future__ import annotations

from typing import Any


def validate_task_graph(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return {"valid", "unknown_dependencies", "cycles"} for tasks with id and depends_on."""
    ids = {str(task.get("id", "")) for task in tasks}
    graph = {str(task.get("id", "")): [str(dep) for dep in task.get("depends_on", [])] for task in tasks}
    unknown: list[dict[str, str]] = []
    for task_id, deps in graph.items():
        for dep in deps:
            if dep not in ids:
                unknown.append({"task": task_id, "missing": dep})

    cycles: list[list[str]] = []
    state: dict[str, int] = {}  # 0 = unvisited, 1 = on stack, 2 = done
    stack: list[str] = []

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for dep in graph.get(node, []):
            if dep not in graph:
                continue
            if state.get(dep, 0) == 1:
                cycles.append(stack[stack.index(dep):] + [dep])
            elif state.get(dep, 0) == 0:
                visit(dep)
        stack.pop()
        state[node] = 2

    for node in graph:
        if state.get(node, 0) == 0:
            visit(node)

    return {"valid": not unknown and not cycles, "unknown_dependencies": unknown, "cycles": cycles}


def independent_batch(ready: list[dict[str, Any]], declared_files: dict[str, list[str]]) -> list[str]:
    """Pick ready tasks that may run together: each declares files, and no two share a file.

    Tasks without a declared write set are never batched, since their edits cannot be
    checked for conflicts. Order follows the ready list, so selection is deterministic.
    """
    claimed: set[str] = set()
    batch: list[str] = []
    for task in ready:
        task_id = str(task.get("id", ""))
        files = {path for path in declared_files.get(task_id, []) if path}
        if not files or files & claimed:
            continue
        claimed |= files
        batch.append(task_id)
    return batch


SLICE_LAYERS = ("ui", "api", "persistence", "tests")


def layer_of(path: str) -> str | None:
    """Classify a repository path into a vertical-slice layer, or None when unknown.

    A path heuristic, not a parser: it matches conventional directory and file names.
    """
    normalized = "/" + str(path).replace("\\", "/").lower().lstrip("/")
    name = normalized.rsplit("/", 1)[-1]
    if (
        "/tests/" in normalized or "/__tests__/" in normalized or name.startswith("test_")
        or ".test." in name or ".spec." in name
    ):
        return "tests"
    if any(token in normalized for token in ("/migrations/", "/models/", "/models.", "/schema", "/repositories/", "/repository.", "/db/", "/sql/", "/storage.", "/persistence")):
        return "persistence"
    if name.endswith((".tsx", ".jsx", ".vue", ".svelte", ".css", ".scss", ".html")) or any(token in normalized for token in ("/components/", "/screens/", "/pages/", "/ui/")):
        return "ui"
    if any(token in normalized for token in ("/routes/", "/routes.", "/api/", "/controllers/", "/handlers/", "/endpoints/", "/backend/")):
        return "api"
    return None


def missing_slice_layers(declared_files: list[str], touched_files: list[str]) -> list[str]:
    """Layers a multi-layer task declared but has not yet changed.

    A task that declares files in one layer only has no slice requirement. A task that
    declares two or more layers must change every one of them before it can be complete.
    """
    required = {layer for layer in (layer_of(path) for path in declared_files) if layer}
    if len(required) < 2:
        return []
    touched = {layer for layer in (layer_of(path) for path in touched_files) if layer}
    return sorted(required - touched)
