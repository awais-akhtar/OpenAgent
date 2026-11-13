"""Measure CodeQ scan, graph build and graph persistence on generated repositories.

Usage: python scripts/measure_codeq_index.py [file_count ...]
Prints one JSON line per size with wall time and peak traced memory.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

from openagent.codebase import graph_store
from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.work_graph import build_code_knowledge_graph
from openagent.storage import OpenAgentStore


def make_repo(root: Path, count: int) -> None:
    for index in range(count):
        package = root / f"pkg{index // 50}"
        package.mkdir(parents=True, exist_ok=True)
        previous = f"from pkg{max(0, index - 1) // 50}.mod{max(0, index - 1)} import value{max(0, index - 1)}\n" if index else ""
        (package / f"mod{index}.py").write_text(
            f"{previous}def value{index}(x):\n    return x + {index}\n\n"
            f"def helper{index}(y):\n    return value{index}(y)\n",
            encoding="utf-8",
        )
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_all.py").write_text("def test_smoke():\n    assert True\n", encoding="utf-8")


def measure(count: int) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        repo_root = base / "repo"
        repo_root.mkdir()
        make_repo(repo_root, count)
        store = OpenAgentStore(base / "data")
        with store.session() as connection:
            graph_store.migrate(connection)
        intelligence = RepositoryIntelligence(repo_root)
        tracemalloc.start()
        started = time.perf_counter()
        project_map = intelligence.scan()
        scanned = time.perf_counter()
        graph = build_code_knowledge_graph(project_map)
        built = time.perf_counter()
        with store.session() as connection:
            graph_store.save_revision(connection, "bench", "local", graph)
        saved = time.perf_counter()
        query_started = time.perf_counter()
        context, retrieval = intelligence.select_task_context(
            "find value and helper functions in the package",
            graph,
            seed_paths=["pkg0/mod0.py"],
            limit=12,
            max_chars=48_000,
            max_file_chars=6_000,
            graph_limit=8,
        )
        queried = time.perf_counter()
        changed_path = repo_root / "pkg0" / "mod0.py"
        changed_path.write_text(
            changed_path.read_text(encoding="utf-8") + "\n# phase8 incremental benchmark change\n",
            encoding="utf-8",
        )
        incremental_started = time.perf_counter()
        incremental_map = intelligence.scan(previous=project_map)
        incremental_scanned = time.perf_counter()
        incremental_graph = build_code_knowledge_graph(incremental_map)
        incremental_built = time.perf_counter()
        refresh = graph_store.refresh_scope(graph, incremental_graph)
        with store.session() as connection:
            graph_store.save_revision(connection, "bench", "local", incremental_graph, previous_graph=graph)
        incremental_saved = time.perf_counter()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "files": count + 1,
            "scan_seconds": round(scanned - started, 3),
            "graph_build_seconds": round(built - scanned, 3),
            "persist_seconds": round(saved - built, 3),
            "context_query_seconds": round(queried - query_started, 3),
            "context_files": len(context),
            "context_chars": retrieval["context_chars"],
            "incremental_scan_seconds": round(incremental_scanned - incremental_started, 3),
            "incremental_graph_build_seconds": round(incremental_built - incremental_scanned, 3),
            "incremental_persist_seconds": round(incremental_saved - incremental_built, 3),
            "incremental_changed_files": refresh["changed_files"],
            "incremental_changed_node_count": len(refresh["changed_node_ids"]),
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "peak_traced_mb": round(peak / (1024 * 1024), 1),
        }


if __name__ == "__main__":
    sizes = [int(value) for value in sys.argv[1:]] or [20, 200, 1000]
    for size in sizes:
        print(json.dumps(measure(size), sort_keys=True), flush=True)
