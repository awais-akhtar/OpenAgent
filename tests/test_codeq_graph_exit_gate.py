"""Phase 1 exit gate: a synthetic multi-language repository.

Retrieval is deterministic; unchanged files keep their node ids and data; a changed file
re-derives only its own nodes and its neighbours; unresolved relations are labelled, not
fabricated; configuration, dependency and command links are present.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openagent.codebase import graph_store
from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.work_graph import build_code_knowledge_graph, prune_stale_references, resolve_local_import

FILES = {
    "pkg/__init__.py": "",
    "pkg/helpers.py": "def h():\n    return 1\n",
    "pkg/core.py": "from .helpers import h\nimport os\n\ndef run():\n    return h()\n",
    "src/main.py": "from pkg.core import run\nimport requests\n\ndef main():\n    return run()\n",
    "tests/test_core.py": "from pkg.core import run\n\ndef test_run():\n    assert run() == 1\n",
    "web/lib/index.js": "export const value = 1;\n",
    "web/lib/util.js": "export const util = 2;\n",
    "web/index.js": "import { value } from './lib';\nexport default value;\n",
    "web/app.ts": "import React from 'react';\nimport { util } from './lib/util';\nexport const app = util;\n",
    "requirements.txt": "requests>=2\n",
    "package.json": json.dumps({"name": "demo", "dependencies": {"react": "^18.0.0"}, "scripts": {"test": "node --test", "build": "tsc"}}),
    "tsconfig.json": json.dumps({"compilerOptions": {"strict": True}}),
}


def _write_repo(root: Path, files: dict[str, str]) -> None:
    for relative, content in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _graph(root: Path) -> dict:
    return build_code_knowledge_graph(RepositoryIntelligence(root).scan())


def _file_nodes(graph: dict) -> dict[str, dict]:
    return {str(node["path"]): node for node in graph["nodes"] if node.get("kind") == "file"}


class MultiLanguageGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "repo"
        self.root.mkdir()
        _write_repo(self.root, FILES)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_retrieval_is_deterministic_across_scans(self) -> None:
        first = _graph(self.root)
        second = _graph(self.root)
        self.assertEqual(first["revision"], second["revision"])
        self.assertEqual(first["nodes"], second["nodes"])
        self.assertEqual(first["edges"], second["edges"])

    def test_unchanged_files_keep_their_ids_and_data_after_an_edit(self) -> None:
        before = _file_nodes(_graph(self.root))
        (self.root / "web" / "app.ts").write_text("import React from 'react';\nexport const app = 3;\n", encoding="utf-8")
        after = _file_nodes(_graph(self.root))
        for path in FILES:
            if path == "web/app.ts":
                continue
            self.assertEqual(before[path]["id"], after[path]["id"], path)
            self.assertEqual(before[path], after[path], path)

    def test_refresh_touches_only_the_changed_file_and_its_neighbours(self) -> None:
        before = _graph(self.root)
        (self.root / "web" / "app.ts").write_text("import React from 'react';\nexport const app = 3;\n", encoding="utf-8")
        after = _graph(self.root)
        scope = graph_store.refresh_scope(before, after)
        self.assertEqual({"added": [], "changed": ["web/app.ts"], "deleted": []}, scope["changed_files"])
        changed_file_paths = {path for node_id in scope["changed_node_ids"] for path in [next((n.get("path") for n in after["nodes"] if n["id"] == node_id), "")] if path}
        self.assertTrue(changed_file_paths <= {"web/app.ts"} | set(scope["neighbour_paths"]), changed_file_paths)
        self.assertNotIn("pkg/core.py", changed_file_paths)

    def test_local_relations_resolve_and_external_ones_are_labelled(self) -> None:
        graph = _graph(self.root)
        files = _file_nodes(graph)
        edges = graph["edges"]

        def has_edge(source_path: str, target_path: str, relation: str) -> bool:
            return any(
                edge["source"] == files[source_path]["id"] and edge["target"] == files[target_path]["id"] and edge["relation"] == relation
                for edge in edges
            )

        self.assertTrue(has_edge("pkg/core.py", "pkg/helpers.py", "imports-local"))
        self.assertTrue(has_edge("src/main.py", "pkg/core.py", "imports-local"))
        self.assertTrue(has_edge("web/index.js", "web/lib/index.js", "imports-local"), "a directory import resolves to its index")
        self.assertTrue(has_edge("web/app.ts", "web/lib/util.js", "imports-local"))

        imports = {node["label"]: node for node in graph["nodes"] if node.get("kind") == "import"}
        self.assertEqual("unresolved", imports["os"]["resolution"], "stdlib module is not fabricated as a local file")
        self.assertFalse(any(edge["relation"] == "imports-local" and edge["source"] == files["pkg/core.py"]["id"] and edge["target"] != files["pkg/helpers.py"]["id"] for edge in edges))

    def test_declared_dependencies_and_commands_are_linked(self) -> None:
        graph = _graph(self.root)
        files = _file_nodes(graph)
        dependencies = {node["label"]: node for node in graph["nodes"] if node.get("kind") == "dependency"}
        self.assertEqual({"python", "npm"}, {node["ecosystem"] for node in dependencies.values() if node["label"] in {"requests", "react"}})
        self.assertTrue(any(edge["relation"] == "declares-dependency" and edge["source"] == files["requirements.txt"]["id"] and edge["target"] == dependencies["requests"]["id"] for edge in graph["edges"]))
        self.assertTrue(any(edge["relation"] == "uses-dependency" and edge["target"] == dependencies["react"]["id"] for edge in graph["edges"]))
        commands = {node["label"]: node["command"] for node in graph["nodes"] if node.get("kind") == "command"}
        self.assertEqual("npm run build", commands.get("build"))
        self.assertTrue(any(edge["relation"] == "configuration-file" and edge["target"] == files["tsconfig.json"]["id"] for edge in graph["edges"]))

    def test_stale_file_references_are_rejected_before_use(self) -> None:
        graph = _graph(self.root)
        live = {path for path in FILES if path != "pkg/helpers.py"}
        pruned = prune_stale_references(graph, live)
        self.assertEqual(1, pruned["rejected_stale_nodes"])
        self.assertNotIn("pkg/helpers.py", {node.get("path") for node in pruned["nodes"] if node.get("kind") == "file"})
        helper_id = _file_nodes(graph)["pkg/helpers.py"]["id"]
        self.assertFalse(any(helper_id in (edge["source"], edge["target"]) for edge in pruned["edges"]))


class ResolverRuleTests(unittest.TestCase):
    KNOWN = {"pkg/core.py", "pkg/helpers.py", "other/helpers.py", "web/lib/index.js", "web/app.ts", "pkg/__init__.py"}

    def test_plain_stdlib_name_never_matches_a_repository_file(self) -> None:
        # A sibling src/os.py would shadow the stdlib when this script runs, so it is a real local match;
        # only a module elsewhere in the repository is refused for a bare name.
        self.assertEqual([], resolve_local_import("src/main.py", "os", {"pkg/os.py"}))
        self.assertEqual(["src/os.py"], resolve_local_import("src/main.py", "os", {"src/os.py", "pkg/os.py"}))

    def test_dotted_name_matching_two_packages_is_ambiguous_not_guessed(self) -> None:
        candidates = resolve_local_import("app.py", "helpers.x", {"pkg/helpers/x.py", "other/helpers/x.py"})
        self.assertEqual(2, len(candidates))

    def test_relative_python_import_climbs_package_levels(self) -> None:
        self.assertEqual(["pkg/helpers.py"], resolve_local_import("pkg/core.py", ".helpers", self.KNOWN))

    def test_js_directory_import_needs_an_index_not_a_prefix(self) -> None:
        self.assertEqual(["web/lib/index.js"], resolve_local_import("web/app.ts", "./lib", self.KNOWN))
        self.assertEqual([], resolve_local_import("web/app.ts", "./lib/util", self.KNOWN))


if __name__ == "__main__":
    unittest.main()
