"""Phase 1: bounded graph queries, deleted files, unresolved imports and path safety."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.work_graph import MAX_QUERY_LIMIT, build_code_knowledge_graph, query_graph, safe_relative_path


class GraphQueryAndSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "utils.py").write_text("def calculate(value):\n    return value * 2\n", encoding="utf-8")
        (self.root / "src" / "app.py").write_text("from src.utils import calculate\nimport missing_package_xyz\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_deleted_file_disappears_from_the_refreshed_graph(self) -> None:
        before = build_code_knowledge_graph(RepositoryIntelligence(self.root).scan())
        self.assertTrue(any(node.get("path") == "src/utils.py" for node in before["nodes"]))

        (self.root / "src" / "utils.py").unlink()
        after = build_code_knowledge_graph(RepositoryIntelligence(self.root).scan())
        self.assertFalse(any(node.get("path") == "src/utils.py" for node in after["nodes"]))
        # The importer remains, but no stale edge points at the deleted file.
        target_ids = {node["id"] for node in after["nodes"] if node.get("path") == "src/utils.py"}
        self.assertFalse(any(edge.get("target") in target_ids for edge in after["edges"]))

    def test_unresolved_import_is_labelled_and_not_fabricated(self) -> None:
        graph = build_code_knowledge_graph(RepositoryIntelligence(self.root).scan())
        unresolved = [node for node in graph["nodes"] if node.get("kind") == "import" and node.get("label") == "missing_package_xyz"]
        self.assertEqual(1, len(unresolved))
        self.assertEqual("unresolved", unresolved[0].get("resolution"))
        # No local-file edge is invented for a package that does not exist in the repository.
        import_ids = {node["id"] for node in unresolved}
        self.assertFalse(any(edge.get("source") in import_ids and edge.get("relation") == "imports-local" for edge in graph["edges"]))

    def test_query_is_bounded_on_a_large_graph(self) -> None:
        large = {"nodes": [{"id": f"n{i}", "kind": "symbol", "label": f"sym{i}", "path": f"pkg/mod{i}.py"} for i in range(500)], "edges": []}
        result = query_graph(large, limit=10_000)
        self.assertEqual(MAX_QUERY_LIMIT, len(result["nodes"]))
        self.assertTrue(result["truncated"])
        self.assertEqual(500, result["total"])

    def test_query_filters_by_kind_text_and_path_prefix(self) -> None:
        graph = build_code_knowledge_graph(RepositoryIntelligence(self.root).scan())
        by_kind = query_graph(graph, kind="symbol", limit=50)
        self.assertTrue(by_kind["nodes"])
        self.assertTrue(all(node["kind"] == "symbol" for node in by_kind["nodes"]))
        by_text = query_graph(graph, text="calculate", limit=50)
        self.assertTrue(any(node.get("label") == "calculate" for node in by_text["nodes"]))
        by_prefix = query_graph(graph, path_prefix="src", limit=50)
        self.assertTrue(by_prefix["nodes"])
        self.assertTrue(all(str(node.get("path", "")).startswith("src") for node in by_prefix["nodes"]))

    def test_path_traversal_and_absolute_paths_are_rejected(self) -> None:
        self.assertIsNone(safe_relative_path("../secret.txt"))
        self.assertIsNone(safe_relative_path("src/../../secret.txt"))
        self.assertIsNone(safe_relative_path("/etc/passwd"))
        self.assertIsNone(safe_relative_path("C:/Windows/system.ini"))
        self.assertIsNone(safe_relative_path(""))

    def test_valid_paths_are_normalized(self) -> None:
        self.assertEqual("src/app.py", safe_relative_path("src\\app.py"))
        self.assertEqual("a/b", safe_relative_path("./a//b"))


if __name__ == "__main__":
    unittest.main()
