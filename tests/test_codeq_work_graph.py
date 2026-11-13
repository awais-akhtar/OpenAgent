from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.work_graph import build_code_knowledge_graph, related_code_paths


class CodeQWorkGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "main.py").write_text(
            "from utils import calculate\n\ndef handler():\n    return calculate(2)\n",
            encoding="utf-8",
        )
        (self.root / "src" / "utils.py").write_text(
            "def calculate(value):\n    return value * 2\n",
            encoding="utf-8",
        )
        (self.root / "tests" / "test_utils.py").write_text(
            "from src.utils import calculate\n\ndef test_calculate():\n    assert calculate(2) == 4\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_scan_records_stable_file_identity_and_current_hash(self) -> None:
        intelligence = RepositoryIntelligence(self.root)
        first = intelligence.scan()
        first_graph = build_code_knowledge_graph(first)
        first_symbol = next(node for node in first_graph["nodes"] if node["kind"] == "symbol" and node["label"] == "calculate")
        old_hash = first_symbol["content_sha256"]

        source = self.root / "src" / "utils.py"
        source.write_text("# inserted line\n" + source.read_text(encoding="utf-8"), encoding="utf-8")
        second_graph = build_code_knowledge_graph(intelligence.scan())
        second_symbol = next(node for node in second_graph["nodes"] if node["kind"] == "symbol" and node["label"] == "calculate")

        self.assertEqual(first_symbol["id"], second_symbol["id"])
        self.assertNotEqual(old_hash, second_symbol["content_sha256"])
        self.assertEqual(2, second_symbol["line"])

    def test_graph_retrieval_adds_only_unambiguous_source_and_test_neighbors(self) -> None:
        graph = build_code_knowledge_graph(RepositoryIntelligence(self.root).scan())
        main_neighbors = related_code_paths(graph, ["src/main.py"])
        utility_neighbors = related_code_paths(graph, ["src/utils.py"])

        self.assertIn("src/utils.py", {item["path"] for item in main_neighbors})
        test_edge = next(item for item in utility_neighbors if item["path"] == "tests/test_utils.py" and item["relation"] == "possible-test-for-selected-file")
        self.assertEqual("possible-test-for-selected-file", test_edge["relation"])
        self.assertEqual("candidate", test_edge["confidence"])

    def test_ambiguous_symbol_calls_are_not_promoted_to_verified_edges(self) -> None:
        project_map = {
            "root": str(self.root),
            "file_inventory": [
                {"path": "a.py", "size_bytes": 1, "content_sha256": "a", "index_status": "indexed"},
                {"path": "b.py", "size_bytes": 1, "content_sha256": "b", "index_status": "indexed"},
            ],
            "symbols": [
                {"name": "run", "kind": "functiondef", "path": "a.py", "line": 1},
                {"name": "run", "kind": "functiondef", "path": "b.py", "line": 1},
                {"name": "entry", "kind": "functiondef", "path": "a.py", "line": 5},
            ],
            "call_graph": [{"file": "a.py", "symbol": "entry", "line": 5, "calls": ["run"]}],
            "imports": {}, "routes": [], "test_files": [],
        }
        graph = build_code_knowledge_graph(project_map)
        self.assertFalse(any(edge["relation"] == "calls" for edge in graph["edges"]))


if __name__ == "__main__":
    unittest.main()
