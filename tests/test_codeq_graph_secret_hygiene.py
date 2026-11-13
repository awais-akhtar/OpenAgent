"""Phase 8: a secret value in a project file never reaches the scanned map or the code graph."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openagent.codebase.repository import RepositoryIntelligence
from openagent.codebase.work_graph import build_code_knowledge_graph

SECRET = "sk-test-SECRETVALUE-123"


class GraphSecretHygieneTests(unittest.TestCase):
    def test_env_secret_value_is_absent_from_map_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "app.py").write_text('import os\nKEY = os.environ.get("API_KEY")\n', encoding="utf-8")
            (root / ".env").write_text(f"API_KEY={SECRET}\n", encoding="utf-8")
            project_map = RepositoryIntelligence(root).scan()
            graph = build_code_knowledge_graph(project_map)
            self.assertNotIn(SECRET, json.dumps(project_map))
            self.assertNotIn(SECRET, json.dumps(graph))


if __name__ == "__main__":
    unittest.main()
