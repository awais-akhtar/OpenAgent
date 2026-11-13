"""Two saves of the same graph revision return one row instead of failing on the unique key."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openagent.codebase import graph_store
from openagent.codebase.work_graph import build_code_knowledge_graph
from openagent.storage import OpenAgentStore


class _MissingFirstLookup:
    """Wraps a connection so the first revision lookup misses, as it would before a concurrent insert."""

    def __init__(self, connection) -> None:
        self._connection = connection
        self._missed = False

    def execute(self, sql, params=()):
        if sql.startswith("SELECT id FROM codebase_graph_revisions") and not self._missed:
            self._missed = True
            return _NoRow()
        return self._connection.execute(sql, params)

    def __getattr__(self, name):
        return getattr(self._connection, name)


class _NoRow:
    def fetchone(self):
        return None


class GraphRevisionRaceTests(unittest.TestCase):
    def test_racing_insert_of_the_same_revision_reuses_the_existing_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OpenAgentStore(Path(directory) / "data")
            graph = build_code_knowledge_graph({"root": "/r", "file_inventory": [
                {"path": "a.py", "content_sha256": "h", "index_status": "fresh", "language": "python", "size_bytes": 1}], "symbols": []})
            with store.session() as connection:
                graph_store.migrate(connection)
                first = graph_store.save_revision(connection, "p", "o", graph)
            with store.session() as connection:
                second = graph_store.save_revision(_MissingFirstLookup(connection), "p", "o", graph)
                count = connection.execute("SELECT COUNT(*) FROM codebase_graph_revisions").fetchone()[0]
            self.assertEqual(first, second)
            self.assertEqual(1, count)


if __name__ == "__main__":
    unittest.main()
