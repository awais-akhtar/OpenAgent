from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.knowledge import KnowledgeEngine
from openagent.storage import OpenAgentStore


class KnowledgeEngineTests(unittest.TestCase):
    def test_ingest_search_semantic_graph_compress_and_replay(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            knowledge = KnowledgeEngine(store)

            document = knowledge.ingest_text(
                "OpenAgent Knowledge",
                (
                    "OpenAgent builds a local-first knowledge graph.\n\n"
                    "Awais uses OpenAgent for private AI memory on 2026-06-25.\n\n"
                    "The system supports source attribution and semantic retrieval."
                ),
                project="core",
            )

            self.assertGreaterEqual(document["chunk_count"], 1)
            self.assertIn("openagent", document["tags"])
            text_results = knowledge.search("knowledge graph", project="core")
            self.assertGreaterEqual(len(text_results), 1)
            self.assertIn("source", text_results[0]["metadata"])

            semantic_results = knowledge.semantic_search("private memory", project="core")
            self.assertGreaterEqual(len(semantic_results), 1)
            self.assertIn("score", semantic_results[0])

            hybrid_results = knowledge.hybrid_search("private knowledge graph", project="core")
            self.assertGreaterEqual(len(hybrid_results), 1)
            self.assertIn("retrieval", hybrid_results[0])

            graph = knowledge.graph(project="core")
            self.assertGreaterEqual(len(graph["entities"]), 1)
            tags = knowledge.tags(project="core")
            self.assertGreaterEqual(len(tags), 1)
            timeline = knowledge.timeline(project="core")
            self.assertGreaterEqual(len(timeline), 2)
            synthesis = knowledge.synthesize("private AI memory", project="core")
            self.assertIn("source-attributed", " ".join(synthesis["reasoning_steps"]))
            self.assertGreaterEqual(len(synthesis["sources"]), 1)

            compressed = knowledge.compress_memory(project="core")
            self.assertEqual(compressed["summary"]["project"], "core")

            replay = knowledge.create_replay(project="core", label="Memory replay", query="OpenAgent")
            self.assertEqual(replay["project"], "core")
            self.assertIn("events", replay["snapshot"])
            self.assertIn("chunks", replay["snapshot"])

            status = knowledge.status(project="core")
            self.assertEqual(status["documents"], 1)
            self.assertGreaterEqual(status["chunks"], 1)
            self.assertGreaterEqual(status["tags"], 1)


if __name__ == "__main__":
    unittest.main()
