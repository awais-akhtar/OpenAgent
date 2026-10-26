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
            self.assertGreaterEqual(len(graph["nodes"]), 1)
            self.assertGreaterEqual(len(graph["edges"]), 1)
            self.assertEqual(len(graph["nodes"]), graph["stats"]["nodes"])
            node_ids = {node["id"] for node in graph["nodes"]}
            self.assertTrue(all(edge["source"] in node_ids and edge["target"] in node_ids for edge in graph["edges"]))
            self.assertTrue(any(node["degree"] > 0 for node in graph["nodes"]))
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

            documents = knowledge.list_documents(project="core")
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0]["title"], "OpenAgent Knowledge")
            self.assertGreaterEqual(documents[0]["chunk_count"], 1)
            self.assertIn("openagent", documents[0]["tags"])
            self.assertNotIn("source_path", documents[0])
            detail = knowledge.get_document(document["id"], project="core")
            self.assertEqual(detail["title"], "OpenAgent Knowledge")
            self.assertTrue(detail["content"])
            self.assertEqual(len(detail["chunks"]), document["chunk_count"])
            self.assertIn("openagent", detail["tags"])
            with self.assertRaises(KeyError):
                knowledge.get_document(document["id"], project="other")
            self.assertEqual(knowledge.delete_document(document["id"], project="other")["deleted"], False)
            self.assertEqual(knowledge.get_document(document["id"], project="core")["id"], document["id"])
            self.assertEqual(knowledge.delete_document(document["id"], project="core")["deleted"], True)
            with self.assertRaises(KeyError):
                knowledge.get_document(document["id"], project="core")

    def test_update_rebuilds_index_in_place_and_preserves_project_scope(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            knowledge = KnowledgeEngine(store)
            created = knowledge.ingest_text(
                "Old handbook",
                "LegacyZephyr contains the outdated deployment steps and old owner details.",
                project="core",
                source="manual",
                source_path="docs/handbook.md",
                metadata={"origin": "review"},
            )

            updated = knowledge.update_document(
                created["id"],
                "Current handbook",
                "CurrentQuasar contains the approved deployment steps and current owner details.",
                project="core",
            )

            self.assertEqual(updated["id"], created["id"])
            self.assertEqual(updated["created_at"], created["created_at"])
            self.assertEqual(updated["source_path"], "docs/handbook.md")
            self.assertEqual(updated["source"], "manual")
            detail = knowledge.get_document(created["id"], project="core")
            self.assertEqual(detail["title"], "Current handbook")
            self.assertIn("CurrentQuasar", detail["content"])
            self.assertNotIn("LegacyZephyr", detail["content"])
            self.assertEqual(len(knowledge.list_documents(project="core")), 1)
            self.assertEqual(knowledge.search("LegacyZephyr", project="core"), [])
            current_results = knowledge.search("CurrentQuasar", project="core")
            self.assertEqual(len(current_results), 1)
            self.assertEqual(current_results[0]["document_id"], created["id"])
            with store.session() as connection:
                chunks = connection.execute("SELECT COUNT(*) AS count FROM document_chunks WHERE document_id = ?", (created["id"],)).fetchone()["count"]
                embeddings = connection.execute("SELECT COUNT(*) AS count FROM chunk_embeddings WHERE project = 'core'").fetchone()["count"]
                stale_entities = connection.execute("SELECT COUNT(*) AS count FROM knowledge_entities WHERE project = 'core' AND name LIKE '%LegacyZephyr%'").fetchone()["count"]
            self.assertEqual(chunks, updated["chunk_count"])
            self.assertEqual(embeddings, updated["chunk_count"])
            self.assertEqual(stale_entities, 0)
            with self.assertRaises(KeyError):
                knowledge.update_document(created["id"], "Cross-project edit", "Should not update.", project="other")
            self.assertEqual(knowledge.get_document(created["id"], project="core")["title"], "Current handbook")


if __name__ == "__main__":
    unittest.main()
