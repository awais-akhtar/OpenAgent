from __future__ import annotations

import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from openagent.chat import ChatService
from openagent.external_research import ExternalResearchCollector
from openagent.research import ResearchEngine
from openagent.model_gateway import ModelResult
from openagent.models import ContextEvent
from openagent.storage import OpenAgentStore


ARXIV_FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><id>https://arxiv.org/abs/2608.01234</id><published>2026-08-10T12:00:00Z</published>
  <title>Efficient Local Language Model Research</title><summary>A measured local inference study.</summary>
  <author><name>A. Researcher</name></author></entry>
</feed>'''


class RecordingGateway:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def complete(self, messages, **_kwargs):
        self.messages = list(messages)
        return ModelResult(provider="test", model="test-model", content="Grounded response", metadata={})


class ChatResearchTests(unittest.TestCase):
    def test_research_respects_explicit_source_count_with_a_safe_cap(self) -> None:
        self.assertEqual(20, ResearchEngine._requested_source_limit("Find 20 papers and write a literature review"))
        self.assertEqual(40, ResearchEngine._requested_source_limit("List 80 papers"))

    def test_arxiv_and_crossref_results_include_real_dates_and_urls(self) -> None:
        crossref = {
            "message": {"items": [{
                "DOI": "10.1000/example", "title": ["A current LLM paper"],
                "URL": "https://doi.org/10.1000/example", "published": {"date-parts": [[2026, 8, 9]]},
                "author": [{"given": "Ada", "family": "Lovelace"}], "abstract": "Evidence summary",
            }]}
        }
        collector = ExternalResearchCollector()

        def read(url: str, **_kwargs) -> bytes:
            if "arxiv.org" in url:
                return ARXIV_FEED
            if "crossref.org" in url:
                return json.dumps(crossref).encode("utf-8")
            return json.dumps({"items": []}).encode("utf-8")

        with patch.object(collector, "_read", side_effect=read):
            sources = collector.search("find latest papers on LLM", limit=8)

        titles = {source["title"] for source in sources}
        self.assertIn("Efficient Local Language Model Research", titles)
        self.assertIn("A current LLM paper", titles)
        self.assertTrue(all(source["url"].startswith("https://") for source in sources))
        self.assertTrue(any(source["published_at"].startswith("2026-08-10") for source in sources))

    def test_research_result_is_persisted_as_an_ordinary_conversation(self) -> None:
        with TemporaryDirectory() as tmp:
            service = ChatService(OpenAgentStore(tmp), RecordingGateway())
            result = service.record_research(
                "Find the latest LLM paper",
                "A source-backed report.",
                project="core",
                sources=[{"title": "Paper", "url": "https://example.test/paper"}],
                research_session_id="research-1",
            )
            messages = service.store.list_chat_messages(result["conversation_id"], limit=10)

            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(messages[-1]["provider"], "research")
            self.assertEqual(messages[-1]["metadata"]["research_session_id"], "research-1")

    def test_chat_automatically_retrieves_relevant_local_knowledge(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            store.add_event(ContextEvent(
                title="Private deployment note", content="The deployment codename is Aurora Seven.",
                type="knowledge", source="upload", project="core",
            ))
            gateway = RecordingGateway()
            ChatService(store, gateway).send("What is the deployment codename?", project="core")

            system = gateway.messages[0]["content"]
            self.assertIn("Private deployment note", system)
            self.assertIn("Aurora Seven", system)

    def test_chat_uses_project_conversation_memory_and_current_attachments(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            gateway = RecordingGateway()
            service = ChatService(store, gateway)
            service.send("My preferred report format is a short table.", project="core")
            service.send(
                "Use my preferred format for this file.",
                project="core",
                attachments=[{
                    "id": "asset-1", "name": "evidence.txt", "media_type": "document",
                    "extracted_text": "Candidate Delta reached 92 percent accuracy.",
                }],
            )

            system = gateway.messages[0]["content"]
            self.assertIn("another conversation", system)
            self.assertIn("preferred report format", system)
            self.assertIn("evidence.txt", system)
            self.assertIn("Candidate Delta", system)



if __name__ == "__main__":
    unittest.main()
