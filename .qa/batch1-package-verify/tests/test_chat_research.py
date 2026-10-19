from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.chat import ChatService
from openagent.external_research import ExternalResearchCollector
from openagent.research import ResearchEngine
from openagent.model_gateway import ModelResult
from openagent.models import ChatMessage, ContextEvent
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


class VisionRecordingGateway(RecordingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.vision_call = None

    def complete_vision(self, messages, image, **kwargs):
        self.messages = list(messages)
        self.vision_call = {"image": image, **kwargs}
        return ModelResult(
            provider="ollama", model="qwen2.5vl:3b", content="The image contains a blue square.",
            metadata={"image_transport": "ollama-message-images"},
        )


class AttachmentDenialGateway(RecordingGateway):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, messages, **_kwargs):
        self.calls += 1
        self.messages = list(messages)
        content = (
            "I am unable to access or process attachments."
            if self.calls == 1 else "The attached report says Candidate Delta reached 92 percent accuracy."
        )
        return ModelResult(provider="test", model="test-model", content=content, metadata={})


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
            self.assertIn("processed locally", system)
            self.assertIn("<attached_evidence>", gateway.messages[-1]["content"])
            self.assertIn("evidence.txt", gateway.messages[-1]["content"])
            self.assertIn("Candidate Delta reached 92 percent accuracy", gateway.messages[-1]["content"])

    def test_chat_retrieves_an_early_turn_after_hundreds_of_messages(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            gateway = RecordingGateway()
            service = ChatService(store, gateway)
            first = service.send("Remember that the launch phrase is amber comet.", project="core")
            conversation_id = first["conversation_id"]
            for index in range(620):
                store.add_chat_message(ChatMessage(
                    conversation_id=conversation_id, role="user", content=f"Routine progress note {index}.",
                    provider="user", model="human",
                ))
            service.send("What launch phrase did I ask you to remember?", conversation_id=conversation_id, project="core")

            self.assertIn("amber comet", gateway.messages[0]["content"].lower())
            self.assertGreaterEqual(len(store.list_chat_messages(conversation_id, limit=5000)), 622)

    def test_chat_sends_a_stored_image_to_the_native_vision_gateway(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            gateway = VisionRecordingGateway()
            service = ChatService(store, gateway)
            asset = service.media.ingest_bytes(
                b"\x89PNG\r\n\x1a\nimage-payload", filename="diagram.png", project="core",
            )

            result = service.send(
                "What is shown in this image?", project="core", provider="ollama",
                attachments=[{
                    "id": asset["id"], "name": "diagram.png", "media_type": "image",
                    "mime_type": "image/png", "summary": asset["analysis"]["summary"],
                }],
            )

            self.assertEqual(gateway.vision_call["image"], b"\x89PNG\r\n\x1a\nimage-payload")
            self.assertEqual(gateway.vision_call["media_type"], "image/png")
            self.assertEqual(result["model"], "qwen2.5vl:3b")
            self.assertEqual(result["messages"][-1]["content"], "The image contains a blue square.")

    def test_stored_image_hash_allows_windows_path_ctime_jitter(self) -> None:
        with TemporaryDirectory() as tmp:
            service = ChatService(OpenAgentStore(tmp), VisionRecordingGateway())
            payload = bytes((137, 80, 78, 71, 13, 10, 26, 10)) + b"image-payload"
            asset = service.media.ingest_bytes(payload, filename="diagram.png", project="core")
            original_stat = Path.stat

            def path_stat(path, *args, **kwargs):
                result = original_stat(path, *args, **kwargs)
                if str(path) == asset['stored_path']:
                    return SimpleNamespace(
                        st_dev=result.st_dev, st_ino=result.st_ino,
                        st_size=result.st_size, st_mtime_ns=result.st_mtime_ns,
                        st_ctime_ns=result.st_ctime_ns + 1, st_mode=result.st_mode)
                return result

            with patch('openagent.chat.Path.stat', path_stat):
                observed = service._vision_attachment([{
                    'id': asset['id'], 'media_type': 'image'}])

            self.assertEqual(observed, (payload, asset['mime_type']))

    def test_chat_corrects_a_false_attachment_access_denial(self) -> None:
        with TemporaryDirectory() as tmp:
            gateway = AttachmentDenialGateway()
            service = ChatService(OpenAgentStore(tmp), gateway)
            result = service.send(
                "Read this report.", project="core",
                attachments=[{
                    "id": "asset-1", "name": "report.txt", "media_type": "document",
                    "extracted_text": "Candidate Delta reached 92 percent accuracy.",
                }],
            )

            self.assertEqual(gateway.calls, 2)
            self.assertIn("Candidate Delta", result["messages"][-1]["content"])
            self.assertIn("Correction: the attachment was already processed", gateway.messages[-1]["content"])

    def test_attachment_evidence_survives_follow_up_turns(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            gateway = RecordingGateway()
            service = ChatService(store, gateway)
            asset = service.media.ingest_bytes(
                b"Awais Akhtar is a software engineer with experience in local AI systems.",
                filename="cv-awais-akhtar.txt", project="core",
            )
            first = service.send(
                "Read this file.", project="core",
                attachments=[{"id": asset["id"], "name": "cv-awais-akhtar.txt", "media_type": "document"}],
            )
            service.send(
                "Read that file completely and summarize the experience.",
                conversation_id=first["conversation_id"], project="core",
            )

            self.assertIn("Files from earlier turns", gateway.messages[0]["content"])
            self.assertIn("software engineer", gateway.messages[-1]["content"])
            self.assertIn("cv-awais-akhtar.txt", gateway.messages[-1]["content"])



if __name__ == "__main__":
    unittest.main()
