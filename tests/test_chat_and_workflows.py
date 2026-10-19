from __future__ import annotations

from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from openagent.chat import ChatService
from openagent.model_gateway import ModelGateway
from openagent.storage import OpenAgentStore
from openagent.workflows import WorkflowEngine


class ChatAndWorkflowTests(unittest.TestCase):
    def test_chat_stream_preserves_generation_error_and_notes_close_failure(self) -> None:
        class BrokenStream:
            def __init__(self):
                self.items = iter((SimpleNamespace(
                    content="partial", provider="local", model="fixture",
                    done=False, metadata={},
                ),))
                self.closed = False

            def __iter__(self):
                return self

            def __next__(self):
                try:
                    return next(self.items)
                except StopIteration:
                    raise ValueError("generation failed")

            def close(self):
                self.closed = True
                raise RuntimeError("cleanup failed")

        class Gateway:
            def __init__(self):
                self.stream = BrokenStream()

            def stream_complete(self, *_args, **_kwargs):
                return self.stream

        with TemporaryDirectory() as tmp:
            gateway = Gateway()
            service = ChatService(OpenAgentStore(tmp), gateway)
            events = service.stream("hello")
            self.assertEqual(next(events)["type"], "conversation")
            self.assertEqual(next(events)["type"], "status")
            self.assertEqual(next(events)["content"], "partial")
            with self.assertRaisesRegex(ValueError, "generation failed") as caught:
                next(events)

        self.assertTrue(gateway.stream.closed)
        self.assertTrue(any(
            "chat provider stream cleanup failed: RuntimeError" in note
            for note in getattr(caught.exception, "__notes__", ())
        ))

    def test_recent_history_limit_keeps_latest_messages_in_chronological_order(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = ChatService(store, ModelGateway())
            result = service.send("message 0", project="core")
            conversation_id = str(result["conversation_id"])
            for index in range(1, 26):
                service.send(f"message {index}", conversation_id=conversation_id, project="core")

            messages = store.list_chat_messages(conversation_id, limit=8)

            self.assertEqual(8, len(messages))
            self.assertEqual("message 22", messages[0]["content"])
            self.assertIn("message 25", messages[-2]["content"])
            self.assertEqual("assistant", messages[-1]["role"])

    def test_local_chat_persists_messages_and_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = ChatService(store, ModelGateway())

            result = service.send("Remember that OpenAgent is local-first.", project="core")

            conversation_id = result["conversation"]["id"]
            messages = store.list_chat_messages(conversation_id)
            self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
            self.assertEqual(messages[1]["provider"], "local")
            self.assertEqual(store.health()["conversation_count"], 1)
            self.assertGreaterEqual(store.health()["audit_event_count"], 1)

    def test_chat_branch_search_export_import_and_replay(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = ChatService(store, ModelGateway())

            result = service.send("Track the workspace roadmap and branch this chat.", project="core")
            conversation_id = result["conversation"]["id"]
            messages = store.list_chat_messages(conversation_id)

            branch = service.branch(
                conversation_id,
                at_message_id=messages[0]["id"],
                title="Workspace branch",
            )
            self.assertEqual(len(branch["messages"]), 1)
            self.assertEqual(branch["conversation"]["title"], "Workspace branch")

            search = service.search("workspace roadmap", project="core")
            self.assertGreaterEqual(len(search), 1)
            self.assertEqual(search[0]["project"], "core")

            exported = service.export_conversation(conversation_id)
            self.assertTrue(exported["path"].endswith(".json"))
            imported = service.import_conversation(exported, project="core")
            self.assertEqual(len(imported["messages"]), 2)

            replay = service.replay(conversation_id)
            self.assertEqual(replay["replay"]["message_count"], 2)
            self.assertEqual(replay["replay"]["timeline"][0]["role"], "user")

    def test_workflow_runs_local_steps(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            engine = WorkflowEngine(store)
            workflow = engine.create(
                name="Test workflow",
                project="core",
                steps=[
                    {"action": "add_event", "title": "Workflow memory", "content": "Recorded locally."},
                    {"action": "run_agent", "agent": "planning", "goal": "Plan tests"},
                    {"action": "search_context", "query": "Workflow"},
                ],
            )

            run = engine.run(workflow["id"])

            self.assertEqual(run["status"], "completed")
            self.assertEqual(len(run["result"]["steps"]), 3)
            self.assertEqual(store.health()["workflow_count"], 1)
            self.assertEqual(store.health()["agent_run_count"], 1)


if __name__ == "__main__":
    unittest.main()
