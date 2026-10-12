from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.chat import ChatService
from openagent.chat_artifacts import ChatArtifactService
from openagent.model_gateway import ModelGateway
from openagent.routes import StreamRouteResult
from openagent.routes.chat_routes import ChatRouteDependencies, handle_get, handle_post
from openagent.storage import OpenAgentStore


class ChatRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.store = OpenAgentStore(self.temp.name)
        self.chat = ChatService(self.store, ModelGateway())

    def tearDown(self) -> None:
        self.temp.cleanup()

    def dependencies(self, payload: dict[str, object] | None = None) -> ChatRouteDependencies:
        return ChatRouteDependencies(
            store=self.store,
            chat=self.chat,
            artifacts=ChatArtifactService(self.store),
            first=lambda params, name: params.get(name, [None])[0],
            parse_bool=lambda value, default=False: default if value is None else str(value).lower() in {"1", "true"},
            parse_int=lambda value, default=0: default if value is None else int(value),
            read_json=lambda: dict(payload or {}),
        )

    def test_get_validation_and_conversation_listing(self) -> None:
        missing = handle_get("/api/messages", "", self.dependencies())
        self.assertEqual(missing.status, 400)
        listing = handle_get("/api/conversations", "project=core&limit=10", self.dependencies())
        self.assertEqual(listing.payload, {"conversations": []})

    def test_post_chat_and_stream_use_the_same_domain_handler(self) -> None:
        payload = {"message": "Summarize local boundaries", "project": "core", "provider": "local"}
        result = handle_post("/api/chat", self.dependencies(payload))
        self.assertEqual(result.status, 201)
        self.assertEqual(result.payload["provider"], "local")
        stream = handle_post("/api/chat/stream", self.dependencies(payload))
        self.assertIsInstance(stream, StreamRouteResult)
        self.assertTrue(any(event.get("type") == "message" for event in stream.events))

    def test_unowned_path_is_not_handled(self) -> None:
        self.assertIsNone(handle_get("/api/health", "", self.dependencies()))
        self.assertIsNone(handle_post("/api/workflows", self.dependencies({})))


if __name__ == "__main__":
    unittest.main()
