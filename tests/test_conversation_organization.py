from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.chat import ChatService
from openagent.conversation_organization import ConversationOrganization
from openagent.model_gateway import ModelGateway
from openagent.models import ChatMessage
from openagent.routes.conversation_routes import ConversationRouteDependencies, ROUTES, handle_delete
from openagent.storage import OpenAgentStore


class ConversationOrganizationTests(unittest.TestCase):
    def test_organization_compaction_semantic_sort_and_branch_graph(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            chat = ChatService(store, ModelGateway())
            organization = ConversationOrganization(store)
            alpha = chat.send("Investigate vector retrieval quality", project="core")
            beta = chat.send("Prepare the calendar brief", project="core")
            alpha_id = str(alpha["conversation_id"])
            beta_id = str(beta["conversation_id"])

            organization.update(alpha_id, patch={"pinned": True, "folder": "Research", "compaction_mode": "manual"})
            folder = organization.create_folder("Research", project="core")
            self.assertEqual(folder["name"], "Research")
            ranked = organization.list(project="core", sort="semantic", query="vector retrieval")
            self.assertEqual(ranked[0]["id"], alpha_id)
            self.assertTrue(ranked[0]["pinned"])

            compaction = organization.compact(alpha_id, preserve=2)
            self.assertEqual(compaction["preserved_message_count"], 2)
            branch = chat.branch(alpha_id, title="Retrieval branch")
            graph = organization.branch_graph(alpha_id)
            self.assertEqual(graph["edges"][0]["target"], branch["conversation"]["id"])

            result = organization.bulk([beta_id], action="archive")
            self.assertEqual(result["count"], 1)
            self.assertTrue(organization.get(beta_id)["archived"])
            self.assertNotIn(beta_id, {item["id"] for item in organization.list(project="core")})
            archived = organization.list(project="core", include_archived=True, archived_only=True)
            self.assertEqual([beta_id], [item["id"] for item in archived])

    def test_route_contract_covers_full_organization_lifecycle(self) -> None:
        paths = {path for _, path in ROUTES}
        self.assertIn("/api/conversations/bulk", paths)
        self.assertIn("/api/conversations/{id}/branches", paths)
        self.assertIn("/api/conversations/{id}/compact", paths)
        self.assertIn("/api/conversations/{id}", paths)

    def test_delete_route_removes_conversation_messages_and_returns_not_found_for_missing_id(self) -> None:
        with TemporaryDirectory() as temp:
            store = OpenAgentStore(temp)
            organization = ConversationOrganization(store)
            conversation = store.create_conversation("Disposable chat", project="default", conversation_id="chat-to-delete")
            store.add_chat_message(ChatMessage(conversation_id=conversation["id"], role="user", content="Synthetic test message"))
            deps = ConversationRouteDependencies(
                organization=organization,
                first=lambda params, key: (params.get(key) or [""])[0],
                parse_bool=lambda _value, *, default=False: default,
                parse_int=lambda _value, *, default=100: default,
                query_owner=lambda _params: "local",
                payload_owner=lambda _payload: "local",
                read_json=lambda: {},
            )

            deleted = handle_delete("/api/conversations/chat-to-delete", deps)
            self.assertIsNotNone(deleted)
            self.assertEqual(deleted.status, 200)
            self.assertEqual(deleted.payload["changed"], [{"id": "chat-to-delete", "deleted": True}])
            self.assertIsNone(store.get_conversation("chat-to-delete"))
            self.assertEqual(store.list_chat_messages("chat-to-delete"), [])
            self.assertEqual(organization.list(project="default"), [])

            missing = handle_delete("/api/conversations/not-found", deps)
            self.assertIsNotNone(missing)
            self.assertEqual(missing.status, 404)


if __name__ == "__main__":
    unittest.main()
