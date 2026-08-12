from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.chat import ChatService
from openagent.conversation_organization import ConversationOrganization
from openagent.model_gateway import ModelGateway
from openagent.routes.conversation_routes import ROUTES
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


if __name__ == "__main__":
    unittest.main()
