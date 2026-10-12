from __future__ import annotations

import unittest

from openagent.routes.handlers import handler_specs


class RouteHandlerRegistryTests(unittest.TestCase):
    def test_physical_domain_handlers_are_declared(self) -> None:
        specs = handler_specs()
        domains = {spec["domain"] for spec in specs}
        self.assertEqual(
            {
                "chat",
                "agents",
                "workspace",
                "assistant",
                "twin",
                "semantic",
                "vault",
                "voice",
                "knowledge",
                "finetune",
                "scraper",
                "drafts",
                "media",
                "calendar",
                "communication",
                "local_ai",
                "security",
                "developer",
                "documents",
                "workflows",
            },
            domains,
        )
        for spec in specs:
            self.assertTrue(spec["routes_module"].startswith("openagent.routes."))
            self.assertGreater(len(spec["owned_prefixes"]), 0)


if __name__ == "__main__":
    unittest.main()
