from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.models import ContextEvent
from openagent.storage import OpenAgentStore


class StorageTests(unittest.TestCase):
    def test_add_list_and_search_events(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            store.add_event(
                ContextEvent(
                    title="OpenAgent vision",
                    content="Build a self-hosted AI operating system.",
                    project="core",
                )
            )

            events = store.list_events(project="core")
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["title"], "OpenAgent vision")

            matches = store.search_events("self-hosted", project="core")
            self.assertEqual(len(matches), 1)

            missing = store.search_events("calendar", project="core")
            self.assertEqual(missing, [])

    def test_health_counts_records(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            store.add_event(ContextEvent(title="Memory", content="Local only."))
            health = store.health()
            self.assertEqual(health["status"], "ok")
            self.assertEqual(health["event_count"], 1)


if __name__ == "__main__":
    unittest.main()

