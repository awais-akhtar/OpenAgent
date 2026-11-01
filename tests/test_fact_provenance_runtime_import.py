from __future__ import annotations

from tempfile import TemporaryDirectory
from unittest import TestCase

from openagent.fact_provenance import FactProvenanceService
from openagent.routes.runtime import build_http_server
from openagent.storage import OpenAgentStore


class FactProvenanceRuntimeImportTests(TestCase):
    def test_runtime_import_resolves_fact_provenance_dependency(self):
        self.assertTrue(callable(build_http_server))

    def test_fact_source_create_retract_restore_persists_after_migration(self):
        with TemporaryDirectory() as directory:
            store = OpenAgentStore(directory)
            service = FactProvenanceService(store)
            fact = service.record_fact(
                fact="A verified local preference",
                project="default",
                sources=[{"source_type": "test", "source_reference": "fixture-1"}],
            )
            self.assertEqual(fact["fact"], "A verified local preference")
            self.assertEqual(fact["project"], "default")
            self.assertEqual(fact["sources"][0]["source_reference"], "fixture-1")
            retracted = service.retract_fact(
                fact["id"], "default", retracted_by="test", reason="verification"
            )
            self.assertTrue(retracted["retracted_at"])
            restored = service.restore_fact(fact["id"], "default")
            self.assertEqual(restored["retracted_at"], "")
