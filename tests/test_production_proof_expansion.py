from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.production_proof import run_production_proof


class ProductionProofExpansionTests(unittest.TestCase):
    def test_release_proof_covers_operational_failure_categories(self) -> None:
        with TemporaryDirectory() as temp:
            result = run_production_proof(temp, queue_jobs=20)
        self.assertTrue(result["ok"], result)
        names = {item["name"] for item in result["checks"]}
        self.assertTrue({"migrations", "crash-recovery", "queue-soak", "multi-user-isolation",
                         "provider-certification", "accessibility-contract", "parser-fuzz",
                         "installer-upgrade-uninstall", "http-e2e", "browser-e2e-contract"}.issubset(names))


if __name__ == "__main__":
    unittest.main()
