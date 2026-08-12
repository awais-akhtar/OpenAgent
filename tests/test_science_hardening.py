from __future__ import annotations

import unittest

from openagent.model_gateway import ModelResult
from openagent.science.execution import ScientificCodeRunner, execution_policy_violations
from openagent.science.planner import ScientificPlanner
from openagent.science.tool_router import ScientificToolRouter
from openagent.science.verification.claims import check_claim_evidence
from openagent.science.verification.units import check_units


class _PlanGateway:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = 0

    def complete(self, messages, **kwargs):
        self.calls += 1
        return ModelResult("local", "test", self.content, {})


class ScientificHardeningTests(unittest.TestCase):
    def test_valid_local_model_plan_is_used(self) -> None:
        gateway = _PlanGateway('{"workflow":"beam_optimization","steps":["collect inputs","solve and verify"],"assumptions":[]}')
        plan = ScientificPlanner(gateway).create_plan("Design a beam")
        self.assertEqual(plan.workflow, "beam_optimization")
        self.assertEqual(plan.source, "model")
        self.assertEqual(gateway.calls, 1)

    def test_invalid_model_plan_uses_deterministic_fallback(self) -> None:
        plan = ScientificPlanner(_PlanGateway('{"workflow":"shell","steps":["run anything","trust it"]}')).create_plan("Design a lighter beam")
        self.assertEqual(plan.workflow, "beam_optimization")
        self.assertEqual(plan.source, "deterministic")

    def test_execution_policy_blocks_dangerous_source_before_launch(self) -> None:
        self.assertIn("blocked import 'os'", execution_policy_violations("import os\nos.remove('important')"))
        self.assertIn("blocked import 'requests'", execution_policy_violations("import requests\nrequests.get('https://example.com')"))
        self.assertTrue(execution_policy_violations("print((1).__class__.__mro__)"))
        record = ScientificCodeRunner().run("import subprocess\nsubprocess.run(['echo', 'unsafe'])", retries=1)
        self.assertEqual(record.status, "failed")
        self.assertEqual(record.returncode, 126)
        self.assertEqual(record.attempts, 1)
        self.assertEqual(record.lifecycle[1]["stage"], "policy")
        self.assertEqual(record.lifecycle[1]["status"], "rejected")

    def test_repaired_source_is_checked_by_policy_again(self) -> None:
        record = ScientificCodeRunner().run(
            "print(missing)",
            retries=1,
            repair=lambda _code, _traceback: "import socket\nprint('blocked')",
        )
        self.assertEqual(record.returncode, 126)
        self.assertEqual(record.repairs, 1)
        self.assertEqual(record.lifecycle[-2]["stage"], "policy")

    def test_execution_memory_limit_is_bounded(self) -> None:
        with self.assertRaisesRegex(ValueError, "memory_limit_mb"):
            ScientificCodeRunner().run("print('ok')", memory_limit_mb=16)

    def test_claims_require_resolvable_evidence(self) -> None:
        supported = check_claim_evidence([{"id": "c1", "text": "Energy decayed.", "evidence_ids": ["run:energy"]}], {"run:energy": {"value": 1.0}})
        unsupported = check_claim_evidence([{"id": "c2", "text": "A broad claim", "evidence_ids": ["missing"]}], {})
        self.assertTrue(supported.passed)
        self.assertFalse(unsupported.passed)

    def test_unit_check_records_dimensional_backend(self) -> None:
        finding = check_units({"stress": {"unit": "Pa", "expected": "N/m2"}, "length": "m"})
        self.assertTrue(finding.passed)
        self.assertIn(finding.evidence["engine"], {"pint", "known-unit-fallback"})

    def test_heat_workflow_enforces_energy_decay_and_provenance(self) -> None:
        plan = ScientificPlanner().create_plan("simulate the heat equation")
        result, verification = ScientificToolRouter().execute(plan)
        checks = {item.check: item for item in verification.findings}
        self.assertTrue(result["constraints"][0]["passed"])
        self.assertLessEqual(result["energy_final"], result["energy_initial"])
        self.assertTrue(checks["design_constraints"].passed)
        self.assertTrue(checks["claim_evidence_provenance"].passed)


if __name__ == "__main__":
    unittest.main()
