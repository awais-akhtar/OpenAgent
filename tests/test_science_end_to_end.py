from __future__ import annotations

import base64
from io import BytesIO
import json
from tempfile import TemporaryDirectory
import unittest

from openagent.agents import AgentRegistry
from openagent.routes.science_routes import ScienceRouteDependencies, handle_post
from openagent.science.agent import ScientificAgent
from openagent.science.citations import ScientificCitationResolver
from openagent.science.service import ScientificService
from openagent.storage import OpenAgentStore


class _Context:
    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store

    def search(self, *_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return []


class _VisionResponse:
    content = '{"observations":[{"statement":"The image is uniformly dark blue.","region":"full frame","confidence":0.9,"evidence":"uniform RGB pixels"}],"limitations":[]}'
    provider = "local-vision"
    model = "test-vlm"


class _VisionGateway:
    def complete_vision(self, *_args: object, **_kwargs: object) -> _VisionResponse:
        return _VisionResponse()


class ScienceEndToEndTests(unittest.TestCase):
    def _dependencies(
        self,
        payload: dict[str, object],
        store: OpenAgentStore,
        *,
        service: ScientificService | None = None,
        gateway: object = None,
    ) -> ScienceRouteDependencies:
        agent = ScientificAgent(_Context(store))
        return ScienceRouteDependencies(
            agent=agent,
            run_agent=lambda goal, project="default": AgentRegistry(store).run("scientific", goal, project=project),
            read_json=lambda: payload,
            service=service or ScientificService(store),
            gateway=gateway,
        )

    def test_deeponet_route_exposes_real_training_and_held_out_evaluation(self) -> None:
        inputs = [[float(index + offset) / 10.0 for index in range(6)] for offset in range(8)]
        outputs = [[value * 0.7 + 0.1 for value in row] for row in inputs]
        evaluation_inputs = [[float(index + 9) / 10.0 for index in range(6)]]
        evaluation_outputs = [[value * 0.7 + 0.1 for value in evaluation_inputs[0]]]
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/operators/deeponet", self._dependencies({
                "project": "lab",
                "backend": "numpy",
                "inputs": inputs,
                "outputs": outputs,
                "evaluation_inputs": evaluation_inputs,
                "evaluation_outputs": evaluation_outputs,
            }, store))
        self.assertIsNotNone(result)
        self.assertEqual(result.status, 201)
        operator = result.payload["operator"]
        self.assertTrue(operator["training"]["is_full_implementation"])
        self.assertEqual(operator["evaluation_scope"], "held-out")
        self.assertIn("relative_rmse", operator["evaluation"])
        self.assertEqual(operator["progress"][-1]["stage"], "report")
        json.dumps(result.payload)

    def test_run_route_preserves_agent_record_and_surfaces_planner_provenance(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/run", self._dependencies({"project": "lab", "goal": "Design a lighter beam"}, store))
        self.assertIsNotNone(result)
        self.assertEqual(result.payload["run"]["agent"], "scientific")
        self.assertIn(result.payload["provenance"]["planner"], {"model", "deterministic"})
        self.assertEqual(result.payload["provenance"]["workflow"], "beam_optimization")
        self.assertIsInstance(result.payload["verification_evidence"], list)

    def test_model_planner_provenance_is_not_flattened_away(self) -> None:
        run = {
            "agent": "scientific",
            "context_used": ["paper"],
            "metadata": {
                "external_network": False,
                "science": {
                    "workflow": "heat_equation",
                    "plan": {"workflow": "heat_equation", "source": "model"},
                    "verification": {"findings": [{"evidence": {"energy_decay": True}}]},
                },
            },
        }
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            deps = ScienceRouteDependencies(
                agent=ScientificAgent(_Context(store)),
                run_agent=lambda *_args, **_kwargs: run,
                read_json=lambda: {"project": "lab", "goal": "Solve heat flow"},
                service=ScientificService(store),
            )
            result = handle_post("/api/science/run", deps)
        self.assertIsNotNone(result)
        self.assertEqual(result.payload["provenance"]["planner"], "model")
        self.assertEqual(result.payload["verification_evidence"], [{"energy_decay": True}])

    def test_materials_route_exposes_progress_metrics_report_and_uncertainty(self) -> None:
        records = [
            {"density": 6.8 + index * 0.1, "hardness": 150.0 + index * 12.0, "strength": 430.0 + index * 31.0}
            for index in range(12)
        ]
        candidates = [{"density": 7.05, "hardness": 178.0}, {"density": 8.8, "hardness": 360.0}]
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/materials/analyse", self._dependencies({
                "project": "lab", "target": "strength", "records": records, "candidates": candidates,
            }, store))
        self.assertIsNotNone(result)
        analysis = result.payload["analysis"]
        self.assertEqual([step["stage"] for step in analysis["progress"]], ["validate", "fit", "evaluate", "rank", "verify"])
        self.assertGreater(analysis["report"]["test_rows"], 0)
        self.assertIn("rmse", analysis["report"]["test_metrics"])
        self.assertIn("uncertainty", analysis["ranked_candidates"][0])

    def test_multimodal_route_returns_typed_observations(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        buffer = BytesIO()
        Image.new("RGB", (2, 2), (20, 40, 60)).save(buffer, format="PNG")
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/multimodal", self._dependencies({
                "project": "lab",
                "goal": "Review the supplied evidence",
                "images": [{"name": "sample.png", "base64": base64.b64encode(buffer.getvalue()).decode("ascii"), "vlm_model": "test-vlm"}],
                "sensor_csv": [{"name": "sensor.csv", "content": "time,temp\n0,20\n1,25\n"}],
            }, store, gateway=_VisionGateway()))
        self.assertIsNotNone(result)
        self.assertEqual(result.payload["context"]["count"], 2)
        image = next(item for item in result.payload["context"]["items"] if item["kind"] == "image")
        self.assertEqual(image["local_analysis"]["status"], "completed")
        self.assertEqual(result.payload["observations"][0]["observations"][0]["statement"], "The image is uniformly dark blue.")

    def test_optimizer_route_exposes_available_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/engineering/optimize", self._dependencies({
                "project": "lab", "method": "random", "iterations": 40,
            }, store))
        self.assertIsNotNone(result)
        optimization = result.payload["optimization"]
        self.assertEqual(optimization["result"]["method"], "random")
        self.assertIn("differential_evolution", optimization["available_methods"])

    def test_verification_requires_structured_claim_evidence(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/verify", self._dependencies({
                "project": "lab",
                "result": {"stress": 125e6},
                "units": {"stress": "Pa"},
                "claims": [{"id": "claim:stress", "text": "Stress is 125 MPa.", "evidence_ids": ["calculation:stress"]}],
                "evidence": {"calculation:stress": {"value": 125e6, "unit": "Pa"}},
            }, store))
        self.assertIsNotNone(result)
        verification = result.payload["verification"]
        self.assertTrue(verification["passed"])
        checks = {item["check"] for item in verification["evidence"]}
        self.assertIn("claim_evidence_provenance", checks)
        self.assertEqual({item["check"] for item in verification["provenance_checks"]}, {"claim_evidence_provenance", "citation_provenance"})

    def test_external_citation_route_requires_approval_and_uses_default_resolver_contract(self) -> None:
        citation = {"id": "paper:1", "url": "https://example.test/paper"}
        calls: list[str] = []

        class Response:
            status = 200
            headers = {"Content-Type": "text/html", "Content-Length": "49"}

            def read(self, limit: int = -1) -> bytes:
                return b"<html><title>Test publication</title></html>"[:limit]

            def geturl(self) -> str:
                return "https://example.test/paper"

        def transport(request: object, timeout: float) -> Response:
            calls.append(request.full_url)  # type: ignore[attr-defined]
            self.assertLessEqual(timeout, 10.0)
            return Response()

        resolver = ScientificCitationResolver(
            transport=transport,
            address_resolver=lambda host, port, **kwargs: [(2, 1, 6, "", ("93.184.216.34", port))],
        )
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = ScientificService(store, citation_resolver=resolver)
            unapproved = handle_post("/api/science/verify", self._dependencies({
                "project": "lab", "result": {"value": 1.0}, "citations": [citation],
            }, store, service=service))
            self.assertEqual(calls, [])
            approved = handle_post("/api/science/verify", self._dependencies({
                "project": "lab",
                "result": {"value": 1.0},
                "citations": [citation],
                "external_citation_lookup_approved": True,
            }, store, service=service))
        unapproved_provenance = unapproved.payload["verification"]["safety"]["citation_provenance"]
        approved_provenance = approved.payload["verification"]["safety"]["citation_provenance"]
        self.assertEqual(unapproved_provenance["status"], "unverified")
        self.assertFalse(unapproved_provenance["external_lookup"])
        self.assertEqual(approved_provenance["status"], "verified")
        self.assertFalse(approved_provenance["claim_entailment_checked"])
        self.assertEqual(len(calls), 1)

    def test_citation_approval_must_be_a_boolean(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            result = handle_post("/api/science/verify", self._dependencies({
                "project": "lab",
                "result": {"value": 1.0},
                "citations": [{"id": "paper:1", "doi": "10.5555/example"}],
                "external_citation_lookup_approved": "yes",
            }, store))
        self.assertEqual(result.status, 400)
        self.assertIn("must be a boolean", result.payload["error"])


if __name__ == "__main__":
    unittest.main()
