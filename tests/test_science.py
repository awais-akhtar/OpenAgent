from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from openagent.agents import AgentRegistry
from openagent.science.computing.heat_equation import HeatEquation1D
from openagent.science.computing.surrogate import ScientificSurrogate
from openagent.science.computing.neural_operator import DeepONet, FNO, PINN
from openagent.science.engineering.optimizer import EngineeringOptimizer, beam_problem
from openagent.science.experiment import ExperimentPlanner
from openagent.science.execution import ScientificCodeRunner
from openagent.science.memory import ScientificMemory
from openagent.science.materials.active_learning import ActiveLearner
from openagent.science.materials.predictor import MaterialPropertyPredictor
from openagent.science.multimodal.fusion import MultimodalScientificContext
from openagent.science.verification.scientific_verifier import ScientificVerifier
from openagent.science.verification.equations import check_symbolic_equations
from openagent.science.planner import ScientificPlanner
from openagent.science.tools import discover_scientific_tools
from openagent.science.papers import PaperSearch
from openagent.science.safety import check_citations, check_consistency, detect_unsupported_claims
from openagent.science.service import ScientificService
from openagent.storage import OpenAgentStore


class ScientificWorkflowTests(unittest.TestCase):
    def test_heat_solver_is_stable_and_surrogate_is_comparable(self) -> None:
        solver = HeatEquation1D(points=12, steps=10)
        initial = solver.initial_condition()
        final = solver.solve(initial)
        self.assertLessEqual(solver.energy(final), solver.energy(initial))
        surrogate = ScientificSurrogate(modes=3)
        surrogate.fit(solver.generate_dataset(samples=20))
        comparison = surrogate.compare_with_solver(solver, initial)
        self.assertIn("relative_rmse", comparison)

    def test_beam_optimizer_returns_a_reviewable_design(self) -> None:
        result = EngineeringOptimizer(iterations=120).solve(beam_problem())
        self.assertIn(result["status"], {"feasible", "infeasible"})
        self.assertEqual(len(result["constraints"]), 2)
        self.assertIn("mass", result["metrics"])

    def test_heat_exchanger_workflow_executes_and_profiles_python(self) -> None:
        with TemporaryDirectory() as tmp:
            run = AgentRegistry(OpenAgentStore(tmp)).run("scientific", "optimize a heat exchanger", project="science")
            science = run["metadata"]["science"]
            self.assertEqual(science["workflow"], "heat_exchanger_optimization")
            self.assertEqual(science["result"]["status"], "completed")
            self.assertGreater(science["result"]["execution"]["elapsed_ms"], 0)
            self.assertIn("peak_memory_bytes", science["result"]["execution"])

    def test_problem_planner_exposes_five_reviewable_stages(self) -> None:
        plan = ScientificPlanner().create_plan("Design a lighter beam")
        self.assertEqual(plan.workflow, "beam_optimization")
        self.assertEqual(len(plan.steps), 5)
        self.assertEqual(plan.steps[0], "gather parameters")
        self.assertEqual(plan.steps[-1], "produce the profiled report")

    def test_symbolic_stress_equation_is_checked_with_units(self) -> None:
        finding = check_symbolic_equations([
            {"expression": "stress = force/area", "symbols": {"stress": 125e6, "force": 250000.0, "area": 0.002}}
        ])
        self.assertTrue(finding.passed)
        verification = ScientificVerifier().verify(
            {"stress": 125e6},
            units={"stress": "Pa"},
            equations=[{"expression": "stress = force/area", "symbols": {"stress": 125e6, "force": 250000.0, "area": 0.002}}],
        )
        self.assertTrue(verification.passed)
        self.assertFalse(ScientificVerifier().verify({"unstable": float("nan")}).passed)

    def test_bridge_executes_profiles_and_keeps_limit_warning(self) -> None:
        with TemporaryDirectory() as tmp:
            run = AgentRegistry(OpenAgentStore(tmp)).run("scientific", "Optimize bridge", project="science")
            science = run["metadata"]["science"]
            self.assertEqual(science["workflow"], "bridge_optimization")
            self.assertEqual(science["result"]["status"], "completed")
            self.assertTrue(science["verification"]["passed"])
            self.assertTrue(science["result"]["limitations"])
            self.assertGreater(science["result"]["execution"]["elapsed_ms"], 0)

    def test_agent_reads_project_scientific_memory(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            service = ScientificService(store)
            service.remember(project="science", category="property", name="density", value=2700, unit="kg/m3")
            service.remember(project="science", category="property", name="Young modulus", value=70, unit="GPa")
            run = AgentRegistry(store).run("scientific", "Design a lighter beam", project="science")
            used = run["metadata"]["science"]["result"]["memory_used"]
            self.assertEqual({item["name"] for item in used}, {"density", "Young modulus"})

    def test_materials_predictor_selects_uncertain_candidate(self) -> None:
        predictor = MaterialPropertyPredictor()
        predictor.fit(
            [
                {"density": 7.8, "hardness": 220.0, "strength": 620.0},
                {"density": 7.4, "hardness": 180.0, "strength": 510.0},
                {"density": 8.1, "hardness": 260.0, "strength": 710.0},
            ],
            target="strength",
        )
        candidates = [{"density": 7.2, "hardness": 170.0}, {"density": 8.5, "hardness": 320.0}]
        selected = ActiveLearner().select(predictor, candidates)
        self.assertEqual(len(selected), 1)
        self.assertIn("uncertainty", selected[0])

    def test_multimodal_context_and_verifier_are_local_and_typed(self) -> None:
        context = MultimodalScientificContext()
        context.add_text("lab note", "The motor warmed during the test.")
        context.add_sensor_csv("motor.csv", "time,temp\n0,21\n1,44\n")
        context.add_image("thermal", [1, 2, 3, 4], width=2, height=2)
        context.add_simulation("run.json", {"peak": 44.0, "steps": [1, 2]})
        summary = context.summary()
        self.assertEqual(summary["count"], 4)
        verification = ScientificVerifier().verify(summary)
        self.assertTrue(verification.passed)

    def test_scientific_agent_is_registered_and_persists_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            store = OpenAgentStore(tmp)
            run = AgentRegistry(store).run("scientific", "optimise a beam", project="science")
            self.assertEqual(run["agent"], "scientific")
            self.assertIn("Verification:", run["response"])
            self.assertGreaterEqual(store.health()["audit_event_count"], 1)

    def test_scientific_supporting_interfaces(self) -> None:
        memory = ScientificMemory()
        memory.remember("property", "density", 7.8, unit="g/cm3")
        self.assertEqual(memory.search("density")[0].unit, "g/cm3")
        experiments = ExperimentPlanner().propose("lighter alloy", [{"density": 7.1, "priority": 2}, {"density": 6.9, "priority": 1}])
        self.assertEqual(experiments[0].variables["density"], 7.1)
        execution = ScientificCodeRunner().run("print('ok')")
        self.assertEqual(execution.status, "completed")
        repaired = ScientificCodeRunner().run("print(math.sqrt(4))", retries=1)
        self.assertEqual(repaired.status, "completed")
        self.assertEqual(repaired.attempts, 2)
        self.assertGreaterEqual(repaired.peak_memory_bytes, 0)
        self.assertEqual(repaired.repairs, 1)
        self.assertEqual([item["stage"] for item in repaired.lifecycle], ["code_prepared", "execute", "traceback", "repair", "execute", "profile"])
        self.assertTrue(repaired.attempt_history[0]["repair_applied"])
        for operator in (FNO(), DeepONet(), PINN()):
            operator.fit([[0.0]], [[1.0]])
            self.assertEqual(operator.predict([0.0]), [1.0])
        self.assertTrue(any(item.name == "NumPy" for item in discover_scientific_tools()))
        papers = PaperSearch([{"title": "Heat transfer study", "abstract": "A finite difference method."}])
        self.assertEqual(papers.search("heat")[0].title, "Heat transfer study")
        self.assertTrue(check_citations("See [paper-1]", [{"id": "paper-1", "title": "Paper 1"}])["passed"])
        self.assertTrue(check_consistency({"a": 1.0, "b": 2.0})['passed'])
        self.assertTrue(
            detect_unsupported_claims(
                [{"text": "Heat result", "evidence_ids": ["result:heat"]}],
                {"result:heat": 1.0},
            )["passed"]
        )

    def test_scientific_service_persists_memory_and_materials(self) -> None:
        with TemporaryDirectory() as tmp:
            service = ScientificService(OpenAgentStore(tmp))
            service.remember(project="science", category="property", name="Young modulus", value=210, unit="GPa")
            self.assertEqual(len(service.list_memory(project="science")), 1)
            records = [
                {"density": 7.8, "hardness": 220.0, "strength": 620.0},
                {"density": 7.4, "hardness": 180.0, "strength": 510.0},
                {"density": 8.1, "hardness": 260.0, "strength": 710.0},
            ]
            saved = service.save_material_dataset(project="science", name="alloys", target="strength", records=records)
            self.assertEqual(saved["summary"]["rows"], 3)
            analysis = service.analyse_materials(project="science", records=records, target="strength")
            self.assertEqual(len(analysis["chart"]), 3)
            self.assertIn("verification", analysis)


if __name__ == "__main__":
    unittest.main()
