from __future__ import annotations

from io import BytesIO
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from openagent.science.engineering.optimizer import DesignProblem, EngineeringOptimizer, beam_problem
from openagent.science.materials.active_learning import ActiveLearner
from openagent.science.materials.dataset import analyse_dataset_upload
from openagent.science.materials.predictor import MaterialPropertyPredictor
from openagent.science.multimodal.fusion import MultimodalScientificContext
from openagent.science.multimodal.image import analyse_image_bytes, observe_with_gateway
from openagent.science.simulations.fenics_heat import FenicsHeatAdapter
from openagent.science.tools import FenicsTool


def _materials_records(count: int = 80) -> list[dict[str, float]]:
    records = []
    for index in range(count):
        density = 2.0 + (index % 20) * 0.25
        hardness = 70.0 + index * 2.4
        temperature = 250.0 + (index % 7) * 12.0
        strength = 42.0 * density + 1.65 * hardness - 0.08 * temperature + ((index % 5) - 2) * 1.2
        records.append({"density": density, "hardness": hardness, "temperature": temperature, "strength": strength})
    return records


class _GatewayResult:
    provider = "local-vlm"
    model = "vision-test"
    content = '{"observations":[{"statement":"A bright region occupies the upper-left area.","region":"top_left","confidence":0.82,"evidence":"pixel contrast"}],"limitations":["No scale marker was supplied."]}'


class _StructuredGateway:
    def complete_vision(self, messages, image, **kwargs):
        self.messages = messages
        self.image = image
        self.kwargs = kwargs
        return _GatewayResult()


class _TextOnlyGateway:
    def complete(self, messages, **kwargs):
        result = _GatewayResult()
        result.content = "This looks like a sample."
        return result


class ScienceDomainDepthTests(unittest.TestCase):
    def test_materials_uncertainty_uses_standardized_distribution_distance(self) -> None:
        records = _materials_records()
        predictor = MaterialPropertyPredictor(ensemble_size=12, seed=9)
        fit = predictor.fit(records, target="strength")
        self.assertEqual(fit["model"], "standardized-ridge-bootstrap")
        self.assertLess(predictor.evaluate(records)["rmse"], 3.0)
        familiar = {key: records[30][key] for key in ("density", "hardness", "temperature")}
        outlier = {"density": 30.0, "hardness": 900.0, "temperature": 1200.0}
        familiar_distribution = predictor.predict_distribution(familiar)
        outlier_distribution = predictor.predict_distribution(outlier)
        self.assertGreater(outlier_distribution["nearest_training_distance"], familiar_distribution["nearest_training_distance"])
        self.assertGreater(outlier_distribution["uncertainty"], familiar_distribution["uncertainty"])
        selected = ActiveLearner().select(predictor, [familiar, outlier])[0]
        self.assertEqual(selected["features"], outlier)
        self.assertIn("ensemble_std", selected)

    def test_materials_upload_generates_held_out_metrics_and_artifacts(self) -> None:
        fieldnames = ["density", "hardness", "temperature", "strength"]
        string_buffer = __import__("io").StringIO()
        writer = csv.DictWriter(string_buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_materials_records())
        data = string_buffer.getvalue().encode("utf-8")
        with TemporaryDirectory() as tmp:
            report = analyse_dataset_upload(data, name="alloys.csv", target="strength", artifact_dir=tmp, seed=5)
            self.assertEqual(report["status"], "completed")
            self.assertGreater(report["training_rows"], report["test_rows"])
            self.assertLess(report["test_metrics"]["rmse"], 5.0)
            self.assertGreater(report["test_metrics"]["r2"], 0.98)
            self.assertEqual(len(report["artifacts"]), 2)
            self.assertTrue(all(Path(item["path"]).is_file() for item in report["artifacts"]))
            self.assertTrue(all(len(item["sha256"]) == 64 for item in report["artifacts"]))

    def test_local_image_analysis_and_vlm_status_are_truthful(self) -> None:
        image = Image.new("RGB", (12, 12), color=(10, 20, 30))
        for x in range(6):
            for y in range(6):
                image.putpixel((x, y), (240, 240, 240))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        data = buffer.getvalue()
        local = analyse_image_bytes(data, name="thermal.png")
        self.assertEqual(local["status"], "completed")
        self.assertEqual(local["semantic_observations"], [])
        self.assertIn("No object", local["interpretation"])
        unavailable = observe_with_gateway(data, gateway=None, model="")
        self.assertEqual(unavailable["status"], "unavailable")
        rejected = observe_with_gateway(data, gateway=_TextOnlyGateway(), model="vision-test")
        self.assertEqual(rejected["status"], "unavailable")
        observed = observe_with_gateway(data, gateway=_StructuredGateway(), model="vision-test")
        self.assertEqual(observed["status"], "completed")
        self.assertEqual(len(observed["observations"]), 1)
        context = MultimodalScientificContext()
        item = context.add_image_bytes("thermal.png", data)
        self.assertEqual(item["vlm"]["status"], "not-requested")

    def test_engineering_methods_and_benchmark_report_actual_backend(self) -> None:
        optimizer = EngineeringOptimizer(iterations=80, seed=7)
        random_result = optimizer.solve(beam_problem(), method="random")
        self.assertEqual(random_result["method"], "random")
        self.assertEqual(random_result["evaluations"], 80)
        benchmark = optimizer.benchmark(beam_problem(), methods=("random", "differential_evolution", "minimize"))
        self.assertEqual(len(benchmark["methods"]), 3)
        self.assertTrue(all(item["method"] in {"random", "differential_evolution", "minimize"} for item in benchmark["methods"]))
        self.assertGreaterEqual(float(benchmark["best_objective"]), 0.0)

    def test_bayesian_optimizer_is_deterministic_and_reports_acquisition_provenance(self) -> None:
        problem = DesignProblem(
            variables={"x": (0.01, 1.0), "y": (0.01, 1.0)},
            objective="minimum_loss",
            evaluator=lambda design: {
                "loss": (design["x"] - 0.23) ** 2 + (design["y"] - 0.77) ** 2,
            },
        )
        optimizer = EngineeringOptimizer(iterations=24, seed=31)
        first = optimizer.solve(problem, method="bayesian")
        second = optimizer.solve(problem, method="bayesian")

        self.assertEqual(first["design"], second["design"])
        self.assertEqual(first["objective"], second["objective"])
        self.assertEqual(first["backend"], "native-rbf-gaussian-process")
        self.assertEqual(first["acquisition"], "expected_improvement")
        self.assertEqual(first["evaluations"], 24)
        self.assertEqual(first["initial_evaluations"] + first["adaptive_evaluations"], 24)
        self.assertFalse(first["fallback_used"])
        self.assertLess(float(first["objective"]), 0.01)
        self.assertLess(float(first["objective"]), float(first["initial_best_objective"]))
        self.assertGreater(float(first["acquisition_improvement"]), 0.0)
        promising = optimizer._expected_improvement(mean=-0.5, standard_deviation=0.2, best=-0.2)
        poor = optimizer._expected_improvement(mean=0.5, standard_deviation=0.05, best=-0.2)
        self.assertGreater(promising, poor)

    def test_fenics_adapter_runs_real_backend_or_labeled_fallback(self) -> None:
        adapter = FenicsHeatAdapter(points=16, steps=12)
        result = adapter.run()
        self.assertEqual(result["status"], "completed")
        self.assertLessEqual(result["energy_final"], result["energy_initial"] + 1e-9)
        self.assertEqual(result["is_fenics_result"], result["backend"] == "fenics")
        if result["backend"] != "fenics":
            self.assertEqual(result["fidelity"], "reference-fallback")
            self.assertIn("unavailable", result["warning"])
        wrapped = FenicsTool().run(inputs={"points": 16, "steps": 12})
        self.assertEqual(wrapped["backend"], result["backend"])


if __name__ == "__main__":
    unittest.main()
