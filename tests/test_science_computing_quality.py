from __future__ import annotations

import unittest

from openagent.science.computing.heat_equation import HeatEquation1D
from openagent.science.computing.neural_operator import DeepONet, FNO, PINN
from openagent.science.computing.surrogate import ScientificSurrogate


class ScientificComputingQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = HeatEquation1D(points=24, steps=30, alpha=0.12, dt=0.0005)
        training = cls.solver.generate_dataset(samples=96, seed=19)
        held_out = cls.solver.generate_dataset(samples=24, seed=97)
        cls.train_inputs = [row["input"] for row in training]
        cls.train_outputs = [row["target"] for row in training]
        cls.test_inputs = [row["input"] for row in held_out]
        cls.test_outputs = [row["target"] for row in held_out]

    def test_numpy_deeponet_generalises_on_held_out_heat_solutions(self) -> None:
        operator = DeepONet(backend="numpy", hidden=40, basis=24, seed=11)
        report = operator.fit(self.train_inputs, self.train_outputs)
        metrics = operator.evaluate(self.test_inputs, self.test_outputs)

        self.assertEqual(report["backend"], "numpy-random-feature-deeponet")
        self.assertTrue(report["is_full_implementation"])
        self.assertTrue(report["operator_identifiable"])
        self.assertLess(metrics["relative_rmse"], 0.08)
        self.assertGreater(metrics["r2"], 0.95)

    def test_deeponet_is_input_dependent_and_deterministic(self) -> None:
        first = DeepONet(backend="numpy", hidden=32, basis=20, seed=23)
        second = DeepONet(backend="numpy", hidden=32, basis=20, seed=23)
        first.fit(self.train_inputs, self.train_outputs)
        second.fit(self.train_inputs, self.train_outputs)
        prediction_a = first.predict(self.test_inputs[0])
        prediction_b = first.predict(self.test_inputs[1])
        repeated = second.predict(self.test_inputs[0])

        self.assertGreater(max(abs(a - b) for a, b in zip(prediction_a, prediction_b)), 1e-4)
        self.assertLess(max(abs(a - b) for a, b in zip(prediction_a, repeated)), 1e-12)

    def test_unimplemented_operator_families_report_baseline_status(self) -> None:
        for operator in (FNO(), PINN()):
            report = operator.fit([[0.0], [1.0]], [[0.0], [1.0]])
            self.assertFalse(report["is_full_implementation"])
            self.assertIn("baseline", report["implementation"])

    def test_fourier_surrogate_reports_quality_and_timing(self) -> None:
        dataset = [
            {"input": values, "target": targets}
            for values, targets in zip(self.train_inputs, self.train_outputs)
        ]
        surrogate = ScientificSurrogate(modes=10, regularization=1e-7)
        surrogate.fit(dataset)
        comparisons = [
            surrogate.compare_with_solver(self.solver, values)
            for values in self.test_inputs
        ]
        mean_relative_rmse = sum(item["relative_rmse"] for item in comparisons) / len(comparisons)

        self.assertLess(mean_relative_rmse, 0.12)
        self.assertTrue(all(item["r2"] > 0.8 for item in comparisons))
        self.assertTrue(all(item["surrogate_seconds"] >= 0.0 for item in comparisons))
        self.assertTrue(all(item["solver_seconds"] >= 0.0 for item in comparisons))


if __name__ == "__main__":
    unittest.main()
