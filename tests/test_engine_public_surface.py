"""Public engine imports and source-distribution declarations."""
import importlib
from pathlib import Path
import tomllib
import unittest


class EnginePublicSurfaceTests(unittest.TestCase):
    def test_every_lazy_export_resolves_to_its_declared_object(self):
        from openagent import adaptive_engine

        for name, (module_name, attribute) in adaptive_engine._EXPORTS.items():
            with self.subTest(name=name):
                module = importlib.import_module(f"openagent.adaptive_engine.{module_name}")
                self.assertIs(getattr(adaptive_engine, name), getattr(module, attribute))

    def test_compiled_sources_are_declared_for_wheels_and_source_archives(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["openagent.adaptive_engine"]
        package = root / "openagent" / "adaptive_engine"
        declared = {path.resolve() for pattern in patterns for path in package.glob(pattern)}
        sources = {path.resolve() for path in package.rglob("*") if path.suffix in {".c", ".h"}}
        self.assertTrue(sources)
        self.assertFalse(sources - declared, f"Unpackaged compiled sources: {sources - declared}")
        manifest = (root / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        self.assertIn("recursive-include openagent/adaptive_engine *.c *.h", manifest)
