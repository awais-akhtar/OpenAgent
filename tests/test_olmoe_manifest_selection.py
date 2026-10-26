"""OLMoE checkpoints with several packed-expert manifests.

Mirrors the local ``olmoe-1b-7b-instruct`` layout: one checkpoint with I4,
I8, Q4 and Q8 manifests.  Selection must be explicit or unique, must bind the
route's persisted format, rejects an INT4-row manifest at the universal leased
adapter boundary, and every leased-admitted format must generate the same
tokens as the independent NumPy model through the production universal path.
"""
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.olmoe import OlmoeModel, pack_olmoe_experts
from openagent.adaptive_engine.runtime import AdaptiveExecutionEngine
from openagent.adaptive_engine.universal_adapters import (
    expert_manifests, resolve_checkpoint_route, select_expert_manifest)
from openagent.adaptive_engine.universal_engine import ModelRoute
from openagent.storage import OpenAgentStore
from tests.test_native_segment_executor import build_executor
from tests.test_q8_0_olmoe import checkpoint

FORMATS = {"i4": "int4-row", "i8": "int8-row", "q4": "q4_0", "q8": "q8_0"}
ADMITTED = ("i8", "q4", "q8")


class OlmoeManifestSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.mkdtemp(prefix="oa-olmoe-manifests-")
        cls.library = build_executor(cls.compiled)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.compiled, ignore_errors=True)

    def setUp(self):
        self.threads = {thread.ident for thread in threading.enumerate()}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "olmoe"
        self.root.mkdir()
        checkpoint(self.root)
        self.manifests = {}
        for name, dtype in FORMATS.items():
            packed = pack_olmoe_experts(self.root, self.root / f"experts.{name}", dtype=dtype)
            path = self.root / f"manifest-{name}.json"
            packed.write(path)
            self.manifests[name] = (path.resolve(), packed)
        data = Path(temporary.name) / "data"
        data.mkdir()
        self.engine = AdaptiveExecutionEngine(OpenAgentStore(data))
        self.options = dict(library=str(self.library), context_tokens=16, cache_mib=1,
                            resident_mib=4, transfer_mib=4, staging_mib=4)

    def tearDown(self):
        self.engine.close()
        leaked = [t.name for t in threading.enumerate() if t.ident not in self.threads and t.is_alive()]
        self.assertEqual(leaked, [])

    def test_every_manifest_is_listed_with_its_persisted_format(self):
        found = {path: fmt for path, fmt in expert_manifests(self.root).items()
                 if path.name.startswith("manifest-")}
        self.assertEqual(found, {path: FORMATS[name] for name, (path, _) in self.manifests.items()})

    def test_ambiguous_default_fails_closed_and_names_the_choices(self):
        with self.assertRaisesRegex(ValueError, "pass manifest=<path>") as raised:
            resolve_checkpoint_route({"model_type": "olmoe"}, self.root)
        for name, dtype in FORMATS.items():
            self.assertIn(f"manifest-{name}.json={dtype}", str(raised.exception))
        with patch("openagent.adaptive_engine.universal_adapters.OlmoeNativeOwner") as owner:
            with self.assertRaisesRegex(ValueError, "exactly one sparse-expert manifest"):
                self.engine.open_universal_model(self.root, **self.options)
            owner.assert_not_called()

    def test_explicit_manifest_binds_the_route_format(self):
        for name, dtype in FORMATS.items():
            route = resolve_checkpoint_route({"model_type": "olmoe"}, self.root,
                                             options={"manifest": str(self.manifests[name][0])})
            self.assertEqual(route, ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", dtype))

    def test_foreign_manifest_cannot_be_selected_for_checkpoint(self):
        foreign = self.root.parent / "foreign-manifest.json"
        self.manifests["q8"][1].write(foreign)
        with self.assertRaisesRegex(ValueError, "direct file of the admitted checkpoint"):
            resolve_checkpoint_route({"model_type": "olmoe"}, self.root,
                                     options={"manifest": str(foreign)})
        with self.assertRaisesRegex(ValueError, "direct file of the admitted checkpoint"):
            self.engine.open_universal_model(self.root, manifest=str(foreign), **self.options)

    def test_explicit_route_selects_the_unique_matching_manifest(self):
        path, fmt = select_expert_manifest(self.root, persisted_format="q8_0")
        self.assertEqual((path, fmt), (self.manifests["q8"][0], "q8_0"))
        with self.assertRaisesRegex(ValueError, "route admits 'int8-row'"):
            select_expert_manifest(self.root, manifest=self.manifests["q8"][0],
                                   persisted_format="int8-row")
        with self.assertRaisesRegex(ValueError, "found 0"):
            select_expert_manifest(self.root, persisted_format="q5_1")

    def test_foreign_manifests_are_rejected(self):
        source, _ = self.manifests["q8"]
        elsewhere = self.root.parent / "elsewhere"
        elsewhere.mkdir()
        outside = elsewhere / "manifest-q8.json"
        outside.write_bytes(source.read_bytes())
        with self.assertRaisesRegex(ValueError, "direct file of the admitted checkpoint"):
            select_expert_manifest(self.root, manifest=outside)
        # A manifest packed for another checkpoint, copied into this one.
        other = self.root.parent / "other-olmoe"
        other.mkdir()
        checkpoint(other)
        (other / "config.json").write_text(
            (other / "config.json").read_text().replace('"eos_token_id": 22', '"eos_token_id": 21'))
        foreign = pack_olmoe_experts(other, other / "experts.q8", dtype="q8_0")
        copied = self.root / "manifest-foreign.json"
        foreign.write(copied)
        with self.assertRaisesRegex(ValueError, "does not belong to this checkpoint"):
            select_expert_manifest(self.root, manifest=copied)
        with self.assertRaisesRegex(ValueError, "does not belong to this checkpoint"):
            resolve_checkpoint_route({"model_type": "olmoe"}, self.root,
                                     options={"manifest": str(copied)})
        with patch("openagent.adaptive_engine.universal_adapters.OlmoeNativeOwner") as owner:
            with self.assertRaisesRegex(ValueError, "does not belong"):
                self.engine.open_universal_model(self.root, manifest=str(copied), **self.options)
            owner.assert_not_called()
        copied.unlink()
        link = self.root / "manifest-link.json"
        try:
            link.symlink_to(outside)
        except OSError:
            return  # symlinks need extra privileges on some Windows hosts
        with self.assertRaisesRegex(ValueError, "direct file of the admitted checkpoint"):
            select_expert_manifest(self.root, manifest=link)

    def test_gateway_only_int4_row_manifest_is_rejected_by_universal_adapter(self):
        with patch("openagent.adaptive_engine.universal_adapters.OlmoeNativeOwner") as owner:
            with self.assertRaisesRegex(ValueError, "does not implement 'int4-row'.*OlmoeDirectProvider"):
                self.engine.open_universal_model(self.root, manifest=str(self.manifests["i4"][0]),
                                                 **self.options)
            owner.assert_not_called()

    def test_each_admitted_format_matches_the_numpy_model(self):
        prompt = [1, 4, 7]
        for name in ADMITTED:
            path, packed = self.manifests[name]
            with self.subTest(format=FORMATS[name]):
                with OlmoeModel(self.root, packed, context_tokens=16,
                                expert_cache_bytes=4 * 1024 ** 2) as model:
                    pending, expected = prompt, []
                    for _ in range(3):
                        logits = np.asarray(model.forward(pending))
                        token = int(np.argmax(logits[-1] if logits.ndim == 2 else logits))
                        expected.append(token)
                        pending = [token]
                session = self.engine.open_universal_model(self.root, manifest=str(path), **self.options)
                self.assertEqual(session.manifest.route.persisted_format, FORMATS[name])
                self.assertEqual(session.owner.manifest_path, path)
                result = session.generate(prompt, max_tokens=3)
                self.assertEqual(result["generated_ids"], expected[:len(result["generated_ids"])])
                session.close()

    def test_explicit_route_without_manifest_opens_the_matching_format(self):
        route = ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", "q4_0")
        session = self.engine.open_universal_model(self.root, route=route, **self.options)
        self.assertEqual(session.owner.manifest_path, self.manifests["q4"][0])
        session.close()

    def test_different_manifests_are_different_warm_sessions(self):
        first = self.engine.open_universal_model(self.root, manifest=str(self.manifests["i8"][0]),
                                                 **self.options)
        second = self.engine.open_universal_model(self.root, manifest=str(self.manifests["q8"][0]),
                                                  **self.options)
        self.assertIsNot(first, second)
        self.assertNotEqual(first.metrics()["state_directory"], second.metrics()["state_directory"])
        self.assertIs(self.engine.open_universal_model(self.root, manifest=str(self.manifests["i8"][0]),
                                                       **self.options), first)


if __name__ == "__main__":
    unittest.main()
