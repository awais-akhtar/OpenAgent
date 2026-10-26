"""Real route capture accepted by the P15 gate; forged measurements rejected.

A tiny OLMoE checkpoint is compiled and run through the production universal
path, compared with the independent NumPy model, and recorded as a receipt.
The gate must accept it, and must reject each forged or inconsistent
measurement even when every file hash is re-pinned to match.
"""
import copy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from openagent.adaptive_engine import source_scope
from openagent.adaptive_engine.phase1_capture import (
    CaptureError, canonical_row, capture_olmoe_route, compile_segment_runtime)
from openagent.adaptive_engine.phase1_closure_integrated import Phase1ValidatorComplete
from openagent.adaptive_engine.phase1_observations import (
    ObservationError, coverage_applicability, decode_matrix, encode_matrix, load_receipts,
    sha256_file, verify_route_receipt)
from openagent.adaptive_engine.universal_engine import ModelRoute
from tests.test_native_segment_executor import build_executor
from tests.test_olmoe_batch import checkpoint

ROUTE = ModelRoute("olmoe", "olmoe", "native", "fp32", "F32", "int8-row")


class CanonicalCaptureRowTests(unittest.TestCase):
    def test_capture_cli_forwards_exact_publisher_case_id(self):
        from unittest.mock import patch
        from openagent.adaptive_engine import phase1_capture

        with patch.object(phase1_capture, "capture_olmoe_route",
                          return_value={"case_id": "olmoe/exact-pin"}) as capture:
            result = phase1_capture.main([
                "olmoe", "--checkpoint", "checkpoint", "--manifest", "manifest.json",
                "--prompt-ids", "1,2", "--case-id", "olmoe/exact-pin",
            ])

        self.assertEqual(result, 0)
        self.assertEqual(capture.call_args.kwargs["case_id"], "olmoe/exact-pin")

    def test_capture_route_requires_exact_publisher_pin_when_tuple_is_shared(self):
        from openagent.adaptive_engine.source_scope import source_matrix_rows

        route = ModelRoute("mixtral", "mixtral", "native", "fp32", "F32", "float32")
        matching = [row for row in source_matrix_rows(admitted_only=True)
                    if tuple(row[name] for name in (
                        "family", "model_type", "backend", "arithmetic",
                        "source_format", "persisted_format")) == route.as_tuple()]
        self.assertEqual(len(matching), 2)
        with self.assertRaisesRegex(CaptureError, "publisher case ID required"):
            canonical_row(route)
        selected = canonical_row(route, case_id=matching[1]["case_id"])
        self.assertEqual(selected["case_id"], matching[1]["case_id"])
        with self.assertRaisesRegex(CaptureError, "does not resolve"):
            canonical_row(route, case_id="mixtral/foreign/revision")


class Phase1CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiled = tempfile.mkdtemp(prefix="oa-capture-")
        cls.library = build_executor(cls.compiled)
        # The C Segment/Edge registry seals on its first lookup and retains
        # callback pointers for the lifetime of the loaded library. These two
        # public-range exercises register independent Python owners, so they
        # need isolated runtime libraries rather than sharing one sealed
        # process-global registry.
        cls.public_range_library = compile_segment_runtime(
            Path(cls.compiled) / "public-runtime-range")
        cls.public_cli_library = compile_segment_runtime(
            Path(cls.compiled) / "public-runtime-cli")
        cls.base = Path(tempfile.mkdtemp(prefix="oa-capture-model-"))
        cls.root = cls.base / "olmoe"
        cls.root.mkdir()
        checkpoint(cls.root, True).write(cls.root / "manifest.json")
        cls.receipts = cls.base / "phase1_runtime_receipts.json"
        cls.measured = capture_olmoe_route(
            cls.root, manifest=cls.root / "manifest.json", library=cls.library,
            prompt_ids=[1, 3], max_tokens=2, recovery_tokens=2, context_tokens=8,
            cache_mib=1, resident_mib=1, transfer_mib=1, staging_mib=4,
            evidence_root=cls.base / "evidence", receipts_path=cls.receipts)
        cls.row = canonical_row(ROUTE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.compiled, ignore_errors=True)
        shutil.rmtree(cls.base, ignore_errors=True)

    def receipt(self):
        return copy.deepcopy(load_receipts(self.receipts)[self.row["case_id"]])

    def scope(self):
        return SimpleNamespace(validate_phase1_source=source_scope.validate_phase1_source,
                               source_scope_matrix=lambda: {"execution_cases": [self.row]})

    def test_real_capture_is_measured_and_accepted_by_the_gate(self):
        self.assertEqual(self.measured["case_id"], "olmoe/olmoe/native/fp32/F32/int8-row/olmoe/caller-pin")
        self.assertLess(self.measured["max_abs_logit_error"], 1e-3)
        self.assertEqual(self.measured["device"], "cpu")
        self.assertNotEqual(self.measured["implementation"], self.measured["reference"])
        self.assertEqual(self.measured["coverage"], {
            "media_lifecycle": "not-applicable", "speculative_state": "not-applicable",
            "distributed_recovery": "not-applicable", "format_device": "verified",
            "abi_recovery": "verified"})
        validator = Phase1ValidatorComplete(self.scope(), self.receipts)
        result = validator.validate_all()
        for name in ("A04_media_lifecycle", "A05_speculative_rng", "A06_distributed_consensus",
                     "A07_format_validation", "A08_abi_callbacks", "A09_observation_binding",
                     "A10_evidence_requirements"):
            self.assertTrue(result["assertions"][name]["passed"], (name, result["assertions"][name]))

    def test_public_range_backend_captures_the_exact_segment_edge_owner(self):
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        root = self.base / "public-range-f32"
        root.mkdir()
        checkpoint(root, True)
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["torch_dtype"] = "float32"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest = pack_olmoe_experts(root, root / "experts.public-i8", dtype="int8-row")
        manifest_path = root / "manifest.public-i8.json"
        manifest.write(manifest_path)

        measured = capture_olmoe_route(
            root, manifest=manifest_path, library=self.public_range_library, prompt_ids=[1],
            max_tokens=1, recovery_tokens=1, context_tokens=4,
            cache_mib=1, resident_mib=1, transfer_mib=1, staging_mib=1,
            conversion_mib=4, backend="public-range",
            evidence_root=self.base / "public-range-evidence",
            receipts_path=self.base / "public-range-receipts.json")

        self.assertEqual(measured["case_id"],
                         "olmoe/olmoe/public-range/fp32/F32/int8-row/olmoe/caller-pin")
        self.assertLess(measured["max_abs_logit_error"], 1e-3)
        self.assertEqual(measured["device"], "cpu")

    def test_public_native_olmoe_cli_uses_the_pinned_segment_edge_runtime(self):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO
        import hashlib
        from openagent.adaptive_engine.native_olmoe_cli import main
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        root = self.base / "public-cli-f32"
        root.mkdir()
        checkpoint(root, True)
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["torch_dtype"] = "float32"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        manifest = pack_olmoe_experts(root, root / "experts.cli-public-i8", dtype="int8-row")
        manifest_path = manifest.write(root / "manifest.cli-public-i8.json")
        identity = hashlib.sha256(json.dumps(
            manifest.to_dict(), sort_keys=True, separators=(",", ":"),
            allow_nan=False).encode("utf-8")).hexdigest()
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            result = main([
                str(root), str(manifest_path), "--library", str(self.library),
                "--public-runtime-library", str(self.public_cli_library),
                "--checkpoint-identity", identity, "--context-tokens", "4",
                "--cache-mib", "1", "--resident-mib", "1", "--transfer-mib", "1",
                "--staging-mib", "4", "--snapshot-mib", "1",
                "--token-ids", "[1]", "--max-tokens", "1"])
        self.assertEqual(result, 0, errors.getvalue())
        response = json.loads(output.getvalue())
        self.assertEqual(response["schema"], "openagent.native-olmoe-public-generation.v1")
        self.assertEqual(len(response["generated_ids"]), 1)
        self.assertEqual(response["abi_contract"],
                         {"segment": 1, "edge": 2, "certified": False})

    def test_unknown_capture_backend_fails_closed(self):
        with self.assertRaisesRegex(CaptureError, "capture backend must be native or public-range"):
            capture_olmoe_route(
                self.root, manifest=self.root / "manifest.json", library=self.library,
                prompt_ids=[1], max_tokens=1, recovery_tokens=1, context_tokens=4,
                backend="device", evidence_root=self.base / "bad-backend-evidence",
                receipts_path=self.base / "bad-backend-receipts.json")

    def test_public_range_requires_exact_int8_row_manifest(self):
        wrong_path = Path("models/olmoe-1b-7b-instruct-f16/openagent-experts.q8.manifest.json")
        with self.assertRaisesRegex(CaptureError, "public-range owner admits only int8-row"):
            capture_olmoe_route(
                self.root, manifest=wrong_path, library=self.library,
                prompt_ids=[1], max_tokens=1, recovery_tokens=1, context_tokens=4,
                backend="public-range", evidence_root=self.base / "bad-public-evidence",
                receipts_path=self.base / "bad-public-receipts.json")

    def test_capture_passes_its_decode_budget_to_the_native_engine(self):
        from unittest.mock import patch
        from openagent.adaptive_engine import phase1_capture, universal_adapters
        seen = []
        original = universal_adapters.OlmoeNativeOwner.__init__

        def recording(owner, *args, **kwargs):
            seen.append(kwargs.get("conversion_mib"))
            original(owner, *args, **kwargs)

        # Keep this focused test on the leased universal owner. Production
        # captures now prefer the exact direct consumer when one is registered.
        with patch.dict(phase1_capture.OLMOE_DIRECT_CONSUMERS, {}, clear=True), \
                patch.object(universal_adapters.OlmoeNativeOwner, "__init__", recording):
            capture_olmoe_route(
                self.root, manifest=self.root / "manifest.json", library=self.library,
                prompt_ids=[1, 3], max_tokens=1, recovery_tokens=1, context_tokens=8,
                cache_mib=1, resident_mib=1, transfer_mib=1, staging_mib=1, conversion_mib=64,
                evidence_root=self.base / "evidence-budget",
                receipts_path=self.base / "budget-receipts.json")
        self.assertEqual(seen, [64])

    def test_direct_int4_capture_uses_explicit_native_packed_consumer(self):
        from openagent.adaptive_engine.native_packed_expert import NativePackedExpertBackend
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        backend = NativePackedExpertBackend()
        if not backend.info().available:
            self.skipTest("Numba packed INT4 consumer is unavailable")
        root = self.base / "direct-int4"
        root.mkdir()
        checkpoint(root, True)
        manifest = pack_olmoe_experts(root, root / "experts.i4", dtype="int4-row")
        manifest_path = root / "manifest.i4.json"
        manifest.write(manifest_path)
        measured = capture_olmoe_route(
            root, manifest=manifest_path, library=None, prompt_ids=[1], max_tokens=1,
            recovery_tokens=1, context_tokens=4, cache_mib=1, conversion_mib=1,
            evidence_root=self.base / "direct-int4-evidence",
            receipts_path=self.base / "direct-int4-receipts.json")
        self.assertEqual(measured["case_id"],
                         "olmoe/olmoe/native/fp32/F32/int4-row/olmoe/caller-pin")
        self.assertLess(measured["max_abs_logit_error"], 1e-3)
        self.assertIn("packed-int4-numba", measured["implementation"])

    def test_direct_float32_capture_uses_explicit_native_consumer(self):
        from openagent.adaptive_engine.native_expert import NativeExpertBackend
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        backend = NativeExpertBackend()
        if not backend.info().available:
            self.skipTest("NativeExpertBackend is unavailable")
        root = self.base / "direct-float32"
        root.mkdir()
        checkpoint(root, True)
        manifest = pack_olmoe_experts(root, root / "experts.f32", dtype="float32")
        manifest_path = root / "manifest.f32.json"
        manifest.write(manifest_path)
        measured = capture_olmoe_route(
            root, manifest=manifest_path, library=None, prompt_ids=[1], max_tokens=1,
            recovery_tokens=1, context_tokens=4, cache_mib=1, conversion_mib=1,
            evidence_root=self.base / "direct-float32-evidence",
            receipts_path=self.base / "direct-float32-receipts.json")
        self.assertEqual(measured["case_id"],
                         "olmoe/olmoe/native/fp32/F32/float32/olmoe/caller-pin")
        self.assertLess(measured["max_abs_logit_error"], 1e-3)
        self.assertIn("native-cpu-numba", measured["implementation"])

    def test_direct_int2_capture_uses_explicit_native_lowbit_consumer(self):
        from openagent.adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        backend = NativeLowBitExpertBackend("int2-row")
        if not backend.info().available:
            self.skipTest("NativeLowBitExpertBackend is unavailable")
        root = self.base / "direct-int2"
        root.mkdir()
        checkpoint(root, True)
        manifest = pack_olmoe_experts(root, root / "experts.i2", dtype="int2-row")
        manifest_path = root / "manifest.i2.json"
        manifest.write(manifest_path)
        measured = capture_olmoe_route(
            root, manifest=manifest_path, library=None, prompt_ids=[1], max_tokens=1,
            recovery_tokens=1, context_tokens=4, cache_mib=1, conversion_mib=1,
            evidence_root=self.base / "direct-int2-evidence",
            receipts_path=self.base / "direct-int2-receipts.json")
        self.assertEqual(measured["case_id"],
                         "olmoe/olmoe/native/fp32/F32/int2-row/olmoe/caller-pin")
        self.assertLess(measured["max_abs_logit_error"], 1e-3)
        self.assertIn("int2-row-numba", measured["implementation"])

    def test_direct_int3_capture_uses_grouped_native_lowbit_consumer(self):
        from openagent.adaptive_engine.native_lowbit_expert import NativeLowBitExpertBackend
        from openagent.adaptive_engine.olmoe import pack_olmoe_experts

        backend = NativeLowBitExpertBackend("int3-grouped-g64")
        if not backend.info().available:
            self.skipTest("NativeLowBitExpertBackend is unavailable")
        root = self.base / "direct-int3"
        root.mkdir()
        checkpoint(root, True)
        manifest = pack_olmoe_experts(
            root, root / "experts.i3", dtype="int3-grouped-g64")
        manifest_path = root / "manifest.i3.json"
        manifest.write(manifest_path)
        measured = capture_olmoe_route(
            root, manifest=manifest_path, library=None, prompt_ids=[1], max_tokens=1,
            recovery_tokens=1, context_tokens=4, cache_mib=1, conversion_mib=1,
            evidence_root=self.base / "direct-int3-evidence",
            receipts_path=self.base / "direct-int3-receipts.json")
        self.assertEqual(
            measured["case_id"],
            "olmoe/olmoe/native/fp32/F32/int3-grouped-g64/olmoe/caller-pin")
        self.assertLess(measured["max_abs_logit_error"], 1e-3)
        self.assertIn("int3-grouped-g64-numba", measured["implementation"])

    def test_direct_consumer_registry_matches_gateway_format_defaults(self):
        from openagent.adaptive_engine.phase1_capture import (
            OLMOE_DIRECT_CONSUMERS, _olmoe_direct_backend)
        from openagent.model_gateway import _OLMOE_KERNEL_BY_FORMAT

        # The capture's direct-consumer registry intentionally excludes formats
        # whose route is captured through the leased segment engine instead.
        self.assertLessEqual(set(OLMOE_DIRECT_CONSUMERS), set(_OLMOE_KERNEL_BY_FORMAT))
        for persisted_format, consumer in OLMOE_DIRECT_CONSUMERS.items():
            with self.subTest(persisted_format=persisted_format):
                backend = _olmoe_direct_backend(persisted_format)
                self.assertEqual(backend.NAME, _OLMOE_KERNEL_BY_FORMAT[persisted_format])
                self.assertIn(persisted_format, backend.info().capabilities.payload_dtypes)
                self.assertIn(type(backend).__name__, consumer)

    def test_legacy_a10_runner_refuses_to_write_synthetic_receipts(self):
        from contextlib import redirect_stderr
        from io import StringIO
        from openagent.adaptive_engine.run_a10_evidence import (
            create_test_evidence_binding, run_a10_evidence)

        output = self.base / "synthetic-a10-must-not-exist"
        with redirect_stderr(StringIO()):
            self.assertEqual(run_a10_evidence(output_dir=output), 1)
        self.assertFalse(output.exists())
        with self.assertRaisesRegex(RuntimeError, "synthetic A10 evidence is prohibited"):
            create_test_evidence_binding("olmoe", 1)

    def test_claimed_booleans_are_ignored(self):
        receipt = self.receipt()
        receipt.update(full_generation_success=False, numerical_parity_verified=False,
                       platform_certified=False)
        self.assertEqual(verify_route_receipt(self.row, receipt)["case_id"], self.row["case_id"])

    def test_receipt_requires_the_exact_admitted_tuple(self):
        receipt = self.receipt()
        receipt["route"][5] = "q8_0"
        self.assertRejected(receipt, "receipt route differs")
        receipt = self.receipt()
        candidate = dict(self.row, combination_status="candidate-unverified")
        with self.assertRaisesRegex(ObservationError, "unadmitted source route"):
            verify_route_receipt(candidate, receipt)

    def test_receipt_metadata_must_match_measured_outputs(self):
        receipt = self.receipt()
        receipt["placement_constraints"]["device"] = "xpu:0"
        self.assertRejected(receipt, "placement constraints differ")
        receipt = self.receipt()
        receipt["comparison_adapter"] = "unobserved reference"
        self.assertRejected(receipt, "comparison adapter differs")
        receipt = self.receipt()
        receipt["cache_schema"]["layers"] += 1
        self.assertRejected(receipt, "cache schema differs")

    def test_checkpoint_artifact_cannot_come_from_another_directory(self):
        receipt = self.receipt()
        weight = next(pin for pin in receipt["artifacts"] if pin["role"] == "weights")
        foreign = self.base / "foreign-weight.safetensors"
        shutil.copyfile(weight["path"], foreign)
        weight["path"] = str(foreign)
        self.assertRejected(receipt, "outside the pinned checkpoint")

    def test_large_checkpoint_artifacts_use_artifact_limit_not_observation_limit(self):
        from unittest.mock import patch
        from openagent.adaptive_engine import phase1_observations

        receipt = self.receipt()
        large = self.root / "large-shard.safetensors"
        try:
            with large.open("wb") as stream:
                stream.truncate(2 * 1024 * 1024)
            receipt["artifacts"].append({"role": "weights", "path": str(large.resolve()),
                                         "sha256": sha256_file(large)})
            # The deliberately small observation limit represents the JSON/logit
            # payload bound; a larger but independently bounded artifact remains
            # valid because verification hashes it incrementally.
            with patch.object(phase1_observations, "MAX_OBSERVATION_BYTES", 1024 * 1024), \
                    patch.object(phase1_observations, "MAX_ARTIFACT_BYTES", 3 * 1024 * 1024):
                self.assertEqual(verify_route_receipt(self.row, receipt)["case_id"], self.row["case_id"])
        finally:
            large.unlink(missing_ok=True)

    def forge(self, kind, mutate):
        """Rewrite one observation, re-pin its hash, and return the receipt."""
        receipt = self.receipt()
        directory = Path(tempfile.mkdtemp(dir=self.base))
        for pin in receipt["observations"]:
            if pin["kind"] == kind:
                payload = json.loads(Path(pin["path"]).read_text(encoding="utf-8"))
                mutate(payload)
                target = directory / f"{kind}.json"
                target.write_text(json.dumps(payload), encoding="utf-8")
                pin["path"], pin["sha256"] = str(target), sha256_file(target)
        return receipt

    def assertRejected(self, receipt, message):
        with self.assertRaisesRegex(ObservationError, message):
            verify_route_receipt(self.row, receipt)

    def test_forged_reference_beyond_tolerance_is_rejected(self):
        def shift(payload):
            logits = decode_matrix(payload["reference"]["logits"], label="x").copy()
            logits += 0.5
            payload["reference"]["logits"] = encode_matrix(logits)
        self.assertRejected(self.forge("comparison", shift), "exceeds tolerance")

    def test_tolerance_above_policy_is_rejected(self):
        self.assertRejected(self.forge("comparison", lambda p: p["tolerance"].update(max_abs_logit=1.0)),
                            "declared logit tolerance")

    def test_reference_must_be_a_different_implementation(self):
        tokens = json.loads(Path(next(pin for pin in self.receipt()["observations"]
                                      if pin["kind"] == "tokens")["path"]).read_text())
        self.assertRejected(self.forge("comparison", lambda p: p["reference"].update(
            implementation=tokens["implementation"])), "different implementation")

    def test_reference_must_be_the_reviewed_source_at_its_current_hash(self):
        self.assertRejected(self.forge("comparison", lambda p: p["reference"].update(
            source={"module": "somewhere/else.py", "sha256": "0" * 64})), "reviewed in-repository")
        self.assertRejected(self.forge("comparison", lambda p: p["reference"]["source"].update(
            sha256="0" * 64)), "changed since capture")
        self.assertRejected(self.forge("comparison", lambda p: p["reference"].pop("source")),
                            "reviewed in-repository")

    def test_tokens_must_be_the_argmax_of_the_observed_logits(self):
        def swap(payload):
            payload["generated_ids"] = [(token + 1) % 12 for token in payload["generated_ids"]]
        self.assertRejected(self.forge("tokens", swap), "argmax|differ")

    def test_cache_routing_recovery_and_placement_measurements_are_checked(self):
        self.assertRejected(self.forge("cache", lambda p: p.update(position=p["position"] + 1)),
                            "cache position")
        self.assertRejected(self.forge("routing", lambda p: next(iter(p["layers"].values())).pop()),
                            "one row per processed position")
        self.assertRejected(self.forge("recovery", lambda p: p.update(replay_ids=[0] * len(p["replay_ids"]))
                                       if p["replay_ids"] != [0] * len(p["replay_ids"])
                                       else p.update(replay_ids=[1] * len(p["replay_ids"]))),
                            "replay after restore")
        self.assertRejected(self.forge("placement", lambda p: p.update(host_fallback_operations=3)),
                            "host-fallback")
        self.assertRejected(self.forge("placement", lambda p: p.update(device="xpu:0")),
                            "non-CPU device")

    def test_routing_must_cover_the_configured_layer_indices(self):
        def replace_layer(payload):
            layers = payload["layers"]
            first = next(iter(layers))
            layers["model.layers.99"] = layers.pop(first)
        self.assertRejected(self.forge("routing", replace_layer),
                            "every configured layer index exactly once")

    def test_one_file_cannot_serve_several_kinds(self):
        receipt = self.receipt()
        tokens = next(pin for pin in receipt["observations"] if pin["kind"] == "tokens")
        for pin in receipt["observations"]:
            if pin["kind"] == "cache":
                pin["path"], pin["sha256"] = tokens["path"], tokens["sha256"]
        self.assertRejected(receipt, "one file cannot serve")

    def test_observation_bound_to_another_route_or_config_is_rejected(self):
        self.assertRejected(self.forge("logits", lambda p: p.update(case_id="olmoe/other")),
                            "another route")
        self.assertRejected(self.forge("logits", lambda p: p.update(config_sha256="0" * 64)),
                            "another checkpoint config")

    def test_receipt_must_pin_every_checkpoint_weight_shard(self):
        receipt = self.receipt()
        weight = next(index for index, artifact in enumerate(receipt["artifacts"])
                      if artifact["role"] == "weights")
        receipt["artifacts"].pop(weight)
        with self.assertRaisesRegex(ObservationError, "every SafeTensors checkpoint weight shard"):
            verify_route_receipt(self.row, receipt)

    def test_receipt_cannot_pin_one_checkpoint_weight_shard_twice(self):
        receipt = self.receipt()
        weight = next(artifact for artifact in receipt["artifacts"]
                      if artifact["role"] == "weights")
        receipt["artifacts"].append(copy.deepcopy(weight))
        with self.assertRaisesRegex(ObservationError, "repeats a SafeTensors checkpoint weight shard pin"):
            verify_route_receipt(self.row, receipt)

    def test_not_applicable_must_match_route_applicability(self):
        receipt = self.receipt()
        receipt["checks"]["format_device"] = {"status": "not-applicable", "reason": "skip"}
        self.assertRejected(receipt, "format_device applies")
        media_row = next(case for case in source_scope.source_scope_matrix()["execution_cases"]
                         if case["model_type"] == "qwen3_5_moe" and
                         case["combination_status"] == "verified-admitted" and case["backend"] == "native")
        applicability = coverage_applicability(media_row)
        self.assertTrue(applicability["media_lifecycle"])
        self.assertTrue(coverage_applicability(next(
            case for case in source_scope.source_scope_matrix()["execution_cases"]
            if case["model_type"] == "deepseek_v41" and case["combination_status"] == "verified-admitted"
        ))["speculative_state"])
        self.assertFalse(any(v for k, v in coverage_applicability(self.row).items()
                             if k in ("media_lifecycle", "speculative_state", "distributed_recovery")))

    def test_gate_requires_a_receipt_for_every_admitted_route(self):
        other = dict(self.row, case_id="olmoe/olmoe/native/fp32/F32/q8_0/olmoe/caller-pin")
        scope = SimpleNamespace(validate_phase1_source=lambda: True,
                                source_scope_matrix=lambda: {"execution_cases": [self.row, other]})
        validator = Phase1ValidatorComplete(scope, self.receipts)
        result = validator.validate_all()
        self.assertFalse(result["assertions"]["A10_evidence_requirements"]["passed"])
        self.assertIn("1 of 2 admitted routes lack runtime receipts",
                      result["assertions"]["A10_evidence_requirements"]["error"])


if __name__ == "__main__":
    unittest.main()
