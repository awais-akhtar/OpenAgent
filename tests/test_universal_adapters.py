"""Publisher-family production adapter wiring.

Uses a real SafeTensors checkpoint, the real gateway operator parser and the
real P02 inventory gate.  Only ``PublisherSession`` is replaced, because the
family C libraries are not part of this test; the replacement records exactly
what the production opener hands to it.
"""
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from openagent.adaptive_engine.publisher_provider import (
    parse_publisher_operator, publisher_operator_backend)
from openagent.adaptive_engine.universal_adapters import (
    PublisherServiceOwner, register_production_universal_adapters, resolve_checkpoint_route)
from openagent.adaptive_engine.universal_engine import ModelRoute, UniversalEngine


def write_checkpoint(root: Path, model_type: str, dtypes: dict[str, str]) -> None:
    (root / "config.json").write_text(json.dumps(
        {"model_type": model_type, "architectures": ["Fixture"]}), encoding="utf-8")
    header, data = {}, bytearray()
    for name, dtype in dtypes.items():
        width = {"F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1}[dtype]
        begin = len(data)
        data.extend(np.zeros(4 * width, dtype=np.uint8).tobytes())
        header[name] = {"dtype": dtype, "shape": [4], "data_offsets": [begin, len(data)]}
    encoded = json.dumps(header).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    (root / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + bytes(data))


class FakeService:
    def __init__(self):
        self.calls = []

    def snapshot(self, session_id):
        self.calls.append(("snapshot", session_id))
        return ("receipt", session_id)

    def restore(self, session_id, receipt):
        self.calls.append(("restore", session_id, receipt))

    def release_snapshot(self, session_id, receipt):
        self.calls.append(("release", session_id, receipt))


class RecordingSession:
    instances = []

    def __init__(self, args, *, cancelled=None):
        self.args = args
        self.service = FakeService()
        self.session_id = "session-7"
        self.requests = []
        self.closed = 0
        self.checkpoint_identity = "a" * 64
        RecordingSession.instances.append(self)

    def run(self, args, *, emit=None, cancelled=None):
        self.requests.append(args)
        if emit is not None:
            emit({"event": "tokens", "token_ids": [9], "committed": False})
        return {"generated_ids": [9], "finish_reason": "length"}

    def close(self):
        self.closed += 1


ARGV = ["--runtime-library", "runtime.so", "--kernel-library", "kernels.so", "--context", "256",
        "--runtime-mib", "8", "--staging-mib", "8", "--snapshot-mib", "8", "--transfer-mib", "8",
        "--layers-per-range", "2"]


class PublisherAdapterTests(unittest.TestCase):
    def setUp(self):
        RecordingSession.instances = []
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_checkpoint(self.root, "mixtral", {"model.embed_tokens.weight": "BF16",
                                                "model.layers.0.mlp.weight": "BF16"})
        self.engine = UniversalEngine()
        register_production_universal_adapters(self.engine)
        patcher = patch("openagent.adaptive_engine.native_family_cli.PublisherSession", RecordingSession)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_resolver_names_the_checkpoint_source_dtype(self):
        self.assertEqual(resolve_checkpoint_route({"model_type": "mixtral"}, self.root),
                         ModelRoute("mixtral", "mixtral", "native", "fp32", "BF16", "float32"))

    def test_opener_builds_gateway_arguments_for_the_admitted_checkpoint(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        self.assertIsInstance(session.owner, PublisherServiceOwner)
        created = RecordingSession.instances[0]
        self.assertEqual(created.args.checkpoint, self.root.resolve())
        self.assertEqual(created.args.family, "mixtral")
        result = session.generate([1, 2], max_tokens=3, temperature=0.5, stop_token_id=[4])
        self.assertEqual(result["generated_ids"], [9])
        request = created.requests[0]
        self.assertEqual(json.loads(request.token_ids), [1, 2])
        self.assertEqual((request.max_tokens, request.temperature, request.stop_token_id), (3, 0.5, [4]))
        self.assertEqual(created.args.token_ids, "[]")
        self.assertEqual([call[0] for call in created.service.calls], ["snapshot", "release"])
        self.assertTrue(all(call[1] == "session-7" for call in created.service.calls))
        events = list(session.stream([1], max_tokens=1))
        self.assertEqual(events[0]["token_ids"], [9])
        self.assertEqual(events[-1]["event"], "result")
        session.close()
        self.assertEqual(created.closed, 1)

    def test_publisher_owner_constructor_failure_keeps_session_cleanup_retryable(self):
        original_close = RecordingSession.close

        def fail_first_close(session):
            attempts = getattr(session, "close_attempts", 0) + 1
            session.close_attempts = attempts
            if attempts == 1:
                raise OSError("injected session close failure")
            original_close(session)

        with patch.object(RecordingSession, "close", fail_first_close), \
             patch("openagent.adaptive_engine.universal_adapters.PublisherServiceOwner",
                   side_effect=RuntimeError("injected owner construction failure")):
            with self.assertRaisesRegex(RuntimeError, "injected owner construction failure") as raised:
                self.engine.open(self.root, argv=list(ARGV))

        session = RecordingSession.instances[-1]
        error = raised.exception
        self.assertEqual(session.close_attempts, 1)
        self.assertTrue(callable(getattr(error, "cleanup", None)))
        self.assertTrue(any("cleanup remains retryable" in note
                            for note in getattr(error, "__notes__", ())))
        error.cleanup()
        self.assertEqual(session.close_attempts, 2)
        self.assertEqual(session.closed, 1)

    def test_request_rollback_uses_the_service_receipt(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        created = RecordingSession.instances[0]
        with self.assertRaises(Exception):
            session.generate([1], publish=lambda result: (_ for _ in ()).throw(OSError("sink")))
        kinds = [call[0] for call in created.service.calls]
        self.assertEqual(kinds, ["snapshot", "restore", "release"])
        self.assertEqual(created.service.calls[1][2], ("receipt", "session-7"))

    def test_closing_after_terminal_result_keeps_committed_stream_state(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        created = RecordingSession.instances[0]
        stream = session.stream([1], max_tokens=1)
        events = []
        for event in stream:
            events.append(event)
            if event.get("event") == "result":
                break
        stream.close()

        self.assertEqual(events[-1]["event"], "result")
        self.assertTrue(session.is_usable())
        self.assertEqual(session.status()["epoch"], 1)
        self.assertNotIn("restore", [call[0] for call in created.service.calls])
        session.close()

    def test_undeclared_or_unsafe_options_are_rejected_before_construction(self):
        cases = (
            ({}, "required adapter options missing"),
            ({"argv": list(ARGV), "model_family": "mixtral"}, "undeclared adapter options"),
            ({"argv": [*ARGV, "--checkpoint", "/elsewhere"]}, "supplies the admitted checkpoint"),
            ({"argv": [*ARGV, "--token-ids", "[1]"]}, "prompt inputs"),
        )
        for options, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(Exception, message):
                self.engine.open(self.root, **options)
        self.assertEqual(RecordingSession.instances, [])

    def test_request_fields_outside_the_contract_are_rejected(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        with self.assertRaisesRegex(ValueError, "unsupported universal request fields"):
            session.generate([1], save_cache="/tmp/x")
        self.assertEqual(RecordingSession.instances[0].requests, [])

    def test_stream_cancellation_predicate_stays_on_consumer_thread(self):
        class PollingSession(RecordingSession):
            def run(self, args, *, emit=None, cancelled=None):
                if cancelled is not None:
                    cancelled()
                return super().run(args, emit=emit, cancelled=cancelled)

        patcher = patch("openagent.adaptive_engine.native_family_cli.PublisherSession",
                        PollingSession)
        patcher.start()
        self.addCleanup(patcher.stop)
        session = self.engine.open(self.root, argv=list(ARGV))
        self.addCleanup(session.close)

        consumer_thread = threading.get_ident()
        callback_threads = []
        events = list(session.stream(
            [1], max_tokens=1,
            cancelled=lambda: callback_threads.append(threading.get_ident()) or False))

        self.assertEqual(events[-1]["event"], "result")
        self.assertTrue(callback_threads)
        self.assertEqual(set(callback_threads), {consumer_thread})

    def test_operator_backend_must_match_the_admitted_route(self):
        with self.assertRaisesRegex(ValueError, "route admits 'native'"):
            self.engine.open(self.root, argv=[*ARGV, "--device", "xpu:0"])
        self.assertEqual(RecordingSession.instances, [])
        device = ModelRoute("mixtral", "mixtral", "device", "fp32", "BF16", "float32")
        session = self.engine.open(self.root, route=device, argv=[*ARGV, "--device", "xpu:0"])
        self.assertEqual(RecordingSession.instances[0].args.device, "xpu:0")
        session.close()

    def test_explicit_canonical_source_dtype_must_match_checkpoint_inventory(self):
        # F32 is a valid canonical Mixtral source route, but this checkpoint
        # actually contains BF16 tensors. Admission must not confuse route
        # eligibility with checkpoint-to-route identity.
        wrong_source = ModelRoute("mixtral", "mixtral", "native", "fp32", "F32", "float32")
        with self.assertRaisesRegex(ValueError, "differs from checkpoint inventory 'BF16'"):
            self.engine.open(self.root, route=wrong_source, argv=list(ARGV))
        self.assertEqual(RecordingSession.instances, [])

    def test_checkpoint_inventory_gate_runs_before_construction(self):
        write_checkpoint(self.root, "mixtral", {"model.embed_tokens.weight": "BF16",
                                                "model.layers.0.mlp.weight": "F8_E5M2"})
        route = ModelRoute("mixtral", "mixtral", "native", "fp32", "BF16", "float32")
        with self.assertRaisesRegex(ValueError, "unsupported floating source encoding"):
            self.engine.open(self.root, route=route, argv=list(ARGV))
        self.assertEqual(RecordingSession.instances, [])


class IgnoringCancellationSession(RecordingSession):
    """Backend that keeps running after its consumer abandons a stream."""
    release = threading.Event()

    def run(self, args, *, emit=None, cancelled=None):
        self.requests.append(args)
        emit({"event": "tokens", "token_ids": [9], "committed": False})
        IgnoringCancellationSession.release.wait(10)
        return {"generated_ids": [9], "finish_reason": "length"}


class StreamQuarantineTests(unittest.TestCase):
    def setUp(self):
        RecordingSession.instances = []
        IgnoringCancellationSession.release = threading.Event()
        self.addCleanup(IgnoringCancellationSession.release.set)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_checkpoint(self.root, "mixtral", {"model.embed_tokens.weight": "BF16"})
        self.engine = UniversalEngine()
        register_production_universal_adapters(self.engine)
        patcher = patch("openagent.adaptive_engine.native_family_cli.PublisherSession",
                        IgnoringCancellationSession)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_abandoned_stream_with_running_worker_quarantines_instead_of_racing(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        owner = session.owner
        owner._guard.join_seconds = 0.2
        backend = RecordingSession.instances[0]
        stream = session.stream([1], max_tokens=1)
        self.assertEqual(next(stream)["token_ids"], [9])
        with self.assertRaisesRegex(RuntimeError, "stream worker ignored cancellation"):
            stream.close()
        self.assertEqual(owner._guard.quarantine_reason, "stream worker ignored cancellation")
        status = session.status()
        self.assertTrue(status["closing"])
        self.assertFalse(status["closed"])
        self.assertEqual(status["pending_cleanup_receipts"], 1)
        # Rollback was refused while the worker ran; no restore reached the service.
        self.assertNotIn("restore", [call[0] for call in backend.service.calls])
        with self.assertRaisesRegex(RuntimeError, "close is pending"):
            session.generate([1])
        with self.assertRaisesRegex(RuntimeError, "still running"):
            session.close()
        self.assertEqual(backend.closed, 0)
        IgnoringCancellationSession.release.set()
        session.close()
        self.assertTrue(session.status()["closed"])
        self.assertEqual(session.status()["pending_cleanup_receipts"], 0)
        self.assertEqual(backend.closed, 1)

    def test_cooperative_abandonment_still_rolls_back(self):
        session = self.engine.open(self.root, argv=list(ARGV))
        backend = RecordingSession.instances[0]
        IgnoringCancellationSession.release.set()
        stream = session.stream([1], max_tokens=1)
        next(stream)
        stream.close()
        self.assertIsNone(session.owner._guard.quarantine_reason)
        self.assertIn("restore", [call[0] for call in backend.service.calls])
        self.assertTrue(session.is_usable())
        session.close()


class DenseQwen35ScopeTests(unittest.TestCase):
    """Dense ``qwen3_5`` is an explicit Phase 1 rejection, never an MoE alias."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        write_checkpoint(self.root, "qwen3_5", {"model.embed_tokens.weight": "BF16",
                                                "model.layers.0.mlp.up_proj.weight": "BF16"})
        self.engine = UniversalEngine()
        register_production_universal_adapters(self.engine)

    def test_dense_identity_reports_its_recorded_scope_reason_everywhere(self):
        from openagent.adaptive_engine.source_scope import (
            PHASE1_OUT_OF_SCOPE_IDENTITIES, source_family_for_identity, source_scope_matrix)
        self.assertIn("qwen3_5", PHASE1_OUT_OF_SCOPE_IDENTITIES)
        self.assertFalse(any(case["model_type"] == "qwen3_5"
                             for case in source_scope_matrix()["execution_cases"]))
        for call in (lambda: source_family_for_identity("qwen3_5"),
                     lambda: resolve_checkpoint_route({"model_type": "qwen3_5"}, self.root),
                     lambda: self.engine.open(self.root, argv=list(ARGV))):
            with self.assertRaisesRegex(ValueError, "outside Phase 1 scope: qwen3_5"):
                call()
        self.assertNotIn("qwen3_5", self.engine.registered_model_types())
        self.assertEqual(source_family_for_identity("qwen3_5_moe"), "qwen35")

    def test_dense_identity_cannot_be_forced_through_an_moe_route(self):
        route = ModelRoute("qwen35", "qwen3_5_moe", "native", "fp32", "BF16", "float32")
        with self.assertRaisesRegex(ValueError, "outside Phase 1 scope|differs from checkpoint identity"):
            self.engine.open(self.root, route=route, argv=list(ARGV))


class OperatorBackendTests(unittest.TestCase):
    def test_backend_follows_each_operator_flag(self):
        namespace = SimpleNamespace
        self.assertEqual(publisher_operator_backend("glm5_next_text", namespace(public_runtime_library="x")),
                         "public-range")
        self.assertEqual(publisher_operator_backend("qwen3_5_moe", namespace(segment_runtime_library="x")),
                         "public-range")
        self.assertEqual(publisher_operator_backend("qwen3_5_moe", namespace(text_device="xpu")), "device")
        self.assertEqual(publisher_operator_backend("mixtral", namespace(device="cpu")), "device")
        self.assertEqual(publisher_operator_backend("mixtral", namespace()), "native")
        self.assertEqual(publisher_operator_backend("deepseek_v41", namespace(device="xpu")), "device")

    def test_gateway_and_universal_paths_share_one_parser(self):
        operator, args = parse_publisher_operator("mixtral", ["--checkpoint", "/m", *ARGV])
        self.assertEqual(operator.__name__, "openagent.adaptive_engine.native_family_cli")
        self.assertEqual((args.family, args.token_ids), ("mixtral", "[]"))


if __name__ == "__main__":
    unittest.main()
