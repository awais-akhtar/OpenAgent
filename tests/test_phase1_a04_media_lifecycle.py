from __future__ import annotations

import unittest

from openagent.adaptive_engine.phase1_a04_media_lifecycle import (
    FamilyMediaLifecycle,
    FamilyMediaLifecycleRegistry,
)


class Phase1A04MediaLifecycleTests(unittest.TestCase):
    def test_text_only_families_reject_media_ownership(self) -> None:
        with self.assertRaisesRegex(ValueError, "text-only family"):
            FamilyMediaLifecycle("olmoe")

    def test_qwen35_rejects_audio_without_an_admitted_tower(self) -> None:
        lifecycle = FamilyMediaLifecycle("qwen35")

        with self.assertRaisesRegex(ValueError, "no admitted consumer"):
            lifecycle.admit({"audio": b"audio"})

    def test_video_requires_finite_positive_rate(self) -> None:
        lifecycle = FamilyMediaLifecycle("kimi-k3")

        for fps in (True, 0, float("nan"), float("inf")):
            with self.subTest(fps=fps), self.assertRaisesRegex(ValueError, "finite positive fps"):
                lifecycle.admit({"video": b"12345678", "fps": fps})

    def test_raw_payload_count_includes_images_and_memoryview_bytes(self) -> None:
        lifecycle = FamilyMediaLifecycle("qwen35")
        image = memoryview(bytearray(32)).cast("I")
        video = memoryview(bytearray(24)).cast("I")

        image_id = lifecycle.admit({"images": [b"12345678", image]})
        image_frame = lifecycle.complete(image_id)
        self.assertEqual(image_frame.input_byte_count, 40)
        lifecycle.release(image_id)

        video_id = lifecycle.admit({"video": video, "fps": 24})
        video_frame = lifecycle.complete(video_id)
        self.assertEqual(video_frame.input_byte_count, 24)
        lifecycle.release(video_id)

    def test_processed_frame_obeys_transition_and_release_contract(self) -> None:
        lifecycle = FamilyMediaLifecycle("qwen4")
        frame_id = lifecycle.admit_processed(("image",), input_byte_count=64)

        lifecycle.on_stream_chunk(frame_id)
        frame = lifecycle.complete(frame_id)
        self.assertEqual(frame.state, "complete")
        self.assertEqual(frame.stream_chunks, 1)
        frame.state = "streaming"
        lifecycle.release(frame_id)
        self.assertEqual(lifecycle.active_count(), 0)

    def test_backend_error_preserves_primary_exception_and_terminal_frame(self) -> None:
        lifecycle = FamilyMediaLifecycle("inkling")
        frame_id = lifecycle.admit_processed(("audio",), input_byte_count=64)
        primary = RuntimeError("decoder failed")

        frame = lifecycle.error(frame_id, primary)

        self.assertEqual(frame.state, "error")
        self.assertTrue(any("media frame" in note for note in primary.__notes__))
        lifecycle.release(frame_id)

    def test_registry_close_attempts_every_handler_and_retains_retry(self) -> None:
        class Handler:
            def __init__(self, family, fail_first):
                self.family = family
                self.fail_first = fail_first
                self.calls = 0

            def close(self):
                self.calls += 1
                if self.fail_first and self.calls == 1:
                    raise RuntimeError(self.family + " close failed")

        registry = FamilyMediaLifecycleRegistry()
        failing = Handler("qwen35", True)
        healthy = Handler("qwen4", False)
        registry._handlers = {"qwen35": failing, "qwen4": healthy}

        with self.assertRaisesRegex(RuntimeError, "qwen35 close failed"):
            registry.close()

        self.assertEqual(healthy.calls, 1)
        self.assertEqual(set(registry._handlers), {"qwen35", "qwen4"})
        registry.close()
        self.assertEqual(failing.calls, 2)
        self.assertEqual(registry._handlers, {})

    def test_failed_media_frame_must_be_terminal_before_release(self) -> None:
        lifecycle = FamilyMediaLifecycle("inkling")
        frame_id = lifecycle.admit_processed(("audio",), input_byte_count=64)

        with self.assertRaisesRegex(RuntimeError, "cannot release active"):
            lifecycle.release(frame_id)
        lifecycle.cancel(frame_id)
        lifecycle.release(frame_id)

    def test_inkling_mtp_is_not_admitted_as_media(self) -> None:
        lifecycle = FamilyMediaLifecycle("inkling")

        with self.assertRaisesRegex(ValueError, "target-verified P05"):
            lifecycle.admit({"mtp_draft_tokens": [1, 2]})
        with self.assertRaisesRegex(ValueError, "no admitted consumer"):
            lifecycle.admit_processed(("mtp",), input_byte_count=8)

    def test_deepseek_dspark_and_mtp_are_not_media_payloads(self) -> None:
        lifecycle = FamilyMediaLifecycle("deepseek-v41")

        with self.assertRaisesRegex(ValueError, "target-verified P05"):
            lifecycle.admit({"mtp_draft_tokens": [1]})
        with self.assertRaisesRegex(ValueError, "owner-bound P12"):
            lifecycle.admit({"dspark_continuation": b"state"})
        for media_type in ("mtp", "dspark"):
            with self.subTest(media_type=media_type), self.assertRaisesRegex(
                    ValueError, "no admitted consumer"):
                lifecycle.admit_processed((media_type,), input_byte_count=8)


if __name__ == "__main__":
    unittest.main()
