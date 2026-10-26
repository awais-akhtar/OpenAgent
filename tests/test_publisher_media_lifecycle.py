from __future__ import annotations

import unittest
from unittest.mock import Mock

import numpy as np

from openagent.adaptive_engine.execution_service import ExecutionCancelled
from openagent.adaptive_engine.publisher_provider import NativePublisherProvider
from openagent.model_gateway import ProviderConfigurationError


class PublisherMediaLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.provider = NativePublisherProvider()
        self.provider._configuration = lambda: {"family": "qwen4_exp"}
        self.message = [{"role": "user", "content": "describe this"}]
        self.image = np.zeros((2, 3, 3), dtype=np.uint8)

    def _lifecycle(self):
        lifecycle = Mock()
        lifecycle.admit_processed.return_value = "frame-1"
        return lifecycle

    def test_media_frame_tracks_native_dispatch_and_completes(self):
        lifecycle = self._lifecycle()
        emit = Mock()

        def run(_messages, **kwargs):
            kwargs["_media_chunk_callback"]()
            kwargs["emit"]("token event")
            return "result"

        self.provider._run_impl = run
        self.provider._media_lifecycle_registry.enable = Mock(return_value=lifecycle)
        result = self.provider._run(
            self.message,
            media_items=[self.image],
            emit=emit,
            max_output_tokens=4,
        )

        self.assertEqual(result, "result")
        self.provider._media_lifecycle_registry.enable.assert_called_once_with("qwen4")
        lifecycle.admit_processed.assert_called_once_with(
            ("image",), input_byte_count=self.image.shape[0] * self.image.shape[1] * 16
        )
        self.assertEqual(lifecycle.on_stream_chunk.call_count, 1)
        emit.assert_called_once_with("token event")
        lifecycle.complete.assert_called_once_with("frame-1")
        lifecycle.release.assert_called_once_with("frame-1")

    def test_media_cancellation_marks_cancelled_and_releases(self):
        lifecycle = self._lifecycle()

        def run(_messages, **_kwargs):
            _kwargs["_media_chunk_callback"]()
            raise ExecutionCancelled("cancelled by caller")

        self.provider._run_impl = run
        self.provider._media_lifecycle_registry.enable = Mock(return_value=lifecycle)
        with self.assertRaises(ExecutionCancelled):
            self.provider._run(
                self.message,
                media_items=[self.image],
                cancelled=lambda: True,
            )

        lifecycle.cancel.assert_called_once_with("frame-1", "cancelled by caller")
        lifecycle.error.assert_not_called()
        lifecycle.release.assert_called_once_with("frame-1")

    def test_backend_error_marks_error_and_releases(self):
        lifecycle = self._lifecycle()
        failure = RuntimeError("backend failed")

        def run(_messages, **_kwargs):
            _kwargs["_media_chunk_callback"]()
            raise failure

        self.provider._run_impl = run
        self.provider._media_lifecycle_registry.enable = Mock(return_value=lifecycle)
        with self.assertRaisesRegex(RuntimeError, "backend failed"):
            self.provider._run(self.message, media_items=[self.image])

        lifecycle.error.assert_called_once_with("frame-1", failure)
        lifecycle.cancel.assert_not_called()
        lifecycle.release.assert_called_once_with("frame-1")

    def test_pre_dispatch_error_does_not_claim_media_was_consumed(self):
        lifecycle = self._lifecycle()
        failure = ValueError("request rejected before decoder dispatch")

        def run(_messages, **_kwargs):
            raise failure

        self.provider._run_impl = run
        self.provider._media_lifecycle_registry.enable = Mock(return_value=lifecycle)
        with self.assertRaisesRegex(ValueError, "before decoder dispatch"):
            self.provider._run(self.message, media_items=[self.image])

        lifecycle.on_stream_chunk.assert_not_called()
        lifecycle.error.assert_called_once_with("frame-1", failure)
        lifecycle.release.assert_called_once_with("frame-1")

    def test_unsupported_cross_modality_request_fails_before_lifecycle_admission(self):
        from openagent.adaptive_engine.publisher_media_inputs import PublisherVideoInput

        video = PublisherVideoInput(
            (np.zeros((2, 3, 3), dtype=np.uint8),), (0,), 1, 1.0)
        registry = Mock()
        self.provider._media_lifecycle_registry = registry
        self.provider._run_impl = Mock(side_effect=AssertionError("decoder path must not run"))

        with self.assertRaisesRegex(ProviderConfigurationError,
                                    "one media modality per request"):
            self.provider._run(self.message, media_items=[self.image, video])

        registry.enable.assert_not_called()
        self.provider._run_impl.assert_not_called()

    def test_audio_waveform_is_rejected_at_gateway_media_boundary(self):
        waveform = np.zeros((16000,), dtype=np.float32)

        with self.assertRaisesRegex(ProviderConfigurationError,
                                    "sampling-rate and placeholder-order contracts are required"):
            self.provider._validate_decoded_media([waveform], max_output_tokens=4)

    def test_real_provider_registry_reuses_family_owner_and_releases_frames(self):
        self.provider._run_impl = lambda _messages, **kwargs: (
            kwargs["_media_chunk_callback"](), "result"
        )[1]

        self.assertEqual(
            self.provider._run(self.message, media_items=[self.image]), "result"
        )
        self.assertEqual(
            self.provider._run(self.message, media_items=[self.image]), "result"
        )

        lifecycle = self.provider._media_lifecycle_registry.require("qwen4")
        self.assertEqual(lifecycle.active_count(), 0)
        self.assertEqual(self.provider._media_lifecycle_registry.active_counts(), {
            "qwen4": 0,
        })

    def test_provider_close_shuts_registry_even_when_decoder_cleanup_fails(self):
        failure = RuntimeError("decoder close failed")
        registry = Mock()
        self.provider._media_lifecycle_registry = registry
        self.provider._release_session = Mock(side_effect=failure)

        with self.assertRaisesRegex(RuntimeError, "decoder close failed"):
            self.provider.close()

        self.assertTrue(self.provider._closed)
        registry.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
