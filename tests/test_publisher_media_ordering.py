"""Publisher media control-token placement regressions."""
from contextlib import nullcontext
import sys
from types import ModuleType
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from openagent.adaptive_engine.publisher_provider import (
    NativePublisherProvider, ProviderConfigurationError,
)
from openagent.adaptive_engine.publisher_media_inputs import PublisherVideoInput
import numpy as np


class PublisherMediaOrderingTests(unittest.TestCase):
    def test_cross_modality_order_is_rejected_before_processor_dispatch(self):
        video = PublisherVideoInput(
            (np.zeros((1, 1, 3), dtype=np.uint8),), (0,), 1, 1.0)
        still = np.zeros((1, 1, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ProviderConfigurationError,
                'one media modality per request'):
            NativePublisherProvider._validate_media_modality_order(
                'qwen4_exp', (still, video))

        # K3's typed media path preserves one shared input order, so its
        # image/video mixture remains eligible for its own processor.
        NativePublisherProvider._validate_media_modality_order(
            'kimi_k3', (still, video))

    def test_configured_video_placeholder_is_rejected_without_video_attachment(self):
        provider = NativePublisherProvider()
        config = {
            'family': 'deepseek_v41',
            'model': 'model',
            'image_placeholder': '<image>',
            'video_placeholder': '<video>',
            'video_token_id': 17,
            'chat_template': '{{ messages[0].content }}',
            'template_variables': {},
        }
        args = SimpleNamespace(image_token_id=16, image_mib=1, max_tokens=8)
        class Template:
            def render(self, **_variables):
                return 'rendered prompt'

        class Environment:
            def __init__(self, **_options):
                self.globals = {}

            def from_string(self, _template):
                return Template()

        jinja = ModuleType('jinja2')
        jinja.StrictUndefined = object
        jinja.TemplateError = type('TemplateError', (Exception,), {})
        sandbox = ModuleType('jinja2.sandbox')
        sandbox.SandboxedEnvironment = Environment
        with patch.object(provider, '_configuration', return_value=config), \
             patch.object(provider, '_operator', return_value=(object(), args)), \
             patch.object(provider, '_request_lock', return_value=nullcontext()), \
             patch.dict(sys.modules, {'jinja2': jinja, 'jinja2.sandbox': sandbox}):
            with self.assertRaisesRegex(ProviderConfigurationError,
                    'vision placeholder must be inserted by the gateway'):
                provider._run(
                    [{'role': 'user', 'content': 'Summarize this: <video>'}],
                    image=b'encoded-image',
                )


if __name__ == '__main__':
    unittest.main()
