from __future__ import annotations

import base64
import os
import unittest
from unittest.mock import patch

from openagent.model_gateway import ModelGateway, ProviderConfigurationError, ProviderUnavailableError
from openagent.science.multimodal.image import observe_with_gateway


class ModelGatewayVisionTests(unittest.TestCase):
    def test_ollama_uses_native_message_images_and_reports_provenance(self) -> None:
        gateway = ModelGateway()
        gateway.providers["ollama"].resolve_model = lambda requested: requested or "llava:latest"
        captured = {}

        def fake_post(url, payload, *, headers):
            captured.update(url=url, payload=payload, headers=headers)
            return {"message": {"content": '{"observations":[{"statement":"bright edge","confidence":0.8}]}'}}

        with patch("openagent.model_gateway.post_json", side_effect=fake_post):
            result = gateway.complete_vision(
                [{"role": "user", "content": "Inspect this image."}],
                b"\x89PNG\r\n",
                provider_id="ollama",
                model="llava:latest",
                media_type="image/png",
                max_output_tokens=120,
            )

        message = captured["payload"]["messages"][0]
        self.assertEqual(message["content"], "Inspect this image.")
        self.assertEqual(message["images"], [base64.b64encode(b"\x89PNG\r\n").decode("ascii")])
        self.assertNotIn("data:image", str(message["content"]))
        self.assertEqual(captured["payload"]["options"]["num_predict"], 120)
        self.assertEqual(result.provider, "ollama")
        self.assertEqual(result.model, "llava:latest")
        self.assertEqual(result.metadata["image_transport"], "ollama-message-images")
        self.assertFalse(result.metadata["external_approved"])

    def test_openai_uses_image_url_content_and_requires_external_approval(self) -> None:
        environment = {
            "OPENAGENT_OPENAI_BASE_URL": "https://vision.example/v1",
            "OPENAGENT_OPENAI_API_KEY": "secret",
            "OPENAGENT_OPENAI_MODEL": "vision-model",
        }
        with patch.dict(os.environ, environment, clear=False):
            gateway = ModelGateway()
        with self.assertRaisesRegex(ProviderConfigurationError, "crosses the local data boundary"):
            gateway.complete_vision(
                [{"role": "user", "content": "Inspect."}],
                b"jpeg",
                provider_id="openai-compatible",
                model="vision-model",
                media_type="image/jpeg",
            )

        captured = {}

        def fake_post(url, payload, *, headers):
            captured.update(url=url, payload=payload, headers=headers)
            return {"choices": [{"message": {"content": "vision answer"}}]}

        with patch("openagent.model_gateway.post_json", side_effect=fake_post):
            result = gateway.complete_vision(
                [{"role": "system", "content": "Be precise."}, {"role": "user", "content": "Inspect."}],
                b"jpeg",
                provider_id="openai-compatible",
                model="vision-model",
                media_type="image/jpeg",
                allow_external=True,
                max_output_tokens=75,
            )

        content = captured["payload"]["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Inspect."})
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(captured["headers"], {"Authorization": "Bearer secret"})
        self.assertEqual(captured["payload"]["max_tokens"], 75)
        self.assertEqual(result.metadata["provider"], "openai-compatible")
        self.assertEqual(result.metadata["model"], "vision-model")
        self.assertTrue(result.metadata["external_approved"])

    def test_text_only_provider_is_rejected_without_a_request(self) -> None:
        gateway = ModelGateway()
        with patch("openagent.model_gateway.post_json") as request:
            with self.assertRaisesRegex(ProviderUnavailableError, "does not support native vision"):
                gateway.complete_vision(
                    [{"role": "user", "content": "Inspect."}],
                    b"image",
                    provider_id="local",
                )
        request.assert_not_called()

    def test_scientific_adapter_uses_only_complete_vision(self) -> None:
        class VisionGateway:
            def __init__(self):
                self.vision_calls = []

            def complete(self, *args, **kwargs):
                raise AssertionError("plain text completion must not be used")

            def complete_vision(self, messages, image, **kwargs):
                self.vision_calls.append((messages, image, kwargs))
                return type(
                    "Result",
                    (),
                    {
                        "provider": "ollama",
                        "model": "llava:latest",
                        "content": '{"observations":[{"statement":"visible line","region":"center","confidence":0.9,"evidence":"pixels"}],"limitations":[]}',
                    },
                )()

        gateway = VisionGateway()
        result = observe_with_gateway(b"raw-image", gateway=gateway, model="llava:latest", provider_id="ollama")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["provider"], "ollama")
        self.assertEqual(gateway.vision_calls[0][1], b"raw-image")
        prompt = gateway.vision_calls[0][0][-1]["content"]
        self.assertNotIn("base64", prompt)
        self.assertNotIn("image_data_uri", prompt)


if __name__ == "__main__":
    unittest.main()
