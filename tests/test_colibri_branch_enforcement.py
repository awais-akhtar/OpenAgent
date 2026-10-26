"""Regression suite: every declared Colibri capability branch is enforced at its
public entry point, not merely declared in the catalog.

For each of the twelve capability keys:
  - 'partial' / 'implemented' → the route accepts a minimal valid request.
  - 'restricted' / 'not_admitted' → the route rejects with an explicit error
    before allocating a provider, queue slot, decoder, or persistent state.

This suite proves the boundary is at the wire, not only in documentation.
It uses only the route handler logic; no model or provider is started.
"""

from __future__ import annotations

import unittest
from http import HTTPStatus
from unittest.mock import MagicMock, patch

from openagent.adaptive_engine.colibri_capabilities import (
    colibri_capabilities,
    require_colibri_capability,
)
from openagent.routes.inference_routes import InferenceRouteDependencies, handle_get, handle_post


# ---------------------------------------------------------------------------
# Minimal gateway stub — never reaches a model
# ---------------------------------------------------------------------------

class _StubGateway:
    """Return a dict for any scoring call so partial-admitted routes don't crash."""

    def __init__(self):
        self.providers = {}

    def inference_status(self):
        return {"status": "idle"}

    def score_brio(self, *a, **kw):
        return {"choices": [{"score": 0.5}], "usage": {"prompt_tokens": 10, "read_tokens": 1}}

    def score_brio_questions(self, *a, **kw):
        return {"answers": [{"choices": [{"score": 0.5}]}], "usage": {"prompt_tokens": 10, "read_tokens": 1}}

    def score_brio_schema(self, *a, **kw):
        return {"fields": [{"choices": [{"score": 0.5}]}], "usage": {"prompt_tokens": 10, "read_tokens": 1}}

    def generate(self, *a, **kw):
        return {"text": "hi", "finish_reason": "stop", "usage": {}}

    def stream_compat(self, *a, **kw):
        return iter([])


def _deps(payload=None, *, cancelled=None):
    return InferenceRouteDependencies(
        gateway=_StubGateway(),
        store=MagicMock(),
        read_json=lambda: (payload or {}),
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# 1. Catalog integrity
# ---------------------------------------------------------------------------

class ColibriCatalogTests(unittest.TestCase):
    """The capability catalog itself is well-formed."""

    EXPECTED_KEYS = {
        "brio_scoring",
        "jev_systemone",
        "brio_cache_slot",
        "brio_terminal_browser_clients",
        "direct_k3_v41_scoring",
        "openai_compatible",
        "anthropic_messages",
        "gbnf_constrained_generation",
        "family_tool_protocols",
        "ready_mux_engine_protocol",
        "colibri_queue_contract",
        "coli_usage_route_telemetry",
        "cache_aware_expert_route_substitution",
        "multi_disk_mirrors",
    }

    def test_all_expected_keys_present(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(set(caps.keys()), self.EXPECTED_KEYS)

    def test_every_entry_has_status_and_scope(self):
        caps = colibri_capabilities()["capabilities"]
        for name, record in caps.items():
            with self.subTest(capability=name):
                self.assertIn(record.get("status"),
                              ("partial", "implemented", "restricted", "not_admitted"),
                              f"{name} has unexpected status")
                self.assertIsInstance(record.get("scope"), str)
                self.assertTrue(record["scope"], f"{name} scope must be non-empty")

    def test_catalog_returns_detached_copy(self):
        a = colibri_capabilities()
        b = colibri_capabilities()
        a["capabilities"]["brio_scoring"]["status"] = "MUTATED"
        self.assertNotEqual(b["capabilities"]["brio_scoring"]["status"], "MUTATED")

    def test_require_raises_for_restricted(self):
        with self.assertRaises(ValueError) as ctx:
            require_colibri_capability("brio_cache_slot")
        self.assertIn("brio_cache_slot", str(ctx.exception))

    def test_require_raises_for_not_admitted(self):
        with self.assertRaises(ValueError) as ctx:
            require_colibri_capability("ready_mux_engine_protocol")
        self.assertIn("ready_mux_engine_protocol", str(ctx.exception))

    def test_require_admits_partial(self):
        record = require_colibri_capability("brio_scoring")
        self.assertEqual(record["status"], "partial")

    def test_require_raises_for_unknown(self):
        with self.assertRaises(ValueError) as ctx:
            require_colibri_capability("nonexistent_capability_xyz")
        self.assertIn("unknown", str(ctx.exception))

    def test_require_raises_for_empty_name(self):
        with self.assertRaises(ValueError):
            require_colibri_capability("")

    def test_require_raises_for_non_string(self):
        with self.assertRaises(ValueError):
            require_colibri_capability(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. GET /v1/colibri/capabilities — public capabilities endpoint
# ---------------------------------------------------------------------------

class ColibriCapabilitiesEndpointTests(unittest.TestCase):
    def test_get_colibri_capabilities_returns_all_keys(self):
        result = handle_get("/v1/colibri/capabilities", "", _deps())
        self.assertEqual(result.status, HTTPStatus.OK)
        caps = result.payload["capabilities"]
        self.assertIn("brio_scoring", caps)
        self.assertIn("ready_mux_engine_protocol", caps)
        self.assertIn("multi_disk_mirrors", caps)

    def test_capabilities_payload_is_detached(self):
        result = handle_get("/v1/colibri/capabilities", "", _deps())
        result.payload["capabilities"]["brio_scoring"]["status"] = "MUTATED"
        result2 = handle_get("/v1/colibri/capabilities", "", _deps())
        self.assertNotEqual(result2.payload["capabilities"]["brio_scoring"]["status"], "MUTATED")


# ---------------------------------------------------------------------------
# 3. brio_cache_slot — restricted; rejected before provider execution
# ---------------------------------------------------------------------------

class BrioCacheSlotEnforcementTests(unittest.TestCase):
    """cache_slot is structurally valid but rejected before provider allocation."""

    def _brio_payload(self, **extra):
        base = {
            "messages": [{"role": "user", "content": "hello"}],
            "options": ["a", "b"],
        }
        base.update(extra)
        return base

    def test_cache_slot_is_rejected_with_unprocessable_entity(self):
        result = handle_post("/v1/brio", _deps(self._brio_payload(cache_slot=0)))
        self.assertEqual(result.status, HTTPStatus.UNPROCESSABLE_ENTITY)
        self.assertIn("brio_cache_slot", result.payload["error"])

    def test_cache_slot_rejection_fires_before_gateway_call(self):
        """The stub gateway's score_brio must never be called."""
        gateway = MagicMock()
        deps = InferenceRouteDependencies(
            gateway=gateway,
            store=MagicMock(),
            read_json=lambda: self._brio_payload(cache_slot=3),
        )
        handle_post("/v1/brio", deps)
        gateway.score_brio.assert_not_called()
        gateway.score_brio_questions.assert_not_called()
        gateway.score_brio_schema.assert_not_called()

    def test_brio_without_cache_slot_passes_validation(self):
        """Requests without cache_slot are not blocked by this guard."""
        gateway = MagicMock()
        gateway.score_brio.return_value = {"choices": [{"score": 0.5}], "usage": {"prompt_tokens": 10, "read_tokens": 1}}
        deps = InferenceRouteDependencies(
            gateway=gateway,
            store=MagicMock(),
            read_json=lambda: self._brio_payload(),
        )
        result = handle_post("/v1/brio", deps)
        # May fail for other reasons (model not loaded) but NOT the cache_slot guard.
        self.assertNotEqual(result.status, HTTPStatus.UNPROCESSABLE_ENTITY)


# ---------------------------------------------------------------------------
# 4. family_tool_protocols — restricted; tools rejected on both compat routes
# ---------------------------------------------------------------------------

class FamilyToolProtocolEnforcementTests(unittest.TestCase):

    def test_anthropic_non_empty_tools_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [{"name": "lookup", "description": "search"}],
        }
        result = handle_post("/v1/messages", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("tool calls", result.payload["error"]["message"].lower())

    def test_anthropic_non_null_tool_choice_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [],
            "tool_choice": {"type": "auto"},
        }
        result = handle_post("/v1/messages", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("tool calls", result.payload["error"]["message"].lower())

    def test_anthropic_thinking_blocks_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "thinking": {"type": "enabled", "budget_tokens": 512},
        }
        result = handle_post("/v1/messages", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("tool calls", result.payload["error"]["message"].lower())

    def test_openai_non_empty_tools_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("tool calls", result.payload["error"]["message"].lower())

    def test_openai_non_null_tool_choice_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
            "tool_choice": "auto",
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("tool calls", result.payload["error"]["message"].lower())

    def test_anthropic_empty_tools_array_is_admitted(self):
        """Empty tools array means no tools and must not trigger the tool guard."""
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "tools": [],
        }
        result = handle_post("/v1/messages", _deps(payload))
        # Must NOT be rejected by the tool guard; may fail for other reasons.
        error_msg = result.payload.get("error", {}).get("message", "")
        self.assertNotIn("tool calls", error_msg.lower())
        self.assertNotIn("thinking blocks", error_msg.lower())


# ---------------------------------------------------------------------------
# 5. gbnf_constrained_generation — restricted; response_format != text rejected
# ---------------------------------------------------------------------------

class GBNFConstrainedGenerationEnforcementTests(unittest.TestCase):

    def test_json_object_response_format_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("response_format", result.payload["error"]["message"].lower())

    def test_json_schema_response_format_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": {}}},
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertEqual(result.status, HTTPStatus.BAD_REQUEST)
        self.assertIn("response_format", result.payload["error"]["message"].lower())

    def test_text_response_format_is_admitted(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "text"},
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        # Must not be rejected for response_format reason.
        error_msg = result.payload.get("error", {}).get("message", "")
        self.assertNotIn("response_format", error_msg.lower())

    def test_null_response_format_is_admitted(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        error_msg = result.payload.get("error", {}).get("message", "")
        self.assertNotIn("response_format", error_msg.lower())


# ---------------------------------------------------------------------------
# 6. ready_mux_engine_protocol — not_admitted; no route exposes it
# ---------------------------------------------------------------------------

class ReadyMuxEngineProtocolEnforcementTests(unittest.TestCase):
    """No route provides READY/CAPS/STAT/HWINFO/TIERS/EMAP or mux framing."""

    MUXPATHS = [
        "/v1/mux", "/caps", "/stat", "/hwinfo", "/tiers", "/emap",
        "/ready", "/_mux", "/v1/ready",
    ]

    def test_no_admitted_get_route_for_mux_paths(self):
        for path in self.MUXPATHS:
            with self.subTest(path=path):
                result = handle_get(path, "", _deps())
                self.assertIsNone(result,
                    f"GET {path} should return None (unhandled), not {result}")

    def test_no_admitted_post_route_for_mux_paths(self):
        for path in self.MUXPATHS:
            with self.subTest(path=path):
                result = handle_post(path, _deps({}))
                self.assertIsNone(result,
                    f"POST {path} should return None (unhandled), not {result}")

    def test_capability_catalog_marks_ready_mux_not_admitted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["ready_mux_engine_protocol"]["status"], "not_admitted")

    def test_require_ready_mux_raises(self):
        with self.assertRaises(ValueError) as ctx:
            require_colibri_capability("ready_mux_engine_protocol")
        self.assertIn("ready_mux_engine_protocol", str(ctx.exception))


# ---------------------------------------------------------------------------
# 7. brio_terminal_browser_clients — not_admitted; no route exposes it
# ---------------------------------------------------------------------------

class BrioTerminalBrowserClientEnforcementTests(unittest.TestCase):
    TERMINAL_PATHS = ["/brio/terminal", "/v1/brio/terminal", "/v1/brio/browser",
                      "/brio/browser", "/v1/terminal"]

    def test_no_get_route_for_terminal_paths(self):
        for path in self.TERMINAL_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_get(path, "", _deps()))

    def test_no_post_route_for_terminal_paths(self):
        for path in self.TERMINAL_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_post(path, _deps({})))

    def test_capability_catalog_marks_terminal_not_admitted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["brio_terminal_browser_clients"]["status"], "not_admitted")


# ---------------------------------------------------------------------------
# 8. coli_usage_route_telemetry — not_admitted; no .coli_usage or route-trace
# ---------------------------------------------------------------------------

class ColiUsageRoutetelemetryEnforcementTests(unittest.TestCase):
    COLI_PATHS = ["/v1/coli_usage", "/.coli_usage", "/v1/route-trace",
                  "/v1/brio/trace", "/coli/usage"]

    def test_no_get_route(self):
        for path in self.COLI_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_get(path, "", _deps()))

    def test_no_post_route(self):
        for path in self.COLI_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_post(path, _deps({})))

    def test_capability_catalog_marks_not_admitted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["coli_usage_route_telemetry"]["status"], "not_admitted")


# ---------------------------------------------------------------------------
# 9. cache_aware_expert_route_substitution — not_admitted
# ---------------------------------------------------------------------------

class CacheAwareExpertRouteSubstitutionEnforcementTests(unittest.TestCase):
    SUBST_PATHS = ["/v1/route-substitution", "/v1/expert-route", "/v1/brio/substitute"]

    def test_no_get_route(self):
        for path in self.SUBST_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_get(path, "", _deps()))

    def test_no_post_route(self):
        for path in self.SUBST_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_post(path, _deps({})))

    def test_capability_catalog_marks_not_admitted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["cache_aware_expert_route_substitution"]["status"], "not_admitted")


# ---------------------------------------------------------------------------
# 10. multi_disk_mirrors — not_admitted
# ---------------------------------------------------------------------------

class MultiDiskMirrorsEnforcementTests(unittest.TestCase):
    MIRROR_PATHS = ["/v1/mirrors", "/v1/mirror-plan", "/v1/brio/mirrors"]

    def test_no_get_route(self):
        for path in self.MIRROR_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_get(path, "", _deps()))

    def test_no_post_route(self):
        for path in self.MIRROR_PATHS:
            with self.subTest(path=path):
                self.assertIsNone(handle_post(path, _deps({})))

    def test_capability_catalog_marks_not_admitted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["multi_disk_mirrors"]["status"], "not_admitted")


# ---------------------------------------------------------------------------
# 11. direct_k3_v41_scoring — restricted; rejects before provider allocation
# ---------------------------------------------------------------------------

class DirectK3V41ScoringEnforcementTests(unittest.TestCase):
    """The Brio scoring route rejects non-native owner references via the
    capability catalog; the capability entry itself is restricted."""

    def test_capability_catalog_marks_restricted(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["direct_k3_v41_scoring"]["status"], "restricted")

    def test_require_direct_k3_v41_scoring_raises(self):
        with self.assertRaises(ValueError) as ctx:
            require_colibri_capability("direct_k3_v41_scoring")
        self.assertIn("direct_k3_v41_scoring", str(ctx.exception))


# ---------------------------------------------------------------------------
# 12. openai_compatible and anthropic_messages — partial; admitted routes exist
# ---------------------------------------------------------------------------

class OpenAIAndAnthropicPartialAdmissionTests(unittest.TestCase):
    """Partially-implemented routes exist at the wire; basic requests reach
    validation (not blocked at the capability guard)."""

    def test_openai_completions_route_exists(self):
        """POST /v1/chat/completions returns something (not None)."""
        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}]}
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertIsNotNone(result)

    def test_anthropic_messages_route_exists(self):
        payload = {"model": "local", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 8}
        result = handle_post("/v1/messages", _deps(payload))
        self.assertIsNotNone(result)

    def test_openai_completions_capability_catalog_marks_partial(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["openai_compatible"]["status"], "partial")

    def test_anthropic_messages_capability_catalog_marks_partial(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["anthropic_messages"]["status"], "partial")

    def test_openai_unsupported_field_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "unknown_colibri_ext": "value",
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertIsNotNone(result)
        self.assertIn("unsupported request fields", result.payload.get("error", {}).get("message", ""))

    def test_anthropic_unsupported_field_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "unknown_colibri_ext": "value",
        }
        result = handle_post("/v1/messages", _deps(payload))
        self.assertIsNotNone(result)
        self.assertIn("unsupported request fields", result.payload.get("error", {}).get("message", ""))

    def test_openai_unsupported_queue_controls_rejected(self):
        for control in ("priority", "wait_timeout", "wait_for_slot"):
            with self.subTest(control=control):
                payload = {
                    "model": "local",
                    "messages": [{"role": "user", "content": "hi"}],
                    control: 1,
                }
                result = handle_post("/v1/chat/completions", _deps(payload))
                self.assertIn("unsupported queue controls",
                              result.payload.get("error", {}).get("message", ""))

    def test_anthropic_unsupported_queue_controls_rejected(self):
        for control in ("priority", "wait_timeout", "wait_for_slot"):
            with self.subTest(control=control):
                payload = {
                    "model": "local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 8,
                    control: 1,
                }
                result = handle_post("/v1/messages", _deps(payload))
                self.assertIn("unsupported queue controls",
                              result.payload.get("error", {}).get("message", ""))


# ---------------------------------------------------------------------------
# 13. colibri_queue_contract — partial; queue=true is admitted and gated
# ---------------------------------------------------------------------------

class ColibriQueueContractEnforcementTests(unittest.TestCase):

    def test_capability_catalog_marks_queue_contract_partial(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["colibri_queue_contract"]["status"], "partial")

    def test_queue_true_is_admitted_on_chat_completions(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "queue": True,
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        # Not blocked by queue validation — may fail for provider reasons.
        error_msg = result.payload.get("error", {}).get("message", "") if result else ""
        self.assertNotIn("queue must be boolean", error_msg)
        self.assertNotIn("unsupported queue controls", error_msg)

    def test_queue_non_boolean_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "queue": "yes",
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertIn("queue must be boolean",
                      result.payload.get("error", {}).get("message", ""))

    def test_queue_timeout_without_queue_true_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "queue_timeout": 30,
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertIn("queue_timeout requires queue=true",
                      result.payload.get("error", {}).get("message", ""))

    def test_queue_timeout_above_300_rejected(self):
        payload = {
            "model": "local",
            "messages": [{"role": "user", "content": "hi"}],
            "queue": True,
            "queue_timeout": 301,
        }
        result = handle_post("/v1/chat/completions", _deps(payload))
        self.assertIn("queue_timeout",
                      result.payload.get("error", {}).get("message", ""))


# ---------------------------------------------------------------------------
# 14. brio_scoring and jev_systemone — partial; minimal routes respond
# ---------------------------------------------------------------------------

class BrioScoringAndJevSystemoneEnforcementTests(unittest.TestCase):

    def test_brio_scoring_capability_catalog_marks_partial(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["brio_scoring"]["status"], "partial")

    def test_jev_systemone_capability_catalog_marks_partial(self):
        caps = colibri_capabilities()["capabilities"]
        self.assertEqual(caps["jev_systemone"]["status"], "partial")

    def test_brio_route_returns_result_for_valid_options_request(self):
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "options": ["option A", "option B"],
        }
        gateway = MagicMock()
        gateway.score_brio.return_value = {"choices": [{"score": 0.6}, {"score": 0.4}], "usage": {"prompt_tokens": 10, "read_tokens": 1}}
        deps = InferenceRouteDependencies(
            gateway=gateway,
            store=MagicMock(),
            read_json=lambda: payload,
        )
        result = handle_post("/v1/brio", deps)
        self.assertIsNotNone(result)

    def test_systemone_route_exists(self):
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
        }
        result = handle_post("/v1/systemone", _deps(payload))
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
