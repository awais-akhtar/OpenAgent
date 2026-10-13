"""Public, conservative inventory of the Colibri-specific serving subset.

This is a declaration of the OpenAgent wire surface, not a claim of Colibri
compatibility. Implemented subset routes still have narrower request-level
guards; absent protocols/features are reported as unavailable here.
"""

from copy import deepcopy


_COLIBRI_CAPABILITIES = {
    "brio_scoring": {
        "status": "partial",
        "entrypoint": "/v1/brio",
        "scope": "synchronous configured-native scoring; persistent cache_slot is rejected",
    },
    "jev_systemone": {
        "status": "partial",
        "entrypoint": "/v1/systemone",
        "scope": "synchronous text request and response adapter only",
    },
    "brio_cache_slot": {
        "status": "restricted",
        "entrypoint": "/v1/brio",
        "scope": "requests specifying cache_slot are rejected; no cross-request KV history",
    },
    "brio_terminal_browser_clients": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "no Colibri terminal or browser client contract is exposed",
    },
    "direct_k3_v41_scoring": {
        "status": "restricted",
        "entrypoint": "/v1/brio",
        "scope": "provider rejects non-native owners pending scoring lifecycle composition",
    },
    "openai_compatible": {
        "status": "partial",
        "entrypoint": "/v1/chat/completions, /v1/completions",
        "scope": "synchronous text-only subset; unsupported controls and features are rejected",
    },
    "anthropic_messages": {
        "status": "partial",
        "entrypoint": "/v1/messages",
        "scope": "synchronous text-only subset; no streaming or tool-use parity",
    },
    "gbnf_constrained_generation": {
        "status": "restricted",
        "entrypoint": "/v1/chat/completions",
        "scope": "non-text response_format is rejected; no grammar-constrained decoder",
    },
    "family_tool_protocols": {
        "status": "restricted",
        "entrypoint": "/v1/chat/completions, /v1/messages",
        "scope": "tool definitions and tool choice are rejected",
    },
    "ready_mux_engine_protocol": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "no Colibri READY-gated mux or legacy stdin protocol is exposed",
    },
    "colibri_queue_contract": {
        "status": "restricted",
        "entrypoint": "/health, /metrics",
        "scope": "no Colibri request-queue/admission contract; compatibility calls remain synchronous and expose no queue-wait header, 429/Retry-After policy, or Colibri counters/histograms; /health and /metrics expose only real native scheduler gauges when available",
    },
    "coli_usage_route_telemetry": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "no .coli_usage persistence or Colibri route-trace format is exposed",
    },
    "cache_aware_expert_route_substitution": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "no Colibri top-K route substitution contract is exposed",
    },
    "multi_disk_mirrors": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "mirror identity, fallback, and partial-mirror planning are unavailable",
    },
}


def colibri_capabilities() -> dict:
    """Return a detached status map for the public Colibri scope endpoint."""
    return deepcopy({"compatibility": "partial-openagent-subset", "capabilities": _COLIBRI_CAPABILITIES})
