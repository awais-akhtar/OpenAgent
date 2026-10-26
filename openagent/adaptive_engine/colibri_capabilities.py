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
        "scope": "synchronous configured-native scoring with cooperative cancellation on detected HTTP disconnect; messages accept only role/content text and exact type=text content blocks; extra message fields and multimodal blocks are rejected; persistent cache_slot is rejected",
    },
    "jev_systemone": {
        "status": "partial",
        "entrypoint": "/v1/systemone",
        "scope": "synchronous text request and response adapter with cooperative cancellation on detected HTTP disconnect",
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
        "scope": "text-only JSON and SSE subset with cooperative cancellation on detected HTTP disconnect; OpenAI chat and text completions emit protocol-specific deltas, terminal events and [DONE]; chat stream_options supports only include_usage; text completions reject messages and max_completion_tokens, while chat completions reject prompt; up to four bounded non-empty stop strings are encoded by the configured checkpoint tokenizer and applied to generated token prefixes; best_of is restricted to an exact integer value of 1; tools and unsupported controls/features are not implemented",
    },
    "anthropic_messages": {
        "status": "partial",
        "entrypoint": "/v1/messages",
        "scope": "text-only JSON and SSE subset with cooperative cancellation on detected HTTP disconnect; Messages SSE emits named message/content-block/terminal/error events and usage; up to four bounded non-empty stop_sequences are encoded by the configured checkpoint tokenizer, applied to generated token prefixes, and returned as stop_sequence when matched; stream_options, non-null tool_choice, non-empty tools, and non-null thinking are rejected or unavailable",
    },
    "gbnf_constrained_generation": {
        "status": "restricted",
        "entrypoint": "/v1/chat/completions",
        "scope": "non-text response_format is rejected; no grammar-constrained decoder",
    },
    "family_tool_protocols": {
        "status": "restricted",
        "entrypoint": "/v1/chat/completions, /v1/messages",
        "scope": "tool calls and thinking blocks are not admitted; non-empty tools arrays and non-null tool_choice are rejected; malformed non-array tools values are validation errors; empty tools arrays mean no tools and are accepted",
    },
    "ready_mux_engine_protocol": {
        "status": "not_admitted",
        "entrypoint": None,
        "scope": "no Colibri READY-gated mux or legacy stdin protocol is exposed; internal worker JSONL and HTTP routes are not substitutes and do not advertise CAPS/STAT/HWINFO/TIERS/EMAP or byte-counted mux framing",
    },
    "colibri_queue_contract": {
        "status": "partial",
        "entrypoint": "/health, /metrics",
        "scope": "OpenAI-compatible and Anthropic text JSON/SSE routes accept queue=false for immediate bounded admission or queue=true with optional queue_timeout (default 30 seconds, maximum 300); the native provider has one active slot and a FIFO of at most 64 waiters; overload and queue timeout return 429 with Retry-After: 1; successful responses expose X-Queue-Wait-Ms; /health and /metrics expose provider-owned active/waiting and admission counters plus wait histogram; priority, wait_timeout and wait_for_slot are explicitly rejected; distributed queue cancellation/transport semantics remain outside this subset",
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


def require_colibri_capability(name: str, *, operation: str = "operation") -> dict:
    """Admit only an explicitly exposed Colibri capability.

    Capability declarations are otherwise easy for a new caller to treat as
    documentation and bypass.  Keep the boundary in one helper so restricted
    and not-admitted protocols fail before allocating a provider, queue slot,
    decoder, or persistent state.  A detached record is returned for partial
    and implemented capabilities so callers cannot mutate the catalog.
    """
    if type(name) is not str or not name:
        raise ValueError("named Colibri capability required")
    record = _COLIBRI_CAPABILITIES.get(name)
    if record is None:
        raise ValueError("unknown Colibri capability: " + name)
    status = record.get("status")
    if status in ("restricted", "not_admitted"):
        scope = record.get("scope") or "capability is outside the admitted Colibri subset"
        raise ValueError(f"{operation} is not admitted for Colibri capability {name}: {scope}")
    if status not in ("partial", "implemented"):
        raise ValueError("invalid Colibri capability status: " + name)
    return deepcopy(record)
