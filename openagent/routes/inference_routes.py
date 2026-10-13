"""Adaptive CPU inference engine route contracts and handlers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
import math
from time import time
from typing import Any

from ..model_gateway import ModelGateway, ProviderConfigurationError, ProviderUnavailableError
from ..models import AuditEvent
from ..storage import OpenAgentStore
from . import JsonRouteResult
from ._contracts import route_contract


ROUTE_GROUP = "inference"
ROUTES = (
    ("GET", "/api/inference/status"),
    ("GET", "/health"),
    ("GET", "/metrics"),
    ("GET", "/v1/models"),
    ("GET", "/v1/models/{model}"),
    ("GET", "/v1/colibri/capabilities"),
    ("POST", "/v1/brio"),
    ("POST", "/v1/systemone"),
    ("POST", "/v1/chat/completions"),
    ("POST", "/v1/completions"),
    ("POST", "/v1/messages"),
    ("POST", "/api/inference/benchmark"),
    ("POST", "/api/inference/cache/clear"),
    ("POST", "/api/inference/warmup"),
)


def contract() -> dict[str, Any]:
    return route_contract(ROUTE_GROUP, ROUTES)


@dataclass(frozen=True)
class InferenceRouteDependencies:
    gateway: ModelGateway
    store: OpenAgentStore
    read_json: Any


def handle_get(path: str, query: str, deps: InferenceRouteDependencies) -> JsonRouteResult | None:
    if path == "/v1/colibri/capabilities":
        # This endpoint reports the deliberately narrower local subset. It is
        # not a Colibri compatibility assertion; each route keeps its own
        # request validation and explicit unsupported-feature errors.
        from ..adaptive_engine.colibri_capabilities import colibri_capabilities
        return JsonRouteResult(colibri_capabilities())
    if path == "/api/inference/status":
        return JsonRouteResult(deps.gateway.inference_status())
    if path == "/metrics":
        return JsonRouteResult({"_prometheus": _prometheus_metrics(deps.gateway)}, HTTPStatus.OK)
    if path == "/health":
        native = deps.gateway.providers.get("native-publisher")
        state = {} if native is None else native.status()
        runtime = state.get("execution_runtime")
        scheduler = runtime.get("scheduler") if type(runtime) is dict else None
        queueing = type(scheduler) is dict
        return JsonRouteResult({"status": "ok", "scheduler": {
            "active": scheduler.get("active") if queueing else int(bool(state.get("active"))),
            "queued": scheduler.get("waiting") if queueing else None,
            "capacity": scheduler.get("max_active") if queueing else None,
            "max_queue": scheduler.get("max_waiting") if queueing else None,
            "reserved_bytes": scheduler.get("reserved_bytes") if queueing else None,
            "max_bytes": scheduler.get("max_bytes") if queueing else None,
            "queueing": queueing,
            "queue_metrics_available": queueing,
            "native_provider_configured": native is not None}}, HTTPStatus.OK)
    if path == "/v1/models" or path.startswith("/v1/models/"):
        provider = deps.gateway.providers.get("native-publisher")
        if provider is None:
            return JsonRouteResult({"error": {"message": "native publisher is not configured", "type": "server_error"}},
                                   HTTPStatus.SERVICE_UNAVAILABLE)
        try:
            config = provider._configuration()
        except (ProviderConfigurationError, ProviderUnavailableError, OSError, ValueError) as exc:
            return JsonRouteResult({"error": {"message": str(exc), "type": "server_error"}},
                                   HTTPStatus.SERVICE_UNAVAILABLE)
        model_id = config["model"]
        requested_id = path.removeprefix("/v1/models/") if path != "/v1/models" else None
        try:
            from urllib.parse import unquote
            requested_id = None if requested_id is None else unquote(requested_id)
        except Exception:
            requested_id = ""
        if requested_id is not None and requested_id != model_id:
            return JsonRouteResult({"error": {"message": "model not found", "type": "invalid_request_error"}},
                                   HTTPStatus.NOT_FOUND)
        models = [{"id": model_id, "object": "model", "created": int(datetime.now(timezone.utc).timestamp()),
                  "owned_by": "openagent-local"}]
        if path != "/v1/models":
            return JsonRouteResult(models[0])
        return JsonRouteResult({"object": "list", "data": models})
    return None


def handle_post(path: str, deps: InferenceRouteDependencies) -> JsonRouteResult | None:
    if path not in {
        "/v1/brio",
        "/v1/systemone",
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/messages",
        "/api/inference/benchmark",
        "/api/inference/cache/clear",
        "/api/inference/warmup",
    }:
        return None
    payload = deps.read_json()
    if path in {"/v1/chat/completions", "/v1/completions", "/v1/messages"}:
        return _handle_compat_completion(path, payload, deps)
    if path == "/v1/systemone":
        if (type(payload) is not dict or set(payload) - {"model", "state", "questions"} or
                "state" not in payload or
                type(payload.get("questions")) is not dict or
                not 1 <= len(payload["questions"]) <= 64 or
                ("model" in payload and type(payload["model"]) is not str)):
            return JsonRouteResult({"error": "invalid systemone request"}, HTTPStatus.UNPROCESSABLE_ENTITY)
        entries = []
        metadata = []
        question_bytes = 0
        try:
            state = deps.gateway.serialize_jev_state(payload["state"])
            if not state:
                raise ValueError('systemone state cannot be empty')
            for name, spec in payload["questions"].items():
                if (type(name) is not str or not name.strip() or type(spec) is not dict or
                        set(spec) - {"type", "instructions", "criteria"} or
                        "instructions" not in spec or
                        len(name.encode('utf-8')) > 1024):
                    raise ValueError('each systemone question needs a name, type and instructions')
                kind = spec.get("type")
                criteria = spec.get("criteria")
                instructions = deps.gateway.serialize_jev_state(spec["instructions"])
                if not instructions.strip():
                    raise ValueError('question instructions cannot be empty')
                current_bytes = len(instructions.encode('utf-8'))
                if kind == "noul":
                    if criteria is not None and (type(criteria) is not dict or set(criteria) - {"true", "false"} or
                            any(type(value) is not str for value in criteria.values())):
                        raise ValueError('noul criteria must contain optional true/false descriptions')
                    if criteria:
                        current_bytes += sum(len(value.encode('utf-8')) for value in criteria.values())
                    if current_bytes > 1024**2 or question_bytes + current_bytes > 1024**2:
                        raise ValueError('systemone question text exceeds the 1 MiB aggregate limit')
                    prompt = instructions
                    if criteria:
                        prompt += '\nyes: '+criteria.get('true', '')+'\nno: '+criteria.get('false', '')
                    prompt += '\nAnswer yes or no.'
                    options = ["yes", "no"]
                    details = {"type": kind}
                elif kind == "choice":
                    if (type(criteria) is not dict or not 2 <= len(criteria) <= 255 or
                            any(type(label) is not str or not label.strip() or type(description) is not str
                                for label, description in criteria.items())):
                        raise ValueError('choice criteria require 2 to 255 label/description pairs')
                    current_bytes += sum(len(label.encode('utf-8')) + len(description.encode('utf-8'))
                                         for label, description in criteria.items())
                    if current_bytes > 1024**2 or question_bytes + current_bytes > 1024**2:
                        raise ValueError('systemone question text exceeds the 1 MiB aggregate limit')
                    options = list(criteria)
                    prompt = instructions+'\n'+'\n'.join(label+': '+criteria[label] for label in options)
                    details = {"type": kind, "options": options}
                elif kind == "score":
                    if (type(criteria) is not list or not 2 <= len(criteria) <= 10 or
                            any(type(description) is not str for description in criteria)):
                        raise ValueError('score criteria require 2 to 10 level descriptions')
                    current_bytes += sum(len(description.encode('utf-8')) for description in criteria)
                    if current_bytes > 1024**2 or question_bytes + current_bytes > 1024**2:
                        raise ValueError('systemone question text exceeds the 1 MiB aggregate limit')
                    options = [str(index) for index in range(1, len(criteria)+1)]
                    prompt = instructions+'\n'+'\n'.join(
                        option+': '+description for option, description in zip(options, criteria))
                    prompt += '\nAnswer with the number.'
                    details = {"type": kind, "legend": dict(zip(options, criteria))}
                else:
                    raise ValueError('systemone question type must be noul, choice or score')
                current_bytes = len(prompt.encode('utf-8'))
                question_bytes += current_bytes
                if question_bytes > 1024**2:
                    raise ValueError('systemone question text exceeds the 1 MiB aggregate limit')
                entries.append({"question": prompt, "options": options})
                metadata.append((name, details))
            batch = deps.gateway.score_brio_questions(state, entries,
                model=payload.get("model"), normalize="mean", option_limit=255,
                jev_model_alias=True)
        except (ValueError, TypeError, ProviderConfigurationError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except ProviderUnavailableError as exc:
            return JsonRouteResult({"error": str(exc), "status": "unavailable"},
                                   HTTPStatus.SERVICE_UNAVAILABLE)
        answers = {}
        for (name, details), scored in zip(metadata, batch["answers"]):
            choices = scored["choices"]
            probabilities = {choice["option"]: choice["p"] for choice in choices}
            if details["type"] == "noul":
                answer = {"type": "noul", "noul": probabilities.get("yes", 0.0)}
            elif details["type"] == "choice":
                count = len(details["options"])
                peak = max(probabilities.values())
                confidence = min(1.0, max(0.0, (count*peak-1)/(count-1)))
                answer = {"type": "choice", "choice": scored["answer"],
                    "probabilities": probabilities, "confidence": confidence}
            else:
                count = len(details["legend"])
                peak = max(probabilities.values())
                confidence = min(1.0, max(0.0, (count*peak-1)/(count-1)))
                expected = sum(int(level)*probability for level, probability in probabilities.items())
                answer = {"type": "score", "score": expected, "legend": details["legend"],
                    "probabilities": probabilities, "confidence": confidence}
            answers[name] = answer
        deps.store.add_audit_event(AuditEvent(
            actor="inference-engine", action="inference.systemone.score",
            resource="inference:closed-set-scoring", project="default",
            metadata={"provider": "native-publisher", "question_count": len(answers),
                      "read_tokens": batch["usage"]["read_tokens"]},
        ))
        return JsonRouteResult({"model": batch["model"], "answers": answers,
            "usage": {"input_tokens": batch["usage"]["prompt_tokens"],
                      "output_tokens": batch["usage"]["read_tokens"]}}, HTTPStatus.OK)
    if path == "/v1/brio":
        allowed = {"model", "state", "messages", "question", "options", "questions", "schema", "task", "normalize", "cache_slot"}
        forms = [name for name in ("options", "questions", "schema")
                 if type(payload) is dict and payload.get(name) is not None]
        if (type(payload) is not dict or set(payload) - allowed or
                ("state" in payload) == ("messages" in payload) or len(forms) != 1 or
                ("question" in payload and type(payload["question"]) is not str) or
                ("task" in payload and type(payload["task"]) is not str) or
                (forms and forms[0] != "options" and "question" in payload) or
                (forms and forms[0] != "schema" and "task" in payload) or
                ("model" in payload and type(payload["model"]) is not str) or
                ("cache_slot" in payload and (type(payload["cache_slot"]) is not int or
                    not 0 <= payload["cache_slot"] < 16)) or
                payload.get("normalize", "mean") not in ("mean", "sum")):
            return JsonRouteResult({"error": "invalid Brio request fields"}, HTTPStatus.UNPROCESSABLE_ENTITY)
        try:
            state = payload["state"] if "state" in payload else _brio_messages_to_state(payload["messages"])
            if forms[0] == "options":
                result = deps.gateway.score_brio(state, payload["options"],
                    question=payload.get("question", ""), model=payload.get("model"),
                    normalize=payload.get("normalize", "mean"), cache_slot=payload.get("cache_slot"))
            elif forms[0] == "questions":
                result = deps.gateway.score_brio_questions(state, payload["questions"],
                    model=payload.get("model"), normalize=payload.get("normalize", "mean"),
                    cache_slot=payload.get("cache_slot"))
            else:
                result = deps.gateway.score_brio_schema(state, payload["schema"],
                    task=payload.get("task", ""), model=payload.get("model"),
                    normalize=payload.get("normalize", "mean"), cache_slot=payload.get("cache_slot"))
        except (ValueError, ProviderConfigurationError, TypeError) as exc:
            return JsonRouteResult({"error": str(exc)}, HTTPStatus.UNPROCESSABLE_ENTITY)
        except ProviderUnavailableError as exc:
            return JsonRouteResult({"error": str(exc), "status": "unavailable"},
                                   HTTPStatus.SERVICE_UNAVAILABLE)
        deps.store.add_audit_event(AuditEvent(
            actor="inference-engine", action="inference.brio.score",
            resource="inference:closed-set-scoring", project="default",
            metadata={"provider": "native-publisher",
                      "choice_count": sum(len(answer.get('choices', ())) for answer in
                          ([result] if 'choices' in result else result.get('answers', result.get('fields', ())))),
                      "prompt_tokens": result["usage"]["prompt_tokens"],
                      "read_tokens": result["usage"]["read_tokens"]},
        ))
        return JsonRouteResult(result, HTTPStatus.OK)
    project = str(payload.get("project", "default")).strip() or "default"
    try:
        if path == "/api/inference/benchmark":
            result = deps.gateway.benchmark_inference_engine(
                iterations=_as_int(payload.get("iterations"), 100)
            )
            action = "inference.benchmark"
        elif path == "/api/inference/cache/clear":
            result = deps.gateway.clear_inference_cache()
            action = "inference.cache.clear"
        else:
            result = deps.gateway.warmup(
                provider_id=str(payload.get("provider", "auto")).strip() or "auto",
                model=str(payload.get("model", "")).strip() or None,
            )
            action = "inference.warmup"
    except (ValueError, ProviderConfigurationError) as exc:
        return JsonRouteResult({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
    except ProviderUnavailableError as exc:
        return JsonRouteResult(
            {
                "error": str(exc),
                "status": "unavailable",
                "privacy_boundary": "The failed warmup did not cross the local data boundary.",
            },
            HTTPStatus.SERVICE_UNAVAILABLE,
        )
    deps.store.add_audit_event(
        AuditEvent(
            actor="inference-engine",
            action=action,
            resource="inference:local-runtime",
            project=project,
            metadata={"provider": result.get("provider", "local"), "status": result.get("status", "ok")},
        )
    )
    return JsonRouteResult(result, HTTPStatus.CREATED)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _handle_compat_completion(path: str, payload: object,
                              deps: InferenceRouteDependencies) -> JsonRouteResult:
    """Serve a deliberately local-only OpenAI/Anthropic text compatibility subset."""
    if type(payload) is not dict:
        return JsonRouteResult({"error": {"message": "request must be an object",
            "type": "invalid_request_error"}}, HTTPStatus.BAD_REQUEST)
    anthropic = path == "/v1/messages"
    allowed = ({"model", "messages", "max_tokens", "temperature", "top_p", "stop_sequences",
                "stream", "system", "tools", "tool_choice", "metadata", "thinking", "stream_options"} if anthropic else
               {"model", "messages", "prompt", "max_tokens", "max_completion_tokens", "temperature",
                "top_p", "stop", "stream", "n", "presence_penalty", "frequency_penalty",
                "seed", "user", "response_format", "stream_options", "tools", "tool_choice", "logprobs",
                "top_logprobs", "echo", "best_of", "suffix"})
    def error(message, status=HTTPStatus.BAD_REQUEST, kind="invalid_request_error"):
        if anthropic:
            anthropic_kind = {
                "invalid_request_error": "invalid_request_error",
                "unsupported_feature": "invalid_request_error",
                "server_error": "api_error",
            }.get(kind, "api_error")
            return JsonRouteResult({"type": "error", "error": {
                "type": anthropic_kind, "message": message}}, status)
        return JsonRouteResult({"error": {"message": message, "type": kind}}, status)
    if set(payload) - allowed:
        return error("unsupported request fields: " + ", ".join(sorted(set(payload)-allowed)))
    if payload.get("stream", False) is not False:
        return error("streaming is not supported by this compatibility route", kind="unsupported_feature")
    if payload.get("stream_options") not in (None, {}) and not (
            not anthropic and type(payload.get("stream_options")) is dict and
            set(payload["stream_options"]) == {"include_usage"} and
            type(payload["stream_options"]["include_usage"]) is bool):
        return error("unsupported stream_options fields")
    if anthropic:
        if payload.get("tools") or payload.get("tool_choice") or payload.get("thinking"):
            return error("tool calls and thinking blocks are not admitted on the Anthropic compatibility route",
                         kind="unsupported_feature")
        if "temperature" in payload or "top_p" in payload:
            return error("Anthropic sampling controls are not wired to the native session",
                         kind="unsupported_feature")
        if payload.get("metadata") not in (None, {}):
            return error("Anthropic metadata is not supported", kind="unsupported_feature")
        max_tokens = payload.get("max_tokens")
        raw_messages = payload.get("messages")
    else:
        if payload.get("tools") or payload.get("tool_choice"):
            return error("tool calls are not admitted on the compatibility route", kind="unsupported_feature")
        if payload.get("response_format") is not None:
            response_format = payload["response_format"]
            if (type(response_format) is not dict or response_format.get("type") != "text" or
                    set(response_format) != {"type"}):
                return error("only response_format type=text is supported", kind="unsupported_feature")
        if payload.get("logprobs") not in (None, False) or payload.get("top_logprobs") is not None:
            return error("logprobs are not supported", kind="unsupported_feature")
        if "seed" in payload:
            return error("seed is not wired to the native session", kind="unsupported_feature")
        if "temperature" in payload or "top_p" in payload:
            return error("sampling controls are not wired to the native session",
                         kind="unsupported_feature")
        if any(payload.get(key, 0) != 0 for key in ("presence_penalty", "frequency_penalty")):
            return error("presence and frequency penalties are not wired to the native session",
                         kind="unsupported_feature")
        if path == "/v1/completions":
            if "echo" in payload and type(payload["echo"]) is not bool:
                return error("echo must be boolean")
            raw_prompt = payload.get("prompt")
            if type(raw_prompt) is not str:
                return error("prompt must be a string")
            raw_messages = [{"role": "user", "content": raw_prompt}]
        else:
            raw_messages = payload.get("messages")
        max_tokens = payload.get("max_completion_tokens", payload.get("max_tokens"))
    if max_tokens is None and (anthropic or path == "/v1/chat/completions"):
        return error("max_tokens is required")
    if max_tokens is not None and (type(max_tokens) is not int or not 1 <= max_tokens <= 32768):
        return error("max_tokens must be from 1 to 32768")
    if not anthropic and "max_tokens" in payload and "max_completion_tokens" in payload:
        return error("max_tokens and max_completion_tokens cannot both be specified")
    for key in ("presence_penalty", "frequency_penalty"):
        value = payload.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not -2 <= value <= 2):
            return error(f"{key} must be between -2 and 2")
    if not anthropic and "best_of" in payload and payload["best_of"] != 1:
        return error("only best_of=1 is supported", kind="unsupported_feature")
    if not anthropic and payload.get("suffix") not in (None, ""):
        return error("suffix is not supported", kind="unsupported_feature")
    for key in ("temperature", "top_p"):
        value = payload.get(key)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or
                                  (value < 0 if key == "temperature" else not 0 < value <= 1)):
            return error(f"{key} is outside the supported range")
    if not anthropic and (type(payload.get("n", 1)) is not int or payload.get("n", 1) != 1):
        return error("only n=1 is supported", kind="unsupported_feature")
    # The completion subset emits generated text only. Accepting echo alongside
    # an omitted/empty prompt would silently claim semantics the provider does
    # not implement, so reject it unless this is the text-prompt completion API.
    if not anthropic and path == "/v1/chat/completions" and "echo" in payload:
        return error("echo is supported only by the text completion route",
                     kind="unsupported_feature")
    stop_sequences = payload.get("stop", payload.get("stop_sequences", ()))
    if isinstance(stop_sequences, str):
        stop_sequences = (stop_sequences,)
    if (type(stop_sequences) not in (list, tuple) or len(stop_sequences) > 4 or
            any(type(item) is not str or not item or len(item.encode("utf-8")) > 4096
                for item in stop_sequences)):
        return error("stop must be a string or an array of at most four bounded strings")
    # Native PublisherSession stop-prefix arguments are token-ID sequences,
    # not user-facing text. Do not misroute raw strings as iterable token IDs;
    # until the tokenizer/template-aware stop adapter exists, reject explicitly.
    if stop_sequences:
        return error("text stop sequences are not yet adapted to native token boundaries",
                     kind="unsupported_feature")
    def _text_content(value):
        if type(value) is str:
            return value
        if anthropic and type(value) is list:
            if any(type(block) is not dict or block.get("type") != "text" or
                   type(block.get("text")) is not str or set(block) - {"type", "text"}
                   for block in value):
                return None
            return "".join(block["text"] for block in value)
        return None
    if (type(raw_messages) is not list or not raw_messages or len(raw_messages) > 10000 or
            any(type(item) is not dict or set(item) - {"role", "content"} or
                item.get("role") not in ({"user", "assistant"} if anthropic else {"system", "user", "assistant"}) or
                _text_content(item.get("content")) is None for item in raw_messages)):
        return error("messages must be a bounded array of text role/content objects")
    messages = [{"role": item["role"], "content": _text_content(item["content"])} for item in raw_messages]
    if anthropic and "system" in payload:
        system = _text_content(payload["system"])
        if system is None:
            return error("system must contain text only")
        messages.insert(0, {"role": "system", "content": system})
    if sum(len(item["content"].encode("utf-8")) for item in messages) > 1024**2:
        return error("input exceeds 1 MiB")
    model = payload.get("model")
    if type(model) is not str or not model:
        return error("model is required")
    provider = deps.gateway.providers.get("native-publisher")
    if provider is None:
        return error("native publisher is not configured", HTTPStatus.SERVICE_UNAVAILABLE, "server_error")
    try:
        configured = provider._configuration()["model"]
        if model != configured:
            return error("requested model differs from configured native checkpoint")
        result = provider.complete(messages, model=model, max_output_tokens=max_tokens,
                                   stop_prefixes=())
    except (ProviderConfigurationError, ValueError, TypeError) as exc:
        return error(str(exc))
    except ProviderUnavailableError as exc:
        return error(str(exc), HTTPStatus.SERVICE_UNAVAILABLE, "server_error")
    created = int(time())
    runtime = result.metadata.get("runtime", {})
    prompt_tokens = runtime.get("expanded_prompt_tokens", runtime.get("prompt_tokens", 0))
    completion_tokens = len(runtime.get("generated_ids", ()))
    if anthropic:
        return JsonRouteResult({"id": "msg_" + str(created), "type": "message", "role": "assistant",
            "model": result.model, "content": [{"type": "text", "text": result.content}],
            "stop_reason": "max_tokens" if runtime.get("finish_reason") == "length" else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens}})
    if path == "/v1/completions":
        text = ((payload.get("prompt", "")) if payload.get("echo", False) else "") + result.content
        return JsonRouteResult({"id": "cmpl-" + str(created), "object": "text_completion", "created": created,
            "model": result.model, "choices": [{"text": text, "index": 0,
                "finish_reason": "length" if runtime.get("finish_reason") == "length" else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens}})
    return JsonRouteResult({"id": "chatcmpl-" + str(created), "object": "chat.completion", "created": created,
        "model": result.model, "choices": [{"index": 0, "message": {"role": "assistant", "content": result.content},
                                             "finish_reason": "length" if runtime.get("finish_reason") == "length" else "stop"}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens}})


def _prometheus_metrics(gateway: ModelGateway) -> str:
    provider = gateway.providers.get("native-publisher")
    state = {} if provider is None else provider.status()
    lines = [
        '# HELP openagent_native_publisher_up Whether the configured native publisher is available.',
        '# TYPE openagent_native_publisher_up gauge',
        f'openagent_native_publisher_up {1 if provider is not None and provider.available() else 0}',
        '# HELP openagent_native_publisher_active Whether a native request owns the provider.',
        '# TYPE openagent_native_publisher_active gauge',
        f'openagent_native_publisher_active {1 if state.get("active") else 0}',
        '# HELP openagent_native_publisher_stream_active Whether native streaming is active.',
        '# TYPE openagent_native_publisher_stream_active gauge',
        f'openagent_native_publisher_stream_active {1 if state.get("stream_worker_active") else 0}',
    ]
    runtime = state.get("execution_runtime")
    scheduler = runtime.get("scheduler") if type(runtime) is dict else None
    if type(scheduler) is dict:
        for metric, key, help_text in (
                ('openagent_generation_scheduler_active', 'active', 'Active generation admission leases.'),
                ('openagent_generation_scheduler_waiting', 'waiting', 'Generation admission requests waiting.'),
                ('openagent_generation_scheduler_max_active', 'max_active', 'Generation admission active capacity.'),
                ('openagent_generation_scheduler_max_waiting', 'max_waiting', 'Generation admission queue capacity.'),
                ('openagent_generation_scheduler_reserved_bytes', 'reserved_bytes', 'Bytes reserved by active generation leases.'),
                ('openagent_generation_scheduler_max_bytes', 'max_bytes', 'Generation admission byte capacity.')):
            value = scheduler.get(key)
            if type(value) is int and value >= 0:
                lines.extend((f'# HELP {metric} {help_text}', f'# TYPE {metric} gauge',
                              f'{metric} {value}'))
    return '\n'.join(lines) + '\n'


def _prometheus_text(value: str) -> bytes:
    if type(value) is not str or len(value.encode('utf-8')) > 1024**2:
        raise ValueError('bounded Prometheus text required')
    return value.encode('utf-8')


def _brio_messages_to_state(messages: object) -> str:
    if type(messages) is not list or not messages or len(messages) > 10000:
        raise ValueError('Brio messages must be a nonempty bounded array')
    parts = []
    total_bytes = 0
    for message in messages:
        if type(message) is not dict or message.get('role', 'user') not in (
                'system', 'user', 'assistant', 'tool'):
            raise ValueError('Brio messages require a supported role and object shape')
        content = message.get('content')
        if type(content) is str:
            text = content
        elif type(content) is list:
            pieces = []
            for block in content:
                if type(block) is dict and type(block.get('text')) is str:
                    pieces.append(block['text'])
                else:
                    raise ValueError('Brio message content blocks must contain text only')
            text = ''.join(pieces)
        else:
            raise ValueError('Brio message content must be text')
        total_bytes += len(text.encode('utf-8'))
        if total_bytes > 1024**2:
            raise ValueError('Brio conversation exceeds the 1 MiB input limit')
        if text:
            parts.append(f"{message.get('role', 'user')}: {text}")
    if not parts:
        raise ValueError('Brio messages must contain text')
    return '\n'.join(parts)
