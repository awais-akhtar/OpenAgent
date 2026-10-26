"""Serving Engine — OpenAI-compatible request lifecycle and SSE streaming.

Provides the request/response types and processing pipeline for a
``/v1/completions`` and ``/v1/chat/completions`` compatible inference server.
Designed to be used without a web framework; callers supply their own HTTP
layer (FastAPI, Flask, http.server, etc.) and call ``process_completion`` or
``process_chat_completion`` directly.

Components
----------
``Role``                    -- message role enum (system / user / assistant)
``ChatMessage``             -- one turn in a conversation
``CompletionRequest``       -- /v1/completions parsed request
``ChatCompletionRequest``   -- /v1/chat/completions parsed request
``CompletionChoice``        -- one candidate in the response
``CompletionResponse``      -- full response object (non-streaming)
``StreamChunk``             -- one SSE data frame for streaming responses
``TokenizerAdapter``        -- wraps any callable tokenizer; falls back to
                              ASCII-byte encoding when none is provided
``PromptFormatter``         -- converts chat messages to a flat prompt string
``ServingEngine``           -- high-level entry point; accepts requests,
                              dispatches to Phase 2 service, formats responses
"""
from __future__ import annotations

import logging
from itertools import islice
from operator import index as integer_index
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, List, Optional

from .failure_reporting import add_failure_note, failure_detail

logger = logging.getLogger(__name__)


def _positive_request_integer(value: Any, name: str) -> int:
    if type(value) is not int or not 1 <= value < 2**31:
        raise ValueError(name + ' must be a positive signed 32-bit integer')
    return value


def _request_boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(name + ' must be a boolean')
    return value


def _token_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError('backend token IDs must be nonnegative signed 32-bit integers')
    try:
        token = integer_index(value)
    except TypeError as error:
        raise ValueError('backend token IDs must be integers') from error
    if not 0 <= token < 2**31:
        raise ValueError('backend token ID is outside the signed 32-bit range')
    return token


# ---------------------------------------------------------------------------
# Role
# ---------------------------------------------------------------------------

class Role(str, Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"


# ---------------------------------------------------------------------------
# ChatMessage
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role:    Role
    content: str
    name:    Optional[str] = None

    def to_dict(self) -> dict:
        d: dict = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ChatMessage":
        return cls(
            role=Role(d.get("role", "user")),
            content=str(d.get("content", "")),
            name=d.get("name"),
        )


# ---------------------------------------------------------------------------
# CompletionRequest
# ---------------------------------------------------------------------------

@dataclass
class CompletionRequest:
    """Parsed /v1/completions request."""
    model:            str
    prompt:           str
    max_tokens:       int   = 256
    # None means "use the backend's default". Explicit sampling controls are
    # rejected until Phase 2 carries a sampling policy through its worker.
    temperature:      Optional[float] = None
    top_p:            Optional[float] = None
    n:                int   = 1       # number of completions
    stream:           bool  = False
    stop:             Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logprobs:         Optional[int]  = None
    echo:             bool  = False
    user:             Optional[str]  = None
    request_id:       str   = field(default_factory=lambda: str(uuid.uuid4()))

    @classmethod
    def from_dict(cls, d: dict) -> "CompletionRequest":
        echo = d.get("echo", False)
        if type(echo) is not bool:
            raise ValueError("echo must be a boolean")
        return cls(
            model=d.get("model", "openagent"),
            prompt=str(d.get("prompt", "")),
            max_tokens=_positive_request_integer(d.get("max_tokens", 256), 'max_tokens'),
            temperature=(None if d.get("temperature") is None else float(d["temperature"])),
            top_p=(None if d.get("top_p") is None else float(d["top_p"])),
            n=_positive_request_integer(d.get("n", 1), 'n'),
            stream=_request_boolean(d.get("stream", False), 'stream'),
            stop=d.get("stop"),
            presence_penalty=float(d.get("presence_penalty", 0.0)),
            frequency_penalty=float(d.get("frequency_penalty", 0.0)),
            logprobs=d.get("logprobs"),
            echo=echo,
            user=d.get("user"),
        )


# ---------------------------------------------------------------------------
# ChatCompletionRequest
# ---------------------------------------------------------------------------

@dataclass
class ChatCompletionRequest:
    """Parsed /v1/chat/completions request."""
    model:       str
    messages:    List[ChatMessage]
    max_tokens:  int   = 256
    temperature: Optional[float] = None
    top_p:       Optional[float] = None
    n:           int   = 1
    stream:      bool  = False
    stop:        Optional[List[str]] = None
    user:        Optional[str]  = None
    request_id:  str   = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.messages = [
            ChatMessage.from_dict(m) if isinstance(m, dict) else m
            for m in self.messages
        ]

    @classmethod
    def from_dict(cls, d: dict) -> "ChatCompletionRequest":
        raw_msgs = d.get("messages", [])
        return cls(
            model=d.get("model", "openagent"),
            messages=[ChatMessage.from_dict(m) for m in raw_msgs],
            max_tokens=_positive_request_integer(d.get("max_tokens", 256), 'max_tokens'),
            temperature=(None if d.get("temperature") is None else float(d["temperature"])),
            top_p=(None if d.get("top_p") is None else float(d["top_p"])),
            n=_positive_request_integer(d.get("n", 1), 'n'),
            stream=_request_boolean(d.get("stream", False), 'stream'),
            stop=d.get("stop"),
            user=d.get("user"),
        )


# ---------------------------------------------------------------------------
# CompletionChoice / CompletionResponse
# ---------------------------------------------------------------------------

@dataclass
class CompletionChoice:
    text:          str
    index:         int   = 0
    logprobs:      Optional[Any] = None
    finish_reason: str   = "stop"

    def to_dict(self) -> dict:
        return {
            "text":          self.text,
            "index":         self.index,
            "logprobs":      self.logprobs,
            "finish_reason": self.finish_reason,
        }


@dataclass
class ChatCompletionChoice:
    message:       ChatMessage
    index:         int  = 0
    finish_reason: str  = "stop"

    def to_dict(self) -> dict:
        return {
            "index":         self.index,
            "message":       self.message.to_dict(),
            "finish_reason": self.finish_reason,
        }


@dataclass
class UsageStats:
    prompt_tokens:     int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict:
        return {
            "prompt_tokens":     self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens":      self.total_tokens,
        }


@dataclass
class CompletionResponse:
    id:      str
    model:   str
    choices: List[CompletionChoice]
    usage:   UsageStats
    created: int = field(default_factory=lambda: int(time.time()))
    object:  str = "text_completion"

    def to_dict(self) -> dict:
        return {
            "id":      self.id,
            "object":  self.object,
            "created": self.created,
            "model":   self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage":   self.usage.to_dict(),
        }


@dataclass
class ChatCompletionResponse:
    id:      str
    model:   str
    choices: List[ChatCompletionChoice]
    usage:   UsageStats
    created: int = field(default_factory=lambda: int(time.time()))
    object:  str = "chat.completion"

    def to_dict(self) -> dict:
        return {
            "id":      self.id,
            "object":  self.object,
            "created": self.created,
            "model":   self.model,
            "choices": [c.to_dict() for c in self.choices],
            "usage":   self.usage.to_dict(),
        }


# ---------------------------------------------------------------------------
# StreamChunk
# ---------------------------------------------------------------------------

@dataclass
class StreamChunk:
    """One SSE data frame for a streaming completion."""
    id:           str
    model:        str
    delta_text:   str   = ""
    finish_reason: Optional[str] = None
    index:        int   = 0
    is_chat:      bool  = False

    def to_sse_line(self) -> str:
        import json
        payload: dict
        if self.is_chat:
            payload = {
                "id": self.id,
                "object": "chat.completion.chunk",
                "model": self.model,
                "choices": [{
                    "index": self.index,
                    "delta": {"content": self.delta_text},
                    "finish_reason": self.finish_reason,
                }],
            }
        else:
            payload = {
                "id": self.id,
                "object": "text_completion",
                "model": self.model,
                "choices": [{
                    "index": self.index,
                    "text": self.delta_text,
                    "finish_reason": self.finish_reason,
                }],
            }
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def done_line() -> str:
        return "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# TokenizerAdapter
# ---------------------------------------------------------------------------

class TokenizerAdapter:
    """Wraps any tokenizer; falls back to ASCII ord encoding when none given.

    Args:
        tokenizer: Any object with ``encode(text) → List[int]`` and
                   ``decode(ids) → str`` methods.  If None, uses a simple
                   UTF-8 byte-level fallback (each byte = one token).
    """

    def __init__(self, tokenizer: Optional[Any] = None) -> None:
        self._tok = tokenizer

    def encode(self, text: str) -> List[int]:
        if self._tok is not None:
            try:
                result = self._tok.encode(text)
                # HuggingFace returns list or tensor
                if hasattr(result, "tolist"):
                    return result.tolist()
                if hasattr(result, "input_ids"):
                    ids = result.input_ids
                    return ids.tolist() if hasattr(ids, "tolist") else list(ids)
                return list(result)
            except Exception as exc:
                logger.debug("tokenizer encode failed, using fallback: %s", exc)
        return list(text.encode("utf-8"))

    def encode_strict(self, text: str) -> List[int]:
        """Encode control text without silently switching tokenization schemes."""
        if self._tok is None:
            return list(text.encode("utf-8"))
        encode_fn = getattr(self._tok, "encode", None)
        if not callable(encode_fn):
            raise ValueError("configured tokenizer does not provide encode()")
        try:
            try:
                result = encode_fn(text, add_special_tokens=False)
            except TypeError as error:
                # Small test/custom tokenizers often expose encode(text) only.
                # Retry that exact supported signature, but never fall back to
                # a different tokenizer if its own encoding fails.
                if "add_special_tokens" not in str(error):
                    raise
                result = encode_fn(text)
            if hasattr(result, "tolist"):
                result = result.tolist()
            if hasattr(result, "input_ids"):
                result = result.input_ids
                if hasattr(result, "tolist"):
                    result = result.tolist()
            return list(result)
        except Exception as error:
            raise ValueError("configured tokenizer failed to encode control text") from error

    def decode(self, ids: List[int]) -> str:
        if self._tok is not None:
            try:
                result = self._tok.decode(ids, skip_special_tokens=True)
                return str(result)
            except Exception as exc:
                try:
                    return self._tok.decode(ids)
                except Exception:
                    logger.debug("tokenizer decode failed, using fallback: %s", exc)
        try:
            return bytes(i & 0xFF for i in ids).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def decode_strict(self, ids: List[int]) -> str:
        """Decode generated IDs without silently changing tokenization schemes."""
        if type(ids) not in (list, tuple) or any(
            type(token_id) is not int or token_id < 0 for token_id in ids
        ):
            raise ValueError("generated token IDs must be a sequence of nonnegative integers")
        if self._tok is None:
            try:
                return bytes(token_id & 0xFF for token_id in ids).decode(
                    "utf-8", errors="replace")
            except (OverflowError, ValueError) as error:
                raise ValueError("generated token IDs could not be decoded") from error

        decode_fn = getattr(self._tok, "decode", None)
        if not callable(decode_fn):
            raise ValueError("configured tokenizer does not provide decode()")
        try:
            try:
                result = decode_fn(ids, skip_special_tokens=True)
            except TypeError as error:
                if "skip_special_tokens" not in str(error):
                    raise
                result = decode_fn(ids)
            if type(result) is not str:
                raise TypeError("configured tokenizer decode() must return text")
            return result
        except Exception as error:
            raise ValueError("configured tokenizer failed to decode generated tokens") from error


# ---------------------------------------------------------------------------
# PromptFormatter
# ---------------------------------------------------------------------------

class PromptFormatter:
    """Converts chat messages to a single prompt string for the model.

    Supports a configurable chat template; defaults to a simple
    ``<role>: <content>`` format that works for most instruct models.
    """

    DEFAULT_TEMPLATE = "{role}: {content}\n"
    SEPARATOR = "\n"

    def __init__(self, template: Optional[str] = None) -> None:
        self._template = template or self.DEFAULT_TEMPLATE

    def format(self, messages: List[ChatMessage]) -> str:
        parts = [
            self._template.format(role=msg.role.value, content=msg.content)
            for msg in messages
        ]
        return self.SEPARATOR.join(parts) + "\nassistant:"


# ---------------------------------------------------------------------------
# ServingEngine
# ---------------------------------------------------------------------------

class ServingEngine:
    """Processes completion and chat-completion requests via a backend service.

    Args:
        service: Any object with ``open_session()`` returning a session that
                 has ``generate(context_ids, max_new_tokens)`` and
                 ``stream(context_ids, max_new_tokens)`` methods.
        tokenizer: Optional tokenizer adapter or raw tokenizer.
        model_name: Model identifier returned in responses.
    """

    def __init__(
        self,
        service: Any,
        tokenizer: Optional[Any] = None,
        model_name: str = "openagent",
    ) -> None:
        self._service    = service
        self._tokenizer  = TokenizerAdapter(tokenizer)
        self._formatter  = PromptFormatter()
        self._model_name = model_name

        # Telemetry
        self._total_requests   = 0
        self._total_tokens_in  = 0
        self._total_tokens_out = 0
        self._errors           = 0

    # ---- /v1/completions ----------------------------------------------------

    def process_completion(
        self, request: CompletionRequest
    ) -> CompletionResponse:
        self._validate_completion_options(request)
        prompt_ids  = self._tokenizer.encode(request.prompt)
        stop_prefixes = self._encode_stop_sequences(request.stop)
        self._total_requests   += 1
        self._total_tokens_in  += len(prompt_ids)

        choices: List[CompletionChoice] = []
        completion_token_count = 0
        for i in range(request.n):
            try:
                out_ids, stop_matched = self._run_generate_result(
                    prompt_ids, request.max_tokens, stop_prefixes)
                text = self._tokenizer.decode_strict(out_ids)
                self._total_tokens_out += len(out_ids)
                completion_token_count += len(out_ids)
                choices.append(CompletionChoice(
                    text=(request.prompt if request.echo else "") + text, index=i,
                    finish_reason=("stop" if stop_matched or len(out_ids) < request.max_tokens
                                   else "length")))
            except Exception as exc:
                logger.warning("serving: completion error: %s", failure_detail(exc))
                self._errors += 1
                choices.append(CompletionChoice(text="", index=i, finish_reason="error"))

        return CompletionResponse(
            id=f"cmpl-{request.request_id}",
            model=self._model_name,
            choices=choices,
            usage=UsageStats(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_token_count,
            ),
        )

    def stream_completion(
        self, request: CompletionRequest
    ) -> Iterator[StreamChunk]:
        self._validate_completion_options(request, streaming=True)
        prompt_ids = self._tokenizer.encode(request.prompt)
        stop_prefixes = self._encode_stop_sequences(request.stop)
        self._total_requests  += 1
        self._total_tokens_in += len(prompt_ids)
        stream_id = f"cmpl-{request.request_id}"
        stream_failed = False
        stop_state: dict[str, bool] = {}
        generated_count = 0
        source = self._run_stream(
            prompt_ids, request.max_tokens, stop_prefixes=stop_prefixes,
            stop_state=stop_state,
        )
        primary: BaseException | None = None

        try:
            if request.echo and request.prompt:
                yield StreamChunk(
                    id=stream_id, model=self._model_name,
                    delta_text=request.prompt, is_chat=False)
            for token_id in source:
                text = self._tokenizer.decode_strict([token_id])
                self._total_tokens_out += 1
                generated_count += 1
                yield StreamChunk(
                    id=stream_id, model=self._model_name,
                    delta_text=text, is_chat=False)
        except BaseException as exc:
            primary = exc
            if isinstance(exc, Exception):
                logger.warning("serving: stream_completion error: %s", failure_detail(exc))
                self._errors += 1
                stream_failed = True
            else:
                raise
        finally:
            cleanup = self._close_stream_source(source, primary)
            if cleanup is not None:
                if isinstance(cleanup, Exception):
                    logger.warning("serving: stream_completion cleanup error: %s", failure_detail(cleanup))
                    self._errors += 1
                    stream_failed = True
                else:
                    raise cleanup
        yield StreamChunk(
            id=stream_id, model=self._model_name,
            finish_reason=("error" if stream_failed else
                           "stop" if stop_state.get("matched") or generated_count < request.max_tokens
                           else "length"), is_chat=False)

    # ---- /v1/chat/completions -----------------------------------------------

    def process_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        self._validate_chat_completion_options(request)
        prompt    = self._formatter.format(request.messages)
        prompt_ids = self._tokenizer.encode(prompt)
        stop_prefixes = self._encode_stop_sequences(request.stop)
        self._total_requests   += 1
        self._total_tokens_in  += len(prompt_ids)

        choices: List[ChatCompletionChoice] = []
        completion_token_count = 0
        for i in range(request.n):
            try:
                out_ids, stop_matched = self._run_generate_result(
                    prompt_ids, request.max_tokens, stop_prefixes)
                text = self._tokenizer.decode_strict(out_ids)
                self._total_tokens_out += len(out_ids)
                completion_token_count += len(out_ids)
                choices.append(ChatCompletionChoice(
                    message=ChatMessage(role=Role.ASSISTANT, content=text),
                    index=i,
                    finish_reason=("stop" if stop_matched or len(out_ids) < request.max_tokens
                                   else "length"),
                ))
            except Exception as exc:
                logger.warning("serving: chat completion error: %s", failure_detail(exc))
                self._errors += 1
                choices.append(ChatCompletionChoice(
                    message=ChatMessage(role=Role.ASSISTANT, content=""),
                    index=i, finish_reason="error"))

        return ChatCompletionResponse(
            id=f"chatcmpl-{request.request_id}",
            model=self._model_name,
            choices=choices,
            usage=UsageStats(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_token_count,
            ),
        )

    def stream_chat_completion(
        self, request: ChatCompletionRequest
    ) -> Iterator[StreamChunk]:
        self._validate_chat_completion_options(request, streaming=True)
        prompt     = self._formatter.format(request.messages)
        prompt_ids = self._tokenizer.encode(prompt)
        stop_prefixes = self._encode_stop_sequences(request.stop)
        self._total_requests  += 1
        self._total_tokens_in += len(prompt_ids)
        stream_id = f"chatcmpl-{request.request_id}"
        stream_failed = False
        stop_state: dict[str, bool] = {}
        generated_count = 0
        source = self._run_stream(
            prompt_ids, request.max_tokens, stop_prefixes=stop_prefixes,
            stop_state=stop_state,
        )
        primary: BaseException | None = None

        try:
            for token_id in source:
                text = self._tokenizer.decode_strict([token_id])
                self._total_tokens_out += 1
                generated_count += 1
                yield StreamChunk(
                    id=stream_id, model=self._model_name,
                    delta_text=text, is_chat=True)
        except BaseException as exc:
            primary = exc
            if isinstance(exc, Exception):
                logger.warning("serving: stream_chat error: %s", failure_detail(exc))
                self._errors += 1
                stream_failed = True
            else:
                raise
        finally:
            cleanup = self._close_stream_source(source, primary)
            if cleanup is not None:
                if isinstance(cleanup, Exception):
                    logger.warning("serving: stream_chat cleanup error: %s", failure_detail(cleanup))
                    self._errors += 1
                    stream_failed = True
                else:
                    raise cleanup
        yield StreamChunk(
            id=stream_id, model=self._model_name,
            finish_reason=("error" if stream_failed else
                           "stop" if stop_state.get("matched") or generated_count < request.max_tokens
                           else "length"), is_chat=True)

    # ---- backend dispatch ---------------------------------------------------

    def _get_session(self) -> Any:
        open_fn = getattr(self._service, "open_session", None)
        if callable(open_fn):
            return open_fn()
        return self._service

    @staticmethod
    def _validate_completion_options(request: CompletionRequest, *, streaming: bool = False) -> None:
        ServingEngine._validate_request_limits(request, streaming=streaming)
        if type(request.echo) is not bool:
            raise ValueError("echo must be a boolean")
        if request.logprobs is not None:
            raise ValueError("completion logprobs are not supported by this serving engine")
        if request.temperature is not None or request.top_p is not None:
            raise ValueError("sampling controls are not wired to the native session")
        if request.presence_penalty != 0 or request.frequency_penalty != 0:
            raise ValueError("presence and frequency penalties are not wired to the native session")

    @staticmethod
    def _validate_chat_completion_options(request: ChatCompletionRequest, *, streaming: bool = False) -> None:
        ServingEngine._validate_request_limits(request, streaming=streaming)
        if request.temperature is not None or request.top_p is not None:
            raise ValueError("sampling controls are not wired to the native session")

    @staticmethod
    def _validate_request_limits(request: Any, *, streaming: bool) -> None:
        _positive_request_integer(request.max_tokens, 'max_tokens')
        _positive_request_integer(request.n, 'n')
        if type(request.stream) is not bool:
            raise ValueError('stream must be a boolean')
        if streaming and request.n != 1:
            raise ValueError('streaming supports exactly one choice')

    def _run_generate(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        stop: Optional[List[str]],
    ) -> List[int]:
        stop_prefixes = self._encode_stop_sequences(stop)
        tokens, _ = self._run_generate_result(prompt_ids, max_tokens, stop_prefixes)
        return tokens

    def _run_generate_result(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        stop_prefixes: tuple[tuple[int, ...], ...],
    ) -> tuple[List[int], bool]:
        if stop_prefixes:
            matched: dict[str, bool] = {}
            tokens = list(self._run_stream(
                prompt_ids, max_tokens, stop_prefixes=stop_prefixes,
                stop_state=matched,
            ))
            return tokens, bool(matched.get("matched"))

        session = self._get_session()
        primary: BaseException | None = None
        iterator = None
        try:
            gen_fn = getattr(session, "generate", None)
            if not callable(gen_fn):
                raise RuntimeError("Backend session has no generate() method")
            result = gen_fn(context_ids=prompt_ids, max_new_tokens=max_tokens)
            # A backend may return a lazy iterator backed by this owner. Both
            # materialization and iterator release must precede session close.
            iterator = iter(result) if result is not None else iter(())
            return [_token_id(token) for token in islice(iterator, max_tokens)], False
        except BaseException as exc:
            primary = exc
            raise
        finally:
            cleanup = self._close_stream_source(iterator, primary) if iterator is not None else None
            self._close_session(session, primary if primary is not None else cleanup)
            if primary is None and cleanup is not None:
                raise cleanup

    def _encode_stop_sequences(
        self, stop: Optional[Any]
    ) -> tuple[tuple[int, ...], ...]:
        if stop is None:
            return ()
        if type(stop) is str:
            values = (stop,)
        elif type(stop) in (list, tuple):
            values = tuple(stop)
        else:
            raise ValueError("stop must be a string or a list of strings")
        if (not values or len(values) > 4 or
                any(type(value) is not str or not value for value in values)):
            raise ValueError("stop requires one to four nonempty strings")
        encoded = []
        for value in values:
            try:
                byte_length = len(value.encode("utf-8"))
            except UnicodeEncodeError as error:
                raise ValueError("stop strings must contain valid UTF-8 text") from error
            if byte_length > 4096:
                raise ValueError("stop strings are limited to 4096 UTF-8 bytes")
            token_ids = tuple(self._tokenizer.encode_strict(value))
            if (not token_ids or len(token_ids) > 128 or
                    any(type(token) is not int or not 0 <= token < 2**31 for token in token_ids)):
                raise ValueError("each stop string must encode to 1..128 valid token IDs")
            if token_ids not in encoded:
                encoded.append(token_ids)
        return tuple(encoded)

    def _run_stream(
        self,
        prompt_ids: List[int],
        max_tokens: int,
        *,
        stop_prefixes: tuple[tuple[int, ...], ...] = (),
        stop_state: Optional[dict[str, bool]] = None,
    ):
        session  = self._get_session()
        primary: BaseException | None = None
        iterator = None
        cleanup_error: BaseException | None = None
        try:
            # Attribute lookup may execute a backend proxy/property and can
            # fail just like invocation. Keep it inside the session lifetime
            # guard so this path always closes an owner after successful open.
            stream_fn = getattr(session, "stream", None)
            if callable(stream_fn):
                iterator = iter(stream_fn(context_ids=prompt_ids, max_new_tokens=max_tokens))
            else:
                gen_fn = getattr(session, "generate", None)
                if not callable(gen_fn):
                    raise RuntimeError("Backend session has no stream() or generate() method")
                ids = gen_fn(context_ids=prompt_ids, max_new_tokens=max_tokens)
                iterator = iter(ids) if ids is not None else iter(())
            pending: list[int] = []
            for raw_token in islice(iterator, max_tokens):
                token = _token_id(raw_token)
                if not stop_prefixes:
                    yield token
                    continue
                pending.append(token)
                while pending:
                    candidate = tuple(pending)
                    if candidate in stop_prefixes:
                        if stop_state is not None:
                            stop_state["matched"] = True
                        return
                    if any(prefix[:len(candidate)] == candidate for prefix in stop_prefixes):
                        break
                    yield pending.pop(0)
            for token in pending:
                yield token
        except BaseException as exc:
            primary = exc
            raise
        finally:
            if iterator is not None:
                try:
                    close_iterator = getattr(iterator, "close", None)
                    if callable(close_iterator):
                        close_iterator()
                except BaseException as cleanup:
                    if primary is None:
                        cleanup_error = cleanup
                    else:
                        add_failure_note(
                            primary,
                            "serving stream iterator close failed: " + failure_detail(cleanup),
                        )
            try:
                self._close_session(session, primary or cleanup_error)
            except BaseException as cleanup:
                if primary is None and cleanup_error is None:
                    cleanup_error = cleanup
                else:
                    add_failure_note(
                        primary or cleanup_error,
                        "serving session close failed: " + failure_detail(cleanup),
                    )
            if primary is None and cleanup_error is not None:
                raise cleanup_error

    @staticmethod
    def _close_session(session: Any, primary: BaseException | None) -> None:
        try:
            close_fn = getattr(session, "close", None)
            if callable(close_fn):
                close_fn()
        except BaseException as cleanup:
            if primary is None:
                raise
            add_failure_note(
                primary,
                "serving session close failed: " + failure_detail(cleanup),
            )

    @staticmethod
    def _close_stream_source(source: Any, primary: BaseException | None) -> BaseException | None:
        try:
            close_fn = getattr(source, "close", None)
            if callable(close_fn):
                close_fn()
        except BaseException as cleanup:
            if primary is not None:
                add_failure_note(
                    primary,
                    "serving response stream close failed: " + failure_detail(cleanup),
                )
                return None
            return cleanup
        return None

    # ---- telemetry ----------------------------------------------------------

    def metrics(self) -> dict:
        return {
            "serving_total_requests":    self._total_requests,
            "serving_total_tokens_in":   self._total_tokens_in,
            "serving_total_tokens_out":  self._total_tokens_out,
            "serving_errors_total":      self._errors,
            "serving_model":             self._model_name,
        }


__all__ = [
    "Role",
    "ChatMessage",
    "CompletionRequest",
    "ChatCompletionRequest",
    "CompletionChoice",
    "ChatCompletionChoice",
    "CompletionResponse",
    "ChatCompletionResponse",
    "StreamChunk",
    "UsageStats",
    "TokenizerAdapter",
    "PromptFormatter",
    "ServingEngine",
]
