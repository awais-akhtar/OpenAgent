"""Persistent, bounded serving primitives for autoregressive decoders.

The service intentionally owns one decoder worker.  A scheduling cycle may
select several requests, but decoder calls are made serially unless a future
decoder exposes an explicit batch API.  This keeps lifecycle and telemetry
claims aligned with the work that actually ran.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field
import copy
import math
import sys
import threading
import time
import uuid
from typing import Any, Callable, Iterable, Iterator, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {self.role}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be text")


@runtime_checkable
class StreamingDecoder(Protocol):
    """Minimal decoder contract used by :class:`OlmoeService`.

    The decoder should call ``on_token`` for each text fragment and check
    ``cancelled`` between generation steps.  It may return the complete text,
    an iterable of fragments, or ``None`` when all output was emitted through
    the callback.
    """

    def generate(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int,
        on_token: Callable[[str], None],
        cancelled: Callable[[], bool],
    ) -> str | Iterable[str] | None: ...


@runtime_checkable
class StatefulStreamingDecoder(StreamingDecoder, Protocol):
    def snapshot_state(self) -> object: ...

    def restore_state(self, state: object) -> None: ...

    def reset_state(self) -> None: ...


class OlmoeChatDecoder:
    """Adapt an OlmoeModel to the service while preserving per-session KV."""

    def __init__(self, model: Any) -> None:
        required = ("encode", "decode", "continue_id_stream", "snapshot_state", "restore_state", "reset")
        if any(not callable(getattr(model, name, None)) for name in required):
            raise TypeError("model does not expose the OLMoE incremental decoder contract")
        self.model = model
        self._turn_boundary: list[int] = []

    def generate(self, messages, *, max_tokens, on_token, cancelled):
        if (not isinstance(messages, (tuple, list)) or not messages or
                any(not isinstance(message, ChatMessage) for message in messages)):
            raise ValueError('nonempty validated chat message sequence required')
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError('positive generation limit required')
        if not callable(on_token) or not callable(cancelled):
            raise TypeError('stream callback and cancellation predicate required')
        if self.model.position:
            prompt = _render_new_turn(messages[-1])
        else:
            prompt = _render_conversation(messages)
        generated: list[int] = []
        rendered = ""
        ids = self._turn_boundary + list(self.model.encode(prompt))
        self._turn_boundary = []
        stream = iter(self.model.continue_id_stream(ids, max_tokens=max_tokens))
        try:
            for token in stream:
                if cancelled():
                    return None
                generated.append(token)
                text = self.model.decode(generated)
                # Replacement decoding cannot be represented by append-only events.
                if not text.startswith(rendered):
                    raise ValueError("OLMoE decoding revised already emitted text")
                delta = text[len(rendered):]
                rendered = text
                if delta:
                    on_token(delta)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        # The model stream consumes yielded tokens on resumption, but never
        # consumes EOS. Replay the turn delimiter with the next prompt.
        eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if eos is not None:
            self._turn_boundary = list(self.model.encode("\n")) + [eos] + list(self.model.encode("\n"))
        else:
            self._turn_boundary = list(self.model.encode("\n|||IP_ADDRESS|||\n"))
        return None

    def snapshot_state(self) -> object:
        return (self.model.snapshot_state(), tuple(self._turn_boundary))

    def restore_state(self, state: object) -> None:
        model_state, boundary = state
        self.model.restore_state(model_state)
        self._turn_boundary = list(boundary)

    def reset_state(self) -> None:
        self.model.reset()
        self._turn_boundary = []


@dataclass(frozen=True)
class ServiceEvent:
    request_id: str
    kind: str
    timestamp: float
    session_id: str
    data: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionSnapshot:
    session_id: str
    messages: tuple[ChatMessage, ...]
    revision: int


class ServiceRequestError(RuntimeError):
    def __init__(self, request_id: str, code: str, message: str) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.code = code


class QueueFullError(ServiceRequestError):
    pass


class RequestCancelledError(ServiceRequestError):
    pass


class RequestDeadlineError(ServiceRequestError):
    pass


class ServiceClosedError(RuntimeError):
    pass


@dataclass
class _Request:
    request_id: str
    session_id: str
    prompt: str
    priority: int
    deadline: float | None
    max_tokens: int
    stop_prefixes: tuple[str, ...]
    sequence: int
    created: float
    events: deque[ServiceEvent] = field(default_factory=deque)
    event_condition: threading.Condition = field(default_factory=threading.Condition)
    done: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    result_text: str | None = None
    error: ServiceRequestError | None = None
    started: float | None = None
    finished: float | None = None


class RequestHandle:
    """Thread-safe view of one submitted request."""

    def __init__(self, service: "OlmoeService", request: _Request) -> None:
        self._service = service
        self._request = request

    @property
    def request_id(self) -> str:
        return self._request.request_id

    @property
    def session_id(self) -> str:
        return self._request.session_id

    @property
    def done(self) -> bool:
        return self._request.done.is_set()

    def cancel(self) -> bool:
        with self._service._condition:
            if self._request.done.is_set():
                return False
            return self._service.cancel(self.request_id)

    def result(self, timeout: float | None = None) -> str:
        if not self._request.done.wait(timeout):
            raise TimeoutError(f"request {self.request_id} did not finish in time")
        if self._request.error is not None:
            raise self._request.error
        return self._request.result_text or ""

    def events(self, timeout: float | None = None) -> Iterator[ServiceEvent]:
        """Drain events until terminal; slow consumers lose oldest buffered events."""
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._request.event_condition:
                while not self._request.events and not self._request.done.is_set():
                    remaining = None if end is None else end - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise TimeoutError(f"event stream for {self.request_id} timed out")
                    self._request.event_condition.wait(remaining)
                ready = tuple(self._request.events)
                self._request.events.clear()
                finished = self._request.done.is_set()
            yield from ready
            if finished:
                return


class OlmoeService:
    """Persistent session and scheduling layer for an injected decoder."""

    TERMINAL_EVENTS = {"completed", "cancelled", "failed", "expired"}

    def __init__(
        self,
        decoder: StreamingDecoder,
        *,
        queue_capacity: int = 64,
        microbatch_size: int = 4,
        session_state_capacity_bytes: int = 512 * 1024**2,
        event_capacity: int = 1024,
        session_capacity: int = 1024,
        session_capacity_chars: int = 1024 * 1024,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(queue_capacity) is not int or queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if type(microbatch_size) is not int or microbatch_size <= 0:
            raise ValueError("microbatch_size must be positive")
        if type(session_state_capacity_bytes) is not int or session_state_capacity_bytes < 0:
            raise ValueError("session_state_capacity_bytes cannot be negative")
        if type(event_capacity) is not int or event_capacity < 8:
            raise ValueError("event_capacity must be at least 8")
        if (type(session_capacity) is not int or type(session_capacity_chars) is not int or
                session_capacity <= 0 or session_capacity_chars <= 0):
            raise ValueError("session capacities must be positive")
        if not isinstance(decoder, StreamingDecoder):
            raise TypeError("decoder does not implement the streaming decoder contract")
        self._decoder = decoder
        self._queue_capacity = queue_capacity
        self._microbatch_size = microbatch_size
        self._event_capacity = event_capacity
        self._session_capacity = session_capacity
        self._session_capacity_chars = session_capacity_chars
        self._clock = clock
        self._condition = threading.Condition()
        self._sessions_lock = threading.RLock()
        self._pending: list[_Request] = []
        self._requests: dict[str, _Request] = {}
        self._sessions: dict[str, list[ChatMessage]] = {}
        self._session_revisions: dict[str, int] = {}
        self._decoder_states: OrderedDict[str, object] = OrderedDict()
        self._decoder_state_sizes: dict[str, int] = {}
        self._decoder_state_bytes = 0
        self._decoder_state_capacity = int(session_state_capacity_bytes)
        self._decoder_state_evictions = 0
        self._sequence = 0
        self._accepting = True
        self._stopping = False
        self._active: _Request | None = None
        self._stats_lock = threading.Lock()
        self._stats: dict[str, float] = {
            "submitted": 0,
            "started": 0,
            "completed": 0,
            "cancelled": 0,
            "failed": 0,
            "expired": 0,
            "tokens": 0,
            "selected_batches": 0,
            "selected_requests": 0,
            "queue_wait_seconds": 0.0,
            "service_seconds": 0.0,
        }
        self._worker = threading.Thread(target=self._run, name="olmoe-service", daemon=True)
        self._worker.start()

    def submit(
        self,
        prompt: str,
        *,
        session_id: str | None = None,
        request_id: str | None = None,
        priority: int = 0,
        deadline: float | None = None,
        timeout: float | None = None,
        max_tokens: int = 64,
        stop_prefixes: Sequence[str] = (),
    ) -> RequestHandle:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must contain text")
        if len(prompt) > self._session_capacity_chars:
            raise ValueError("prompt exceeds session character capacity")
        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if (not isinstance(stop_prefixes, (tuple, list)) or
                any(type(prefix) is not str or not prefix or len(prefix) > 4096
                    for prefix in stop_prefixes)):
            raise ValueError("stop_prefixes must contain bounded nonempty strings")
        normalized_stop_prefixes = tuple(dict.fromkeys(stop_prefixes))
        if timeout is not None:
            if (type(timeout) not in (int, float) or isinstance(timeout, bool) or
                    not math.isfinite(timeout) or timeout <= 0):
                raise ValueError("timeout must be positive")
            if deadline is not None:
                raise ValueError("use deadline or timeout, not both")
            deadline = self._clock() + timeout
        if deadline is not None and (
                type(deadline) not in (int, float) or isinstance(deadline, bool) or
                not math.isfinite(deadline)):
            raise ValueError("deadline must be a finite timestamp")
        sid = session_id or uuid.uuid4().hex
        rid = request_id or uuid.uuid4().hex
        now = self._clock()
        with self._condition:
            if not self._accepting:
                raise ServiceClosedError("service is not accepting requests")
            if rid in self._requests:
                raise ValueError(f"duplicate request id: {rid}")
            if len(self._pending) >= self._queue_capacity:
                raise QueueFullError(rid, "queue_full", "request queue is full")
            self._sequence += 1
            request = _Request(
                request_id=rid,
                session_id=sid,
                prompt=prompt,
                priority=int(priority),
                deadline=deadline,
                max_tokens=max_tokens,
                stop_prefixes=normalized_stop_prefixes,
                sequence=self._sequence,
                created=now,
            )
            self._requests[rid] = request
            self._pending.append(request)
            self._increment("submitted")
            self._emit(request, "queued", priority=request.priority, deadline=deadline)
            self._condition.notify()
        return RequestHandle(self, request)

    def cancel(self, request_id: str) -> bool:
        with self._condition:
            request = self._requests.get(request_id)
            if request is None or request.done.is_set():
                return False
            request.cancel_requested.set()
            self._emit(request, "cancellation_requested")
            if request in self._pending:
                self._pending.remove(request)
                self._finish_error(
                    request,
                    RequestCancelledError(request.request_id, "cancelled", "request was cancelled"),
                    "cancelled",
                )
            self._condition.notify_all()
            return True

    def create_session(
        self,
        session_id: str | None = None,
        messages: Sequence[ChatMessage] = (),
    ) -> SessionSnapshot:
        sid = session_id or uuid.uuid4().hex
        initial = list(messages)
        if any(not isinstance(message, ChatMessage) for message in initial):
            raise TypeError("messages must contain ChatMessage values")
        if sum(len(message.content) for message in initial) > self._session_capacity_chars:
            raise ValueError("messages exceed session character capacity")
        with self._sessions_lock:
            if sid in self._sessions:
                raise ValueError(f"session already exists: {sid}")
            if len(self._sessions) >= self._session_capacity:
                raise ValueError("session capacity reached")
            self._sessions[sid] = initial
            self._session_revisions[sid] = 0
        return SessionSnapshot(sid, tuple(initial), 0)

    def session(self, session_id: str) -> SessionSnapshot:
        with self._sessions_lock:
            messages = tuple(self._sessions.get(session_id, ()))
            revision = self._session_revisions.get(session_id, 0)
        return SessionSnapshot(session_id, messages, revision)

    def clear_session(self, session_id: str) -> bool:
        with self._condition:
            if any(item.session_id == session_id for item in self._requests.values()):
                raise RuntimeError("cannot clear a session with outstanding requests")
            with self._sessions_lock:
                existed = session_id in self._sessions
                self._sessions.pop(session_id, None)
                self._session_revisions.pop(session_id, None)
                self._decoder_states.pop(session_id, None)
                self._decoder_state_bytes -= self._decoder_state_sizes.pop(session_id, 0)
                return existed

    def telemetry(self) -> dict[str, object]:
        with self._condition:
            queued = len(self._pending)
            active = self._active.request_id if self._active else None
            accepting = self._accepting
        with self._sessions_lock:
            sessions = len(self._sessions)
            decoder_states = len(self._decoder_states)
            decoder_state_bytes = self._decoder_state_bytes
            decoder_state_evictions = self._decoder_state_evictions
        with self._stats_lock:
            stats = dict(self._stats)
        started = int(stats["started"])
        completed = int(stats["completed"])
        stats.update(
            {
                "queued": queued,
                "active_request_id": active,
                "sessions": sessions,
                "accepting": accepting,
                "worker_alive": self._worker.is_alive(),
                "queue_capacity": self._queue_capacity,
                "microbatch_size": self._microbatch_size,
                "decoder_states": decoder_states,
                "decoder_state_bytes": decoder_state_bytes,
                "decoder_state_capacity_bytes": self._decoder_state_capacity,
                "decoder_state_evictions": decoder_state_evictions,
                "execution_mode": "single-decoder-worker",
                "average_queue_wait_ms": (stats["queue_wait_seconds"] * 1000 / started) if started else 0.0,
                "average_service_ms": (stats["service_seconds"] * 1000 / completed) if completed else 0.0,
            }
        )
        return stats

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        with self._condition:
            self._accepting = False
            self._stopping = True
            if cancel_pending:
                for request in self._requests.values():
                    request.cancel_requested.set()
                if self._active is not None:
                    self._active.cancel_requested.set()
            self._condition.notify_all()
        if wait and threading.current_thread() is not self._worker:
            self._worker.join()

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "OlmoeService":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._stopping:
                    self._condition.wait()
                if not self._pending:
                    if self._stopping:
                        return
                    continue
                selected = self._select_locked()
            for request in selected:
                self._execute(request)
            with self._condition:
                if self._stopping and not self._pending:
                    return

    def _select_locked(self) -> list[_Request]:
        now = self._clock()
        live: list[_Request] = []
        for request in self._pending:
            if request.cancel_requested.is_set():
                self._finish_error(
                    request,
                    RequestCancelledError(request.request_id, "cancelled", "request was cancelled"),
                    "cancelled",
                )
            elif request.deadline is not None and request.deadline <= now:
                self._finish_error(
                    request,
                    RequestDeadlineError(request.request_id, "deadline_exceeded", "request deadline elapsed"),
                    "expired",
                )
            else:
                live.append(request)
        # Earliest deadline wins; requests without deadlines then use higher
        # numeric priority and stable submission order.
        live.sort(key=lambda item: (item.deadline is None, item.deadline or 0.0, -item.priority, item.sequence))
        selected = live[: self._microbatch_size]
        selected_ids = {item.request_id for item in selected}
        self._pending = [item for item in live if item.request_id not in selected_ids]
        if selected:
            self._increment("selected_batches")
            self._increment("selected_requests", len(selected))
            for index, request in enumerate(selected):
                self._emit(request, "selected", batch_size=len(selected), batch_index=index)
        return selected

    def _execute(self, request: _Request) -> None:
        if request.done.is_set():
            return
        if request.cancel_requested.is_set():
            self._finish_error(
                request,
                RequestCancelledError(request.request_id, "cancelled", "request was cancelled"),
                "cancelled",
            )
            return
        if request.deadline is not None and request.deadline <= self._clock():
            self._finish_error(
                request,
                RequestDeadlineError(request.request_id, "deadline_exceeded", "request deadline elapsed"),
                "expired",
            )
            return
        request.started = self._clock()
        with self._condition:
            self._active = request
        self._increment("started")
        self._increment("queue_wait_seconds", request.started - request.created)
        self._emit(request, "started")
        with self._sessions_lock:
            history = tuple(self._sessions.get(request.session_id, ()))
            decoder_state = self._decoder_states.get(request.session_id)
            if decoder_state is not None:
                self._decoder_states.move_to_end(request.session_id)
        messages = history + (ChatMessage("user", request.prompt),)
        chunks: list[str] = []
        output_chars = 0
        stopped_by_prefix = False
        pending_prefix = ""
        max_prefix_chars = max((len(prefix) for prefix in request.stop_prefixes), default=0)

        def cancelled() -> bool:
            return request.cancel_requested.is_set() or (
                request.deadline is not None and request.deadline <= self._clock()
            )

        def publish(text: str) -> None:
            nonlocal output_chars
            if not text:
                return
            output_chars += len(text)
            if output_chars > self._session_capacity_chars:
                raise ServiceRequestError(request.request_id, "resource_limit", "output exceeds character capacity")
            chunks.append(text)
            self._increment("tokens")
            self._emit(request, "token", text=text)

        def on_token(token: str) -> None:
            nonlocal stopped_by_prefix, pending_prefix
            if cancelled():
                return
            text = str(token)
            if stopped_by_prefix:
                return
            candidate = pending_prefix + text
            matched = next((prefix for prefix in request.stop_prefixes
                            if candidate.endswith(prefix)), None)
            if matched is not None:
                publish(candidate[:-len(matched)])
                pending_prefix = ""
                stopped_by_prefix = True
                return
            if max_prefix_chars:
                hold = min(max_prefix_chars - 1, len(candidate))
                publish(candidate[:-hold] if hold else candidate)
                pending_prefix = candidate[-hold:] if hold else ""
            else:
                publish(text)

        try:
            if sum(len(message.content) for message in messages) > self._session_capacity_chars:
                raise ServiceRequestError(request.request_id, "resource_limit", "session character capacity reached")
            with self._sessions_lock:
                if request.session_id not in self._sessions and len(self._sessions) >= self._session_capacity:
                    raise ServiceRequestError(request.request_id, "resource_limit", "session capacity reached")
            if isinstance(self._decoder, StatefulStreamingDecoder):
                if decoder_state is None:
                    self._decoder.reset_state()
                else:
                    self._decoder.restore_state(copy.deepcopy(decoder_state))
            returned = self._decoder.generate(
                messages,
                max_tokens=request.max_tokens,
                on_token=on_token,
                cancelled=cancelled,
            )
            if returned is not None:
                if isinstance(returned, str):
                    if not chunks:
                        on_token(returned)
                else:
                    for chunk in returned:
                        if cancelled() or stopped_by_prefix:
                            break
                        on_token(str(chunk))
            if not stopped_by_prefix:
                publish(pending_prefix)
                pending_prefix = ""
            if request.cancel_requested.is_set():
                raise RequestCancelledError(request.request_id, "cancelled", "request was cancelled")
            if request.deadline is not None and request.deadline <= self._clock():
                raise RequestDeadlineError(request.request_id, "deadline_exceeded", "request deadline elapsed")
            result = "".join(chunks)
            state = None
            if isinstance(self._decoder, StatefulStreamingDecoder):
                state = copy.deepcopy(self._decoder.snapshot_state())
            # Cancellation and successful publication share one linearization lock.
            with self._condition:
                if request.cancel_requested.is_set():
                    raise RequestCancelledError(request.request_id, "cancelled", "request was cancelled")
                if request.deadline is not None and request.deadline <= self._clock():
                    raise RequestDeadlineError(request.request_id, "deadline_exceeded", "request deadline elapsed")
                with self._sessions_lock:
                    if sum(len(message.content) for message in messages) + len(result) > self._session_capacity_chars:
                        raise ServiceRequestError(request.request_id, "resource_limit", "session character capacity reached")
                    if request.session_id not in self._sessions and len(self._sessions) >= self._session_capacity:
                        raise ServiceRequestError(request.request_id, "resource_limit", "session capacity reached")
                    if state is not None:
                        self._store_decoder_state(request.session_id, state)
                    session = self._sessions.setdefault(request.session_id, [])
                    session.extend((ChatMessage("user", request.prompt), ChatMessage("assistant", result)))
                    self._session_revisions[request.session_id] = self._session_revisions.get(request.session_id, 0) + 1
                request.result_text = result
                self._increment("completed")
                self._finish(request, "completed", text=result)
        except RequestCancelledError as exc:
            self._finish_error(request, exc, "cancelled")
        except RequestDeadlineError as exc:
            self._finish_error(request, exc, "expired")
        except ServiceRequestError as exc:
            self._finish_error(request, exc, "failed")
        except Exception as exc:
            error = ServiceRequestError(request.request_id, "decoder_error", str(exc) or type(exc).__name__)
            self._finish_error(request, error, "failed")
        finally:
            if request.started is not None:
                self._increment("service_seconds", self._clock() - request.started)
            with self._condition:
                if self._active is request:
                    self._active = None
                self._condition.notify_all()

    def _finish_error(self, request: _Request, error: ServiceRequestError, event: str) -> None:
        if request.done.is_set():
            return
        request.error = error
        self._increment(event if event != "expired" else "expired")
        self._finish(request, event, code=error.code, message=str(error))

    def _finish(self, request: _Request, event: str, **data: object) -> None:
        with self._condition:
            request.finished = self._clock()
            with request.event_condition:
                self._emit(request, event, **data)
                request.done.set()
                request.event_condition.notify_all()
            self._requests.pop(request.request_id, None)

    def _emit(self, request: _Request, kind: str, **data: object) -> None:
        item = ServiceEvent(
            request_id=request.request_id,
            kind=kind,
            timestamp=time.time(),
            session_id=request.session_id,
            data=data,
        )
        with request.event_condition:
            if len(request.events) >= self._event_capacity:
                # A bounded consumer queue may discard intermediate token
                # events, but it must never discard the terminal boundary.
                # Otherwise a slow consumer can observe ``done`` without the
                # completion/failure event that explains how the request ended.
                if kind in self.TERMINAL_EVENTS:
                    for index, queued in enumerate(request.events):
                        if queued.kind not in self.TERMINAL_EVENTS:
                            del request.events[index]
                            break
                    else:
                        request.events.popleft()
                else:
                    request.events.popleft()
            request.events.append(item)
            request.event_condition.notify_all()

    def _increment(self, name: str, amount: float = 1) -> None:
        with self._stats_lock:
            self._stats[name] += amount

    def _store_decoder_state(self, session_id: str, state: object) -> None:
        size = _state_size_bytes(state)
        previous = self._decoder_state_sizes.pop(session_id, 0)
        if session_id in self._decoder_states:
            self._decoder_states.pop(session_id)
        self._decoder_state_bytes -= previous
        if self._decoder_state_capacity == 0 or size > self._decoder_state_capacity:
            return
        while self._decoder_states and self._decoder_state_bytes + size > self._decoder_state_capacity:
            evicted_id, _ = self._decoder_states.popitem(last=False)
            self._decoder_state_bytes -= self._decoder_state_sizes.pop(evicted_id, 0)
            self._decoder_state_evictions += 1
        self._decoder_states[session_id] = state
        self._decoder_state_sizes[session_id] = size
        self._decoder_state_bytes += size


def _render_conversation(messages: Sequence[ChatMessage]) -> str:
    parts = ["|||IP_ADDRESS|||"]
    for message in messages:
        parts.append(f"<|{message.role}|>\n{message.content}\n")
        if message.role == "assistant":
            parts.append("|||IP_ADDRESS|||\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _render_new_turn(message: ChatMessage) -> str:
    if message.role != "user":
        raise ValueError("an incremental OLMoE turn must end with a user message")
    return f"<|user|>\n{message.content}\n<|assistant|>\n"


def _state_size_bytes(value: object, seen: set[int] | None = None) -> int:
    """Estimate retained array and container payload without serializing it."""

    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int) and nbytes >= 0:
        return nbytes
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_state_size_bytes(key, seen) + _state_size_bytes(item, seen) for key, item in value.items())
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return sys.getsizeof(value) + sum(_state_size_bytes(item, seen) for item in value)
    if isinstance(value, (bytes, bytearray, memoryview, str)):
        return value.nbytes if isinstance(value, memoryview) else len(value.encode("utf-8")) if isinstance(value, str) else len(value)
    return sys.getsizeof(value)

