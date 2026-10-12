"""Continuous, single-worker serving over the real OLMoE batch engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .olmoe_batch import OlmoeBatch, OlmoeBatchSession
from .olmoe_service import (
    ChatMessage, OlmoeChatDecoder, OlmoeService, RequestCancelledError,
    RequestDeadlineError, ServiceRequestError, _Request,
    _render_conversation, _render_new_turn,
)


@dataclass
class _Lane:
    request: _Request
    session: OlmoeBatchSession
    messages: tuple[ChatMessage, ...]
    inputs: list[int]
    offset: int = 0
    generated: list[int] = field(default_factory=list)
    rendered: str = ""
    finish_after_input: bool = False


class ContinuousOlmoeService(OlmoeService):
    """OlmoeService API with token-boundary admission and united expert routes.

    Accepts an OlmoeChatDecoder (or its model directly). The caller owns the
    model's lifetime. All model/tokenizer calls run on one worker; the caller
    must not use that model concurrently. Active KV uses fixed-capacity slots,
    bounded separately from the inherited LRU budget for committed snapshots.
    """

    def __init__(
        self,
        decoder: Any,
        *,
        active_kv_capacity_bytes: int = 512 * 1024**2,
        context_tokens: int | None = None,
        prompt_chunk_tokens: int = 32,
        **kwargs: Any,
    ) -> None:
        decoder = decoder if isinstance(decoder, OlmoeChatDecoder) else OlmoeChatDecoder(decoder)
        model = decoder.model
        context = model.context_tokens if context_tokens is None else context_tokens
        if (not isinstance(context, int) or isinstance(context, bool)
                or not 0 < context <= model.context_tokens):
            raise ValueError("context_tokens must fit the model context")
        if (not isinstance(prompt_chunk_tokens, int) or isinstance(prompt_chunk_tokens, bool)
                or prompt_chunk_tokens <= 0):
            raise ValueError("prompt_chunk_tokens must be a positive integer")
        config = model.config
        self._slot_bytes = (2 * config.num_hidden_layers * config.num_key_value_heads
                            * context * (config.hidden_size // config.num_attention_heads)
                            * np.dtype(np.float32).itemsize)
        if (not isinstance(active_kv_capacity_bytes, int) or isinstance(active_kv_capacity_bytes, bool)
                or active_kv_capacity_bytes < self._slot_bytes):
            raise ValueError("active KV budget must fit at least one session")
        self._batch = OlmoeBatch(model)
        self._context_tokens = context
        self._prompt_chunk_tokens = prompt_chunk_tokens
        self._kv_capacity = active_kv_capacity_bytes
        self._lanes: dict[str, _Lane] = {}
        self._batch_calls = 0
        self._batch_rows = 0
        self._max_batch_sessions = 0
        self._peak_kv_bytes = 0
        super().__init__(decoder, **kwargs)

    def telemetry(self) -> dict[str, object]:
        result = super().telemetry()
        with self._condition:
            result.update(
                execution_mode="continuous-olmoe-batch-worker",
                active_request_ids=list(self._lanes),
                active_requests=len(self._lanes),
                active_kv_bytes=len(self._lanes) * self._slot_bytes,
                active_kv_capacity_bytes=self._kv_capacity,
                peak_active_kv_bytes=self._peak_kv_bytes,
                batch_calls=self._batch_calls,
                batch_token_rows=self._batch_rows,
                max_batch_sessions=self._max_batch_sessions,
                prompt_chunk_tokens=self._prompt_chunk_tokens,
            )
        return result

    def _check_live(self, request: _Request) -> None:
        if request.cancel_requested.is_set():
            raise RequestCancelledError(request.request_id, "cancelled", "request was cancelled")
        if request.deadline is not None and request.deadline <= self._clock():
            raise RequestDeadlineError(request.request_id, "deadline_exceeded", "request deadline elapsed")

    def _fail(self, request: _Request, exc: Exception) -> None:
        if isinstance(exc, RequestCancelledError):
            event = "cancelled"
        elif isinstance(exc, RequestDeadlineError):
            event = "expired"
        else:
            event = "failed"
            if not isinstance(exc, ServiceRequestError):
                exc = ServiceRequestError(request.request_id, "decoder_error", str(exc) or type(exc).__name__)
        # Request handles must not retain worker frames and their active KV.
        exc.__traceback__ = None
        exc.__context__ = None
        exc.__cause__ = None
        self._finish_error(request, exc, event)
        self._release(request)

    def _release(self, request: _Request) -> None:
        with self._condition:
            self._lanes.pop(request.request_id, None)
            if request.started is not None:
                self._increment("service_seconds", self._clock() - request.started)
            self._active = next((lane.request for lane in self._lanes.values()), None)
            self._condition.notify_all()

    def _admit(self) -> None:
        # Selection and cancellation share the inherited lifecycle lock. Once
        # removed from the queue, a request remains in _requests until terminal.
        with self._condition:
            live = []
            for request in self._pending:
                try:
                    self._check_live(request)
                except (RequestCancelledError, RequestDeadlineError) as exc:
                    self._fail(request, exc)
                else:
                    live.append(request)
            live.sort(key=lambda item: (item.deadline is None, item.deadline or 0.0,
                                        -item.priority, item.sequence))
            occupied = {lane.request.session_id for lane in self._lanes.values()}
            slots = min(self._microbatch_size, self._kv_capacity // self._slot_bytes) - len(self._lanes)
            selected = []
            remaining = []
            for request in live:
                if slots > 0 and request.session_id not in occupied:
                    selected.append(request)
                    occupied.add(request.session_id)
                    slots -= 1
                else:
                    remaining.append(request)
            self._pending = remaining
            if selected:
                self._increment("selected_batches")
                self._increment("selected_requests", len(selected))
            for index, request in enumerate(selected):
                self._emit(request, "selected", batch_size=len(selected), batch_index=index)
                request.started = self._clock()
                self._increment("started")
                self._increment("queue_wait_seconds", request.started - request.created)
                self._emit(request, "started")
                try:
                    self._check_live(request)
                    lane = self._prepare(request)
                    self._lanes[request.request_id] = lane
                    self._peak_kv_bytes = max(self._peak_kv_bytes, len(self._lanes) * self._slot_bytes)
                except Exception as exc:
                    self._fail(request, exc)
            self._active = next((lane.request for lane in self._lanes.values()), None)

    def _prepare(self, request: _Request) -> _Lane:
        with self._sessions_lock:
            messages = tuple(self._sessions.get(request.session_id, ())) + (ChatMessage("user", request.prompt),)
            if request.session_id not in self._sessions and len(self._sessions) >= self._session_capacity:
                raise ServiceRequestError(request.request_id, "resource_limit", "session capacity reached")
            saved = self._decoder_states.get(request.session_id)
            if saved is not None:
                self._decoder_states.move_to_end(request.session_id)
        if sum(len(message.content) for message in messages) > self._session_capacity_chars:
            raise ServiceRequestError(request.request_id, "resource_limit", "session character capacity reached")
        session = self._batch.create_session(context_tokens=self._context_tokens)
        boundary = []
        if saved is not None:
            state, boundary = saved
            session.restore_state(state)
        # Evicted snapshots use the same full-history render as OlmoeChatDecoder.
        prompt = _render_new_turn(messages[-1]) if session.position else _render_conversation(messages)
        inputs = list(boundary) + list(self._decoder.model.encode(prompt))
        if not inputs or session.position + len(inputs) > session.context_tokens:
            raise ServiceRequestError(request.request_id, "resource_limit", "OLMoE context capacity exceeded")
        return _Lane(request, session, messages, inputs)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._lanes and not self._stopping:
                    self._condition.wait()
                if not self._pending and not self._lanes and self._stopping:
                    return
            self._tick()

    def _prune(self) -> None:
        for lane in list(self._lanes.values()):
            try:
                self._check_live(lane.request)
            except (RequestCancelledError, RequestDeadlineError) as exc:
                self._fail(lane.request, exc)

    def _tick(self) -> None:
        # A separate frame releases completed lanes before the next admission.
        self._prune()
        self._admit()
        lanes = []
        chunks = []
        for lane in list(self._lanes.values()):
            try:
                self._check_live(lane.request)
                chunk = lane.inputs[lane.offset:lane.offset + self._prompt_chunk_tokens]
                if lane.session.position + len(chunk) > lane.session.context_tokens:
                    raise ServiceRequestError(lane.request.request_id, "resource_limit", "OLMoE context capacity exceeded")
                lanes.append(lane)
                chunks.append(chunk)
            except Exception as exc:
                self._fail(lane.request, exc)
        if not lanes:
            return
        try:
            logits = self._batch.forward([lane.session for lane in lanes], chunks)
        except Exception as exc:
            # The batch engine commits no KV on failure. Every participant
            # fails, but other queued requests and committed history survive.
            for lane in lanes:
                try:
                    self._check_live(lane.request)
                except (RequestCancelledError, RequestDeadlineError) as stopped:
                    self._fail(lane.request, stopped)
                else:
                    self._fail(lane.request, exc)
            return
        with self._condition:
            self._batch_calls += 1
            self._batch_rows += sum(map(len, chunks))
            self._max_batch_sessions = max(self._max_batch_sessions, len(lanes))
        for lane, chunk, row in zip(lanes, chunks, logits):
            try:
                self._check_live(lane.request)
                lane.offset += len(chunk)
                if lane.offset < len(lane.inputs):
                    continue
                if lane.finish_after_input:
                    self._publish(lane)
                else:
                    self._sample(lane, row)
            except Exception as exc:
                self._fail(lane.request, exc)

    def _sample(self, lane: _Lane, logits: np.ndarray) -> None:
        token = int(np.argmax(logits))
        if token == self._decoder.model.config.eos_token_id:
            self._publish(lane)
            return
        lane.generated.append(token)
        text = self._decoder.model.decode(lane.generated)
        if not text.startswith(lane.rendered):
            raise ValueError("OLMoE decoding revised already emitted text")
        if len(text) > self._session_capacity_chars:
            raise ServiceRequestError(lane.request.request_id, "resource_limit", "output exceeds character capacity")
        delta = text[len(lane.rendered):]
        lane.rendered = text
        with self._condition:
            self._check_live(lane.request)
            if delta:
                self._increment("tokens")
                self._emit(lane.request, "token", text=delta, token_id=token)
        # Even the last yielded token must enter KV before a snapshot is saved.
        lane.inputs = [token]
        lane.offset = 0
        lane.finish_after_input = len(lane.generated) >= lane.request.max_tokens

    def _publish(self, lane: _Lane) -> None:
        request = lane.request
        model = self._decoder.model
        boundary = list(model.encode("\n")) + [model.config.eos_token_id] + list(model.encode("\n"))
        state = (lane.session.snapshot_state(), tuple(boundary))
        with self._condition:
            self._check_live(request)
            with self._sessions_lock:
                if sum(len(message.content) for message in lane.messages) + len(lane.rendered) > self._session_capacity_chars:
                    raise ServiceRequestError(request.request_id, "resource_limit", "session character capacity reached")
                if request.session_id not in self._sessions and len(self._sessions) >= self._session_capacity:
                    raise ServiceRequestError(request.request_id, "resource_limit", "session capacity reached")
                self._store_decoder_state(request.session_id, state)
                session = self._sessions.setdefault(request.session_id, [])
                session.extend((ChatMessage("user", request.prompt), ChatMessage("assistant", lane.rendered)))
                self._session_revisions[request.session_id] = self._session_revisions.get(request.session_id, 0) + 1
            request.result_text = lane.rendered
            self._increment("completed")
            self._finish(request, "completed", text=lane.rendered, token_ids=tuple(lane.generated))
            self._release(request)
