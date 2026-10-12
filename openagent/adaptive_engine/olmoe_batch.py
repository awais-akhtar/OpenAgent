"""Single-worker OLMoE batches with shared weights and isolated KV state."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .olmoe import OlmoeModel, _rms_norm, _rope, _softmax


def _token_vector(tokens) -> np.ndarray:
    ids = np.asarray(tokens, dtype=object)
    if (ids.ndim != 1 or not ids.size
            or any(not isinstance(token, (int, np.integer))
                   or isinstance(token, (bool, np.bool_)) for token in ids)):
        raise ValueError("tokens must be a nonempty integer vector")
    return ids


class OlmoeBatchSession:
    """An independently resumable sequence owned by one batch engine."""

    def __init__(self, engine: OlmoeBatch, context_tokens: int) -> None:
        self._engine = engine
        self.config = engine.model.config
        self.manifest = engine.model.manifest
        self._state_identity = engine.model._state_identity
        self.context_tokens = context_tokens
        shape = (self.config.num_key_value_heads, context_tokens,
                 self.config.hidden_size // self.config.num_attention_heads)
        self._k = [np.zeros(shape, dtype=np.float32)
                   for _ in range(self.config.num_hidden_layers)]
        self._v = [np.zeros_like(key) for key in self._k]
        self.position = 0

    def reset(self) -> None:
        OlmoeModel.reset(self)

    def snapshot_state(self) -> dict[str, Any]:
        return OlmoeModel.snapshot_state(self)

    def restore_state(self, state: dict[str, Any]) -> None:
        OlmoeModel.restore_state(self, state)


class OlmoeBatch:
    """Borrow a model's dense weights and expert runtime; never its KV buffers.

    The caller owns the model lifetime and schedules calls on a single worker.
    No concurrent calls or session mutations are supported. Each forward appends
    one nonempty token chunk per session and returns its last-token logits.
    Context capacity is per session; expert cache budgeting remains on the model.
    """

    def __init__(self, model: OlmoeModel) -> None:
        self.model = model

    def create_session(self, *, context_tokens: int | None = None) -> OlmoeBatchSession:
        capacity = self.model.context_tokens if context_tokens is None else context_tokens
        if (not isinstance(capacity, int) or isinstance(capacity, bool)
                or not 0 < capacity <= self.model.context_tokens):
            raise ValueError("session context capacity must fit the model")
        return OlmoeBatchSession(self, capacity)

    def step(self, sessions: Sequence[OlmoeBatchSession], token_ids) -> np.ndarray:
        """Append exactly one token per session, returning [batch, vocabulary]."""
        ids = _token_vector(token_ids)
        return self.forward(sessions, [[token] for token in ids])

    def forward(self, sessions: Sequence[OlmoeBatchSession], token_batches) -> np.ndarray:
        sessions = list(sessions)
        batches = list(token_batches)
        if not sessions or len(sessions) != len(batches):
            raise ValueError("sessions and token batches must have equal nonzero length")
        if len({id(session) for session in sessions}) != len(sessions):
            raise ValueError("a session may appear only once in a batch")
        config = self.model.config
        validated = []
        for session, tokens in zip(sessions, batches):
            if not isinstance(session, OlmoeBatchSession) or session._engine is not self:
                raise ValueError("session belongs to a different batch engine")
            ids = _token_vector(tokens)
            if np.any(ids < 0) or np.any(ids >= config.vocab_size):
                raise ValueError("token id is outside the OLMoE vocabulary")
            if session.position + ids.size > session.context_tokens:
                raise ValueError("OLMoE context capacity exceeded")
            validated.append(ids.astype(np.int64))
        boundaries = np.cumsum([0] + [len(ids) for ids in validated])
        x = self.model.embed[np.concatenate(validated)].copy()
        pending = []
        head_dim = config.hidden_size // config.num_attention_heads
        groups = config.num_attention_heads // config.num_key_value_heads
        scale = np.float32(1.0 / np.sqrt(head_dim))
        for index, layer in enumerate(self.model.layers):
            normalized = _rms_norm(x, layer["input_norm"], config.rms_norm_eps)
            q = _rms_norm(normalized @ layer["q"].T, layer["q_norm"], config.rms_norm_eps)
            k = _rms_norm(normalized @ layer["k"].T, layer["k_norm"], config.rms_norm_eps)
            v = normalized @ layer["v"].T
            context = np.empty_like(q)
            layer_pending = []
            for session, begin, end in zip(sessions, boundaries[:-1], boundaries[1:]):
                count = end - begin
                start = session.position
                positions = np.arange(start, start + count, dtype=np.float32)
                queries = _rope(q[begin:end].reshape(count, config.num_attention_heads, head_dim),
                                positions, config.rope_theta)
                new_keys = _rope(k[begin:end].reshape(count, config.num_key_value_heads, head_dim),
                                 positions, config.rope_theta).transpose(1, 0, 2)
                new_values = v[begin:end].reshape(count, config.num_key_value_heads, head_dim).transpose(1, 0, 2)
                keys = np.repeat(np.concatenate((session._k[index][:, :start], new_keys), axis=1), groups, axis=0)
                values = np.repeat(np.concatenate((session._v[index][:, :start], new_values), axis=1), groups, axis=0)
                for row in range(count):
                    available = start + row + 1
                    scores = np.einsum("hd,htd->ht", queries[row], keys[:, :available]) * scale
                    context[begin + row] = np.einsum("ht,htd->hd", _softmax(scores),
                                                    values[:, :available]).reshape(-1)
                layer_pending.append((new_keys, new_values))
            pending.append(layer_pending)
            x += context @ layer["o"].T
            normalized = _rms_norm(x, layer["post_norm"], config.rms_norm_eps)
            probabilities = _softmax(normalized @ layer["router"].T)
            order = np.argsort(-probabilities, axis=1, kind="stable")[:, :config.num_experts_per_tok]
            weights = np.take_along_axis(probabilities, order, axis=1)
            if config.norm_topk_prob:
                weights /= np.sum(weights, axis=1, keepdims=True)
            routes = [[(str(expert), float(weight)) for expert, weight in zip(ids, row)]
                      for ids, row in zip(order, weights)]
            x += self.model.experts.execute_batch(f"model.layers.{index}", normalized, routes)
        final = _rms_norm(x[boundaries[1:] - 1], self.model.final_norm, config.rms_norm_eps)
        logits = final @ self.model.lm_head.T
        # Keep all session state untouched until every layer and the head succeed.
        for index, layer_pending in enumerate(pending):
            for session, (keys, values) in zip(sessions, layer_pending):
                start = session.position
                session._k[index][:, start:start + keys.shape[1]] = keys
                session._v[index][:, start:start + values.shape[1]] = values
        for session, ids in zip(sessions, validated):
            session.position += len(ids)
        return logits
