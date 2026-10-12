"""Injected, unchanged implementation boundaries; no default oracle."""
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


@dataclass(frozen=True)
class Observation:
    """Detached logits [appended tokens, vocabulary] and complete cache leaves."""
    logits: np.ndarray
    cache: Mapping[str, np.ndarray]
    evidence: Mapping[str, np.ndarray] = field(default_factory=dict)


class Session(Protocol):
    def forward(self, token_ids: Sequence[int]) -> Observation:
        """Append tokens; return every appended token's logits and logical cache."""

    def close(self) -> None:
        """Release this session without modifying another session."""


class Adapter(Protocol):
    def metadata(self) -> Mapping[str, Any]:
        """Provide exact identity, runtime provenance and complete cache_schema.

        Required fields are enforced by runner.metadata. cache_schema maps each
        leaf to a zero-based layer or 'global'. Oracle declares authoritative
        and unchanged true. These are author assertions, not a certification.
        """

    def new_session(self, *, seed: int) -> Session:
        """Independent empty state; seed all RNGs and use deterministic eval.

        No shared mutable caches/scratch, downloads, builds, or source rewriting.
        Cache canonicalization may transpose/slice occupied physical layouts,
        but must not round, recompute, or omit logical continuation state.
        An explicitly declared comparison projection may derive a comparison
        view from observed state and actual weights, retaining all raw state in
        evidence. Such views are numerically uncertified, not raw cache exports.
        """
