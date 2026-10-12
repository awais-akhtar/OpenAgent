"""Reusable codec-driven planning for concrete backend adapters."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import sys

from .facade import ExecutionPlan
from .scheduler import DispatchStep


@dataclass(frozen=True)
class StateCodec:
    """Exact-layout hooks; never infer a native cache representation.

    create(position) -> fresh mutable request-local state
    decode_page(state, readonly_view, CachePageMetadata) -> None
    encode_pages(state, token_ids, page_tokens) -> iterable[(tokens, payload)]
    dispose(state) -> None

    Encoded payloads must carry all persistent state, including recurrent/latent
    state. Hooks must not mutate a persistent backend session. A failing create
    hook must release its own partially allocated resources.
    """

    backend_identity: str
    cache_layouts: tuple
    create: Callable
    decode_page: Callable
    encode_pages: Callable
    dispose: Callable

    def __post_init__(self):
        if not self.backend_identity or not self.cache_layouts:
            raise ValueError("exact backend/cache identity required")
        if any(not callable(hook) for hook in
               (self.create, self.decode_page, self.encode_pages, self.dispose)):
            raise TypeError("all state codec hooks must be callable")


class StateCodecPlanner:
    """Restore leased pages, invoke real backend phases, encode, and dispose.

    phases(context, state) -> iterable[DispatchStep]
    logits(context, state) -> owned numpy [N,vocab] array
    scratch_estimate(tokens, max_context_tokens) -> nonnegative bytes

    The scratch estimate must cover restored state, encoded pages, logits and
    finite-check scratch. Injected math/estimates are not model certification.
    """

    def __init__(self, geometry, codec, *, phases, logits, scratch_estimate):
        if not isinstance(codec, StateCodec):
            raise TypeError("StateCodec required")
        if (codec.backend_identity != geometry.identity or
                codec.cache_layouts != geometry.cache_layouts):
            raise ValueError("codec does not match exact backend/cache contract")
        if any(not callable(hook) for hook in (phases, logits, scratch_estimate)):
            raise TypeError("phase/logit/scratch hooks required")
        self.geometry, self.codec = geometry, codec
        self.phases, self.logits, self.scratch_estimate = phases, logits, scratch_estimate

    def __call__(self, token_ids):
        scratch = self.scratch_estimate(token_ids, self.geometry.max_context_tokens)
        if type(scratch) is not int or scratch < 0:
            raise ValueError("scratch estimate must be nonnegative integer bytes")
        holder = {}

        def cleanup():
            if "state" in holder:
                self.codec.dispose(holder["state"])
            holder.clear()

        def restore(context, _):
            context.defer(cleanup)
            holder["context"] = context
            try:state = self.codec.create(context.position)
            except BaseException as primary:
                cleanup_failed_creation=getattr(primary,'cleanup',None)
                if callable(cleanup_failed_creation):context.defer(cleanup_failed_creation)
                raise
            holder["state"] = state
            for metadata in context.cache.page_metadata():
                context.check_cancel()
                with context.cache.read_page(metadata.ordinal) as payload:
                    self.codec.decode_page(state, payload, metadata)
            context.check_cancel()

        def steps():
            yield DispatchStep(None, restore)
            # Routing can depend on request-local reconstructed state.
            yield from self.phases(holder["context"], holder["state"])

        def finish(context):
            state = holder["state"]
            result = self.logits(context, state)
            stream=None
            try:
                stream=self.codec.encode_pages(state, context.token_ids, context.cache.page_tokens)
                encoded=iter(stream)
            except BaseException as primary:
                closer=getattr(stream,'close',None)
                if callable(closer):context.defer(closer)
                cleanup_failed_encoding=getattr(primary,'cleanup',None)
                if callable(cleanup_failed_encoding):context.defer(cleanup_failed_encoding)
                raise
            closer=getattr(encoded,'close',None)
            if closer is not None and not callable(closer):
                raise TypeError('encoded page iterator close must be callable')
            pending=callable(closer)
            def close_encoded():
                nonlocal pending
                if pending:
                    closer();pending=False
            if pending:context.defer(close_encoded)
            try:
                for count, item in enumerate(encoded):
                    context.check_cancel()
                    if count >= len(context.token_ids):
                        raise ValueError("codec produced too many cache pages")
                    tokens, payload = item
                    context.cache.append_page(tokens, payload)
            finally:
                primary=sys.exc_info()[1]
                try:close_encoded()
                except BaseException as cleanup:
                    if primary is None:raise
                    primary.add_note('encoded page iterator disposal retained: '+repr(cleanup))
            return result

        return ExecutionPlan(steps(), finish, scratch)
