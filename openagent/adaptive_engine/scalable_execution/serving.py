"""Serving integration for explicitly bound durable decoder planners."""
from dataclasses import replace
from uuid import uuid4

from ..execution_service import ExecutionLimits, NativeExecutionService
from .facade import ScalableSession
from .scheduler import ScalableExecutor


def create_scalable_service(executor, planner, limits, *, owner=None,
                            own_executor=False):
    """Create isolated session facades over shared admitted immutable resources.

    Planner must create fresh request state and put persistent state entirely
    in the supplied cache transaction. Existing durable sessions are preserved;
    complete cache transport can recover them into a new serving session.
    """
    if type(executor) is not ScalableExecutor or not callable(planner):
        raise TypeError('concrete scalable executor and decoder planner required')
    if type(limits) is not ExecutionLimits or type(own_executor) is not bool:
        raise TypeError('execution limits and explicit ownership policy required')
    if owner is None:
        owner = uuid4().hex
    if type(owner) is not str or not owner or len(owner) > 4096:
        raise ValueError('bounded durable serving owner required')
    geometry = executor.geometry
    if limits.context > geometry.max_context_tokens or limits.max_sessions > executor.cache.max_sessions:
        raise ValueError('serving limits exceed durable executor capacities')
    bound = getattr(planner, 'geometry', geometry)
    if bound != geometry:
        raise ValueError('planner differs from the complete executor geometry')
    codec = getattr(planner, 'codec', None)
    config = getattr(codec, 'config', None)
    model_vocabulary = getattr(config, 'vocab', None)
    if model_vocabulary is None:
        model_vocabulary = getattr(getattr(config, 'mla', None), 'vocab', None)
    if model_vocabulary is not None and model_vocabulary != limits.vocabulary:
        raise ValueError('serving vocabulary differs from the decoder contract')
    limits = replace(limits, prefill_chunk_tokens=min(limits.prefill_chunk_tokens,
                                                     geometry.max_batch_tokens))
    def factory():
        return ScalableSession(executor, uuid4().hex, owner, planner,
                               vocab_size=limits.vocabulary)
    return NativeExecutionService(factory, limits,
                                  factory_cleanup=executor.close if own_executor else None)


__all__ = ['create_scalable_service']
