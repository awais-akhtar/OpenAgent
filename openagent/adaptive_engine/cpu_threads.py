"""Scoped BLAS limits for dedicated operator processes."""

from contextlib import contextmanager


@contextmanager
def blas_thread_scope(threads=None):
    """Do not use concurrently with unrelated BLAS users in the same process."""
    if threads is None:
        yield
        return
    if type(threads) is not int or threads <= 0:
        raise ValueError("BLAS threads must be a positive integer")
    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("BLAS controls require the engine extra (threadpoolctl)") from exc
    with threadpool_limits(limits=threads, user_api="blas"):
        yield
