"""Exception diagnostics that cannot replace the failure being reported."""


def failure_summary(failure, *, limit=512):
    """Return bounded type-only diagnostics without calling exception formatters."""
    if type(limit) is not int or limit <= 0:
        limit = 512
    try:
        name = type(failure).__name__
    except BaseException:
        name = 'BaseException'
    if type(name) is not str or not name or len(name) > 128:
        name = 'BaseException'
    return (name + ': failure details omitted')[:limit]


def failure_detail(failure, *, limit=2048):
    """Preserve ordinary exception text, but fall back safely if it is hostile."""
    if type(limit) is not int or limit <= 0:
        limit = 2048
    try:
        detail = str(failure)
    except BaseException:
        return failure_summary(failure, limit=limit)
    if type(detail) is not str:
        return failure_summary(failure, limit=limit)
    return detail[:limit] if detail else failure_summary(failure, limit=limit)


def add_failure_note(primary, note):
    """Attach a note best-effort using the builtin method, never masking primary."""
    try:
        BaseException.add_note(primary, note)
    except BaseException:
        pass


def retain_cleanup_failure(primary, cleanup_callback, cleanup_failure, note):
    """Keep a retry callback and annotate its failure without masking primary."""
    try:
        primary.cleanup = cleanup_callback
    except BaseException:
        pass
    add_failure_note(primary, note + failure_summary(cleanup_failure))
