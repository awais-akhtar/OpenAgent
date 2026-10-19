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


def add_failure_note(primary, note):
    """Attach a note best-effort using the builtin method, never masking primary."""
    try:
        BaseException.add_note(primary, note)
    except BaseException:
        pass
