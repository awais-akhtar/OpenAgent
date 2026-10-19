"""Failure-safe transfer of an owned raw descriptor to a Python stream."""

import os


def close_after_failure(descriptor, primary, *, owner):
    """Close an owned descriptor without masking the operation's failure."""
    try:
        os.close(descriptor)
    except BaseException as cleanup:
        primary.add_note(owner+' descriptor cleanup failed: '+repr(cleanup))


def fdopen_owned(descriptor, *args, owner='file', **kwargs):
    """Transfer descriptor ownership only when ``os.fdopen`` succeeds."""
    try:
        return os.fdopen(descriptor, *args, **kwargs)
    except BaseException as primary:
        close_after_failure(descriptor, primary, owner=owner)
        raise


__all__ = ['close_after_failure', 'fdopen_owned']
