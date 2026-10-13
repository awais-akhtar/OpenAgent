"""Cancellation boundaries for bounded binary checkpoint stream operations."""


class CheckpointStream:
    """Borrow a stream and check cancellation before every bounded transfer.

    A short write is intentional: existing counted writers advance until their
    whole frame is consumed. Closing this adapter never closes its borrowed IO.
    """

    def __init__(self, stream, *checks, chunk_bytes=65536):
        if (type(chunk_bytes) is not int or chunk_bytes <= 0 or
                not checks or not all(callable(check) for check in checks)):
            raise ValueError('bounded checkpoint transfer and cancellation owners required')
        if (not callable(getattr(stream, 'read', None)) and
                not callable(getattr(stream, 'write', None))):
            raise TypeError('borrowed checkpoint stream must expose read or write')
        self._stream, self._checks, self._chunk = stream, checks, chunk_bytes

    def check(self):
        for check in self._checks:
            check()

    def read(self, count):
        if type(count) is not int or count < 0:
            raise ValueError('counted checkpoint read required')
        if not callable(getattr(self._stream, 'read', None)):
            raise OSError('checkpoint stream is not readable')
        self.check()
        requested = min(count, self._chunk)
        value = self._stream.read(requested)
        if type(value) is not bytes or len(value) > requested:
            raise ValueError('checkpoint reader returned an invalid bounded frame')
        return value

    def write(self, data):
        if not callable(getattr(self._stream, 'write', None)):
            raise OSError('checkpoint stream is not writable')
        self.check()
        try:
            view = memoryview(data).cast('B')[:self._chunk]
        except (TypeError, ValueError) as error:
            raise TypeError('checkpoint write requires a contiguous byte buffer') from error
        written = self._stream.write(view)
        if type(written) is not int or not 0 < written <= len(view):
            raise OSError('checkpoint writer made no bounded progress')
        return written

