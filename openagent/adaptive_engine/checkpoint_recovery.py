"""Explicit, checksum-gated recovery of immutable checkpoint files."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout, wait, FIRST_COMPLETED
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener


class _HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        if urlsplit(newurl).scheme.lower() != "https":
            raise ValueError("checkpoint redirect must remain HTTPS")
        return super().redirect_request(request, fp, code, message, headers, newurl)


class RecoveryCancelled(RuntimeError):
    """Cooperative cancellation; durable partial prefixes remain resumable."""


def _check_cancel(cancel_event=None, stop_event=None):
    if ((cancel_event is not None and cancel_event.is_set()) or
            (stop_event is not None and stop_event.is_set())):
        raise RecoveryCancelled("checkpoint recovery cancelled")


def _wait_cancel(delay, cancel_event, stop_event):
    deadline = time.monotonic() + delay
    while True:
        _check_cancel(cancel_event, stop_event)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        stop_event.wait(min(remaining, 0.1))


def _sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(value, output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(path, cancel_event=None):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            _check_cancel(cancel_event)
            digest.update(block)
    _check_cancel(cancel_event)
    return digest.hexdigest()


@contextmanager
def _destination_lock(target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_name(target.name + ".recovery.lock").open("a+b") as lock:
        if lock.seek(0, os.SEEK_END) == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            lock.seek(0)
            if os.name == "nt":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def recover_file(url: str, destination: Path, **options) -> dict:
    """Recover one destination while preventing simultaneous writers."""
    target = Path(destination).resolve()
    _check_cancel(options.get("cancel_event"))
    with _destination_lock(target):
        return _recover_file(url, target, **options)


def _recover_file(url: str, destination: Path, *, size: int, sha256: str,
                 block_bytes: int = 16 * 1024**2, retries: int = 5,
                 workers: int = 4, opener=None, progress=None, cancel_event=None,
                 request_timeout: float = 60, backoff_max: float = 10) -> dict:
    if type(size) is not int or size <= 0 or type(block_bytes) is not int or block_bytes <= 0:
        raise ValueError("file size and block size must be positive integers")
    if type(retries) is not int or retries < 0:
        raise ValueError("retries must be nonnegative")
    if type(workers) is not int or not 1 <= workers <= 16 or block_bytes > 64 * 1024**2:
        raise ValueError("workers must be 1..16 and blocks no larger than 64 MiB")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) or v <= 0 for v in (request_timeout, backoff_max)):
        raise ValueError("request timeout and maximum backoff must be finite and positive")
    _check_cancel(cancel_event)
    if not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise ValueError("SHA-256 digest must be lowercase hexadecimal")
    if not url.startswith("https://"):
        raise ValueError("checkpoint downloads require HTTPS")
    if opener is None:
        opener = build_opener(_HTTPSRedirectHandler()).open
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".recovery")
    identity_path = partial.with_name(partial.name + ".json")
    progress_path = partial.with_name(partial.name + ".progress.json")
    identity = {"url": url, "size": size, "sha256": sha256}
    if partial.exists():
        if not identity_path.is_file() or json.loads(identity_path.read_text()) != identity:
            raise ValueError("partial checkpoint identity differs; preserve it and choose another destination")
        if progress_path.exists():
            persisted = json.loads(progress_path.read_text(encoding="utf-8"))
            committed = persisted.get("bytes")
            if (persisted.get("identity") != identity or type(committed) is not int or
                    not 0 <= committed <= partial.stat().st_size <= size):
                raise ValueError("invalid recovery progress; partial checkpoint preserved")
            with partial.open("r+b") as output:
                output.truncate(committed)
                output.flush()
                os.fsync(output.fileno())
        if partial.stat().st_size == size and _digest(partial, cancel_event) != sha256:
            failed = partial.with_name(partial.name + ".failed-" + uuid.uuid4().hex)
            os.replace(partial, failed)
            partial.touch(exist_ok=False)
            _atomic_json(progress_path, {"identity": identity, "bytes": 0})
    else:
        _atomic_json(identity_path, identity)
        temporary = partial.with_name(partial.name + ".tmp-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as output:
                if target.is_file() and target.stat().st_size <= size:
                    with target.open("rb") as source:
                        while True:
                            _check_cancel(cancel_event)
                            block = source.read(1024**2)
                            if not block:
                                break
                            output.write(block)
                output.flush()
                os.fsync(output.fileno())
            _check_cancel(cancel_event)
            os.replace(temporary, partial)
            _sync_directory(partial.parent)
            _atomic_json(progress_path, {"identity": identity, "bytes": partial.stat().st_size})
        finally:
            temporary.unlink(missing_ok=True)
    if partial.stat().st_size > size:
        raise ValueError("partial checkpoint exceeds the declared size")
    _atomic_json(progress_path, {"identity": identity, "bytes": partial.stat().st_size})
    if progress is not None:
        progress(partial.stat().st_size, size)
    _check_cancel(cancel_event)
    stop_event = threading.Event()
    def fetch(start, end):
        request = Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
        for attempt in range(retries + 1):
            _check_cancel(cancel_event, stop_event)
            try:
                with opener(request, timeout=request_timeout) as response:
                    content_range = response.headers.get("Content-Range", "")
                    if response.status != 206 or content_range != f"bytes {start}-{end}/{size}":
                        raise ValueError(f"server did not honor the exact byte range: {content_range}")
                    chunks = []
                    remaining = end - start + 2
                    while remaining:
                        _check_cancel(cancel_event, stop_event)
                        block = response.read(min(remaining, 1024**2))
                        if not block:
                            break
                        chunks.append(block)
                        remaining -= len(block)
                    data = b"".join(chunks)
                    if len(data) != end - start + 1:
                        raise OSError("checkpoint range is incomplete or oversized")
                _check_cancel(cancel_event, stop_event)
                return data
            except (OSError, TimeoutError):
                _check_cancel(cancel_event, stop_event)
                if attempt == retries:
                    raise
                _wait_cancel(min(2.0 ** min(attempt + 1, 30), backoff_max), cancel_event, stop_event)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        try:
            while partial.stat().st_size < size:
                _check_cancel(cancel_event)
                start = partial.stat().st_size
                ranges = [(offset, min(size, offset + block_bytes) - 1)
                          for offset in range(start, min(size, start + workers * block_bytes), block_bytes)]
                futures = [pool.submit(fetch, first, last) for first, last in ranges]
                # Commit in range order; publish progress only after data and journal sync.
                for future in futures:
                    while True:
                        _check_cancel(cancel_event)
                        try:
                            data = future.result(timeout=0.1)
                            break
                        except FutureTimeout:
                            if future.done():
                                data = future.result()
                                break
                    _check_cancel(cancel_event)
                    committed = partial.stat().st_size
                    try:
                        with partial.open("ab") as output:
                            output.write(data)
                            output.flush()
                            os.fsync(output.fileno())
                        _atomic_json(progress_path, {"identity": identity, "bytes": committed + len(data)})
                    except BaseException:
                        # If journal replacement completed, keep its durable prefix.
                        durable = json.loads(progress_path.read_text(encoding="utf-8"))["bytes"]
                        with partial.open("r+b") as output:
                            output.truncate(durable)
                            output.flush()
                            os.fsync(output.fileno())
                        raise
                    if progress is not None:
                        progress(committed + len(data), size)
        finally:
            stop_event.set()
            for future in futures:
                future.cancel()
    if _digest(partial, cancel_event) != sha256:
        raise ValueError("checkpoint checksum mismatch; original and recovery file preserved")
    _check_cancel(cancel_event)
    os.replace(partial, target)
    _sync_directory(target.parent)
    identity_path.unlink()
    progress_path.unlink(missing_ok=True)
    return {"path": str(target), "size": size, "sha256": sha256, "verified": True}


class _Cancellation:
    def __init__(self, external):
        self.external = external
        self.local = threading.Event()

    def is_set(self):
        return self.local.is_set() or (self.external is not None and self.external.is_set())


def recover_manifest(specification, destination=None, *, file_workers=3,
                     cancel_event=None, progress=None, on_complete=None, **options) -> list:
    """Recover a manifest dict or JSON path; return results in manifest order.

    progress(name, done, total) and on_complete(result) may run concurrently.
    Cancellation cannot interrupt a blocking read; request_timeout bounds
    socket inactivity and injected openers must honor it. Durable prefixes stay.
    """
    _check_cancel(cancel_event)
    if type(file_workers) is not int or not 1 <= file_workers <= 4:
        raise ValueError("file-workers must be 1..4")
    if isinstance(specification, (str, Path)):
        path = Path(specification).resolve()
        specification = json.loads(path.read_text(encoding="utf-8"))
        if destination is None:
            destination = path.parent
    if destination is None:
        raise ValueError("a destination is required for an in-memory manifest")
    files = specification["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("manifest files must be a nonempty list")
    names = set()
    owned_paths = set()
    for item in files:
        name = item["name"]
        if (not isinstance(name, str) or not name or name in (".", "..") or
                any(c in name for c in ('/', '\\', ':', '\0')) or Path(name).name != name):
            raise ValueError("checkpoint filenames must not contain directory components")
        key = os.path.normcase(name)
        if key in names:
            raise ValueError("checkpoint filenames must be unique")
        names.add(key)
        paths = {os.path.normcase(name + suffix) for suffix in (
            "", ".recovery", ".recovery.json", ".recovery.progress.json", ".recovery.lock")}
        if paths & owned_paths:
            raise ValueError("checkpoint names overlap recovery state files")
        owned_paths.update(paths)
        if type(item["size"]) is not int or item["size"] <= 0:
            raise ValueError("file size must be a positive integer")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise ValueError("SHA-256 digest must be lowercase hexadecimal")
        if not isinstance(item["url"], str) or not item["url"].startswith("https://"):
            raise ValueError("checkpoint downloads require HTTPS")
    root = Path(destination).resolve()
    for name in (item["name"] for item in files):
        if not (root / name).resolve().is_relative_to(root):
            raise ValueError("checkpoint destination escapes manifest directory")
    cancellation = _Cancellation(cancel_event)
    def recover(item):
        _check_cancel(cancellation)
        def notify(done, total):
            if progress is not None:
                progress(item["name"], done, total)
        result = recover_file(item["url"], root / item["name"], size=item["size"],
                              sha256=item["sha256"], cancel_event=cancellation,
                              progress=notify, **options)
        if on_complete is not None:
            on_complete(result)
        return result
    results = [None] * len(files)
    with ThreadPoolExecutor(max_workers=file_workers) as pool:
        futures = {}
        try:
            for index, item in enumerate(files):
                _check_cancel(cancellation)
                futures[pool.submit(recover, item)] = index
            pending = set(futures)
            while pending:
                _check_cancel(cancellation)
                done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    results[futures[future]] = future.result()
        finally:
            cancellation.local.set()
            for future in futures:
                future.cancel()
    _check_cancel(cancel_event)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specification", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--file-workers", type=int, default=3)
    args = parser.parse_args(argv)
    if not args.allow_download:
        parser.error("external checkpoint downloads require --allow-download")
    if not 1 <= args.file_workers <= 4:
        parser.error("file-workers must be 1..4")
    try:
        output_lock = threading.Lock()
        def emit(payload):
            with output_lock:
                print(json.dumps(payload), flush=True)
        def progress(name, done, total):
            emit({"file": name, "bytes": done, "total": total})
        recover_manifest(args.specification, file_workers=args.file_workers,
                         progress=progress, on_complete=emit)
        return 0
    except (RecoveryCancelled, KeyboardInterrupt):
        print(json.dumps({"cancelled": True, "error": "checkpoint recovery cancelled"}), flush=True)
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc)}), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
