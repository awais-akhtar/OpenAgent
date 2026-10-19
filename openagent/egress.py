"""Approved outbound HTTP transport with a public-network boundary."""

from __future__ import annotations

from dataclasses import dataclass
import codecs
import http.client
import ipaddress
import json
import logging
from pathlib import Path
import re
import socket
import ssl
import threading
import time
from typing import Callable, Mapping, Sequence
from urllib.parse import urljoin, urlsplit, urlunsplit


class EgressError(RuntimeError):
    """An outbound request violates policy or cannot complete safely."""


@dataclass(frozen=True, slots=True)
class EgressDecision:
    url: str
    hostname: str
    port: int
    addresses: tuple[str, ...]
    local: bool


@dataclass(frozen=True, slots=True)
class EgressResponse:
    url: str
    status: int
    headers: Mapping[str, str]
    body: bytes


Resolver = Callable[..., Sequence[tuple[object, ...]]]
AuditSink = Callable[[Mapping[str, object]], None]

_LOGGER = logging.getLogger("openagent.egress")
_METRICS_LOCK = threading.Lock()
_METRICS: dict[str, int | float | str] = {
    "requests": 0,
    "failures": 0,
    "bytes": 0,
    "last_status": 0,
    "last_destination": "",
    "updated_at": 0.0,
}


def egress_metrics() -> dict[str, int | float | str]:
    """Return process-local counters without exposing URL queries or credentials."""

    with _METRICS_LOCK:
        return dict(_METRICS)


class EgressPolicy:
    """Validate a destination and return the addresses allowed for connection."""

    def __init__(self, *, resolver: Resolver | None = None) -> None:
        self._resolver = resolver or socket.getaddrinfo

    def authorize(
        self,
        url: str,
        *,
        approved: bool,
        allow_loopback: bool = False,
    ) -> EgressDecision:
        parsed = urlsplit(str(url).strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise EgressError("Outbound destinations must use a complete HTTP or HTTPS URL.")
        if parsed.username or parsed.password:
            raise EgressError("Outbound URLs cannot contain credentials.")
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
            if not allow_loopback:
                raise EgressError("Local and private outbound destinations are blocked.")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            literal = ipaddress.ip_address(hostname)
            addresses = [literal]
        except ValueError:
            try:
                answers = self._resolver(hostname, port, type=socket.SOCK_STREAM)
            except OSError as exc:
                raise EgressError("Outbound hostname could not be resolved.") from exc
            addresses = []
            for answer in answers:
                sockaddr = answer[4] if len(answer) > 4 else None
                if sockaddr:
                    address = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
                    if address not in addresses:
                        addresses.append(address)
        if not addresses:
            raise EgressError("Outbound hostname returned no usable addresses.")
        is_loopback = all(address.is_loopback for address in addresses)
        if is_loopback:
            if not allow_loopback:
                raise EgressError("Loopback outbound destinations are blocked for this feature.")
        elif any(not address.is_global for address in addresses):
            raise EgressError("Private, link-local, reserved, multicast, and mixed DNS destinations are blocked.")
        elif not approved:
            raise PermissionError(f"Outbound access to '{hostname}' requires explicit approval.")
        clean_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", parsed.query, ""))
        return EgressDecision(
            url=clean_url,
            hostname=hostname,
            port=port,
            addresses=tuple(str(address) for address in addresses),
            local=is_loopback,
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname: str, address: str, port: int, timeout: float) -> None:
        super().__init__(hostname, port=port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


class EgressClient:
    """Perform bounded requests after policy approval and address pinning."""

    def __init__(self, *, policy: EgressPolicy | None = None, audit: AuditSink | None = None) -> None:
        self.policy = policy or EgressPolicy()
        self.audit = audit

    def fetch(
        self,
        url: str,
        *,
        approved: bool,
        purpose: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 15.0,
        max_response_bytes: int = 5 * 1024 * 1024,
        allowed_content_types: set[str] | None = None,
        allow_loopback: bool = False,
        max_redirects: int = 4,
        on_line: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> EgressResponse:
        if cancelled is not None and not callable(cancelled):
            raise TypeError("egress cancellation must be callable")
        timeout = min(120.0, max(0.25, float(timeout_seconds)))
        byte_limit = min(100 * 1024 * 1024, max(1, int(max_response_bytes)))
        current = str(url).strip()
        request_method = method.upper().strip() or "GET"
        for redirect_count in range(max_redirects + 1):
            try:
                decision = self.policy.authorize(current, approved=approved, allow_loopback=allow_loopback)
                response = self._request_once(
                    decision,
                    method=request_method,
                    body=body,
                    headers=headers or {},
                    timeout=timeout,
                    max_bytes=byte_limit,
                    on_line=on_line,
                    cancelled=cancelled,
                    allowed_content_types=allowed_content_types,
                )
            except (EgressError, PermissionError):
                self._record_failure(current, purpose=purpose)
                raise
            if response.status in {301, 302, 303, 307, 308}:
                location = response.headers.get("location", "").strip()
                if not location:
                    raise EgressError("Outbound redirect did not include a destination.")
                if redirect_count >= max_redirects:
                    raise EgressError("Outbound request exceeded its redirect limit.")
                current = urljoin(decision.url, location)
                if response.status == 303 or (response.status in {301, 302} and request_method == "POST"):
                    request_method = "GET"
                    body = None
                continue
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().casefold()
            if allowed_content_types and content_type not in {item.casefold() for item in allowed_content_types}:
                raise EgressError(f"Outbound response content type '{content_type or 'unknown'}' is not allowed.")
            self._record(decision, purpose=purpose, status=response.status, byte_count=len(response.body))
            return response
        raise EgressError("Outbound request did not reach a final response.")

    def download(
        self,
        url: str,
        target: str | Path,
        *,
        approved: bool,
        purpose: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float = 120.0,
        max_total_bytes: int = 512 * 1024 * 1024 * 1024,
        resume: bool = True,
        max_redirects: int = 4,
    ) -> dict[str, object]:
        """Stream an approved response to a file with resume and hard size limits."""

        destination = Path(target).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        downloaded = destination.stat().st_size if resume and destination.exists() else 0
        if downloaded > max_total_bytes:
            raise EgressError("Existing partial download exceeds the configured byte limit.")
        request_headers = {str(key): str(value) for key, value in (headers or {}).items()}
        if downloaded:
            request_headers["Range"] = f"bytes={downloaded}-"
        current = str(url).strip()
        timeout = min(3600.0, max(0.25, float(timeout_seconds)))
        for redirect_count in range(max_redirects + 1):
            try:
                decision = self.policy.authorize(current, approved=approved, allow_loopback=False)
                connection, raw = self._open_response(
                    decision, method="GET", body=None, headers=request_headers, timeout=timeout
                )
            except (EgressError, PermissionError):
                self._record_failure(current, purpose=purpose)
                raise
            try:
                response_headers = {key.casefold(): value for key, value in raw.getheaders()}
                if raw.status in {301, 302, 303, 307, 308}:
                    location = response_headers.get("location", "").strip()
                    if not location or redirect_count >= max_redirects:
                        raise EgressError("Outbound download redirect is missing or exceeded its limit.")
                    current = urljoin(decision.url, location)
                    continue
                if raw.status < 200 or raw.status >= 300:
                    raise EgressError(f"Outbound download returned HTTP {raw.status}.")
                append = bool(downloaded and raw.status == 206)
                if raw.status == 206:
                    content_range = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(\d+|\*)",
                        response_headers.get("content-range", "").strip(),
                    )
                    if not append or content_range is None or int(content_range.group(1)) != downloaded:
                        raise EgressError("Outbound download returned an unexpected partial range.")
                    start, end = int(content_range.group(1)), int(content_range.group(2))
                    if end < start:
                        raise EgressError("Outbound download returned an invalid partial range.")
                    if content_range.group(3) != "*" and int(content_range.group(3)) > max_total_bytes:
                        raise EgressError("Outbound download exceeds the configured byte limit.")
                    declared_length = response_headers.get("content-length", "")
                    if declared_length.isdigit() and int(declared_length) != end - start + 1:
                        raise EgressError("Outbound download range length does not match its headers.")
                if not append:
                    downloaded = 0
                declared = response_headers.get("content-length", "")
                if declared.isdigit() and downloaded + int(declared) > max_total_bytes:
                    raise EgressError("Outbound download exceeds the configured byte limit.")
                expected_bytes = downloaded + int(declared) if declared.isdigit() else None
                mode = "ab" if append else "wb"
                with destination.open(mode) as handle:
                    while True:
                        chunk = raw.read(1024 * 1024)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        if downloaded > max_total_bytes:
                            raise EgressError("Outbound download exceeds the configured byte limit.")
                        handle.write(chunk)
                if expected_bytes is not None and downloaded != expected_bytes:
                    raise EgressError("Outbound download ended before the declared byte count.")
                self._record(decision, purpose=purpose, status=int(raw.status), byte_count=downloaded)
                return {
                    "url": decision.url,
                    "status": int(raw.status),
                    "bytes": downloaded,
                    "resumed": append,
                    "path": str(destination),
                }
            except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                self._record_failure(current, purpose=purpose)
                raise EgressError("Outbound download failed at the approved network boundary.") from exc
            finally:
                connection.close()
        raise EgressError("Outbound download did not reach a final response.")

    def _request_once(
        self,
        decision: EgressDecision,
        *,
        method: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
        max_bytes: int,
        on_line: Callable[[str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
        allowed_content_types: set[str] | None = None,
    ) -> EgressResponse:
        connection, raw = self._open_response(
            decision, method=method, body=body, headers=headers, timeout=timeout
        )
        try:
            header_map = {key.casefold(): value for key, value in raw.getheaders()}
            content_type = header_map.get("content-type", "").split(";", 1)[0].strip().casefold()
            if (
                allowed_content_types
                and int(raw.status) not in {301, 302, 303, 307, 308}
                and content_type not in {item.casefold() for item in allowed_content_types}
            ):
                raise EgressError(f"Outbound response content type '{content_type or 'unknown'}' is not allowed.")
            declared = header_map.get("content-length", "")
            if declared.isdigit() and int(declared) > max_bytes:
                raise EgressError("Outbound response exceeded the configured byte limit.")
            if on_line is None or not 200 <= int(raw.status) < 300:
                payload = raw.read(max_bytes + 1)
                if len(payload) > max_bytes:
                    raise EgressError("Outbound response exceeded the configured byte limit.")
            else:
                chunks: list[bytes] = []
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                pending = ""
                total = 0
                watcher_stop = threading.Event()
                watcher_cancelled = threading.Event()

                def close_response_on_cancel() -> None:
                    # The reader may be blocked in readline() while the caller
                    # closes its stream. Poll the caller-owned cancellation
                    # predicate off-thread so cancellation also interrupts that
                    # socket read instead of waiting for the network timeout.
                    while not watcher_stop.wait(0.05):
                        try:
                            should_cancel = bool(cancelled and cancelled())
                        except BaseException:
                            # A broken cancellation owner cannot safely keep an
                            # outbound request alive; fail closed.
                            should_cancel = True
                        if should_cancel:
                            watcher_cancelled.set()
                            try:
                                connection.close()
                            except BaseException:
                                pass
                            return

                watcher: threading.Thread | None = None
                if cancelled is not None:
                    watcher = threading.Thread(
                        target=close_response_on_cancel,
                        name="openagent-egress-cancel",
                        daemon=True,
                    )
                    watcher.start()
                try:
                    while True:
                        if watcher_cancelled.is_set() or (cancelled is not None and cancelled()):
                            raise InterruptedError("outbound streaming request cancelled")
                        chunk = raw.readline(64 * 1024)
                        if not chunk:
                            if watcher_cancelled.is_set() or (cancelled is not None and cancelled()):
                                raise InterruptedError("outbound streaming request cancelled")
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise EgressError("Outbound response exceeded the configured byte limit.")
                        chunks.append(chunk)
                        pending += decoder.decode(chunk)
                        while "\n" in pending:
                            if watcher_cancelled.is_set() or (cancelled is not None and cancelled()):
                                raise InterruptedError("outbound streaming request cancelled")
                            line, pending = pending.split("\n", 1)
                            on_line(line.rstrip("\r"))
                    if watcher_cancelled.is_set() or (cancelled is not None and cancelled()):
                        raise InterruptedError("outbound streaming request cancelled")
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        on_line(pending.rstrip("\r"))
                    payload = b"".join(chunks)
                except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
                    if watcher_cancelled.is_set() or (cancelled is not None and cancelled()):
                        raise InterruptedError("outbound streaming request cancelled") from exc
                    raise
                finally:
                    watcher_stop.set()
            return EgressResponse(decision.url, int(raw.status), header_map, payload)
        except InterruptedError:
            # Cancellation is part of the caller-visible stream contract, not
            # a transport failure.  Preserve it before the broader OSError
            # handler (InterruptedError subclasses OSError).
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            raise EgressError("Outbound request failed at the approved network boundary.") from exc
        finally:
            connection.close()

    @staticmethod
    def _open_response(
        decision: EgressDecision,
        *,
        method: str,
        body: bytes | None,
        headers: Mapping[str, str],
        timeout: float,
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        parsed = urlsplit(decision.url)
        address = decision.addresses[0]
        request_headers = {str(key): str(value) for key, value in headers.items()}
        request_headers.setdefault("Host", parsed.netloc)
        request_headers.setdefault("Connection", "close")
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"
        if parsed.scheme == "https":
            connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
                decision.hostname, address, decision.port, timeout
            )
        else:
            connection = http.client.HTTPConnection(address, port=decision.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=request_headers)
            return connection, connection.getresponse()
        except Exception:
            connection.close()
            raise

    def _record(self, decision: EgressDecision, *, purpose: str, status: int, byte_count: int) -> None:
        parsed = urlsplit(decision.url)
        record = {
            "destination": urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", "")),
            "purpose": str(purpose)[:160],
            "status": status,
            "bytes": byte_count,
            "local": decision.local,
        }
        with _METRICS_LOCK:
            _METRICS["requests"] = int(_METRICS["requests"]) + 1
            _METRICS["bytes"] = int(_METRICS["bytes"]) + int(byte_count)
            _METRICS["last_status"] = int(status)
            _METRICS["last_destination"] = str(record["destination"])
            _METRICS["updated_at"] = time.time()
        _LOGGER.info(json.dumps({"event": "egress.completed", **record}, sort_keys=True))
        if self.audit is not None:
            self.audit(record)

    @staticmethod
    def _record_failure(url: str, *, purpose: str) -> None:
        parsed = urlsplit(str(url))
        destination = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
        with _METRICS_LOCK:
            _METRICS["failures"] = int(_METRICS["failures"]) + 1
            _METRICS["last_destination"] = destination
            _METRICS["updated_at"] = time.time()
        _LOGGER.warning(json.dumps({"event": "egress.failed", "destination": destination, "purpose": str(purpose)[:160]}, sort_keys=True))
