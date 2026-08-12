"""Approved, bounded source-identity checks for scientific citations."""

from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
import ipaddress
import json
import re
import socket
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_USER_AGENT = "OpenAgent/0.13 scientific-citation-resolver"


class CitationResolutionError(ValueError):
    """Raised when a citation cannot safely cross the network boundary."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class _TitleParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_title = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_title and sum(map(len, self.parts)) < 1024:
            self.parts.append(data)

    @property
    def title(self) -> str:
        return " ".join(" ".join(self.parts).split())[:512]


Transport = Callable[[Request, float], Any]
AddressResolver = Callable[..., Sequence[tuple[Any, ...]]]


class ScientificCitationResolver:
    """Resolve citation identity after the caller has approved network access.

    A successful result confirms that a DOI or URL returned source metadata. It
    does not establish that the source entails, supports, or contradicts a
    scientific claim.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 4.0,
        max_response_bytes: int = 131_072,
        transport: Transport | None = None,
        address_resolver: AddressResolver | None = None,
    ) -> None:
        self.timeout_seconds = min(10.0, max(0.25, float(timeout_seconds)))
        self.max_response_bytes = min(1_048_576, max(1024, int(max_response_bytes)))
        self._transport = transport or self._open
        self._address_resolver = address_resolver or socket.getaddrinfo

    def __call__(self, citations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        records = [self.resolve_one(item) for item in citations]
        verified = bool(records) and all(item["identity_status"] == "verified" for item in records)
        return {
            "verified": verified,
            "resolver": "openagent-scientific-citations-v1",
            "citations": records,
            "reason": (
                "Every citation returned source identity metadata. This is not an entailment check."
                if verified
                else "One or more citations did not return verifiable source identity metadata."
            ),
            "claim_entailment_checked": False,
            "network_policy": {
                "timeout_seconds": self.timeout_seconds,
                "max_response_bytes": self.max_response_bytes,
                "redirects_followed": False,
                "public_addresses_only": True,
            },
        }

    def resolve_one(self, citation: Mapping[str, Any]) -> dict[str, Any]:
        supplied = {str(key): value for key, value in citation.items()}
        record: dict[str, Any] = {
            **supplied,
            "identity_status": "unverified",
            "claim_entailment_checked": False,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            doi = self._normalise_doi(str(citation.get("doi", "")))
            if not doi:
                doi = self._normalise_doi(str(citation.get("source", "")))
            if doi:
                record.update(self._resolve_doi(doi))
            else:
                url = str(citation.get("url") or citation.get("source") or "").strip()
                record.update(self._resolve_url(url))
        except (CitationResolutionError, HTTPError, OSError, ValueError, json.JSONDecodeError) as exc:
            record["reason"] = str(exc)
        return record

    def _resolve_doi(self, doi: str) -> dict[str, Any]:
        endpoint = f"https://api.crossref.org/works/{quote(doi, safe='')}"
        payload, response = self._fetch(endpoint, accept="application/json")
        document = json.loads(payload.decode("utf-8"))
        message = document.get("message") if isinstance(document, Mapping) else None
        if not isinstance(message, Mapping):
            raise CitationResolutionError("Crossref returned no work metadata.")
        returned_doi = self._normalise_doi(str(message.get("DOI", "")))
        if returned_doi.casefold() != doi.casefold():
            raise CitationResolutionError("Crossref returned metadata for a different DOI.")
        titles = message.get("title") if isinstance(message.get("title"), list) else []
        return {
            "identity_status": "verified",
            "source_kind": "doi",
            "doi": returned_doi,
            "canonical_url": str(message.get("URL", f"https://doi.org/{returned_doi}")),
            "metadata": {
                "title": str(titles[0])[:512] if titles else "",
                "publisher": str(message.get("publisher", ""))[:256],
                "type": str(message.get("type", ""))[:128],
            },
            "provenance": {
                "provider": "Crossref REST API",
                "request_url": endpoint,
                "http_status": self._status(response),
                "bytes_read": len(payload),
            },
            "reason": "Crossref returned metadata for the supplied DOI; scientific claim support was not assessed.",
        }

    def _resolve_url(self, url: str) -> dict[str, Any]:
        payload, response = self._fetch(url, accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1")
        final_url = str(response.geturl()) if hasattr(response, "geturl") else url
        self._validate_public_url(final_url)
        headers = response.headers if hasattr(response, "headers") else {}
        content_type = str(headers.get("Content-Type", "")).split(";", 1)[0].strip().lower()
        parser = _TitleParser()
        if content_type in {"text/html", "application/xhtml+xml"} or payload.lstrip().startswith(b"<"):
            parser.feed(payload.decode("utf-8", errors="replace"))
        return {
            "identity_status": "verified",
            "source_kind": "url",
            "canonical_url": final_url,
            "metadata": {
                "title": parser.title,
                "content_type": content_type,
                "content_length": str(headers.get("Content-Length", "")),
                "last_modified": str(headers.get("Last-Modified", ""))[:128],
            },
            "provenance": {
                "provider": "direct HTTP metadata",
                "request_url": url,
                "http_status": self._status(response),
                "bytes_read": len(payload),
            },
            "reason": "The public URL returned HTTP metadata; scientific claim support was not assessed.",
        }

    def _fetch(self, url: str, *, accept: str) -> tuple[bytes, Any]:
        self._validate_public_url(url)
        request = Request(url, headers={"Accept": accept, "User-Agent": _USER_AGENT, "Range": f"bytes=0-{self.max_response_bytes - 1}"})
        response = self._transport(request, self.timeout_seconds)
        try:
            final_url = str(response.geturl()) if hasattr(response, "geturl") else url
            self._validate_public_url(final_url)
            status = self._status(response)
            if status < 200 or status >= 300:
                raise CitationResolutionError(f"Citation endpoint returned HTTP {status}.")
            payload = response.read(self.max_response_bytes + 1)
        finally:
            if hasattr(response, "close"):
                response.close()
        if len(payload) > self.max_response_bytes:
            raise CitationResolutionError("Citation response exceeded the configured byte limit.")
        return payload, response

    def _validate_public_url(self, url: str) -> None:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise CitationResolutionError("Citation URL must use HTTP or HTTPS.")
        if parsed.username or parsed.password:
            raise CitationResolutionError("Citation URLs cannot contain credentials.")
        hostname = parsed.hostname.rstrip(".").casefold()
        if hostname == "localhost" or hostname.endswith(".localhost") or hostname.endswith(".local"):
            raise CitationResolutionError("Local and private citation addresses are blocked.")
        try:
            addresses = [ipaddress.ip_address(hostname)]
        except ValueError:
            try:
                answers = self._address_resolver(hostname, parsed.port or (443 if parsed.scheme == "https" else 80), type=socket.SOCK_STREAM)
            except socket.gaierror as exc:
                raise CitationResolutionError("Citation hostname could not be resolved.") from exc
            addresses = []
            for answer in answers:
                sockaddr = answer[4]
                if sockaddr:
                    addresses.append(ipaddress.ip_address(sockaddr[0]))
        if not addresses or any(not address.is_global for address in addresses):
            raise CitationResolutionError("Local, private, reserved, and non-public citation addresses are blocked.")

    @staticmethod
    def _normalise_doi(value: str) -> str:
        candidate = value.strip()
        for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
            if candidate.casefold().startswith(prefix):
                candidate = candidate[len(prefix):]
                break
        return candidate if _DOI_PATTERN.fullmatch(candidate) else ""

    @staticmethod
    def _status(response: Any) -> int:
        return int(getattr(response, "status", response.getcode() if hasattr(response, "getcode") else 0))

    @staticmethod
    def _open(request: Request, timeout: float) -> Any:
        opener = build_opener(_RejectRedirects())
        return opener.open(request, timeout=timeout)


def resolve_scientific_citations(citations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Default resolver used by the scientific service after explicit approval."""

    return ScientificCitationResolver()(citations)
