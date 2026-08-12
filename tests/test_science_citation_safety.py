from __future__ import annotations

import json
import unittest

from openagent.science.citations import ScientificCitationResolver
from openagent.science.safety import check_citations, detect_unsupported_claims


class _Response:
    def __init__(self, body: bytes, *, url: str, content_type: str = "application/json", status: int = 200) -> None:
        self._body = body
        self._url = url
        self.status = status
        self.headers = {"Content-Type": content_type, "Content-Length": str(len(body))}

    def read(self, limit: int = -1) -> bytes:
        return self._body if limit < 0 else self._body[:limit]

    def geturl(self) -> str:
        return self._url


def _public_dns(host: str, port: int, **_kwargs: object) -> list[tuple[object, ...]]:
    return [(2, 1, 6, "", ("93.184.216.34", port))]


class ScientificCitationSafetyTests(unittest.TestCase):
    def test_citations_are_not_described_as_externally_verified_without_approval(self) -> None:
        result = check_citations("Result follows prior work [paper-1].", [{"id": "paper-1", "doi": "10.1000/test"}])
        self.assertTrue(result["passed"])
        self.assertFalse(result["external_checked"])
        self.assertFalse(result["external_verified"])

    def test_approved_resolver_controls_external_verification(self) -> None:
        result = check_citations(
            "Result follows prior work [paper-1].",
            [{"id": "paper-1", "doi": "10.1000/test"}],
            resolver=lambda citation: {"verified": citation["doi"] == "10.1000/test"},
            allow_external=True,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["external_verified"])

    def test_claims_require_explicit_evidence_ids(self) -> None:
        supported = detect_unsupported_claims([{"text": "Energy fell.", "evidence_ids": ["run:energy"]}], {"run:energy": 2.0})
        unsupported = detect_unsupported_claims("Energy fell.", {"run:energy": 2.0})
        self.assertTrue(supported["passed"])
        self.assertFalse(unsupported["passed"])

    def test_crossref_doi_resolution_preserves_provenance_without_entailment_claim(self) -> None:
        calls: list[tuple[str, float]] = []

        def transport(request: object, timeout: float) -> _Response:
            url = request.full_url  # type: ignore[attr-defined]
            calls.append((url, timeout))
            body = json.dumps({"message": {
                "DOI": "10.5555/example.1",
                "title": ["A measured result"],
                "publisher": "Example Press",
                "type": "journal-article",
                "URL": "https://doi.org/10.5555/example.1",
            }}).encode()
            return _Response(body, url=url)

        resolver = ScientificCitationResolver(transport=transport, address_resolver=_public_dns)
        result = resolver([{"id": "paper:1", "doi": "10.5555/example.1", "note": "supplied by user"}])
        record = result["citations"][0]
        self.assertTrue(result["verified"])
        self.assertEqual(record["identity_status"], "verified")
        self.assertEqual(record["note"], "supplied by user")
        self.assertEqual(record["provenance"]["provider"], "Crossref REST API")
        self.assertFalse(record["claim_entailment_checked"])
        self.assertFalse(result["claim_entailment_checked"])
        self.assertIn("not an entailment", result["reason"])
        self.assertEqual(len(calls), 1)
        self.assertLessEqual(calls[0][1], 10.0)

    def test_public_url_uses_bounded_http_metadata(self) -> None:
        seen_headers: dict[str, str] = {}

        def transport(request: object, _timeout: float) -> _Response:
            seen_headers.update(dict(request.header_items()))  # type: ignore[attr-defined]
            return _Response(
                b"<html><head><title>  Laboratory result  </title></head></html>",
                url=request.full_url,  # type: ignore[attr-defined]
                content_type="text/html; charset=utf-8",
            )

        resolver = ScientificCitationResolver(
            max_response_bytes=4096,
            transport=transport,
            address_resolver=_public_dns,
        )
        result = resolver([{"id": "web:1", "url": "https://example.org/result"}])
        record = result["citations"][0]
        self.assertEqual(record["metadata"]["title"], "Laboratory result")
        self.assertEqual(record["provenance"]["provider"], "direct HTTP metadata")
        self.assertIn("Range", seen_headers)
        self.assertFalse(result["network_policy"]["redirects_followed"])

    def test_private_and_local_targets_are_blocked_before_transport(self) -> None:
        calls: list[str] = []

        def transport(request: object, _timeout: float) -> _Response:
            calls.append(request.full_url)  # type: ignore[attr-defined]
            raise AssertionError("transport must not be reached")

        private_dns = lambda host, port, **kwargs: [(2, 1, 6, "", ("10.2.3.4", port))]
        resolver = ScientificCitationResolver(transport=transport, address_resolver=private_dns)
        targets = [
            "http://localhost/admin",
            "http://127.0.0.1/admin",
            "http://[::1]/admin",
            "https://internal.example/admin",
            "file:///etc/passwd",
            "https://user:secret@example.org/private",
        ]
        result = resolver([{"id": f"url:{index}", "url": url} for index, url in enumerate(targets)])
        self.assertFalse(result["verified"])
        self.assertEqual(calls, [])
        self.assertTrue(all(item["identity_status"] == "unverified" for item in result["citations"]))

    def test_oversized_response_and_mismatched_doi_are_not_verified(self) -> None:
        oversized = ScientificCitationResolver(
            max_response_bytes=1024,
            transport=lambda request, timeout: _Response(b"x" * 1025, url=request.full_url),
            address_resolver=_public_dns,
        )
        mismatch = ScientificCitationResolver(
            transport=lambda request, timeout: _Response(
                json.dumps({"message": {"DOI": "10.5555/different"}}).encode(),
                url=request.full_url,
            ),
            address_resolver=_public_dns,
        )
        self.assertFalse(oversized([{"id": "web:1", "url": "https://example.org"}])["verified"])
        self.assertFalse(mismatch([{"id": "paper:1", "doi": "10.5555/requested"}])["verified"])


if __name__ == "__main__":
    unittest.main()
