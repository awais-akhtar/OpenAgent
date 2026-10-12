"""Public-source discovery for explicit research requests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from html import unescape
import json
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
import xml.etree.ElementTree as ET

from .egress import EgressClient, EgressError


USER_AGENT = "OpenAgent/0.13 local research client"


class ExternalResearchCollector:
    """Collect metadata from public scholarly, code, and web indexes."""

    def __init__(self, *, timeout_seconds: float = 8.0, approved: bool = False) -> None:
        self.timeout_seconds = max(2.0, min(float(timeout_seconds), 20.0))
        self.approved = bool(approved)
        self.egress = EgressClient()

    def search(self, query: str, *, limit: int = 12) -> list[dict[str, Any]]:
        query = _search_terms(query)
        if not query:
            return []
        per_source = max(3, min(10, (limit + 2) // 3))
        collectors = [
            ("openalex", self._openalex, per_source),
            ("arxiv", self._arxiv, per_source),
            ("crossref", self._crossref, per_source),
            ("web", self._web, per_source),
        ]
        if re.search(r"\b(code|repository|github|implementation|library)\b", query, re.IGNORECASE):
            collectors.append(("github", self._github, min(3, per_source)))
        found: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=len(collectors), thread_name_prefix="research-source") as pool:
            futures = {pool.submit(method, query, count): name for name, method, count in collectors}
            for future in as_completed(futures):
                try:
                    found.extend(future.result())
                except (OSError, ValueError, EgressError, ET.ParseError, json.JSONDecodeError):
                    continue
        return _deduplicate(found)[: max(1, min(limit, 40))]

    def _openalex(self, query: str, limit: int) -> list[dict[str, Any]]:
        today = date.today().isoformat()
        start = f"{max(1900, date.today().year - 2)}-01-01"
        title_query = _expand_acronyms(query)
        filters = quote_plus(
            f"title.search:{title_query},from_publication_date:{start},to_publication_date:{today}"
        )
        url = (
            "https://api.openalex.org/works?"
            f"filter={filters}&sort=publication_date:desc&per-page={limit}"
        )
        payload = json.loads(self._read(url).decode("utf-8"))
        results = []
        for item in payload.get("results", []):
            title = _clean(item.get("display_name", ""))
            identifiers = item.get("ids") or {}
            paper_url = str(item.get("doi") or identifiers.get("openalex") or item.get("id") or "")
            published = str(item.get("publication_date") or "")
            authors = [
                _clean(((entry.get("author") or {}).get("display_name", "")))
                for entry in item.get("authorships") or []
            ]
            if title and paper_url and _is_available_date(published):
                location = item.get("primary_location") or {}
                results.append(
                    _source(
                        source_id=str(item.get("id") or paper_url), title=title,
                        snippet=_openalex_abstract(item.get("abstract_inverted_index") or {}),
                        url=paper_url, source_type="openalex", published_at=published,
                        metadata={
                            "authors": [name for name in authors if name],
                            "venue": ((location.get("source") or {}).get("display_name", "")),
                            "cited_by_count": item.get("cited_by_count", 0), "index": "OpenAlex",
                        },
                    )
                )
        return results

    def _arxiv(self, query: str, limit: int) -> list[dict[str, Any]]:
        encoded = quote_plus(f'all:"{query}"')
        url = (
            "https://export.arxiv.org/api/query?"
            f"search_query={encoded}&start=0&max_results={limit}&sortBy=submittedDate&sortOrder=descending"
        )
        root = ET.fromstring(self._read(url))
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        results = []
        for entry in root.findall("atom:entry", namespace):
            title = _clean(entry.findtext("atom:title", default="", namespaces=namespace))
            paper_url = entry.findtext("atom:id", default="", namespaces=namespace).strip()
            published = entry.findtext("atom:published", default="", namespaces=namespace).strip()
            authors = [
                _clean(node.findtext("atom:name", default="", namespaces=namespace))
                for node in entry.findall("atom:author", namespace)
            ]
            if title and paper_url and _is_available_date(published):
                results.append(
                    _source(
                        source_id=paper_url.rsplit("/", 1)[-1], title=title,
                        snippet=_clean(entry.findtext("atom:summary", default="", namespaces=namespace)),
                        url=paper_url, source_type="arxiv", published_at=published,
                        metadata={"authors": [name for name in authors if name], "index": "arXiv"},
                    )
                )
        return results

    def _crossref(self, query: str, limit: int) -> list[dict[str, Any]]:
        fields = "DOI,title,author,published,created,URL,abstract,type,publisher"
        url = (
            "https://api.crossref.org/works?"
            f"query.bibliographic={quote_plus(query)}&sort=published&order=desc&rows={limit}&select={fields}"
        )
        payload = json.loads(self._read(url).decode("utf-8"))
        results = []
        for item in payload.get("message", {}).get("items", []):
            title = _clean(" ".join(item.get("title") or []))
            doi = str(item.get("DOI", "")).strip()
            paper_url = str(item.get("URL", "")).strip() or (f"https://doi.org/{doi}" if doi else "")
            date_parts = ((item.get("published") or item.get("created") or {}).get("date-parts") or [[]])[0]
            published = "-".join(str(part) for part in date_parts[:3])
            authors = [
                " ".join(part for part in (author.get("given", ""), author.get("family", "")) if part).strip()
                for author in item.get("author") or []
            ]
            if title and paper_url and _is_available_date(published):
                results.append(
                    _source(
                        source_id=doi or paper_url, title=title,
                        snippet=_clean(item.get("abstract", "")), url=paper_url,
                        source_type="crossref", published_at=published,
                        metadata={
                            "authors": [name for name in authors if name], "doi": doi,
                            "publisher": item.get("publisher", ""), "work_type": item.get("type", ""),
                            "index": "Crossref",
                        },
                    )
                )
        return results

    def _github(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = f"https://api.github.com/search/repositories?q={quote_plus(query)}&sort=updated&order=desc&per_page={limit}"
        payload = json.loads(self._read(url, accept="application/vnd.github+json").decode("utf-8"))
        return [
            _source(
                source_id=str(item.get("id", "")), title=str(item.get("full_name", "")),
                snippet=str(item.get("description", "") or ""), url=str(item.get("html_url", "")),
                source_type="github", published_at=str(item.get("updated_at", "")),
                metadata={
                    "language": item.get("language"), "stars": item.get("stargazers_count", 0),
                    "license": (item.get("license") or {}).get("spdx_id", ""), "index": "GitHub",
                },
            )
            for item in payload.get("items", [])
            if item.get("full_name") and item.get("html_url")
        ]

    def _web(self, query: str, limit: int) -> list[dict[str, Any]]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        page = self._read(url).decode("utf-8", errors="replace")
        links = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            page, flags=re.IGNORECASE | re.DOTALL,
        )
        snippets = re.findall(
            r'<(?:a|div)[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</(?:a|div)>',
            page, flags=re.IGNORECASE | re.DOTALL,
        )
        results = []
        for index, (href, title) in enumerate(links[:limit]):
            target = _unwrap_search_url(unescape(href))
            if not target.startswith(("http://", "https://")):
                continue
            results.append(
                _source(
                    source_id=target, title=_clean(title),
                    snippet=_clean(snippets[index] if index < len(snippets) else ""),
                    url=target, source_type="web", published_at="",
                    metadata={"domain": urlparse(target).netloc, "index": "Web search"},
                )
            )
        return results

    def _read(self, url: str, *, accept: str = "application/json, application/atom+xml, text/html") -> bytes:
        response = self.egress.fetch(
            url,
            approved=self.approved,
            purpose="public research source discovery",
            headers={"User-Agent": USER_AGENT, "Accept": accept},
            timeout_seconds=self.timeout_seconds,
            max_response_bytes=2_000_000,
        )
        if response.status < 200 or response.status >= 300:
            raise EgressError(f"Research source returned HTTP {response.status}.")
        return response.body


def _source(
    *, source_id: str, title: str, snippet: str, url: str,
    source_type: str, published_at: str, metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": source_id, "title": title, "snippet": snippet[:1200], "url": url,
        "type": source_type, "published_at": published_at, "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata,
    }


def _search_terms(query: str) -> str:
    text = re.sub(
        r"\b(find|search|look up|research|latest|recent|newest|papers?|articles?|studies|study|on|about|for me)\b",
        " ", query, flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip(" .,:;?-") or query.strip()


def _expand_acronyms(query: str) -> str:
    if re.search(r"\bllms?\b", query, re.IGNORECASE) and "large language" not in query.casefold():
        return re.sub(r"\bllms?\b", "large language model", query, flags=re.IGNORECASE)
    return query


def _is_available_date(value: str) -> bool:
    match = re.match(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", value or "")
    if not match:
        return True
    year, month, day = (int(part) if part else default for part, default in zip(match.groups(), (0, 1, 1)))
    try:
        return date(year, month, day) <= date.today()
    except ValueError:
        return False


def _openalex_abstract(index: dict[str, list[int]]) -> str:
    if not index:
        return ""
    positioned = sorted((position, word) for word, positions in index.items() for position in positions)
    return " ".join(word for _, word in positioned)


def _clean(value: Any) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _unwrap_search_url(url: str) -> str:
    parsed = urlparse(url)
    target = parse_qs(parsed.query).get("uddg", [""])[0]
    return unquote(target) if target else url


def _deduplicate(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result = []
    for source in sorted(sources, key=lambda item: item.get("published_at", ""), reverse=True):
        key = str(source.get("url") or source.get("title", "")).casefold().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(source)
    return result
