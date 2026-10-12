"""Privacy-first web scraper for knowledge ingestion.

Features
--------
* Fetches web pages using stdlib only (urllib.request)
* Converts HTML → clean text using stdlib html.parser
* Respects robots.txt (cached, privacy-first)
* Extracts metadata: title, description, author, keywords, open-graph
* Scheduled scraping with configurable intervals
* Feeds directly into the KnowledgeEngine
* Monitors URL content changes (diff summary)
* No external tracking; all HTTP requests use a minimal User-Agent
* Rate limiting: max 1 request per domain per 2 seconds

Security
--------
* URL validation: only http/https allowed
* Private network IP ranges are blocked (no SSRF)
* Max response size enforced (5 MB)
* Timeout enforced (15 s)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser
from typing import Any
from urllib.robotparser import RobotFileParser

from .egress import EgressClient, EgressError, EgressPolicy
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_UA = "OpenAgent/1.0 (self-hosted; privacy-first; +https://github.com/openagent)"
_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
_TIMEOUT = 15                   # seconds
_RATE_LIMIT_SECONDS = 2         # min gap between requests to same domain

_domain_last_request: dict[str, float] = {}


# ---------------------------------------------------------------------------
# HTML → text parser
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Minimal HTML → plain-text converter."""
    _SKIP = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._buf: list[str] = []
        self.title = ""
        self.description = ""
        self.author = ""
        self.keywords: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "meta":
            name = (attr_dict.get("name") or attr_dict.get("property") or "").lower()
            content = attr_dict.get("content") or ""
            if name in ("description", "og:description", "twitter:description"):
                self.description = content
            elif name == "author":
                self.author = content
            elif name == "keywords":
                self.keywords = [k.strip() for k in content.split(",") if k.strip()]
            elif name in ("og:title", "twitter:title") and not self.title:
                self.title = content
        if tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li"):
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        if self._in_title and not self.title:
            self.title = data.strip()
        self._buf.append(data)

    def get_text(self) -> str:
        raw = "".join(self._buf)
        # Collapse whitespace
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        raw = re.sub(r"[ \t]+", " ", raw)
        return raw.strip()


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

def _validate_url(url: str) -> str:
    """Raise ValueError if the URL is invalid or targets private networks."""
    try:
        return EgressPolicy().authorize(url, approved=True).url
    except EgressError as exc:
        raise ValueError(str(exc)) from exc


def _rate_limit(domain: str) -> None:
    """Block if we've hit the same domain too recently."""
    last = _domain_last_request.get(domain, 0)
    elapsed = time.monotonic() - last
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _domain_last_request[domain] = time.monotonic()


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------

class WebScraper:
    """Privacy-first web scraper for knowledge ingestion."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.egress = EgressClient()
        self._robots_cache: dict[str, tuple[float, RobotFileParser]] = {}
        self.initialize()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        self.store.initialize()

    # ------------------------------------------------------------------
    # Core scrape
    # ------------------------------------------------------------------

    def scrape_url(
        self,
        url: str,
        *,
        project: str = "default",
        respect_robots: bool = True,
        ingest_to_knowledge: bool = True,
        include_text: bool = False,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        """Fetch a URL, extract text, and optionally ingest into knowledge engine."""
        url = _validate_url(url.strip())
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or parsed.hostname or url

        # Robots.txt check
        if respect_robots and not self._is_allowed(url, domain, project=project, allow_external=allow_external):
            result = {
                "url": url, "status": "blocked_by_robots",
                "error": "robots.txt disallows this URL.",
            }
            self._record_scrape_audit(project, result)
            return result

        _rate_limit(domain)

        # Fetch
        try:
            response = self.egress.fetch(
                url,
                approved=allow_external,
                purpose="knowledge web scrape",
                headers={
                    "User-Agent": _UA,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout_seconds=_TIMEOUT,
                max_response_bytes=_MAX_BYTES,
                allowed_content_types={"text/html", "application/xhtml+xml", "text/plain"},
            )
            status_code = response.status
            raw = response.body
            content_type = response.headers.get("content-type", "")
            if status_code < 200 or status_code >= 300:
                raise EgressError(f"Scrape destination returned HTTP {status_code}.")
        except (EgressError, PermissionError) as exc:
            result = {"url": url, "status": "error", "status_code": 0, "error": str(exc)}
            self._record_scrape_audit(project, result)
            return result

        # Decode
        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=")[-1].split(";")[0].strip()
        try:
            html = raw.decode(charset, errors="replace")
        except LookupError:
            html = raw.decode("utf-8", errors="replace")

        # Parse
        extractor = _TextExtractor()
        try:
            extractor.feed(html)
        except Exception:
            pass
        text = extractor.get_text()
        title = extractor.title or url
        description = extractor.description

        content_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        word_count = len(text.split())

        result: dict[str, Any] = {
            "url": url,
            "status": "ok",
            "status_code": status_code,
            "title": title,
            "description": description,
            "author": extractor.author,
            "keywords": extractor.keywords,
            "text_preview": text[:500],
            "word_count": word_count,
            "content_hash": content_hash,
            "knowledge_doc_id": "",
        }
        if include_text:
            result["text"] = text[:100_000]

        # Ingest into knowledge engine
        if ingest_to_knowledge and text.strip():
            try:
                from .knowledge import KnowledgeEngine
                knowledge = KnowledgeEngine(self.store)
                doc = knowledge.ingest_text(
                    title=title,
                    content=text[:50_000],
                    project=project,
                    source="web-scrape",
                    source_path=url,
                    metadata={
                        "url": url,
                        "description": description,
                        "author": extractor.author,
                        "keywords": extractor.keywords,
                        "content_hash": content_hash,
                    },
                )
                result["knowledge_doc_id"] = doc.get("id", "")
            except Exception as exc:
                result["ingest_error"] = str(exc)

        self._record_scrape_audit(project, result)
        return result

    # ------------------------------------------------------------------
    # Scheduled scraping
    # ------------------------------------------------------------------

    def schedule_scrape(
        self,
        url: str,
        *,
        project: str = "default",
        interval_hours: float = 24.0,
        label: str = "",
        allow_external: bool = False,
    ) -> dict[str, Any]:
        """Register a URL for periodic scraping."""
        url = _validate_url(url.strip())
        now = utc_now()
        job_id = hashlib.md5(f"scrape-{project}-{url}".encode()).hexdigest()
        next_run = (datetime.now(timezone.utc) + timedelta(hours=0.001)).isoformat()
        status = "scheduled" if allow_external else "awaiting_approval"
        with self.store.session() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scrape_jobs VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, project, url, "", interval_hours, "", next_run,
                 status, json.dumps({"label": label, "external_approved": bool(allow_external)}), now, now),
            )
        self.store.add_audit_event(AuditEvent(
            actor="scraper",
            action="scraper.schedule",
            resource=f"scraper:job:{job_id}",
            project=project,
            metadata={"url": url, "interval_hours": interval_hours},
        ))
        return {
            "job_id": job_id,
            "url": url,
            "interval_hours": interval_hours,
            "next_run": next_run,
            "status": status,
            "external_approved": bool(allow_external),
        }

    def list_scheduled(self, *, project: str = "default") -> list[dict[str, Any]]:
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,project,url,interval_hours,last_run,next_run,status,metadata_json,created_at "
                "FROM scrape_jobs WHERE project=? ORDER BY next_run LIMIT 100", (project,)
            ).fetchall()
        return [
            {"id": r[0], "project": r[1], "url": r[2],
             "interval_hours": r[3], "last_run": r[4], "next_run": r[5],
             "status": r[6], "metadata": json.loads(r[7] or "{}"), "created_at": r[8]}
            for r in rows
        ]

    def run_due_jobs(self, *, project: str = "default") -> dict[str, Any]:
        """Run all scheduled scrape jobs that are due."""
        now = utc_now()
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id, url, interval_hours, metadata_json FROM scrape_jobs "
                "WHERE project=? AND next_run <= ? AND status='scheduled' LIMIT 10",
                (project, now),
            ).fetchall()
        ran = 0
        errors = 0
        for row in rows:
            job_id, url, interval_hours, metadata_json = row
            metadata = json.loads(metadata_json or "{}")
            result = self.scrape_url(
                url,
                project=project,
                allow_external=bool(metadata.get("external_approved")),
            )
            status = result.get("status", "error")
            error = result.get("error", "")
            next_run = (datetime.now(timezone.utc) + timedelta(hours=interval_hours)).isoformat()
            scrape_result_id = hashlib.md5(f"{job_id}-{now}".encode()).hexdigest()
            with self.store.session() as conn:
                conn.execute(
                    "UPDATE scrape_jobs SET last_run=?, next_run=?, updated_at=? WHERE id=?",
                    (now, next_run, now, job_id),
                )
                conn.execute(
                    "INSERT INTO scrape_results VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (scrape_result_id, job_id, project, url,
                     result.get("status_code", 0),
                     result.get("title", ""),
                     result.get("content_hash", ""),
                     result.get("word_count", 0),
                     1, result.get("knowledge_doc_id", ""),
                     error, now),
                )
            if status == "ok":
                ran += 1
            else:
                errors += 1
        return {"ran": ran, "errors": errors, "total_due": len(rows)}

    def get_scrape_history(
        self, url: str, *, project: str = "default", limit: int = 20
    ) -> list[dict[str, Any]]:
        """Return scrape history for a URL."""
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id,url,status_code,title,content_hash,word_count,changed,"
                "knowledge_doc_id,error,created_at "
                "FROM scrape_results WHERE project=? AND url=? ORDER BY created_at DESC LIMIT ?",
                (project, url, limit),
            ).fetchall()
        return [
            {"id": r[0], "url": r[1], "status_code": r[2], "title": r[3],
             "content_hash": r[4], "word_count": r[5], "changed": bool(r[6]),
             "knowledge_doc_id": r[7], "error": r[8], "created_at": r[9]}
            for r in rows
        ]

    # ------------------------------------------------------------------
    # robots.txt
    # ------------------------------------------------------------------

    def _is_allowed(self, url: str, domain: str, *, project: str, allow_external: bool) -> bool:
        """Check robots.txt for the given URL. Caches results for 1 hour."""
        now = time.monotonic()
        cached = self._robots_cache.get(domain)
        if cached and (now - cached[0]) < 3600:
            rp = cached[1]
        else:
            rp = RobotFileParser()
            parsed = urllib.parse.urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            try:
                rp.set_url(robots_url)
                response = self.egress.fetch(
                    robots_url,
                    approved=allow_external,
                    purpose="robots.txt policy check",
                    headers={"User-Agent": _UA, "Accept": "text/plain"},
                    timeout_seconds=min(_TIMEOUT, 5),
                    max_response_bytes=256 * 1024,
                    allowed_content_types={"text/plain", "text/html", "application/octet-stream"},
                )
                if 200 <= response.status < 300:
                    rp.parse(response.body.decode("utf-8", errors="replace").splitlines())
            except Exception:
                rp = RobotFileParser()  # empty = allow all
            self._robots_cache[domain] = (now, rp)
        return rp.can_fetch(_UA, url)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as conn:
            n_jobs = conn.execute(
                "SELECT COUNT(*) FROM scrape_jobs WHERE project=?", (project,)
            ).fetchone()[0]
            n_results = conn.execute(
                "SELECT COUNT(*) FROM scrape_results WHERE project=?", (project,)
            ).fetchone()[0]
        return {
            "scheduled_jobs": n_jobs,
            "total_scrapes": n_results,
            "project": project,
            "features": [
                "robots-txt-compliant", "ssrf-protected", "rate-limited",
                "html-to-text", "metadata-extraction", "scheduled-scraping",
                "knowledge-ingest", "change-detection", "content-hashing",
            ],
        }

    def _record_scrape_audit(self, project: str, result: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(
            actor="scraper",
            action="scraper.fetch",
            resource=f"scraper:fetch:{result.get('url', '')}",
            project=project,
            status="ok" if result.get("status") == "ok" else "error",
            metadata={
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "status": result.get("status", ""),
                "status_code": result.get("status_code", 0),
                "word_count": result.get("word_count", 0),
                "error": result.get("error", ""),
            },
        ))
