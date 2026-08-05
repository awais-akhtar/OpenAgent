"""Unified context engine foundation."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

from .models import ContextEvent
from .storage import OpenAgentStore

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")

STOPWORDS = {
    "and",
    "are",
    "but",
    "for",
    "from",
    "has",
    "have",
    "into",
    "not",
    "our",
    "that",
    "the",
    "their",
    "this",
    "with",
    "you",
    "your",
}


class ContextEngine:
    """Builds local context from durable events."""

    def __init__(self, store: OpenAgentStore):
        self.store = store

    def ingest(
        self,
        title: str,
        content: str,
        *,
        type: str = "note",
        source: str = "manual",
        project: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = ContextEvent(
            title=title.strip(),
            content=content.strip(),
            type=type.strip() or "note",
            source=source.strip() or "manual",
            project=project.strip() or "default",
            metadata=metadata or {},
        )
        return self.store.add_event(event)

    def snapshot(self, *, project: str | None = None, limit: int = 100) -> dict[str, Any]:
        events = self.store.list_events(limit=limit, project=project)
        sources = Counter(event["source"] for event in events)
        types = Counter(event["type"] for event in events)
        projects = Counter(event["project"] for event in events)
        terms = Counter()

        for event in events:
            text = f"{event['title']} {event['content']}".lower()
            for token in TOKEN_RE.findall(text):
                if token not in STOPWORDS:
                    terms[token] += 1

        return {
            "project": project or "all",
            "event_count": len(events),
            "sources": dict(sources.most_common()),
            "types": dict(types.most_common()),
            "projects": dict(projects.most_common()),
            "top_terms": [{"term": term, "count": count} for term, count in terms.most_common(12)],
            "recent_events": events[:8],
        }

    def search(self, query: str, *, project: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        return self.store.search_events(query, limit=limit, project=project)

