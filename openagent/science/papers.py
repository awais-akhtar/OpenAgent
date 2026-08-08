"""Searchable local paper records with an explicit external-fetch boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping


@dataclass(frozen=True)
class PaperRecord:
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    citation: str = ""
    source: str = "local"

    def to_dict(self) -> dict[str, object]:
        return {"title": self.title, "abstract": self.abstract, "authors": list(self.authors), "citation": self.citation, "source": self.source}


class PaperSearch:
    """Search local paper metadata; external retrieval must be supplied by the caller."""

    def __init__(self, records: Iterable[Mapping[str, object]] = ()) -> None:
        self.records = [
            PaperRecord(
                str(record.get("title", "")),
                str(record.get("abstract", "")),
                tuple(str(item) for item in record.get("authors", ()) if item),
                str(record.get("citation", "")),
                str(record.get("source", "local")),
            )
            for record in records
            if str(record.get("title", "")).strip()
        ]

    def search(self, query: str, *, limit: int = 10) -> list[PaperRecord]:
        terms = {term.lower() for term in query.split() if term}
        scored = []
        for record in self.records:
            haystack = f"{record.title} {record.abstract} {' '.join(record.authors)}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, record))
        return [record for _, record in sorted(scored, key=lambda item: (-item[0], item[1].title))[: max(1, limit)]]

    def search_with_approved_fetch(self, query: str, fetcher: Callable[[str], Iterable[Mapping[str, object]]]) -> list[PaperRecord]:
        """Use an already-approved caller fetcher, then keep the results local."""

        fetched = PaperSearch(fetcher(query)).records
        return PaperSearch([record.to_dict() for record in [*self.records, *fetched]]).search(query)
