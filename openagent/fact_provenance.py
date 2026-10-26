"""Fact provenance service for tracking sources and retractions."""

from __future__ import annotations

import json
from typing import Any

from .storage import OpenAgentStore
from .utils import stable_id, utc_now


class FactProvenanceService:
    """Manages fact provenance tracking with sources and retractions."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store

    def record_fact(
        self,
        *,
        fact: str,
        project: str,
        fact_type: str = "statement",
        confidence: float = 1.0,
        sources: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Record a fact with provenance sources.

        Args:
            fact: The fact statement to record
            project: Project ID for scoping
            fact_type: Type of fact (statement, observation, inference, etc)
            confidence: Confidence score 0.0-1.0
            sources: List of source references {source_type, source_reference, confidence}

        Returns:
            Recorded fact record with ID
        """
        if not fact or not fact.strip():
            raise ValueError("Fact cannot be empty.")
        if not project or not project.strip():
            raise ValueError("Project is required.")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0.")

        fact = fact.strip()
        fact_id = stable_id("fact", project, fact)
        now = utc_now()

        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO fact_provenance (
                    id, project, fact, fact_type, confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (fact_id, project, fact, fact_type, confidence, now),
            )

            # Record sources
            if sources:
                for source in sources:
                    source_type = str(source.get("source_type", "")).strip()
                    source_ref = str(source.get("source_reference", "")).strip()
                    source_conf = float(source.get("confidence", 1.0))
                    if source_type and source_ref and 0.0 <= source_conf <= 1.0:
                        source_id = stable_id("source", fact_id, source_type, source_ref)
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO fact_sources (
                                id, fact_id, source_type, source_reference, confidence, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (source_id, fact_id, source_type, source_ref, source_conf, now),
                        )

        return self.get_fact(fact_id, project=project) or {}

    def get_fact(self, fact_id: str, project: str | None = None) -> dict[str, Any] | None:
        """Get a fact with its sources and retraction status."""
        query = "SELECT * FROM fact_provenance WHERE id = ?"
        params: tuple[Any, ...] = (fact_id,)
        if project:
            query += " AND project = ?"
            params = (fact_id, project)

        with self.store.session() as connection:
            row = connection.execute(query, params).fetchone()
            if not row:
                return None

            fact = dict(row)
            # Get sources
            sources = connection.execute(
                """
                SELECT id, source_type, source_reference, confidence, created_at
                FROM fact_sources WHERE fact_id = ? ORDER BY created_at ASC
                """,
                (fact_id,),
            ).fetchall()
            fact["sources"] = [dict(s) for s in sources]

            # Get retractions
            retractions = connection.execute(
                """
                SELECT id, retracted_by, reason, evidence, restored_at, created_at
                FROM fact_retractions WHERE fact_id = ? ORDER BY created_at DESC
                """,
                (fact_id,),
            ).fetchall()
            fact["retractions"] = [dict(r) for r in retractions]

        return fact

    def list_facts(self, project: str, limit: int = 100, active_only: bool = True) -> list[dict[str, Any]]:
        """List facts for a project."""
        limit = max(1, min(limit, 500))
        query = "SELECT * FROM fact_provenance WHERE project = ?"
        if active_only:
            query += " AND retracted_at = ''"
        query += " ORDER BY created_at DESC LIMIT ?"

        with self.store.session() as connection:
            rows = connection.execute(query, (project, limit)).fetchall()

        facts = []
        for row in rows:
            fact = dict(row)
            with self.store.session() as connection:
                sources = connection.execute(
                    "SELECT source_type, source_reference, confidence FROM fact_sources WHERE fact_id = ?",
                    (fact["id"],),
                ).fetchall()
                fact["sources"] = [dict(s) for s in sources]
            facts.append(fact)

        return facts

    def retract_fact(
        self,
        fact_id: str,
        project: str,
        *,
        retracted_by: str,
        reason: str,
        evidence: str = "",
    ) -> dict[str, Any]:
        """Retract a fact with reasoning."""
        if not reason or not reason.strip():
            raise ValueError("Retraction reason is required.")

        fact = self.get_fact(fact_id, project=project)
        if not fact:
            raise ValueError(f"Fact {fact_id} not found.")
        if fact.get("retracted_at"):
            raise ValueError(f"Fact {fact_id} is already retracted.")

        now = utc_now()
        retraction_id = stable_id("retraction", fact_id, now)

        with self.store.session() as connection:
            # Update fact with retraction timestamp
            connection.execute(
                """
                UPDATE fact_provenance SET retracted_at = ?, retraction_reason = ?
                WHERE id = ? AND project = ?
                """,
                (now, reason.strip(), fact_id, project),
            )

            # Record retraction details
            connection.execute(
                """
                INSERT INTO fact_retractions (
                    id, fact_id, retracted_by, reason, evidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (retraction_id, fact_id, retracted_by, reason.strip(), evidence.strip(), now),
            )

        return self.get_fact(fact_id, project=project) or {}

    def restore_fact(self, fact_id: str, project: str) -> dict[str, Any]:
        """Restore a retracted fact."""
        fact = self.get_fact(fact_id, project=project)
        if not fact:
            raise ValueError(f"Fact {fact_id} not found.")
        if not fact.get("retracted_at"):
            raise ValueError(f"Fact {fact_id} is not retracted.")

        now = utc_now()

        with self.store.session() as connection:
            # Clear retraction on fact
            connection.execute(
                """
                UPDATE fact_provenance SET retracted_at = '', retraction_reason = ''
                WHERE id = ? AND project = ?
                """,
                (fact_id, project),
            )

            # Mark restoration in retractions
            connection.execute(
                """
                UPDATE fact_retractions SET restored_at = ? WHERE fact_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                (now, fact_id),
            )

        return self.get_fact(fact_id, project=project) or {}

    def search_facts(self, project: str, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """Search facts by content."""
        query = query.strip()
        if not query:
            return self.list_facts(project, limit=limit)

        limit = max(1, min(limit, 200))
        search_term = f"%{query.lower()}%"

        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM fact_provenance
                WHERE project = ? AND lower(fact) LIKE ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (project, search_term, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_fact_statistics(self, project: str) -> dict[str, Any]:
        """Get provenance statistics for a project."""
        with self.store.session() as connection:
            total = connection.execute(
                "SELECT COUNT(*) as count FROM fact_provenance WHERE project = ?",
                (project,),
            ).fetchone()["count"]

            active = connection.execute(
                "SELECT COUNT(*) as count FROM fact_provenance WHERE project = ? AND retracted_at = ''",
                (project,),
            ).fetchone()["count"]

            retracted = total - active

            total_sources = connection.execute(
                """
                SELECT COUNT(*) as count FROM fact_sources
                WHERE fact_id IN (SELECT id FROM fact_provenance WHERE project = ?)
                """,
                (project,),
            ).fetchone()["count"]

            total_retractions = connection.execute(
                """
                SELECT COUNT(*) as count FROM fact_retractions
                WHERE fact_id IN (SELECT id FROM fact_provenance WHERE project = ?)
                """,
                (project,),
            ).fetchone()["count"]

        return {
            "total_facts": total,
            "active_facts": active,
            "retracted_facts": retracted,
            "total_sources": total_sources,
            "total_retractions": total_retractions,
        }
