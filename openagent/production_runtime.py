"""Stage 12 production runtime foundations.

This module keeps the project dependency-light while turning the remaining
production checklist into local, testable, auditable runtime surfaces.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
import base64
import hashlib
import hmac
import json
import math
import secrets
from pathlib import Path
from typing import Any

from .communication import CommunicationHub
from .developer import DeveloperPlatform
from .local_ai import LocalAIInfrastructure
from .media import MediaManager
from .model_gateway import ModelGateway, local_hash_embedding
from .models import AuditEvent, ContextEvent, utc_now
from .security_privacy import SecurityPrivacyManager, decrypt_text, encrypt_text
from .storage import OpenAgentStore


class ProductionRuntimeError(RuntimeError):
    """Raised when a production runtime operation is invalid or unsafe."""


class ProductionRuntime:
    """Local production-depth systems with explicit privacy boundaries."""

    def __init__(self, store: OpenAgentStore, gateway: ModelGateway | None = None):
        self.store = store
        self.gateway = gateway or ModelGateway()
        self.package_root = (self.store.data_dir / "packages").resolve()
        self.package_root.mkdir(parents=True, exist_ok=True)
        self.initialize()
        self.seed_defaults()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS production_policy_rules (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    conditions_json TEXT NOT NULL DEFAULT '{}',
                    priority INTEGER NOT NULL DEFAULT 100,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_production_policy_scope
                    ON production_policy_rules(project, scope, priority);

                CREATE TABLE IF NOT EXISTS production_queue_jobs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    queue TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'queued',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    next_run_at TEXT NOT NULL DEFAULT '',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_production_queue_due
                    ON production_queue_jobs(queue, status, next_run_at, created_at);

                CREATE TABLE IF NOT EXISTS production_audit_anchors (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    from_event_id TEXT NOT NULL,
                    to_event_id TEXT NOT NULL,
                    event_count INTEGER NOT NULL,
                    merkle_root TEXT NOT NULL,
                    previous_anchor_hash TEXT NOT NULL,
                    anchor_hash TEXT NOT NULL,
                    storage_mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_vector_indexes (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    metric TEXT NOT NULL DEFAULT 'cosine',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_vector_records (
                    id TEXT PRIMARY KEY,
                    index_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    external_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    magnitude REAL NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    tombstoned INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(index_id) REFERENCES production_vector_indexes(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_production_vectors_index_namespace
                    ON production_vector_records(index_id, namespace, tombstoned);

                CREATE TABLE IF NOT EXISTS production_connector_adapters (
                    id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    endpoint_url TEXT NOT NULL DEFAULT '',
                    poll_interval_seconds INTEGER NOT NULL DEFAULT 300,
                    credentials_ref TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_connector_poll_runs (
                    id TEXT PRIMARY KEY,
                    adapter_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    status TEXT NOT NULL,
                    imported_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_crypto_envelopes (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    label TEXT NOT NULL,
                    recipients_json TEXT NOT NULL DEFAULT '[]',
                    envelope_json TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_hardware_keys (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key_type TEXT NOT NULL,
                    public_key TEXT NOT NULL,
                    enclave_hint TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_automation_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    runtime TEXT NOT NULL,
                    status TEXT NOT NULL,
                    actions_json TEXT NOT NULL DEFAULT '[]',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_app_packages (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    app_type TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    package_path TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_agent_marketplace (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    author TEXT NOT NULL DEFAULT '',
                    manifest_json TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'available',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_gpu_profiles (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    profile_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_finetune_jobs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    dataset_ref TEXT NOT NULL,
                    hyperparameters_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL,
                    artifact_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_proactive_jobs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    query TEXT NOT NULL,
                    cadence_seconds INTEGER NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    last_run_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_proactive_runs (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    insight_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_federated_rounds (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    participant_count INTEGER NOT NULL,
                    aggregate_json TEXT NOT NULL,
                    privacy_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_sync_devices (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    name TEXT NOT NULL,
                    public_key TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_sync_bundles (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    from_device_id TEXT NOT NULL,
                    to_device_id TEXT NOT NULL,
                    envelope_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_sdk_packages (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    language TEXT NOT NULL,
                    version TEXT NOT NULL,
                    artifact_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS production_sandbox_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    plugin_id TEXT NOT NULL,
                    profile_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def seed_defaults(self) -> None:
        if not self.policy_rules(project="default"):
            self.upsert_policy_rule(
                scope="external.model",
                effect="deny",
                conditions={"unless": "allow_external"},
                description="External model calls require explicit approval.",
                audit=False,
            )
            self.upsert_policy_rule(
                scope="connector.poll",
                effect="deny",
                conditions={"unless": "allow_external"},
                description="Live provider polling requires explicit approval.",
                audit=False,
            )

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            tables = {
                "policy_rules": "production_policy_rules",
                "queued_jobs": "production_queue_jobs",
                "audit_anchors": "production_audit_anchors",
                "vector_indexes": "production_vector_indexes",
                "crypto_envelopes": "production_crypto_envelopes",
                "automation_runs": "production_automation_runs",
                "app_packages": "production_app_packages",
                "agent_packages": "production_agent_marketplace",
                "finetune_jobs": "production_finetune_jobs",
                "proactive_jobs": "production_proactive_jobs",
                "federated_rounds": "production_federated_rounds",
                "sync_bundles": "production_sync_bundles",
                "sdk_packages": "production_sdk_packages",
                "sandbox_runs": "production_sandbox_runs",
            }
            counts = {
                key: connection.execute(f"SELECT count(*) FROM {table} WHERE project = ?", (project,)).fetchone()[0]
                for key, table in tables.items()
            }
            counts["connector_adapters"] = connection.execute(
                "SELECT count(*) FROM production_connector_adapters"
            ).fetchone()[0]
            counts["hardware_keys"] = connection.execute(
                "SELECT count(*) FROM production_hardware_keys WHERE project = ?",
                (project,),
            ).fetchone()[0]
        return {
            "project": project,
            **counts,
            "privacy_boundary": "Stage 12 production runtime stores all records locally and gates external effects.",
        }

    def upsert_policy_rule(
        self,
        *,
        scope: str,
        effect: str,
        project: str = "default",
        conditions: dict[str, Any] | None = None,
        priority: int = 100,
        description: str = "",
        audit: bool = True,
    ) -> dict[str, Any]:
        if effect not in {"allow", "deny"}:
            raise ValueError("Policy effect must be allow or deny.")
        rule_id = stable_id("production-policy", project, scope, str(priority), effect)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_policy_rules (
                    id, project, scope, effect, conditions_json, priority,
                    description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM production_policy_rules WHERE id = ?), ?), ?)
                """,
                (
                    rule_id,
                    project,
                    scope,
                    effect,
                    json.dumps(conditions or {}, sort_keys=True),
                    priority,
                    description,
                    rule_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit("production.policy.upsert", rule_id, project=project, metadata={"scope": scope, "effect": effect})
        return self.policy_rule(rule_id) or {}

    def policy_rule(self, rule_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_policy_rules WHERE id = ?", (rule_id,)).fetchone()
        return policy_rule_row(row) if row else None

    def policy_rules(self, *, project: str = "default", scope: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if scope:
            clauses.append("scope = ?")
            params.append(scope)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM production_policy_rules WHERE {' AND '.join(clauses)} ORDER BY priority ASC",
                tuple(params),
            ).fetchall()
        return [policy_rule_row(row) for row in rows]

    def evaluate_policy(
        self,
        *,
        scope: str,
        context: dict[str, Any] | None = None,
        project: str = "default",
    ) -> dict[str, Any]:
        context = context or {}
        matched = []
        decision = "allow"
        for rule in self.policy_rules(project=project, scope=scope):
            if conditions_match(rule["conditions"], context):
                matched.append(rule)
                decision = rule["effect"]
                break
        result = {"allowed": decision == "allow", "decision": decision, "scope": scope, "matched_rules": matched}
        self._audit("production.policy.evaluate", scope, project=project, metadata=result)
        return result

    def enqueue_job(
        self,
        *,
        kind: str,
        payload: dict[str, Any],
        project: str = "default",
        queue: str = "default",
        max_attempts: int = 3,
        next_run_at: str = "",
    ) -> dict[str, Any]:
        job_id = stable_id("production-job", project, queue, kind, json.dumps(payload, sort_keys=True), utc_now())
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_queue_jobs (
                    id, project, queue, kind, payload_json, status, attempts,
                    max_attempts, next_run_at, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project,
                    queue,
                    kind,
                    json.dumps(payload, sort_keys=True),
                    "queued",
                    0,
                    max(1, int(max_attempts)),
                    next_run_at,
                    "{}",
                    now,
                    now,
                ),
            )
        self._audit("production.queue.enqueue", job_id, project=project, metadata={"kind": kind, "queue": queue})
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_queue_jobs WHERE id = ?", (job_id,)).fetchone()
        return queue_job_row(row) if row else None

    def jobs(self, *, project: str = "default", status: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, min(limit, 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM production_queue_jobs WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [queue_job_row(row) for row in rows]

    def run_next_job(self, *, queue: str = "default", project: str = "default") -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT * FROM production_queue_jobs
                WHERE project = ? AND queue = ? AND status = 'queued'
                  AND (next_run_at = '' OR next_run_at <= ?)
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (project, queue, now),
            ).fetchone()
        if row is None:
            return {"ran": False, "reason": "no_due_job"}
        job = queue_job_row(row)
        attempts = job["attempts"] + 1
        try:
            result = self._run_job_payload(job)
            status = "completed"
        except Exception as exc:
            result = {"error": str(exc)}
            status = "failed" if attempts >= job["max_attempts"] else "queued"
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE production_queue_jobs
                SET status = ?, attempts = ?, result_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, attempts, json.dumps(result, sort_keys=True), utc_now(), job["id"]),
            )
        self._audit(
            "production.queue.run",
            job["id"],
            project=project,
            status=status,
            metadata={"kind": job["kind"], "attempts": attempts},
        )
        return {"ran": True, "job": self.get_job(job["id"])}

    def anchor_audit_log(self, *, project: str = "default", storage_mode: str = "local-append-only") -> dict[str, Any]:
        events = self.store.list_audit_events(project=None if project == "*" else project, limit=500)
        if not events:
            raise ValueError("No audit events are available to anchor.")
        event_hashes = [hash_json(event) for event in sorted(events, key=lambda item: (item["created_at"], item["id"]))]
        merkle_root = merkle(event_hashes)
        previous = self.audit_anchors(project=project, limit=1)
        previous_hash = previous[0]["anchor_hash"] if previous else "0" * 64
        anchor_id = stable_id("audit-anchor", project, merkle_root, previous_hash, utc_now())
        anchor_hash = sha256(f"{previous_hash}:{merkle_root}:{len(event_hashes)}")
        created_at = utc_now()
        ordered_events = sorted(events, key=lambda item: (item["created_at"], item["id"]))
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_audit_anchors (
                    id, project, from_event_id, to_event_id, event_count, merkle_root,
                    previous_anchor_hash, anchor_hash, storage_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anchor_id,
                    project,
                    ordered_events[0]["id"],
                    ordered_events[-1]["id"],
                    len(event_hashes),
                    merkle_root,
                    previous_hash,
                    anchor_hash,
                    storage_mode,
                    created_at,
                ),
            )
        self._audit("production.audit.anchor", anchor_id, project=project, metadata={"event_count": len(event_hashes)})
        return self.audit_anchors(project=project, limit=1)[0]

    def audit_anchors(self, *, project: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM production_audit_anchors WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, max(1, min(limit, 200))),
            ).fetchall()
        return [audit_anchor_row(row) for row in rows]

    def create_vector_index(
        self,
        *,
        name: str,
        project: str = "default",
        dimensions: int = 64,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip() or "default"
        dimensions = max(1, min(int(dimensions), 4096))
        index_id = stable_id("production-vector-index", project, name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_vector_indexes (
                    id, project, name, dimensions, metric, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM production_vector_indexes WHERE id = ?), ?), ?)
                """,
                (index_id, project, name, dimensions, "cosine", json.dumps(metadata or {}, sort_keys=True), index_id, now, now),
            )
        self._audit("production.vector.index.upsert", index_id, project=project, metadata={"dimensions": dimensions})
        return self.get_vector_index(index_id) or {}

    def get_vector_index(self, index_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_vector_indexes WHERE id = ?", (index_id,)).fetchone()
        return vector_index_row(row) if row else None

    def upsert_vector(
        self,
        index_id: str,
        *,
        external_id: str,
        content: str,
        namespace: str = "default",
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index = self.get_vector_index(index_id)
        if index is None:
            raise ValueError(f"Unknown vector index '{index_id}'.")
        vector = vector or local_hash_embedding(content, dimensions=index["dimensions"])
        if len(vector) != index["dimensions"]:
            raise ValueError(f"Vector must have {index['dimensions']} dimensions.")
        record_id = stable_id("production-vector", index_id, namespace, external_id)
        now = utc_now()
        magnitude = vector_magnitude(vector)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_vector_records (
                    id, index_id, project, namespace, external_id, content, vector_json,
                    magnitude, metadata_json, tombstoned, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, COALESCE((SELECT created_at FROM production_vector_records WHERE id = ?), ?), ?)
                """,
                (
                    record_id,
                    index_id,
                    index["project"],
                    namespace,
                    external_id,
                    content,
                    json.dumps([float(item) for item in vector]),
                    magnitude,
                    json.dumps(metadata or {}, sort_keys=True),
                    record_id,
                    now,
                    now,
                ),
            )
        self._audit("production.vector.upsert", record_id, project=index["project"], metadata={"index_id": index_id})
        return self.get_vector(record_id) or {}

    def get_vector(self, record_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_vector_records WHERE id = ?", (record_id,)).fetchone()
        return vector_record_row(row) if row else None

    def vector_search(
        self,
        index_id: str,
        *,
        query: str,
        namespace: str = "default",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        index = self.get_vector_index(index_id)
        if index is None:
            raise ValueError(f"Unknown vector index '{index_id}'.")
        query_vector = local_hash_embedding(query, dimensions=index["dimensions"])
        query_mag = vector_magnitude(query_vector)
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT * FROM production_vector_records
                WHERE index_id = ? AND namespace = ? AND tombstoned = 0
                """,
                (index_id, namespace),
            ).fetchall()
        scored = []
        for row in rows:
            item = vector_record_row(row)
            score = cosine_with_magnitude(query_vector, query_mag, item["vector"], item["magnitude"])
            scored.append((score, item))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {**without_vector(item), "score": round(score, 6)}
            for score, item in scored[: max(1, min(limit, 100))]
        ]

    def semantic_memory_science(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT memory_type, content, tags_json, confidence FROM semantic_memories WHERE project = ?",
                (project,),
            ).fetchall()
        type_counts = Counter(row["memory_type"] for row in rows)
        tag_counts: Counter[str] = Counter()
        lengths = []
        confidence = []
        for row in rows:
            tag_counts.update(json.loads(row["tags_json"]))
            lengths.append(len(row["content"]))
            confidence.append(float(row["confidence"]))
        result = {
            "project": project,
            "memory_count": len(rows),
            "type_distribution": dict(type_counts),
            "top_tags": tag_counts.most_common(20),
            "average_content_length": round(sum(lengths) / len(lengths), 3) if lengths else 0,
            "average_confidence": round(sum(confidence) / len(confidence), 4) if confidence else 0,
            "diversity_score": round(shannon(type_counts), 6),
            "recommendations": memory_recommendations(len(rows), type_counts, tag_counts),
        }
        self._audit("production.semantic.science", f"project:{project}", project=project, metadata=result)
        return result

    def rerank_sources(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        project: str = "default",
        limit: int = 10,
    ) -> dict[str, Any]:
        query_terms = terms(query)
        query_vector = local_hash_embedding(query, dimensions=64)
        ranked = []
        for index, candidate in enumerate(candidates):
            content = str(candidate.get("content") or candidate.get("text") or "")
            title = str(candidate.get("title") or candidate.get("id") or index)
            candidate_vector = local_hash_embedding(f"{title}\n{content}", dimensions=64)
            lexical = lexical_score(query_terms, f"{title} {content}")
            semantic = cosine(query_vector, candidate_vector)
            authority = float(candidate.get("authority", 0.5) or 0.5)
            recency = float(candidate.get("recency", 0.5) or 0.5)
            score = 0.45 * semantic + 0.30 * lexical + 0.15 * authority + 0.10 * recency
            ranked.append({**candidate, "score": round(score, 6), "ranker": "production-local-reranker-v1"})
        ranked.sort(key=lambda item: float(item["score"]), reverse=True)
        result = {"project": project, "query": query, "results": ranked[: max(1, min(limit, 100))]}
        self._audit("production.rag.rerank", f"query:{query[:80]}", project=project, metadata={"candidates": len(candidates)})
        return result

    def analyze_media_production(self, asset_id: str) -> dict[str, Any]:
        media = MediaManager(self.store)
        asset = media.get_asset(asset_id)
        if asset is None:
            raise ValueError(f"Unknown media asset '{asset_id}'.")
        analyses = media.analyses(project=asset["project"], asset_id=asset_id, limit=1)
        latest = analyses[0] if analyses else media.analyze_asset(asset_id)
        findings = dict(latest["findings"])
        metadata = asset["metadata"]
        if metadata.get("ocr_text"):
            findings["ocr_text"] = str(metadata["ocr_text"])
            findings["ocr_provider"] = "local-metadata"
        if metadata.get("transcript"):
            findings["transcript"] = str(metadata["transcript"])
            findings["transcription_provider"] = "local-metadata"
        result = {
            "asset_id": asset_id,
            "project": asset["project"],
            "media_type": asset["media_type"],
            "findings": findings,
            "summary": latest["summary"],
            "privacy_boundary": "Production media analysis uses local metadata, local text extraction, and explicit user-provided transcripts/OCR.",
        }
        self._audit("production.media.analyze", asset_id, project=asset["project"], metadata={"media_type": asset["media_type"]})
        return result

    def register_connector_adapter(
        self,
        *,
        platform: str,
        mode: str = "local_fixture",
        endpoint_url: str = "",
        credentials_ref: str = "",
        enabled: bool = True,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        adapter_id = stable_id("connector-adapter", platform, mode, endpoint_url or "local")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_connector_adapters (
                    id, platform, mode, endpoint_url, poll_interval_seconds,
                    credentials_ref, enabled, config_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM production_connector_adapters WHERE id = ?), ?), ?)
                """,
                (
                    adapter_id,
                    platform,
                    mode,
                    endpoint_url,
                    int((config or {}).get("poll_interval_seconds", 300)),
                    credentials_ref,
                    1 if enabled else 0,
                    json.dumps(config or {}, sort_keys=True),
                    adapter_id,
                    now,
                    now,
                ),
            )
        self._audit("production.connector.adapter.upsert", adapter_id, metadata={"platform": platform, "mode": mode})
        return self.connector_adapter(adapter_id) or {}

    def connector_adapter(self, adapter_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_connector_adapters WHERE id = ?", (adapter_id,)).fetchone()
        return connector_adapter_row(row) if row else None

    def poll_connector(self, adapter_id: str, *, project: str = "default", allow_external: bool = False) -> dict[str, Any]:
        adapter = self.connector_adapter(adapter_id)
        if adapter is None:
            raise ValueError(f"Unknown connector adapter '{adapter_id}'.")
        if not adapter["enabled"]:
            raise ProductionRuntimeError("Connector adapter is disabled.")
        if adapter["mode"] != "local_fixture" and not allow_external:
            raise ProductionRuntimeError("Live connector polling requires allow_external=true.")
        payload = adapter["config"].get("fixture_payload", {})
        run = CommunicationHub(self.store).sync_connector(
            adapter["platform"],
            project=project,
            payload=payload if isinstance(payload, dict) else {},
            allow_external=allow_external,
        )
        poll_id = stable_id("connector-poll", adapter_id, project, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_connector_poll_runs (
                    id, adapter_id, project, status, imported_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (poll_id, adapter_id, project, "completed", json.dumps(run, sort_keys=True), "", created_at),
            )
        self._audit("production.connector.poll", adapter_id, project=project, metadata={"run_id": run["id"]})
        return {"id": poll_id, "adapter_id": adapter_id, "project": project, "sync_run": run, "created_at": created_at}

    def validate_oidc_jwks(
        self,
        *,
        token: str,
        jwks: dict[str, Any],
        issuer: str,
        audience: str,
        project: str = "security",
    ) -> dict[str, Any]:
        result = validate_jwt_with_jwks(token, jwks, issuer=issuer, audience=audience)
        self._audit("production.oidc.jwks.validate", issuer, project=project, status="allowed" if result["valid"] else "denied", metadata={"valid": result["valid"]})
        return result

    def encrypt_envelope(
        self,
        *,
        label: str,
        plaintext: str,
        passphrase: str,
        recipients: list[str],
        project: str = "default",
    ) -> dict[str, Any]:
        envelope_id = stable_id("crypto-envelope", project, label, utc_now())
        payload = encrypt_text(plaintext, passphrase)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_crypto_envelopes (
                    id, project, label, recipients_json, envelope_json, provider, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    envelope_id,
                    project,
                    label,
                    json.dumps(recipients, sort_keys=True),
                    json.dumps(payload, sort_keys=True),
                    "aes-256-gcm+scrypt",
                    created_at,
                ),
            )
        self._audit("production.crypto.encrypt", envelope_id, project=project, metadata={"recipients": recipients})
        return self.crypto_envelope(envelope_id) or {}

    def decrypt_envelope(self, envelope_id: str, *, passphrase: str) -> dict[str, Any]:
        envelope = self.crypto_envelope(envelope_id)
        if envelope is None:
            raise ValueError(f"Unknown crypto envelope '{envelope_id}'.")
        plaintext = decrypt_text(envelope["envelope"], passphrase)
        migrated = envelope.get("provider") != "aes-256-gcm+scrypt"
        if migrated:
            replacement = encrypt_text(plaintext, passphrase)
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE production_crypto_envelopes SET envelope_json = ?, provider = ? WHERE id = ?",
                    (json.dumps(replacement, sort_keys=True), "aes-256-gcm+scrypt", envelope_id),
                )
            envelope = self.crypto_envelope(envelope_id) or envelope
        self._audit(
            "production.crypto.decrypt",
            envelope_id,
            project=envelope["project"],
            metadata={"status": "allowed", "envelope_migrated": migrated},
        )
        return {**envelope, "plaintext": plaintext, "envelope_migrated": migrated}

    def crypto_envelope(self, envelope_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_crypto_envelopes WHERE id = ?", (envelope_id,)).fetchone()
        return crypto_envelope_row(row) if row else None

    def register_hardware_key(
        self,
        *,
        name: str,
        public_key: str,
        project: str = "default",
        key_type: str = "webauthn",
        enclave_hint: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        key_id = stable_id("hardware-key", project, name, public_key)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_hardware_keys (
                    id, project, name, key_type, public_key, enclave_hint, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (key_id, project, name, key_type, public_key, enclave_hint, json.dumps(metadata or {}, sort_keys=True), created_at),
            )
        self._audit("production.hardware_key.register", key_id, project=project, metadata={"key_type": key_type})
        return {
            "id": key_id,
            "project": project,
            "name": name,
            "key_type": key_type,
            "public_key": public_key,
            "enclave_hint": enclave_hint,
            "metadata": metadata or {},
            "created_at": created_at,
        }

    def run_desktop_automation(
        self,
        *,
        actions: list[dict[str, Any]],
        project: str = "default",
        allow_desktop: bool = False,
    ) -> dict[str, Any]:
        desktop_root = (self.store.data_dir / "desktop_automation" / project).resolve()
        desktop_root.mkdir(parents=True, exist_ok=True)
        result_actions = []
        for index, action in enumerate(actions, start=1):
            kind = str(action.get("action", "record"))
            if kind in {"record", "wait", "note"}:
                result_actions.append({"index": index, "action": kind, "status": "completed", "metadata": action})
            elif kind == "write_file":
                target = safe_child_path(desktop_root, str(action.get("path", "")))
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(action.get("content", "")), encoding="utf-8")
                result_actions.append({"index": index, "action": kind, "status": "completed", "path": str(target)})
            elif kind == "read_file":
                target = safe_child_path(desktop_root, str(action.get("path", "")))
                content = target.read_text(encoding="utf-8")
                result_actions.append({"index": index, "action": kind, "status": "completed", "path": str(target), "content": content})
            elif kind == "list_dir":
                target = safe_child_path(desktop_root, str(action.get("path", ".")))
                items = sorted(child.name for child in target.iterdir()) if target.exists() and target.is_dir() else []
                result_actions.append({"index": index, "action": kind, "status": "completed", "path": str(target), "items": items})
            elif not allow_desktop:
                result_actions.append({"index": index, "action": kind, "status": "blocked", "reason": "desktop approval required"})
            else:
                result_actions.append({"index": index, "action": kind, "status": "approved_recorded", "metadata": action})
        return self._save_automation_run(
            project=project,
            runtime="desktop",
            actions=actions,
            result={
                "root": str(desktop_root),
                "actions": result_actions,
                "privacy_boundary": "Desktop automation executes only approved local file actions inside the OpenAgent data directory.",
            },
        )

    def run_browser_automation(
        self,
        *,
        actions: list[dict[str, Any]],
        project: str = "default",
        html: str = "",
    ) -> dict[str, Any]:
        document = BrowserDocument.parse(html)
        state = {
            "url": "",
            "title": document.title or extract_title(html),
            "fields": dict(document.fields),
            "clicks": [],
            "forms": document.forms,
            "links": document.links,
            "text": document.text,
        }
        outputs = []
        for index, action in enumerate(actions, start=1):
            kind = str(action.get("action", "inspect"))
            if kind == "open":
                state["url"] = str(action.get("url", ""))
            elif kind == "fill":
                selector = str(action.get("selector", ""))
                value = str(action.get("value", ""))
                field_key = resolve_field_selector(selector, state["fields"])
                state["fields"][field_key] = value
                outputs.append({"index": index, "action": kind, "selector": selector, "field": field_key, "value": value, "status": "completed"})
            elif kind == "click":
                state["clicks"].append(str(action.get("selector", "")))
            elif kind == "extract_title":
                outputs.append({"title": state["title"]})
            elif kind == "extract_text":
                outputs.append({"text": state["text"]})
            elif kind == "assert_text":
                expected = str(action.get("text", ""))
                if expected not in state["text"] and expected not in json.dumps(state["fields"], sort_keys=True):
                    outputs.append({"index": index, "action": kind, "status": "failed", "expected": expected})
                    continue
            if not (kind == "fill" and outputs and outputs[-1].get("index") == index):
                outputs.append({"index": index, "action": kind, "status": "completed"})
        filled_html = render_filled_html(html, state["fields"])
        return self._save_automation_run(
            project=project,
            runtime="browser",
            actions=actions,
            result={
                "state": state,
                "outputs": outputs,
                "filled_html": filled_html,
                "privacy_boundary": "Browser automation parses and mutates caller-provided local HTML; no browser or network is opened.",
            },
        )

    def package_app(
        self,
        *,
        app_type: str,
        platform: str,
        project: str = "default",
        manifest: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if app_type not in {"desktop", "mobile", "pwa"}:
            raise ValueError("app_type must be desktop, mobile, or pwa.")
        package_id = stable_id("app-package", project, app_type, platform, utc_now())
        entrypoint = "openagent-desktop" if app_type == "desktop" else "python -m openagent --port 8765"
        manifest = {
            "name": "OpenAgent",
            "app_type": app_type,
            "platform": platform,
            "entrypoint": entrypoint,
            "offline_capable": True,
            "browser_required": app_type != "desktop",
            "privacy_boundary": "Package manifest keeps OpenAgent self-hosted and local-first.",
            **(manifest or {}),
        }
        package_path = self.package_root / package_id
        package_path.mkdir(parents=True, exist_ok=True)
        write_app_package_files(package_path, manifest)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_app_packages (
                    id, project, app_type, platform, manifest_json, package_path, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (package_id, project, app_type, platform, json.dumps(manifest, sort_keys=True), str(package_path), "packaged", created_at),
            )
        self._audit("production.app.package", package_id, project=project, metadata={"app_type": app_type, "platform": platform})
        return self.app_package(package_id) or {}

    def app_package(self, package_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_app_packages WHERE id = ?", (package_id,)).fetchone()
        return app_package_row(row) if row else None

    def publish_agent_package(
        self,
        *,
        name: str,
        version: str,
        manifest: dict[str, Any],
        project: str = "default",
        author: str = "",
    ) -> dict[str, Any]:
        package_id = stable_id("agent-package", name, version, author)
        signature = hmac.new(b"openagent-agent-marketplace", json.dumps(manifest, sort_keys=True).encode("utf-8"), hashlib.sha256).hexdigest()
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_agent_marketplace (
                    id, project, name, version, author, manifest_json, signature, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (package_id, project, name, version, author, json.dumps(manifest, sort_keys=True), signature, "available", created_at),
            )
        self._audit("production.agent_marketplace.publish", package_id, project=project, metadata={"name": name, "version": version})
        return self.agent_package(package_id) or {}

    def agent_package(self, package_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_agent_marketplace WHERE id = ?", (package_id,)).fetchone()
        return agent_package_row(row) if row else None

    def plan_gpu_management(self, *, model_id: str, project: str = "default", target_tokens_per_second: float = 20.0) -> dict[str, Any]:
        local_ai = LocalAIInfrastructure(self.store, self.gateway)
        hardware = local_ai.detect_hardware()
        adapters = local_ai.adapters()
        available = [adapter for adapter in adapters if adapter["available"]]
        plan = {
            "model_id": model_id,
            "target_tokens_per_second": target_tokens_per_second,
            "hardware": hardware,
            "selected_adapter": available[0]["id"] if available else "cpu",
            "fallback_adapter": "cpu",
            "privacy_boundary": "GPU planning uses local hardware detection only.",
        }
        profile_id = stable_id("gpu-plan", project, model_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO production_gpu_profiles (id, project, profile_json, plan_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (profile_id, project, json.dumps(hardware, sort_keys=True), json.dumps(plan, sort_keys=True), created_at),
            )
        self._audit("production.gpu.plan", profile_id, project=project, metadata={"model_id": model_id})
        return {"id": profile_id, "project": project, "plan": plan, "created_at": created_at}

    def create_finetune_job(
        self,
        *,
        model_id: str,
        dataset_ref: str,
        project: str = "default",
        job_type: str = "lora",
        hyperparameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = stable_id("finetune", project, model_id, dataset_ref, job_type, utc_now())
        artifact = {
            "adapter_type": job_type,
            "status": "planned",
            "output_dir": str(self.package_root / f"{job_id}-adapter"),
            "privacy_boundary": "Fine-tuning job metadata is local; training executes only when a local trainer is attached.",
        }
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_finetune_jobs (
                    id, project, model_id, job_type, dataset_ref, hyperparameters_json,
                    status, artifact_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    project,
                    model_id,
                    job_type,
                    dataset_ref,
                    json.dumps(hyperparameters or {}, sort_keys=True),
                    "planned",
                    json.dumps(artifact, sort_keys=True),
                    now,
                    now,
                ),
            )
        self._audit("production.finetune.create", job_id, project=project, metadata={"model_id": model_id, "job_type": job_type})
        return self.finetune_job(job_id) or {}

    def finetune_job(self, job_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_finetune_jobs WHERE id = ?", (job_id,)).fetchone()
        return finetune_job_row(row) if row else None

    def create_proactive_job(
        self,
        *,
        name: str,
        query: str,
        project: str = "default",
        cadence_seconds: int = 3600,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        job_id = stable_id("proactive-job", project, name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO production_proactive_jobs (
                    id, project, name, query, cadence_seconds, enabled, last_run_at,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT last_run_at FROM production_proactive_jobs WHERE id = ?), ''), ?, COALESCE((SELECT created_at FROM production_proactive_jobs WHERE id = ?), ?), ?)
                """,
                (
                    job_id,
                    project,
                    name,
                    query,
                    int(cadence_seconds),
                    1,
                    job_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    job_id,
                    now,
                    now,
                ),
            )
        self._audit("production.proactive.upsert", job_id, project=project, metadata={"query": query})
        return self.proactive_job(job_id) or {}

    def proactive_job(self, job_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM production_proactive_jobs WHERE id = ?", (job_id,)).fetchone()
        return proactive_job_row(row) if row else None

    def run_proactive_job(self, job_id: str) -> dict[str, Any]:
        job = self.proactive_job(job_id)
        if job is None:
            raise ValueError(f"Unknown proactive job '{job_id}'.")
        events = self.store.search_events(job["query"], project=job["project"], limit=10)
        insight = {
            "query": job["query"],
            "matched_events": len(events),
            "top_titles": [event["title"] for event in events[:5]],
            "recommendation": "Review matched local context and update goals if needed.",
        }
        run_id = stable_id("proactive-run", job_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO production_proactive_runs (id, job_id, project, insight_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, job_id, job["project"], json.dumps(insight, sort_keys=True), created_at),
            )
            connection.execute("UPDATE production_proactive_jobs SET last_run_at = ?, updated_at = ? WHERE id = ?", (created_at, created_at, job_id))
        self.store.add_event(
            ContextEvent(
                title=f"Proactive insight: {job['name']}",
                content=json.dumps(insight, sort_keys=True),
                type="proactive_insight",
                source="production-runtime",
                project=job["project"],
                metadata={"job_id": job_id, "run_id": run_id},
            )
        )
        self._audit("production.proactive.run", run_id, project=job["project"], metadata={"job_id": job_id})
        return {"id": run_id, "job_id": job_id, "project": job["project"], "insight": insight, "created_at": created_at}

    def aggregate_federated_round(
        self,
        *,
        objective: str,
        updates: list[dict[str, float]],
        project: str = "default",
    ) -> dict[str, Any]:
        if not updates:
            raise ValueError("At least one federated update is required.")
        keys = sorted({key for update in updates for key in update})
        aggregate = {
            key: round(sum(float(update.get(key, 0.0)) for update in updates) / len(updates), 8)
            for key in keys
        }
        privacy = {
            "raw_updates_stored": False,
            "secure_aggregation": "local-average",
            "participant_count": len(updates),
        }
        round_id = stable_id("federated-round", project, objective, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_federated_rounds (
                    id, project, objective, participant_count, aggregate_json, privacy_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (round_id, project, objective, len(updates), json.dumps(aggregate, sort_keys=True), json.dumps(privacy, sort_keys=True), created_at),
            )
        self._audit("production.federated.aggregate", round_id, project=project, metadata={"participant_count": len(updates)})
        return {"id": round_id, "project": project, "objective": objective, "aggregate": aggregate, "privacy": privacy, "created_at": created_at}

    def register_sync_device(
        self,
        *,
        name: str,
        public_key: str,
        project: str = "default",
        trust_level: str = "trusted",
    ) -> dict[str, Any]:
        device_id = stable_id("sync-device", project, name, public_key)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO production_sync_devices (id, project, name, public_key, trust_level, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (device_id, project, name, public_key, trust_level, created_at),
            )
        self._audit("production.sync.device.register", device_id, project=project, metadata={"name": name})
        return {"id": device_id, "project": project, "name": name, "public_key": public_key, "trust_level": trust_level, "created_at": created_at}

    def create_sync_bundle(
        self,
        *,
        from_device_id: str,
        to_device_id: str,
        data: dict[str, Any],
        passphrase: str,
        project: str = "default",
    ) -> dict[str, Any]:
        envelope = self.encrypt_envelope(
            label=f"sync:{from_device_id}:{to_device_id}",
            plaintext=json.dumps(data, sort_keys=True),
            passphrase=passphrase,
            recipients=[to_device_id],
            project=project,
        )
        bundle_id = stable_id("sync-bundle", project, from_device_id, to_device_id, envelope["id"])
        manifest = {
            "tables": sorted(data.keys()),
            "encrypted": True,
            "privacy_boundary": "Sync bundles are encrypted before transport and stored locally.",
        }
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_sync_bundles (
                    id, project, from_device_id, to_device_id, envelope_id, status, manifest_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (bundle_id, project, from_device_id, to_device_id, envelope["id"], "created", json.dumps(manifest, sort_keys=True), created_at),
            )
        self._audit("production.sync.bundle.create", bundle_id, project=project, metadata={"to_device_id": to_device_id})
        return {"id": bundle_id, "project": project, "from_device_id": from_device_id, "to_device_id": to_device_id, "envelope_id": envelope["id"], "manifest": manifest, "status": "created", "created_at": created_at}

    def publish_sdk_package(
        self,
        *,
        language: str,
        version: str,
        project: str = "default",
    ) -> dict[str, Any]:
        developer = DeveloperPlatform(self.store)
        artifacts = developer.generate_sdk(language=language)
        package_id = stable_id("sdk-package", project, language, version)
        artifact = {
            "language": language,
            "version": version,
            "files": artifacts["files"],
            "publish_mode": "local-artifact",
            "privacy_boundary": "SDK package publishing writes a local artifact manifest only.",
        }
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO production_sdk_packages (id, project, language, version, artifact_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (package_id, project, language, version, json.dumps(artifact, sort_keys=True), "published-local", created_at),
            )
        self._audit("production.sdk.publish", package_id, project=project, metadata={"language": language, "version": version})
        return {"id": package_id, "project": project, "language": language, "version": version, "artifact": artifact, "status": "published-local", "created_at": created_at}

    def run_sandboxed_plugin(
        self,
        *,
        plugin_id: str,
        profile_id: str,
        input: dict[str, Any],
        project: str = "default",
    ) -> dict[str, Any]:
        security = SecurityPrivacyManager(self.store)
        evaluation = security.evaluate_sandbox_request(
            profile_id=profile_id,
            permissions=[str(item) for item in input.get("permissions", [])] if isinstance(input.get("permissions"), list) else [],
            network=bool(input.get("network", False)),
        )
        status = "completed" if evaluation["allowed"] else "blocked"
        output = {"echo": input, "sandbox": evaluation}
        run_id = stable_id("sandbox-run", project, plugin_id, profile_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO production_sandbox_runs (id, project, plugin_id, profile_id, status, result_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, project, plugin_id, profile_id, status, json.dumps(output, sort_keys=True), created_at),
            )
        self._audit("production.plugin.sandbox.run", run_id, project=project, status=status, metadata={"plugin_id": plugin_id})
        return {"id": run_id, "project": project, "plugin_id": plugin_id, "profile_id": profile_id, "status": status, "result": output, "created_at": created_at}

    def _run_job_payload(self, job: dict[str, Any]) -> dict[str, Any]:
        payload = job["payload"]
        if job["kind"] == "connector.poll":
            return self.poll_connector(str(payload["adapter_id"]), project=job["project"], allow_external=bool(payload.get("allow_external", False)))
        if job["kind"] == "webhook.retry":
            return DeveloperPlatform(self.store).dispatch_webhook(
                event=str(payload.get("event", "")),
                payload=payload.get("payload") if isinstance(payload.get("payload"), dict) else {},
                allow_network=bool(payload.get("allow_network", False)),
            )
        if job["kind"] == "proactive.run":
            return self.run_proactive_job(str(payload["job_id"]))
        return {"status": "completed", "payload": payload}

    def _save_automation_run(
        self,
        *,
        project: str,
        runtime: str,
        actions: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> dict[str, Any]:
        run_id = stable_id("automation-run", project, runtime, json.dumps(actions, sort_keys=True), utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO production_automation_runs (
                    id, project, runtime, status, actions_json, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, project, runtime, "completed", json.dumps(actions, sort_keys=True), json.dumps(result, sort_keys=True), created_at),
            )
        self._audit("production.automation.run", run_id, project=project, metadata={"runtime": runtime, "action_count": len(actions)})
        return {"id": run_id, "project": project, "runtime": runtime, "status": "completed", "actions": actions, "result": result, "created_at": created_at}

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str = "default",
        status: str = "allowed",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="production-runtime",
                action=action,
                resource=resource,
                project=project,
                status=status,
                metadata=metadata or {},
            )
        )


def stable_id(*parts: Any) -> str:
    return hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:32]


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")))


def conditions_match(conditions: dict[str, Any], context: dict[str, Any]) -> bool:
    unless = conditions.get("unless")
    if unless and bool(context.get(str(unless))):
        return False
    equals = conditions.get("equals")
    if isinstance(equals, dict):
        for key, expected in equals.items():
            if context.get(key) != expected:
                return False
    return True


def merkle(hashes: list[str]) -> str:
    if not hashes:
        return "0" * 64
    layer = hashes
    while len(layer) > 1:
        if len(layer) % 2:
            layer.append(layer[-1])
        layer = [sha256(layer[index] + layer[index + 1]) for index in range(0, len(layer), 2)]
    return layer[0]


def vector_magnitude(vector: list[float]) -> float:
    return sum(value * value for value in vector) ** 0.5


def cosine(left: list[float], right: list[float]) -> float:
    return cosine_with_magnitude(left, vector_magnitude(left), right, vector_magnitude(right))


def cosine_with_magnitude(left: list[float], left_mag: float, right: list[float], right_mag: float) -> float:
    if not left or not right or len(left) != len(right) or not left_mag or not right_mag:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_mag * right_mag)


def without_vector(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "vector"}


def terms(text: str) -> list[str]:
    return [part.lower() for part in text.replace("\n", " ").split() if len(part) > 2]


def lexical_score(query_terms: list[str], text: str) -> float:
    if not query_terms:
        return 0.0
    counts = Counter(terms(text))
    hits = sum(counts.get(term, 0) for term in query_terms)
    return min(1.0, hits / len(query_terms))


def shannon(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counter.values())


def memory_recommendations(count: int, type_counts: Counter[str], tag_counts: Counter[str]) -> list[str]:
    recommendations = []
    if count < 5:
        recommendations.append("Add more typed memories before relying on long-horizon predictions.")
    if "preference" not in type_counts:
        recommendations.append("Capture explicit user preferences.")
    if not tag_counts:
        recommendations.append("Add tags to improve retrieval and consolidation.")
    return recommendations or ["Memory profile has enough local structure for current foundation workflows."]


def extract_title(html: str) -> str:
    lower = html.lower()
    start = lower.find("<title>")
    end = lower.find("</title>")
    if start == -1 or end == -1 or end <= start:
        return ""
    return html[start + len("<title>") : end].strip()


def safe_child_path(root: Path, relative: str) -> Path:
    relative = relative.strip().replace("\\", "/")
    if not relative or relative == ".":
        candidate = root
    else:
        candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError("Desktop automation path escaped the OpenAgent data directory.")
    return candidate


def write_app_package_files(package_path: Path, manifest: dict[str, Any]) -> None:
    app_type = str(manifest.get("app_type", "desktop"))
    run_command = "openagent-desktop" if app_type == "desktop" else "python -m openagent --port 8765"
    (package_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (package_path / "README.md").write_text(
        "\n".join(
            [
                "# OpenAgent App Package",
                "",
                f"Type: {manifest['app_type']}",
                f"Platform: {manifest['platform']}",
                "",
                "Run locally:",
                "",
                "```bash",
                run_command,
                "```",
                "",
                "Desktop packages launch the native OpenAgent shell. Web and PWA packages point at the self-hosted OpenAgent runtime.",
                "",
                "Privacy: this package does not add telemetry. Data remains in the configured local OpenAgent data directory.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (package_path / "openagent-desktop.ps1").write_text(
        "$ErrorActionPreference = \"Stop\"\nopenagent-desktop @args\n",
        encoding="utf-8",
    )
    (package_path / "OpenAgent.cmd").write_text(
        "@echo off\nopenagent-desktop %*\n",
        encoding="utf-8",
    )
    (package_path / "openagent-desktop.sh").write_text(
        "#!/usr/bin/env sh\nopenagent-desktop \"$@\"\n",
        encoding="utf-8",
    )
    (package_path / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>OpenAgent</title><script>location.href="http://127.0.0.1:8765"</script>',
        encoding="utf-8",
    )
    (package_path / "manifest.webmanifest").write_text(
        json.dumps(
            {
                "name": manifest.get("name", "OpenAgent"),
                "short_name": "OpenAgent",
                "start_url": "http://127.0.0.1:8765",
                "display": "standalone",
                "background_color": "#101418",
                "theme_color": "#101418",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (package_path / "service-worker.js").write_text(
        "self.addEventListener('install', event => self.skipWaiting());\n"
        "self.addEventListener('fetch', event => event.respondWith(fetch(event.request)));\n",
        encoding="utf-8",
    )


class BrowserDocument(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.title = ""
        self._in_title = False
        self.forms: list[dict[str, Any]] = []
        self.fields: dict[str, str] = {}
        self.links: list[str] = []
        self._text: list[str] = []
        self._current_form: dict[str, Any] | None = None

    @classmethod
    def parse(cls, html: str) -> "BrowserDocument":
        parser = cls()
        parser.feed(html or "")
        return parser

    @property
    def text(self) -> str:
        return " ".join(part for part in self._text if part).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {key.lower(): value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "form":
            self._current_form = {"id": values.get("id", ""), "action": values.get("action", ""), "method": values.get("method", "get"), "fields": []}
            self.forms.append(self._current_form)
        elif tag in {"input", "textarea", "select"}:
            key = values.get("name") or values.get("id") or values.get("placeholder") or f"{tag}:{len(self.fields) + 1}"
            self.fields[key] = values.get("value", "")
            if self._current_form is not None:
                self._current_form["fields"].append(key)
        elif tag == "a" and values.get("href"):
            self.links.append(values["href"])

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "form":
            self._current_form = None

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text
        else:
            self._text.append(text)


def resolve_field_selector(selector: str, fields: dict[str, str]) -> str:
    selector = selector.strip()
    if selector.startswith("#"):
        selector = selector[1:]
    if selector.startswith("[name=") and selector.endswith("]"):
        selector = selector[6:-1].strip("'\"")
    if selector in fields:
        return selector
    return selector or f"field:{len(fields) + 1}"


def render_filled_html(html: str, fields: dict[str, str]) -> str:
    if not html:
        return json.dumps({"fields": fields}, sort_keys=True)
    rendered = html
    for key, value in fields.items():
        escaped = value.replace('"', "&quot;")
        rendered = replace_input_value(rendered, key, escaped)
    return rendered


def replace_input_value(html: str, key: str, value: str) -> str:
    patterns = [
        f'name="{key}"',
        f"id=\"{key}\"",
        f"name='{key}'",
        f"id='{key}'",
    ]
    for pattern in patterns:
        index = html.find(pattern)
        if index == -1:
            continue
        tag_start = html.rfind("<", 0, index)
        tag_end = html.find(">", index)
        if tag_start == -1 or tag_end == -1:
            continue
        tag = html[tag_start : tag_end + 1]
        if "value=" in tag:
            new_tag = replace_attr_value(tag, "value", value)
        else:
            new_tag = tag[:-1] + f' value="{value}">'
        return html[:tag_start] + new_tag + html[tag_end + 1 :]
    return html + f"\n<!-- OpenAgent filled {key}={value} -->"


def replace_attr_value(tag: str, attr: str, value: str) -> str:
    for quote in ("\"", "'"):
        marker = f"{attr}={quote}"
        start = tag.find(marker)
        if start != -1:
            value_start = start + len(marker)
            value_end = tag.find(quote, value_start)
            if value_end != -1:
                return tag[:value_start] + value + tag[value_end:]
    return tag


def b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def b64url_json(value: str) -> dict[str, Any]:
    return json.loads(b64url_decode(value).decode("utf-8"))


def validate_jwt_with_jwks(token: str, jwks: dict[str, Any], *, issuer: str, audience: str) -> dict[str, Any]:
    try:
        header_raw, payload_raw, signature_raw = token.split(".")
        header = b64url_json(header_raw)
        claims = b64url_json(payload_raw)
        signature = b64url_decode(signature_raw)
    except Exception as exc:
        return {"valid": False, "reason": f"malformed_token:{exc}"}
    signing_input = f"{header_raw}.{payload_raw}".encode("ascii")
    key = find_jwk(jwks, str(header.get("kid", "")), str(header.get("alg", "")))
    if key is None:
        return {"valid": False, "reason": "key_not_found", "header": header}
    alg = str(header.get("alg", ""))
    if alg == "HS256":
        secret = b64url_decode(str(key.get("k", ""))) if key.get("k") else str(key.get("secret", "")).encode("utf-8")
        expected = hmac.new(secret, signing_input, hashlib.sha256).digest()
        signature_ok = hmac.compare_digest(expected, signature)
    elif alg == "RS256":
        signature_ok = verify_rs256(signing_input, signature, key)
    else:
        return {"valid": False, "reason": f"unsupported_alg:{alg}", "header": header}
    if not signature_ok:
        return {"valid": False, "reason": "signature_invalid", "header": header}
    now = int(datetime.now(timezone.utc).timestamp())
    if claims.get("iss") != issuer:
        return {"valid": False, "reason": "issuer_mismatch", "claims": claims}
    aud = claims.get("aud")
    if isinstance(aud, list):
        aud_ok = audience in aud
    else:
        aud_ok = aud == audience
    if not aud_ok:
        return {"valid": False, "reason": "audience_mismatch", "claims": claims}
    if int(claims.get("exp", now + 1)) < now:
        return {"valid": False, "reason": "expired", "claims": claims}
    if int(claims.get("nbf", 0)) > now:
        return {"valid": False, "reason": "not_before", "claims": claims}
    return {"valid": True, "header": header, "claims": claims, "key_id": key.get("kid", "")}


def find_jwk(jwks: dict[str, Any], kid: str, alg: str) -> dict[str, Any] | None:
    keys = jwks.get("keys", [])
    if not isinstance(keys, list):
        return None
    for key in keys:
        if not isinstance(key, dict):
            continue
        if kid and key.get("kid") != kid:
            continue
        if alg and key.get("alg", alg) != alg:
            continue
        return key
    return None


def verify_rs256(signing_input: bytes, signature: bytes, jwk: dict[str, Any]) -> bool:
    try:
        n = int.from_bytes(b64url_decode(str(jwk["n"])), "big")
        e = int.from_bytes(b64url_decode(str(jwk["e"])), "big")
    except Exception:
        return False
    key_len = (n.bit_length() + 7) // 8
    digest = hashlib.sha256(signing_input).digest()
    digest_info = bytes.fromhex("3031300d060960864801650304020105000420") + digest
    decoded = pow(int.from_bytes(signature, "big"), e, n).to_bytes(key_len, "big")
    if not decoded.startswith(b"\x00\x01"):
        return False
    try:
        separator = decoded.index(b"\x00", 2)
    except ValueError:
        return False
    padding = decoded[2:separator]
    return len(padding) >= 8 and all(byte == 0xFF for byte in padding) and decoded[separator + 1 :] == digest_info


def policy_rule_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "scope": row["scope"],
        "effect": row["effect"],
        "conditions": json.loads(row["conditions_json"]),
        "priority": row["priority"],
        "description": row["description"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def queue_job_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "queue": row["queue"],
        "kind": row["kind"],
        "payload": json.loads(row["payload_json"]),
        "status": row["status"],
        "attempts": row["attempts"],
        "max_attempts": row["max_attempts"],
        "next_run_at": row["next_run_at"],
        "result": json.loads(row["result_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def audit_anchor_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "from_event_id": row["from_event_id"],
        "to_event_id": row["to_event_id"],
        "event_count": row["event_count"],
        "merkle_root": row["merkle_root"],
        "previous_anchor_hash": row["previous_anchor_hash"],
        "anchor_hash": row["anchor_hash"],
        "storage_mode": row["storage_mode"],
        "created_at": row["created_at"],
    }


def vector_index_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "dimensions": row["dimensions"],
        "metric": row["metric"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def vector_record_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "index_id": row["index_id"],
        "project": row["project"],
        "namespace": row["namespace"],
        "external_id": row["external_id"],
        "content": row["content"],
        "vector": json.loads(row["vector_json"]),
        "magnitude": row["magnitude"],
        "metadata": json.loads(row["metadata_json"]),
        "tombstoned": bool(row["tombstoned"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def connector_adapter_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "platform": row["platform"],
        "mode": row["mode"],
        "endpoint_url": row["endpoint_url"],
        "poll_interval_seconds": row["poll_interval_seconds"],
        "credentials_ref": row["credentials_ref"],
        "enabled": bool(row["enabled"]),
        "config": json.loads(row["config_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def crypto_envelope_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "label": row["label"],
        "recipients": json.loads(row["recipients_json"]),
        "envelope": json.loads(row["envelope_json"]),
        "provider": row["provider"],
        "created_at": row["created_at"],
    }


def app_package_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "app_type": row["app_type"],
        "platform": row["platform"],
        "manifest": json.loads(row["manifest_json"]),
        "package_path": row["package_path"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def agent_package_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "version": row["version"],
        "author": row["author"],
        "manifest": json.loads(row["manifest_json"]),
        "signature": row["signature"],
        "status": row["status"],
        "created_at": row["created_at"],
    }


def finetune_job_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "model_id": row["model_id"],
        "job_type": row["job_type"],
        "dataset_ref": row["dataset_ref"],
        "hyperparameters": json.loads(row["hyperparameters_json"]),
        "status": row["status"],
        "artifact": json.loads(row["artifact_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def proactive_job_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "name": row["name"],
        "query": row["query"],
        "cadence_seconds": row["cadence_seconds"],
        "enabled": bool(row["enabled"]),
        "last_run_at": row["last_run_at"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
