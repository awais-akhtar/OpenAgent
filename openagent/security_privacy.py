"""Stage 9 security and privacy hardening layer."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import secrets
from typing import Any

from .crypto import (
    CURRENT_ENVELOPE_VERSION,
    decrypt_legacy_security_envelope,
    decrypt_text as decrypt_current_text,
    encrypt_text as encrypt_current_text,
    is_current_envelope,
)
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


class SecurityPrivacyError(RuntimeError):
    """Raised when a security or privacy operation is invalid."""


RETENTION_TABLES = {
    "context_events": ("events", "created_at"),
    "agent_runs": ("agent_runs", "created_at"),
    "workflow_runs": ("workflow_runs", "created_at"),
    "developer_webhook_deliveries": ("developer_webhook_deliveries", "created_at"),
}


class SecurityPrivacyManager:
    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.initialize()
        self.seed_defaults()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS security_secrets (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    ciphertext_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_retention_policies (
                    id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    retention_days INTEGER NOT NULL,
                    action TEXT NOT NULL DEFAULT 'delete',
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_share_grants (
                    id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    grantee TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    expires_at TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_security_share_resource
                    ON security_share_grants(resource_type, resource_id, status);

                CREATE TABLE IF NOT EXISTS security_egress_requests (
                    id TEXT PRIMARY KEY,
                    requester TEXT NOT NULL,
                    purpose TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    risk_level TEXT NOT NULL DEFAULT 'medium',
                    approved_by TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_security_egress_status
                    ON security_egress_requests(status, created_at DESC);

                CREATE TABLE IF NOT EXISTS security_audit_chain (
                    audit_event_id TEXT PRIMARY KEY,
                    previous_hash TEXT NOT NULL,
                    event_hash TEXT NOT NULL,
                    chain_hash TEXT NOT NULL,
                    sealed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_privacy_reports (
                    id TEXT PRIMARY KEY,
                    report_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS security_sandbox_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    network_allowed INTEGER NOT NULL DEFAULT 0,
                    filesystem_json TEXT NOT NULL DEFAULT '{}',
                    env_json TEXT NOT NULL DEFAULT '{}',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def seed_defaults(self) -> None:
        if not self.sandbox_profiles():
            self.create_sandbox_profile(
                name="default-deny",
                permissions=[],
                network_allowed=False,
                filesystem={"read": [], "write": []},
                env={},
                metadata={"description": "Default profile denies network and grants no tool permissions."},
                audit=False,
            )

    def status(self) -> dict[str, Any]:
        chain = self.verify_audit_chain()
        return {
            "secrets": len(self.secrets()),
            "retention_policies": len(self.retention_policies()),
            "share_grants": len(self.share_grants()),
            "egress_requests": len(self.egress_requests()),
            "sealed_audit_events": len(self.audit_chain()),
            "audit_chain_ok": chain["ok"],
            "privacy_reports": len(self.privacy_reports()),
            "sandbox_profiles": len(self.sandbox_profiles()),
            "network_default": False,
        }

    def store_secret(
        self,
        *,
        name: str,
        value: str,
        passphrase: str,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Secret name is required.")
        if not passphrase:
            raise ValueError("A passphrase is required to encrypt the secret.")
        secret_id = stable_id("secret", name)
        now = utc_now()
        payload = encrypt_text(value, passphrase)
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO security_secrets (
                    id, name, description, ciphertext_json, metadata_json,
                    status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM security_secrets WHERE id = ?), ?), ?)
                """,
                (
                    secret_id,
                    name,
                    description,
                    json.dumps(payload, sort_keys=True),
                    json.dumps(metadata or {}, sort_keys=True),
                    "active",
                    secret_id,
                    now,
                    now,
                ),
            )
        self._audit("security.secret.store", secret_id, metadata={"name": name})
        return self.get_secret(secret_id) or {}

    def get_secret(self, secret_id: str) -> dict[str, Any] | None:
        row = self._secret_row(secret_id)
        return secret_row(row) if row else None

    def secrets(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_secrets ORDER BY name").fetchall()
        return [secret_row(row) for row in rows]

    def reveal_secret(self, *, secret_id: str, passphrase: str) -> dict[str, Any]:
        row = self._secret_row(secret_id)
        if row is None:
            raise ValueError(f"Unknown secret '{secret_id}'.")
        try:
            encrypted_payload = json.loads(row["ciphertext_json"])
            value = decrypt_text(encrypted_payload, passphrase)
        except ValueError as exc:
            self._audit("security.secret.reveal", row["id"], metadata={"status": "denied"})
            raise SecurityPrivacyError(str(exc)) from exc
        migrated = not is_current_envelope(encrypted_payload)
        if migrated:
            replacement = encrypt_text(value, passphrase)
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE security_secrets SET ciphertext_json = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(replacement, sort_keys=True), utc_now(), str(row["id"])),
                )
        self._audit("security.secret.reveal", row["id"], metadata={"status": "allowed"})
        return {**secret_row(row), "value": value, "envelope_migrated": migrated}

    def rotate_secret(self, *, secret_id: str, value: str, passphrase: str) -> dict[str, Any]:
        row = self._secret_row(secret_id)
        if row is None:
            raise ValueError(f"Unknown secret '{secret_id}'.")
        payload = encrypt_text(value, passphrase)
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE security_secrets
                SET ciphertext_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(payload, sort_keys=True), utc_now(), row["id"]),
            )
        self._audit("security.secret.rotate", row["id"], metadata={"name": row["name"]})
        return self.get_secret(row["id"]) or {}

    def _secret_row(self, secret_id: str) -> Any | None:
        with self.store.session() as connection:
            return connection.execute(
                "SELECT * FROM security_secrets WHERE id = ? OR name = ?",
                (secret_id, secret_id),
            ).fetchone()

    def create_retention_policy(
        self,
        *,
        resource_type: str,
        retention_days: int,
        action: str = "delete",
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        resource_type = resource_type.strip()
        if resource_type not in RETENTION_TABLES:
            known = ", ".join(sorted(RETENTION_TABLES))
            raise ValueError(f"Unsupported retention resource_type '{resource_type}'. Known: {known}")
        if retention_days < 0:
            raise ValueError("retention_days must be zero or greater.")
        if action != "delete":
            raise ValueError("Only delete retention action is implemented.")
        policy_id = stable_id("retention", resource_type)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO security_retention_policies (
                    id, resource_type, retention_days, action, status, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM security_retention_policies WHERE id = ?), ?), ?)
                """,
                (
                    policy_id,
                    resource_type,
                    retention_days,
                    action,
                    status,
                    json.dumps(metadata or {}, sort_keys=True),
                    policy_id,
                    now,
                    now,
                ),
            )
        self._audit(
            "security.retention.upsert",
            policy_id,
            metadata={"resource_type": resource_type, "retention_days": retention_days},
        )
        return self.get_retention_policy(policy_id) or {}

    def get_retention_policy(self, policy_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM security_retention_policies WHERE id = ?",
                (policy_id,),
            ).fetchone()
        return retention_policy_row(row) if row else None

    def retention_policies(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_retention_policies ORDER BY resource_type").fetchall()
        return [retention_policy_row(row) for row in rows]

    def apply_retention(self, *, policy_id: str, dry_run: bool = True) -> dict[str, Any]:
        policy = self.get_retention_policy(policy_id)
        if policy is None:
            raise ValueError(f"Unknown retention policy '{policy_id}'.")
        if policy["status"] != "active":
            raise ValueError(f"Retention policy '{policy_id}' is not active.")
        table, column = RETENTION_TABLES[policy["resource_type"]]
        cutoff = retention_cutoff(policy["retention_days"])
        with self.store.session() as connection:
            planned = connection.execute(
                f"SELECT count(*) FROM {table} WHERE {column} < ?",
                (cutoff,),
            ).fetchone()[0]
            deleted = 0
            if not dry_run and planned:
                cursor = connection.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))
                deleted = cursor.rowcount
        result = {
            "policy_id": policy_id,
            "resource_type": policy["resource_type"],
            "cutoff": cutoff,
            "dry_run": dry_run,
            "planned": planned,
            "deleted": deleted,
        }
        self._audit("security.retention.apply", policy_id, metadata=result)
        return result

    def create_share_grant(
        self,
        *,
        resource_type: str,
        resource_id: str,
        grantee: str,
        scope: str,
        expires_at: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not resource_type.strip() or not resource_id.strip():
            raise ValueError("resource_type and resource_id are required.")
        if not grantee.strip():
            raise ValueError("grantee is required.")
        grant_id = unique_id("share", resource_type, resource_id, grantee, scope)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO security_share_grants (
                    id, resource_type, resource_id, grantee, scope, status,
                    expires_at, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    grant_id,
                    resource_type,
                    resource_id,
                    grantee,
                    scope,
                    "active",
                    expires_at,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )
        self._audit("security.share.create", grant_id, metadata={"resource_type": resource_type, "grantee": grantee})
        return self.get_share_grant(grant_id) or {}

    def get_share_grant(self, grant_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM security_share_grants WHERE id = ?", (grant_id,)).fetchone()
        return share_grant_row(row) if row else None

    def share_grants(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_share_grants ORDER BY created_at DESC").fetchall()
        return [share_grant_row(row) for row in rows]

    def resolve_share_grant(self, *, grant_id: str, status: str) -> dict[str, Any]:
        if status not in {"revoked", "expired"}:
            raise ValueError("Share grant status must be revoked or expired.")
        if self.get_share_grant(grant_id) is None:
            raise ValueError(f"Unknown share grant '{grant_id}'.")
        with self.store.session() as connection:
            connection.execute(
                "UPDATE security_share_grants SET status = ?, resolved_at = ? WHERE id = ?",
                (status, utc_now(), grant_id),
            )
        self._audit("security.share.resolve", grant_id, metadata={"status": status})
        return self.get_share_grant(grant_id) or {}

    def request_egress(
        self,
        *,
        requester: str,
        purpose: str,
        destination: str,
        resource_type: str,
        resource_id: str,
        risk_level: str = "medium",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not destination.strip():
            raise ValueError("destination is required.")
        request_id = unique_id("egress", requester, destination, resource_type, resource_id)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO security_egress_requests (
                    id, requester, purpose, destination, resource_type, resource_id,
                    status, risk_level, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    requester or "unknown",
                    purpose,
                    destination,
                    resource_type,
                    resource_id,
                    "pending",
                    risk_level,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )
        self._audit("security.egress.request", request_id, metadata={"destination": destination, "risk_level": risk_level})
        return self.get_egress_request(request_id) or {}

    def get_egress_request(self, request_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM security_egress_requests WHERE id = ?", (request_id,)).fetchone()
        return egress_request_row(row) if row else None

    def egress_requests(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_egress_requests ORDER BY created_at DESC").fetchall()
        return [egress_request_row(row) for row in rows]

    def resolve_egress_request(self, *, request_id: str, approved: bool, approved_by: str) -> dict[str, Any]:
        if self.get_egress_request(request_id) is None:
            raise ValueError(f"Unknown egress request '{request_id}'.")
        status = "approved" if approved else "denied"
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE security_egress_requests
                SET status = ?, approved_by = ?, resolved_at = ?
                WHERE id = ?
                """,
                (status, approved_by, utc_now(), request_id),
            )
        self._audit("security.egress.resolve", request_id, metadata={"status": status, "approved_by": approved_by})
        return self.get_egress_request(request_id) or {}

    def seal_audit_events(self) -> dict[str, Any]:
        events = self.store.list_audit_events(limit=500)
        events = sorted(events, key=lambda event: (event["created_at"], event["id"]))
        sealed_ids = {item["audit_event_id"] for item in self.audit_chain()}
        chain = self.audit_chain()
        previous_hash = chain[-1]["chain_hash"] if chain else "0" * 64
        sealed = []
        with self.store.session() as connection:
            for event in events:
                if event["id"] in sealed_ids:
                    continue
                event_hash = hash_json(event)
                chain_hash = sha256(f"{previous_hash}:{event_hash}")
                connection.execute(
                    """
                    INSERT INTO security_audit_chain (
                        audit_event_id, previous_hash, event_hash, chain_hash, sealed_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (event["id"], previous_hash, event_hash, chain_hash, utc_now()),
                )
                sealed.append(event["id"])
                previous_hash = chain_hash
        self._audit("security.audit_chain.seal", "audit-chain", metadata={"sealed": len(sealed)})
        return {"sealed": len(sealed), "audit_event_ids": sealed, "head": previous_hash}

    def audit_chain(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM security_audit_chain ORDER BY rowid ASC"
            ).fetchall()
        return [audit_chain_row(row) for row in rows]

    def verify_audit_chain(self) -> dict[str, Any]:
        chain = self.audit_chain()
        if not chain:
            return {"ok": True, "sealed": 0, "issues": []}
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM audit_events").fetchall()
        events = {row["id"]: audit_event_dict(row) for row in rows}
        expected_previous = "0" * 64
        issues = []
        for item in chain:
            event = events.get(item["audit_event_id"])
            if event is None:
                issues.append({"audit_event_id": item["audit_event_id"], "issue": "missing_event"})
                continue
            event_hash = hash_json(event)
            chain_hash = sha256(f"{expected_previous}:{event_hash}")
            if item["previous_hash"] != expected_previous:
                issues.append({"audit_event_id": item["audit_event_id"], "issue": "previous_hash_mismatch"})
            if item["event_hash"] != event_hash:
                issues.append({"audit_event_id": item["audit_event_id"], "issue": "event_hash_mismatch"})
            if item["chain_hash"] != chain_hash:
                issues.append({"audit_event_id": item["audit_event_id"], "issue": "chain_hash_mismatch"})
            expected_previous = item["chain_hash"]
        return {"ok": not issues, "sealed": len(chain), "head": expected_previous, "issues": issues}

    def create_sandbox_profile(
        self,
        *,
        name: str,
        permissions: list[str] | None = None,
        network_allowed: bool = False,
        filesystem: dict[str, Any] | None = None,
        env: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Sandbox profile name is required.")
        profile_id = stable_id("sandbox", name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO security_sandbox_profiles (
                    id, name, permissions_json, network_allowed, filesystem_json,
                    env_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM security_sandbox_profiles WHERE id = ?), ?), ?)
                """,
                (
                    profile_id,
                    name,
                    json.dumps(permissions or [], sort_keys=True),
                    1 if network_allowed else 0,
                    json.dumps(filesystem or {"read": [], "write": []}, sort_keys=True),
                    json.dumps(env or {}, sort_keys=True),
                    json.dumps(metadata or {}, sort_keys=True),
                    profile_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit("security.sandbox.upsert", profile_id, metadata={"name": name, "network_allowed": network_allowed})
        return self.get_sandbox_profile(profile_id) or {}

    def get_sandbox_profile(self, profile_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT * FROM security_sandbox_profiles WHERE id = ? OR name = ?",
                (profile_id, profile_id),
            ).fetchone()
        return sandbox_profile_row(row) if row else None

    def sandbox_profiles(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_sandbox_profiles ORDER BY name").fetchall()
        return [sandbox_profile_row(row) for row in rows]

    def evaluate_sandbox_request(
        self,
        *,
        profile_id: str,
        permissions: list[str] | None = None,
        network: bool = False,
    ) -> dict[str, Any]:
        profile = self.get_sandbox_profile(profile_id)
        if profile is None:
            raise ValueError(f"Unknown sandbox profile '{profile_id}'.")
        requested = set(permissions or [])
        allowed_permissions = set(profile["permissions"])
        denied_permissions = sorted(requested - allowed_permissions)
        denied = []
        if denied_permissions:
            denied.append({"type": "permissions", "values": denied_permissions})
        if network and not profile["network_allowed"]:
            denied.append({"type": "network", "values": ["external"]})
        result = {
            "profile": profile,
            "allowed": not denied,
            "denied": denied,
            "requested_permissions": sorted(requested),
            "requested_network": network,
        }
        self._audit("security.sandbox.evaluate", profile["id"], metadata={"allowed": result["allowed"]})
        return result

    def create_privacy_report(self) -> dict[str, Any]:
        report = {
            "generated_at": utc_now(),
            "health": self.store.health(),
            "data_inventory": self.data_inventory(),
            "secrets": {"count": len(self.secrets()), "names": [secret["name"] for secret in self.secrets()]},
            "retention_policies": self.retention_policies(),
            "share_grants": self.share_grants(),
            "egress_requests": self.egress_requests(),
            "audit_chain": self.verify_audit_chain(),
            "sandbox_profiles": self.sandbox_profiles(),
            "network_default": False,
        }
        report_id = unique_id("privacy-report")
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO security_privacy_reports (id, report_json, created_at) VALUES (?, ?, ?)",
                (report_id, json.dumps(report, sort_keys=True), report["generated_at"]),
            )
        self._audit("security.privacy_report.create", report_id)
        return {"id": report_id, "report": report, "created_at": report["generated_at"]}

    def privacy_reports(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM security_privacy_reports ORDER BY created_at DESC").fetchall()
        return [privacy_report_row(row) for row in rows]

    def data_inventory(self) -> dict[str, int]:
        tables = [
            "events",
            "agent_runs",
            "conversations",
            "chat_messages",
            "audit_events",
            "workflows",
            "workflow_runs",
            "security_secrets",
            "security_share_grants",
            "security_egress_requests",
        ]
        inventory: dict[str, int] = {}
        with self.store.session() as connection:
            for table in tables:
                exists = connection.execute(
                    "SELECT count(*) FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()[0]
                inventory[table] = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] if exists else 0
        return inventory

    def _audit(self, action: str, resource: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="security-privacy",
                action=action,
                resource=resource,
                project="security",
                metadata=metadata or {},
            )
        )


def encrypt_text(value: str, passphrase: str) -> dict[str, Any]:
    """Encrypt a security secret with the shared current envelope."""

    return encrypt_current_text(value, passphrase, purpose="openagent.security.secret")


def decrypt_text(payload: dict[str, Any], passphrase: str) -> str:
    if int(payload.get("version", 0)) == CURRENT_ENVELOPE_VERSION:
        return decrypt_current_text(payload, passphrase, purpose="openagent.security.secret")
    return decrypt_legacy_security_envelope(payload, passphrase)


def stable_id(*parts: str) -> str:
    return sha256("::".join(str(part) for part in parts))


def unique_id(*parts: str) -> str:
    return stable_id(*parts, utc_now(), secrets.token_hex(8))


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_json(value: dict[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")))


def retention_cutoff(days: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0) - timedelta(days=days)).isoformat()


def secret_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "metadata": json.loads(row["metadata_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "encrypted": True,
    }


def retention_policy_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "resource_type": row["resource_type"],
        "retention_days": row["retention_days"],
        "action": row["action"],
        "status": row["status"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def share_grant_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "resource_type": row["resource_type"],
        "resource_id": row["resource_id"],
        "grantee": row["grantee"],
        "scope": row["scope"],
        "status": row["status"],
        "expires_at": row["expires_at"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }


def egress_request_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "requester": row["requester"],
        "purpose": row["purpose"],
        "destination": row["destination"],
        "resource_type": row["resource_type"],
        "resource_id": row["resource_id"],
        "status": row["status"],
        "risk_level": row["risk_level"],
        "approved_by": row["approved_by"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }


def audit_chain_row(row: Any) -> dict[str, Any]:
    return {
        "audit_event_id": row["audit_event_id"],
        "previous_hash": row["previous_hash"],
        "event_hash": row["event_hash"],
        "chain_hash": row["chain_hash"],
        "sealed_at": row["sealed_at"],
    }


def audit_event_dict(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "actor": row["actor"],
        "action": row["action"],
        "resource": row["resource"],
        "project": row["project"],
        "status": row["status"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def sandbox_profile_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "permissions": json.loads(row["permissions_json"]),
        "network_allowed": bool(row["network_allowed"]),
        "filesystem": json.loads(row["filesystem_json"]),
        "env": json.loads(row["env_json"]),
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def privacy_report_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "report": json.loads(row["report_json"]),
        "created_at": row["created_at"],
    }
