"""Stage 7 enterprise and team deployment features."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from pathlib import Path
import shutil
import time
from typing import Any

from .models import AuditEvent, ContextEvent, utc_now
from .storage import OpenAgentStore


class EnterprisePolicyError(RuntimeError):
    """Raised when a policy denies an action."""


class EnterpriseManager:
    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.export_root = (self.store.data_dir / "exports").resolve()
        self.backup_root = (self.store.data_dir / "backups").resolve()
        self.export_root.mkdir(parents=True, exist_ok=True)
        self.backup_root.mkdir(parents=True, exist_ok=True)
        self.initialize()
        self.seed_defaults()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS enterprise_users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS enterprise_roles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    permissions_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS enterprise_workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS enterprise_workspace_members (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    UNIQUE(workspace_id, user_id, role_id)
                );

                CREATE TABLE IF NOT EXISTS organization_memory (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_org_memory_workspace_created
                    ON organization_memory(workspace_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS enterprise_policies (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT NOT NULL DEFAULT '',
                    rules_json TEXT NOT NULL DEFAULT '[]',
                    enforcement TEXT NOT NULL DEFAULT 'enforce',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sso_configs (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    issuer TEXT NOT NULL,
                    client_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS backup_records (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS disaster_recovery_plans (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def seed_defaults(self) -> None:
        roles = [
            ("owner", "Owner", ["*"]),
            ("admin", "Admin", ["workspace.read", "workspace.write", "user.manage", "policy.manage", "audit.export"]),
            ("member", "Member", ["workspace.read", "workspace.write", "agent.run", "knowledge.read"]),
            ("viewer", "Viewer", ["workspace.read", "knowledge.read"]),
            ("auditor", "Auditor", ["workspace.read", "audit.export", "policy.read"]),
        ]
        for role_id, name, permissions in roles:
            self.upsert_role(role_id=role_id, name=name, permissions=permissions, audit=False)
        if not self.policies():
            self.upsert_policy(
                name="Local Boundary",
                description="Deny external effects unless explicitly approved.",
                rules=[
                    {"scope": "network.external", "effect": "deny", "unless": "allow_external"},
                    {"scope": "audit.export", "effect": "allow"},
                ],
                audit=False,
            )

    def upsert_user(
        self,
        *,
        email: str,
        name: str,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> dict[str, Any]:
        email = email.strip().lower()
        if not email:
            raise ValueError("User email is required.")
        user_id = stable_id("user", email)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO enterprise_users (
                    id, email, name, status, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM enterprise_users WHERE id = ?), ?), ?)
                """,
                (
                    user_id,
                    email,
                    name.strip() or email,
                    status,
                    json.dumps(metadata or {}, sort_keys=True),
                    user_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit("enterprise.user.upsert", user_id, metadata={"email": email, "status": status})
        return self.get_user(user_id) or {}

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM enterprise_users WHERE id = ?", (user_id,)).fetchone()
        return user_row(row) if row else None

    def users(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM enterprise_users ORDER BY email").fetchall()
        return [user_row(row) for row in rows]

    def upsert_role(
        self,
        *,
        role_id: str,
        name: str,
        permissions: list[str],
        description: str = "",
        audit: bool = True,
    ) -> dict[str, Any]:
        role_id = role_id.strip()
        name = name.strip()
        if not role_id:
            raise ValueError("Role id is required.")
        if not name:
            raise ValueError("Role name is required.")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO enterprise_roles (
                    id, name, description, permissions_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM enterprise_roles WHERE id = ?), ?), ?)
                """,
                (role_id, name, description, json.dumps(permissions), role_id, now, now),
            )
        if audit:
            self._audit("enterprise.role.upsert", role_id, metadata={"permissions": permissions})
        return self.get_role(role_id) or {}

    def get_role(self, role_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM enterprise_roles WHERE id = ?", (role_id,)).fetchone()
        return role_row(row) if row else None

    def roles(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM enterprise_roles ORDER BY name").fetchall()
        return [role_row(row) for row in rows]

    def create_workspace(
        self,
        *,
        name: str,
        owner_user_id: str,
        metadata: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> dict[str, Any]:
        workspace_name = name.strip() or "Workspace"
        if self.get_user(owner_user_id) is None:
            raise ValueError(f"Unknown owner user '{owner_user_id}'.")
        workspace_id = stable_id("workspace", workspace_name, owner_user_id)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO enterprise_workspaces (
                    id, name, owner_user_id, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, COALESCE((SELECT created_at FROM enterprise_workspaces WHERE id = ?), ?), ?)
                """,
                (
                    workspace_id,
                    workspace_name,
                    owner_user_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    workspace_id,
                    now,
                    now,
                ),
            )
        self.add_member(workspace_id=workspace_id, user_id=owner_user_id, role_id="owner", audit=audit)
        if audit:
            self._audit(
                "enterprise.workspace.create",
                workspace_id,
                project=workspace_id,
                metadata={"owner_user_id": owner_user_id, "name": workspace_name},
            )
        return self.get_workspace(workspace_id) or {}

    def get_workspace(self, workspace_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM enterprise_workspaces WHERE id = ?", (workspace_id,)).fetchone()
        return workspace_row(row) if row else None

    def workspaces(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM enterprise_workspaces ORDER BY name").fetchall()
        return [workspace_row(row) for row in rows]

    def add_member(self, *, workspace_id: str, user_id: str, role_id: str, audit: bool = True) -> dict[str, Any]:
        if self.get_workspace(workspace_id) is None:
            raise ValueError(f"Unknown workspace '{workspace_id}'.")
        if self.get_user(user_id) is None:
            raise ValueError(f"Unknown user '{user_id}'.")
        if self.get_role(role_id) is None:
            raise ValueError(f"Unknown role '{role_id}'.")
        member_id = stable_id("member", workspace_id, user_id, role_id)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO enterprise_workspace_members (
                    id, workspace_id, user_id, role_id, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (member_id, workspace_id, user_id, role_id, "active", created_at),
            )
        if audit:
            self._audit(
                "enterprise.member.add",
                member_id,
                project=workspace_id,
                metadata={"workspace_id": workspace_id, "user_id": user_id, "role_id": role_id},
            )
        return {
            "id": member_id,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "role_id": role_id,
            "status": "active",
            "created_at": created_at,
        }

    def members(self, workspace_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM enterprise_workspace_members WHERE workspace_id = ? ORDER BY created_at",
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def check_access(self, *, user_id: str, workspace_id: str, permission: str) -> dict[str, Any]:
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT r.* FROM enterprise_workspace_members m
                JOIN enterprise_roles r ON r.id = m.role_id
                WHERE m.workspace_id = ? AND m.user_id = ? AND m.status = 'active'
                """,
                (workspace_id, user_id),
            ).fetchall()
        roles = [role_row(row) for row in rows]
        allowed = any("*" in role["permissions"] or permission in role["permissions"] for role in roles)
        return {"allowed": allowed, "roles": roles, "permission": permission}

    def add_org_memory(
        self,
        *,
        workspace_id: str,
        title: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> dict[str, Any]:
        if self.get_workspace(workspace_id) is None:
            raise ValueError(f"Unknown workspace '{workspace_id}'.")
        title = title.strip()
        content = content.strip()
        if not title:
            raise ValueError("Organization memory title is required.")
        if not content:
            raise ValueError("Organization memory content is required.")
        memory_id = stable_id("org-memory", workspace_id, title, content)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO organization_memory (
                    id, workspace_id, title, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (memory_id, workspace_id, title, content, json.dumps(metadata or {}, sort_keys=True), created_at),
            )
        if audit:
            self._audit(
                "enterprise.org_memory.add",
                memory_id,
                project=workspace_id,
                metadata={"workspace_id": workspace_id, "title": title},
            )
        self.store.add_event(
            ContextEvent(
                title=f"Org memory: {title}",
                content=content,
                type="organization_memory",
                source="enterprise",
                project=workspace_id,
                metadata={"workspace_id": workspace_id, "memory_id": memory_id},
            )
        )
        return {
            "id": memory_id,
            "workspace_id": workspace_id,
            "title": title,
            "content": content,
            "metadata": metadata or {},
            "created_at": created_at,
        }

    def org_memory(self, workspace_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM organization_memory WHERE workspace_id = ? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        return [org_memory_row(row) for row in rows]

    def upsert_policy(
        self,
        *,
        name: str,
        rules: list[dict[str, Any]],
        description: str = "",
        enforcement: str = "enforce",
        audit: bool = True,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("Policy name is required.")
        policy_id = stable_id("policy", name)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO enterprise_policies (
                    id, name, description, rules_json, enforcement, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM enterprise_policies WHERE id = ?), ?), ?)
                """,
                (policy_id, name, description, json.dumps(rules, sort_keys=True), enforcement, policy_id, now, now),
            )
        if audit:
            self._audit(
                "enterprise.policy.upsert",
                policy_id,
                metadata={"name": name, "rule_count": len(rules), "enforcement": enforcement},
            )
        return self.get_policy(policy_id) or {}

    def get_policy(self, policy_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM enterprise_policies WHERE id = ?", (policy_id,)).fetchone()
        return policy_row(row) if row else None

    def policies(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM enterprise_policies ORDER BY name").fetchall()
        return [policy_row(row) for row in rows]

    def evaluate_policy(self, *, scope: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        decisions = []
        final = "allow"
        for policy in self.policies():
            for rule in policy["rules"]:
                if rule.get("scope") not in {scope, "*"}:
                    continue
                effect = rule.get("effect", "allow")
                unless_key = rule.get("unless")
                if unless_key and context.get(unless_key):
                    continue
                decisions.append({"policy": policy["name"], "effect": effect, "rule": rule})
                if effect == "deny" and policy["enforcement"] == "enforce":
                    final = "deny"
        return {"scope": scope, "decision": final, "decisions": decisions}

    def configure_sso(
        self,
        *,
        provider: str,
        issuer: str,
        client_id: str,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        audit: bool = True,
    ) -> dict[str, Any]:
        provider = provider.strip()
        issuer = issuer.strip()
        client_id = client_id.strip()
        if not provider:
            raise ValueError("SSO provider is required.")
        if not issuer:
            raise ValueError("SSO issuer is required.")
        if not client_id:
            raise ValueError("SSO client id is required.")
        config_id = stable_id("sso", provider, issuer, client_id)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sso_configs (
                    id, provider, issuer, client_id, enabled, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM sso_configs WHERE id = ?), ?), ?)
                """,
                (
                    config_id,
                    provider,
                    issuer,
                    client_id,
                    1 if enabled else 0,
                    json.dumps(metadata or {}, sort_keys=True),
                    config_id,
                    now,
                    now,
                ),
            )
        if audit:
            self._audit(
                "enterprise.sso.configure",
                config_id,
                metadata={"provider": provider, "issuer": issuer, "enabled": enabled},
            )
        return self.get_sso_config(config_id) or {}

    def get_sso_config(self, config_id: str, *, redact: bool = True) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM sso_configs WHERE id = ?", (config_id,)).fetchone()
        return sso_row(row, redact=redact) if row else None

    def sso_configs(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM sso_configs ORDER BY provider").fetchall()
        return [sso_row(row) for row in rows]

    def validate_sso_token(
        self,
        *,
        token: str,
        config_id: str = "",
        provider: str = "",
        audience: str = "",
        now: int | None = None,
    ) -> dict[str, Any]:
        config = self._select_sso_config(config_id=config_id, provider=provider)
        if config is None:
            result = {"valid": False, "reason": "unknown_sso_config"}
            self._audit("enterprise.sso.validate", "sso", metadata=result)
            return result
        if not config["enabled"]:
            result = {"valid": False, "reason": "sso_config_disabled", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        try:
            header, claims, signing_input, signature = decode_jwt(token)
        except ValueError as exc:
            result = {"valid": False, "reason": str(exc), "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        if header.get("alg") != "HS256":
            result = {"valid": False, "reason": "unsupported_alg", "alg": header.get("alg"), "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        secret = str(config["metadata"].get("client_secret", ""))
        if not secret:
            result = {"valid": False, "reason": "missing_client_secret", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        expected = hmac.new(secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(signature, expected):
            result = {"valid": False, "reason": "signature_invalid", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        expected_issuer = config["issuer"]
        if claims.get("iss") != expected_issuer:
            result = {"valid": False, "reason": "issuer_mismatch", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        expected_audience = audience or config["client_id"]
        claim_audience = claims.get("aud")
        audiences = claim_audience if isinstance(claim_audience, list) else [claim_audience]
        if expected_audience not in audiences:
            result = {"valid": False, "reason": "audience_mismatch", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        current = int(now if now is not None else time.time())
        try:
            not_before = int(claims["nbf"]) if "nbf" in claims else None
            expires_at = int(claims["exp"]) if "exp" in claims else None
        except (TypeError, ValueError):
            result = {"valid": False, "reason": "invalid_time_claim", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        if not_before is not None and not_before > current:
            result = {"valid": False, "reason": "token_not_yet_valid", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        if expires_at is not None and expires_at <= current:
            result = {"valid": False, "reason": "token_expired", "config": redact_sso_config(config)}
            self._audit("enterprise.sso.validate", config["id"], metadata={"valid": False, "reason": result["reason"]})
            return result
        result = {"valid": True, "claims": claims, "config": redact_sso_config(config)}
        self._audit("enterprise.sso.validate", config["id"], metadata={"valid": True, "subject": claims.get("sub", "")})
        return result

    def _select_sso_config(self, *, config_id: str = "", provider: str = "") -> dict[str, Any] | None:
        if config_id:
            return self.get_sso_config(config_id, redact=False)
        configs = self.sso_configs_raw()
        if provider:
            return next((config for config in configs if config["provider"] == provider), None)
        return configs[0] if configs else None

    def sso_configs_raw(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM sso_configs ORDER BY provider").fetchall()
        return [sso_row(row, redact=False) for row in rows]

    def export_audit(self, *, project: str | None = None) -> dict[str, Any]:
        events = self.store.list_audit_events(project=project, limit=500)
        export_id = stable_id("audit-export", project or "all", utc_now())
        target = self.export_root / f"{export_id}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
        self.store.add_audit_event(
            AuditEvent(
                actor="enterprise",
                action="audit.export",
                resource=str(target),
                project=project or "all",
                metadata={"event_count": len(events)},
            )
        )
        return {"id": export_id, "path": str(target), "event_count": len(events), "created_at": utc_now()}

    def create_backup(self) -> dict[str, Any]:
        backup_id = stable_id("backup", utc_now())
        target_dir = self.backup_root / backup_id
        target_dir.mkdir(parents=True, exist_ok=True)
        db_target = target_dir / self.store.db_path.name
        shutil.copy2(self.store.db_path, db_target)
        manifest = {
            "id": backup_id,
            "database": str(db_target),
            "source_database": str(self.store.db_path),
            "size_bytes": db_target.stat().st_size,
            "created_at": utc_now(),
            "restore_steps": [
                "Stop OpenAgent.",
                "Copy the backup database over the active database.",
                "Restart OpenAgent and run health checks.",
            ],
        }
        manifest_path = target_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO backup_records (id, path, manifest_json, status, created_at) VALUES (?, ?, ?, ?, ?)",
                (backup_id, str(target_dir), json.dumps(manifest, sort_keys=True), "completed", manifest["created_at"]),
            )
        self._audit(
            "enterprise.backup.create",
            backup_id,
            metadata={"path": str(target_dir), "size_bytes": manifest["size_bytes"]},
        )
        return {"id": backup_id, "path": str(target_dir), "manifest": manifest, "status": "completed"}

    def backups(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM backup_records ORDER BY created_at DESC").fetchall()
        return [backup_row(row) for row in rows]

    def create_disaster_recovery_plan(self, *, name: str, rpo_minutes: int, rto_minutes: int) -> dict[str, Any]:
        plan = {
            "rpo_minutes": rpo_minutes,
            "rto_minutes": rto_minutes,
            "backup_root": str(self.backup_root),
            "checks": ["database backup exists", "audit export available", "configuration documented"],
            "restore_order": ["database", "model files", "configuration", "health check"],
        }
        plan_id = stable_id("dr", name)
        with self.store.session() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO disaster_recovery_plans (id, name, plan_json, created_at) VALUES (?, ?, ?, ?)",
                (plan_id, name, json.dumps(plan, sort_keys=True), utc_now()),
            )
        self._audit(
            "enterprise.disaster_recovery.upsert",
            plan_id,
            metadata={"name": name, "rpo_minutes": rpo_minutes, "rto_minutes": rto_minutes},
        )
        return {"id": plan_id, "name": name, "plan": plan}

    def disaster_recovery_plans(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM disaster_recovery_plans ORDER BY created_at DESC").fetchall()
        return [disaster_recovery_row(row) for row in rows]

    def status(self) -> dict[str, Any]:
        return {
            "users": len(self.users()),
            "roles": len(self.roles()),
            "workspaces": len(self.workspaces()),
            "policies": len(self.policies()),
            "sso_configs": len(self.sso_configs()),
            "backups": len(self.backups()),
            "disaster_recovery_plans": len(self.disaster_recovery_plans()),
            "export_root": str(self.export_root),
            "backup_root": str(self.backup_root),
        }

    def _audit(
        self,
        action: str,
        resource: str,
        *,
        project: str = "enterprise",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.store.add_audit_event(
            AuditEvent(
                actor="enterprise",
                action=action,
                resource=resource,
                project=project,
                metadata=metadata or {},
            )
        )


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def user_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "status": row["status"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def role_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "permissions": json.loads(row["permissions_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def workspace_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "owner_user_id": row["owner_user_id"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def org_memory_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workspace_id": row["workspace_id"],
        "title": row["title"],
        "content": row["content"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def policy_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "description": row["description"],
        "rules": json.loads(row["rules_json"]),
        "enforcement": row["enforcement"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def sso_row(row: Any, *, redact: bool = True) -> dict[str, Any]:
    metadata = json.loads(row["metadata_json"])
    if redact:
        metadata = redact_secret_metadata(metadata)
    return {
        "id": row["id"],
        "provider": row["provider"],
        "issuer": row["issuer"],
        "client_id": row["client_id"],
        "enabled": bool(row["enabled"]),
        "metadata": metadata,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def backup_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "path": row["path"],
        "manifest": json.loads(row["manifest_json"]),
        "status": row["status"],
        "created_at": row["created_at"],
    }


def disaster_recovery_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "name": row["name"],
        "plan": json.loads(row["plan_json"]),
        "created_at": row["created_at"],
    }


def redact_secret_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(metadata)
    for key in ["client_secret", "secret", "private_key"]:
        if key in redacted:
            redacted[key] = "***"
    return redacted


def redact_sso_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "metadata": redact_secret_metadata(config.get("metadata", {}))}


def decode_jwt(token: str) -> tuple[dict[str, Any], dict[str, Any], str, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid_jwt_format")
    try:
        header = json.loads(base64url_decode(parts[0]).decode("utf-8"))
        claims = json.loads(base64url_decode(parts[1]).decode("utf-8"))
        signature = base64url_decode(parts[2])
    except (binascii.Error, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("invalid_jwt_payload") from exc
    return header, claims, f"{parts[0]}.{parts[1]}", signature


def base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))
