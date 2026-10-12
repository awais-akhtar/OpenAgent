"""Stage 5 local communication hub."""

from __future__ import annotations

import base64
from collections import Counter
from datetime import datetime, timezone
from email.message import EmailMessage
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import imaplib
import json
from pathlib import Path
import re
import secrets
import smtplib
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from .models import AuditEvent, ContextEvent, utc_now
from .egress import EgressClient
from .storage import OpenAgentStore

FOLLOW_UP_RE = re.compile(r"\b(follow up|remind|deadline|due|please|can you|could you|waiting)\b|\?", re.I)
OAUTH_PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "generic": {
        "auth_endpoint": "",
        "token_endpoint": "",
        "scopes": ["openid", "email"],
        "mailboxes": {"inbox": "INBOX", "archive": "Archive", "sent": "Sent", "trash": "Trash"},
    },
    "google": {
        "auth_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "scopes": ["https://mail.google.com/"],
        "mailboxes": {"inbox": "INBOX", "archive": "[Gmail]/All Mail", "sent": "[Gmail]/Sent Mail", "trash": "[Gmail]/Trash"},
        "extra_auth_params": {"access_type": "offline", "prompt": "consent"},
    },
    "microsoft": {
        "auth_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/authorize",
        "token_endpoint": "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        "scopes": ["offline_access", "https://outlook.office.com/IMAP.AccessAsUser.All", "https://outlook.office.com/SMTP.Send"],
        "mailboxes": {"inbox": "INBOX", "archive": "Archive", "sent": "Sent Items", "trash": "Deleted Items"},
    },
}


class CommunicationApprovalRequired(RuntimeError):
    """Raised when a send action needs approval."""


class CommunicationHub:
    def __init__(self, store: OpenAgentStore):
        self.store = store
        self.egress = EgressClient()
        self.initialize()
        self.seed_connectors()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS communication_connectors (
                    id TEXT PRIMARY KEY,
                    platform TEXT NOT NULL,
                    label TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    capabilities_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inbox_messages (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    platform TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    recipients_json TEXT NOT NULL DEFAULT '[]',
                    subject TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    priority TEXT NOT NULL DEFAULT 'normal',
                    follow_up INTEGER NOT NULL DEFAULT 0,
                    read INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_inbox_project_created
                    ON inbox_messages(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS communication_drafts (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    platform TEXT NOT NULL,
                    message_id TEXT NOT NULL DEFAULT '',
                    recipients_json TEXT NOT NULL DEFAULT '[]',
                    subject TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'draft',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS communication_send_approvals (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    draft_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS sent_messages (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    draft_id TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    recipients_json TEXT NOT NULL DEFAULT '[]',
                    subject TEXT NOT NULL DEFAULT '',
                    body TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS calendar_events (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    calendar_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    starts_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    attendees_json TEXT NOT NULL DEFAULT '[]',
                    source TEXT NOT NULL DEFAULT 'local',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_calendar_project_start
                    ON calendar_events(project, starts_at);

                CREATE TABLE IF NOT EXISTS communication_sync_runs (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    platform TEXT NOT NULL,
                    status TEXT NOT NULL,
                    allow_external INTEGER NOT NULL DEFAULT 0,
                    imported_messages INTEGER NOT NULL DEFAULT 0,
                    imported_contacts INTEGER NOT NULL DEFAULT 0,
                    imported_calendar_events INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_communication_sync_runs_project
                    ON communication_sync_runs(project, started_at DESC);

                CREATE TABLE IF NOT EXISTS communication_dead_letters (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL DEFAULT 'local',
                    platform TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    error TEXT NOT NULL DEFAULT '',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_communication_dead_letters_project
                    ON communication_dead_letters(project, created_at DESC);

                CREATE TABLE IF NOT EXISTS communication_oauth_accounts (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    platform TEXT NOT NULL DEFAULT 'email',
                    provider TEXT NOT NULL DEFAULT 'generic',
                    label TEXT NOT NULL DEFAULT '',
                    client_id TEXT NOT NULL DEFAULT '',
                    client_secret TEXT NOT NULL DEFAULT '',
                    auth_endpoint TEXT NOT NULL DEFAULT '',
                    token_endpoint TEXT NOT NULL DEFAULT '',
                    redirect_uri TEXT NOT NULL DEFAULT '',
                    scopes_json TEXT NOT NULL DEFAULT '[]',
                    access_token TEXT NOT NULL DEFAULT '',
                    refresh_token TEXT NOT NULL DEFAULT '',
                    expires_at TEXT NOT NULL DEFAULT '',
                    token_type TEXT NOT NULL DEFAULT 'Bearer',
                    mailbox_policy_json TEXT NOT NULL DEFAULT '{}',
                    status TEXT NOT NULL DEFAULT 'configured',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS communication_oauth_callbacks (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    account_id TEXT NOT NULL DEFAULT '',
                    provider TEXT NOT NULL DEFAULT '',
                    state TEXT NOT NULL DEFAULT '',
                    code_fingerprint TEXT NOT NULL DEFAULT '',
                    error TEXT NOT NULL DEFAULT '',
                    redirect_uri TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS communication_attachments (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    owner TEXT NOT NULL DEFAULT 'local',
                    message_id TEXT NOT NULL DEFAULT '',
                    filename TEXT NOT NULL,
                    content_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    document_id TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS communication_scheduled_sends (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    owner TEXT NOT NULL,
                    draft_id TEXT NOT NULL,
                    approval_id TEXT NOT NULL DEFAULT '',
                    send_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'scheduled',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_communication_scheduled_due ON communication_scheduled_sends(status, send_at);
                """
            )
            for table in (
                "inbox_messages", "communication_drafts",
                "communication_send_approvals", "sent_messages", "calendar_events",
                "communication_sync_runs", "communication_dead_letters",
                "communication_oauth_accounts", "communication_oauth_callbacks",
                "communication_attachments", "communication_scheduled_sends",
            ):
                self._ensure_column(connection, table, "owner", "TEXT NOT NULL DEFAULT 'local'")
            for column, definition in (
                ("folder", "TEXT NOT NULL DEFAULT 'inbox'"),
                ("answered", "INTEGER NOT NULL DEFAULT 0"),
                ("spam", "INTEGER NOT NULL DEFAULT 0"),
                ("archived", "INTEGER NOT NULL DEFAULT 0"),
            ):
                self._ensure_column(connection, "inbox_messages", column, definition)
            connection.executescript(
                """
                CREATE INDEX IF NOT EXISTS idx_inbox_owner_project_created
                    ON inbox_messages(owner, project, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_communication_oauth_owner_project
                    ON communication_oauth_accounts(owner, project, platform);
                CREATE INDEX IF NOT EXISTS idx_communication_oauth_callbacks_project
                    ON communication_oauth_callbacks(project, owner, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_communication_attachments_project
                    ON communication_attachments(project, owner, created_at DESC);
                """
            )

    def _ensure_column(self, connection: Any, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def seed_connectors(self) -> None:
        connectors = {
            "email": ["inbox", "draft", "send", "search"],
            "caldav": ["calendar", "availability"],
            "slack": ["inbox", "draft", "send"],
            "discord": ["inbox", "draft", "send"],
            "telegram": ["inbox", "draft", "send"],
            "signal": ["inbox", "draft", "send"],
            "teams": ["inbox", "draft", "send"],
            "sms": ["inbox", "draft", "send"],
            "whatsapp": ["inbox", "draft", "send"],
        }
        now = utc_now()
        with self.store.session() as connection:
            for platform, capabilities in connectors.items():
                connection.execute(
                    """
                    INSERT OR IGNORE INTO communication_connectors (
                        id, platform, label, enabled, config_json, capabilities_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        platform,
                        platform,
                        platform.replace("-", " ").title(),
                        0,
                        "{}",
                        json.dumps(capabilities),
                        now,
                        now,
                    ),
                )

    def connectors(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM communication_connectors ORDER BY platform").fetchall()
        return [connector_row(row) for row in rows]

    # ------------------------------------------------------------------
    # OAuth provider onboarding and token lifecycle
    # ------------------------------------------------------------------

    def oauth_providers(self) -> list[dict[str, Any]]:
        return [
            {
                "id": provider,
                "auth_endpoint": config["auth_endpoint"],
                "token_endpoint": config["token_endpoint"],
                "scopes": list(config["scopes"]),
                "mailbox_policy": config["mailboxes"],
                "privacy_boundary": "OAuth metadata is local configuration. Token exchange only runs when allow_external is true.",
            }
            for provider, config in sorted(OAUTH_PROVIDER_DEFAULTS.items())
        ]

    def provider_certification_matrix(self) -> dict[str, Any]:
        providers = []
        for provider, config in sorted(OAUTH_PROVIDER_DEFAULTS.items()):
            providers.append(
                {
                    "id": provider,
                    "protocols": ["oauth", "imap", "smtp"] if provider != "generic" else ["oauth", "imap", "smtp", "custom"],
                    "auth_endpoint": config["auth_endpoint"],
                    "token_endpoint": config["token_endpoint"],
                    "scopes": list(config["scopes"]),
                    "mailbox_policy": dict(config["mailboxes"]),
                    "callback_supported": True,
                    "token_refresh_supported": True,
                    "live_cert_env": {
                        "google": "OPENAGENT_CERT_GMAIL",
                        "microsoft": "OPENAGENT_CERT_OUTLOOK",
                        "generic": "OPENAGENT_CERT_IMAP",
                    }.get(provider, "OPENAGENT_CERT_PROVIDER"),
                    "privacy_boundary": "Provider setup stays local. Live network certification is opt-in through environment flags and user credentials.",
                }
            )
        providers.extend(
            [
                {
                    "id": "fastmail-caldav",
                    "protocols": ["caldav"],
                    "live_cert_env": "OPENAGENT_CERT_CALDAV",
                    "privacy_boundary": "CalDAV certification is opt-in and uses local user-supplied credentials.",
                },
                {
                    "id": "nextcloud-caldav",
                    "protocols": ["caldav"],
                    "live_cert_env": "OPENAGENT_CERT_CALDAV",
                    "privacy_boundary": "CalDAV certification is opt-in and uses local user-supplied credentials.",
                },
                {
                    "id": "radicale-caldav",
                    "protocols": ["caldav"],
                    "live_cert_env": "OPENAGENT_CERT_CALDAV",
                    "privacy_boundary": "CalDAV certification is opt-in and uses local user-supplied credentials.",
                },
            ]
        )
        return {
            "status": "local-certification-ready",
            "providers": providers,
            "gated_live_tests": ["OPENAGENT_CERT_GMAIL", "OPENAGENT_CERT_OUTLOOK", "OPENAGENT_CERT_CALDAV"],
            "privacy_boundary": "Certification metadata is local. No provider is contacted by this endpoint.",
        }

    def record_oauth_callback(
        self,
        *,
        state: str,
        code: str = "",
        error: str = "",
        account_id: str = "",
        redirect_uri: str = "",
        owner: str = "local",
        project: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        account: dict[str, Any] | None = None
        if account_id:
            try:
                account = self.get_oauth_account(account_id, owner=owner if owner else "")
                project = str(account.get("project", project))
                owner = str(account.get("owner", owner or "local"))
            except Exception:
                account = None
        callback_id = stable_id("oauth-callback", project, owner, account_id, state, utc_now())
        fingerprint = hashlib.sha256(code.encode("utf-8")).hexdigest() if code else ""
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_oauth_callbacks (
                    id, project, owner, account_id, provider, state, code_fingerprint,
                    error, redirect_uri, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    callback_id,
                    project or "default",
                    owner or "local",
                    account_id,
                    str((account or {}).get("provider", "")),
                    state,
                    fingerprint,
                    error,
                    redirect_uri,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="communication",
                action="oauth.callback",
                resource=f"oauth_callback:{callback_id}",
                project=project or "default",
                metadata={"account_id": account_id, "state": state, "has_code": bool(code), "has_error": bool(error)},
            )
        )
        return {
            "id": callback_id,
            "project": project or "default",
            "owner": owner or "local",
            "account_id": account_id,
            "provider": str((account or {}).get("provider", "")),
            "state": state,
            "code_received": bool(code),
            "code_fingerprint": fingerprint,
            "error": error,
            "redirect_uri": redirect_uri,
            "created_at": created_at,
            "privacy_boundary": "Authorization code text is never returned. Only a local fingerprint is stored.",
        }

    def oauth_callbacks(self, *, project: str = "default", owner: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        args.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_oauth_callbacks WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(args),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "project": row["project"],
                "owner": row["owner"],
                "account_id": row["account_id"],
                "provider": row["provider"],
                "state": row["state"],
                "code_received": bool(row["code_fingerprint"]),
                "code_fingerprint": row["code_fingerprint"],
                "error": row["error"],
                "redirect_uri": row["redirect_uri"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def configure_oauth_account(
        self,
        *,
        provider: str,
        platform: str = "email",
        project: str = "default",
        owner: str = "local",
        label: str = "",
        client_id: str = "",
        client_secret: str = "",
        auth_endpoint: str = "",
        token_endpoint: str = "",
        redirect_uri: str = "",
        scopes: list[str] | None = None,
        mailbox_policy: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        provider = (provider or "generic").strip().lower()
        defaults = OAUTH_PROVIDER_DEFAULTS.get(provider, OAUTH_PROVIDER_DEFAULTS["generic"])
        auth_endpoint = auth_endpoint.strip() or str(defaults["auth_endpoint"])
        token_endpoint = token_endpoint.strip() or str(defaults["token_endpoint"])
        scopes = scopes if scopes is not None else [str(item) for item in defaults["scopes"]]
        mailbox_policy = mailbox_policy if mailbox_policy is not None else dict(defaults["mailboxes"])
        owner = owner or "local"
        project = project or "default"
        platform = (platform or "email").strip().lower()
        account_id = stable_id("oauth-account", project, owner, platform, provider, label or client_id or auth_endpoint)
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO communication_oauth_accounts (
                    id, project, owner, platform, provider, label, client_id, client_secret,
                    auth_endpoint, token_endpoint, redirect_uri, scopes_json, access_token,
                    refresh_token, expires_at, token_type, mailbox_policy_json, status,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE((SELECT access_token FROM communication_oauth_accounts WHERE id = ?), ''),
                          COALESCE((SELECT refresh_token FROM communication_oauth_accounts WHERE id = ?), ''),
                          COALESCE((SELECT expires_at FROM communication_oauth_accounts WHERE id = ?), ''),
                          COALESCE((SELECT token_type FROM communication_oauth_accounts WHERE id = ?), 'Bearer'),
                          ?, 'configured', COALESCE((SELECT created_at FROM communication_oauth_accounts WHERE id = ?), ?), ?)
                """,
                (
                    account_id,
                    project,
                    owner,
                    platform,
                    provider,
                    label or provider.title(),
                    client_id,
                    client_secret,
                    auth_endpoint,
                    token_endpoint,
                    redirect_uri,
                    json.dumps(scopes),
                    account_id,
                    account_id,
                    account_id,
                    account_id,
                    json.dumps(mailbox_policy or {}, sort_keys=True),
                    account_id,
                    now,
                    now,
                ),
            )
        return self.get_oauth_account(account_id, owner=owner)

    def get_oauth_account(self, account_id: str, *, owner: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._oauth_account_row(connection, account_id, owner=owner)
        return oauth_account_row(row)

    def oauth_accounts(self, *, project: str = "default", owner: str = "", platform: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        if platform:
            clauses.append("platform = ?")
            args.append(platform.strip().lower())
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM communication_oauth_accounts WHERE "
                + " AND ".join(clauses)
                + " ORDER BY updated_at DESC, rowid DESC",
                tuple(args),
            ).fetchall()
        return [oauth_account_row(row) for row in rows]

    def oauth_authorization_url(
        self,
        account_id: str,
        *,
        owner: str = "",
        state: str = "",
        code_challenge: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._oauth_account_row(connection, account_id, owner=owner)
        if not row["auth_endpoint"]:
            raise ValueError("OAuth account has no authorization endpoint.")
        state = state or secrets.token_urlsafe(24)
        params: dict[str, str] = {
            "response_type": "code",
            "client_id": row["client_id"],
            "redirect_uri": row["redirect_uri"],
            "scope": " ".join(json.loads(row["scopes_json"] or "[]")),
            "state": state,
        }
        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = "S256"
        provider_defaults = OAUTH_PROVIDER_DEFAULTS.get(row["provider"], {})
        for key, value in dict(provider_defaults.get("extra_auth_params", {})).items():
            params.setdefault(str(key), str(value))
        return {
            "account_id": row["id"],
            "provider": row["provider"],
            "authorization_url": row["auth_endpoint"] + "?" + urlencode(params),
            "state": state,
            "pkce_required": bool(code_challenge),
            "privacy_boundary": "Opening this URL is a user-approved external OAuth step. OpenAgent only stores returned tokens locally.",
        }

    def refresh_oauth_token(
        self,
        account_id: str,
        *,
        owner: str = "",
        allow_external: bool = False,
        authorization_code: str = "",
        code_verifier: str = "",
    ) -> dict[str, Any]:
        with self.store.session() as connection:
            row = self._oauth_account_row(connection, account_id, owner=owner)
        if not allow_external:
            return {
                "account": oauth_account_row(row),
                "status": "approval_required",
                "requires_external_call": True,
                "reason": "Token exchange/refresh contacts the configured OAuth provider and requires allow_external=true.",
            }
        if not row["token_endpoint"]:
            raise ValueError("OAuth account has no token endpoint.")
        form: dict[str, str] = {
            "client_id": row["client_id"],
        }
        if row["client_secret"]:
            form["client_secret"] = row["client_secret"]
        if authorization_code:
            form.update({
                "grant_type": "authorization_code",
                "code": authorization_code,
                "redirect_uri": row["redirect_uri"],
            })
            if code_verifier:
                form["code_verifier"] = code_verifier
        elif row["refresh_token"]:
            form.update({"grant_type": "refresh_token", "refresh_token": row["refresh_token"]})
        else:
            raise ValueError("OAuth account has no refresh_token; exchange an authorization_code first.")
        response = self.egress.fetch(
            row["token_endpoint"], approved=allow_external, purpose="OAuth token exchange or refresh",
            body=urlencode(form).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "OpenAgent-Communication-OAuth"},
            method="POST",
            timeout_seconds=30, max_response_bytes=2 * 1024 * 1024,
            allowed_content_types={"application/json"},
        )
        if not 200 <= response.status < 300:
            raise ValueError(f"OAuth token refresh returned HTTP {response.status}.")
        try:
            data = json.loads(response.body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - converted to operator-visible failure
            raise ValueError(f"OAuth token refresh failed: {exc}") from exc
        access_token = str(data.get("access_token", ""))
        refresh_token = str(data.get("refresh_token") or row["refresh_token"])
        if not access_token:
            raise ValueError("OAuth token response did not contain access_token.")
        expires_at = ""
        try:
            expires_in = int(data.get("expires_in", 0) or 0)
        except (TypeError, ValueError):
            expires_in = 0
        if expires_in > 0:
            expires_at = datetime.fromtimestamp(datetime.now(tz=timezone.utc).timestamp() + expires_in, tz=timezone.utc).isoformat()
        token_type = str(data.get("token_type") or row["token_type"] or "Bearer")
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE communication_oauth_accounts
                SET access_token = ?, refresh_token = ?, expires_at = ?, token_type = ?, status = 'token_ready', updated_at = ?
                WHERE id = ?
                """,
                (access_token, refresh_token, expires_at, token_type, now, account_id),
            )
            updated = self._oauth_account_row(connection, account_id, owner=owner)
        self.store.add_audit_event(
            AuditEvent(
                actor="communication",
                action="oauth.token_refresh",
                resource=f"oauth:{account_id}",
                project=row["project"],
                status="completed",
                metadata={"provider": row["provider"], "platform": row["platform"], "owner": row["owner"]},
            )
        )
        return {"account": oauth_account_row(updated), "status": "token_ready"}

    def mailbox_policy(self, *, provider: str = "generic", account_id: str = "", owner: str = "") -> dict[str, Any]:
        if account_id:
            with self.store.session() as connection:
                row = self._oauth_account_row(connection, account_id, owner=owner)
            policy = json.loads(row["mailbox_policy_json"] or "{}")
            provider = row["provider"]
        else:
            defaults = OAUTH_PROVIDER_DEFAULTS.get((provider or "generic").strip().lower(), OAUTH_PROVIDER_DEFAULTS["generic"])
            policy = dict(defaults["mailboxes"])
        return {
            "provider": provider,
            "mailboxes": policy,
            "search_defaults": {"inbox": "UNSEEN", "archive": "ALL"},
            "edge_cases": [
                "Provider folder names are normalized before IMAP SELECT.",
                "OAuth XOAUTH2 is preferred when oauth_account_id is configured.",
                "Message-ID remains the dedupe thread anchor when present.",
            ],
        }

    def _oauth_account_row(self, connection: Any, account_id: str, *, owner: str = "") -> Any:
        if owner:
            row = connection.execute(
                "SELECT * FROM communication_oauth_accounts WHERE id = ? AND owner = ?",
                (account_id, owner),
            ).fetchone()
        else:
            row = connection.execute("SELECT * FROM communication_oauth_accounts WHERE id = ?", (account_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown OAuth account '{account_id}'.")
        return row

    def _fresh_oauth_access_token(self, account_id: str, *, owner: str = "") -> str:
        with self.store.session() as connection:
            row = self._oauth_account_row(connection, account_id, owner=owner)
        token = str(row["access_token"] or "")
        if token and not self._token_is_expired(str(row["expires_at"] or "")):
            return token
        refreshed = self.refresh_oauth_token(account_id, owner=owner, allow_external=True)
        account = refreshed.get("account", {}) if isinstance(refreshed, dict) else {}
        if not account.get("token_present"):
            raise ValueError("OAuth token refresh did not produce an access token.")
        with self.store.session() as connection:
            updated = self._oauth_account_row(connection, account_id, owner=owner)
        return str(updated["access_token"])

    def _token_is_expired(self, expires_at: str) -> bool:
        if not expires_at:
            return False
        try:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return True
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=timezone.utc)
        return expiry <= datetime.now(tz=timezone.utc)

    def _xoauth2_payload(self, username: str, token: str) -> bytes:
        return f"user={username}\x01auth=Bearer {token}\x01\x01".encode("utf-8")

    def configure_connector(
        self,
        platform: str,
        *,
        enabled: bool,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                UPDATE communication_connectors
                SET enabled = ?, config_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (1 if enabled else 0, json.dumps(config or {}, sort_keys=True), now, platform),
            )
        return next(item for item in self.connectors() if item["id"] == platform)

    def connector_config(self, platform: str) -> dict[str, Any]:
        platform = platform.strip().lower()
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT config_json FROM communication_connectors WHERE id = ?",
                (platform,),
            ).fetchone()
        if row is None:
            raise ValueError(f"Unknown connector '{platform}'.")
        return json.loads(row["config_json"] or "{}")

    def upsert_contact(
        self,
        name: str,
        *,
        project: str = "default",
        owner: str = "local",
        handles: dict[str, str] | None = None,
        notes: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        contact_id = stable_id("contact", project, owner, name, json.dumps(handles or {}, sort_keys=True))
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO contacts (
                    id, project, name, handles_json, notes, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, COALESCE((SELECT created_at FROM contacts WHERE id = ?), ?), ?)
                """,
                (
                    contact_id,
                    project,
                    name.strip() or "Unknown",
                    json.dumps(handles or {}, sort_keys=True),
                    notes,
                    json.dumps(metadata or {}, sort_keys=True),
                    contact_id,
                    now,
                    now,
                ),
            )
            connection.execute("UPDATE contacts SET owner = ? WHERE id = ?", (owner or "local", contact_id))
        return self.get_contact(contact_id) or {}

    def get_contact(self, contact_id: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        return contact_row(row) if row else None

    def contacts(self, *, project: str = "default", query: str = "", owner: str = "") -> list[dict[str, Any]]:
        pattern = f"%{query.lower()}%"
        clauses = ["project = ?", "(? = '%%' OR lower(name) LIKE ? OR lower(handles_json) LIKE ?)"]
        params: list[Any] = [project, pattern, pattern, pattern]
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM contacts WHERE {' AND '.join(clauses)} ORDER BY name",
                tuple(params),
            ).fetchall()
        return [contact_row(row) for row in rows]

    def ingest_attachment(
        self,
        *,
        filename: str,
        content: bytes,
        project: str = "default",
        owner: str = "local",
        message_id: str = "",
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
        create_document: bool = True,
    ) -> dict[str, Any]:
        filename = Path(filename or "attachment.bin").name
        project = project or "default"
        owner = owner or "local"
        digest = hashlib.sha256(content).hexdigest()
        attachment_id = stable_id("attachment", project, owner, message_id, filename, digest)
        media_dir = self.store.data_dir / "communication_attachments" / project / owner
        media_dir.mkdir(parents=True, exist_ok=True)
        local_path = media_dir / f"{attachment_id}-{filename}"
        local_path.write_bytes(content)
        document_id = ""
        extracted_text = _attachment_text(filename, content)
        if create_document and extracted_text.strip():
            try:
                from .knowledge import KnowledgeEngine

                document = KnowledgeEngine(self.store).ingest_text(
                    title=filename,
                    content=extracted_text,
                    project=project,
                    source="communication-attachment",
                    source_path=str(local_path),
                    metadata={"message_id": message_id, "attachment_id": attachment_id},
                )
                document_id = str(document.get("id", ""))
            except Exception:
                document_id = ""
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO communication_attachments (
                    id, project, owner, message_id, filename, content_type, size_bytes,
                    sha256, local_path, document_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attachment_id,
                    project,
                    owner,
                    message_id,
                    filename,
                    content_type,
                    len(content),
                    digest,
                    str(local_path),
                    document_id,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="communication",
                action="attachment.ingest",
                resource=f"attachment:{attachment_id}",
                project=project,
                metadata={"message_id": message_id, "filename": filename, "document_id": document_id},
            )
        )
        return {
            "id": attachment_id,
            "project": project,
            "owner": owner,
            "message_id": message_id,
            "filename": filename,
            "content_type": content_type,
            "size_bytes": len(content),
            "sha256": digest,
            "local_path": str(local_path),
            "document_id": document_id,
            "created_at": created_at,
            "privacy_boundary": "Attachment bytes are stored under the local OpenAgent data directory.",
        }

    def attachments(self, *, project: str = "default", owner: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        args.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_attachments WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(args),
            ).fetchall()
        return [attachment_row(row) for row in rows]

    def import_message(
        self,
        *,
        platform: str,
        sender: str,
        body: str,
        project: str = "default",
        owner: str = "local",
        recipients: list[str] | None = None,
        subject: str = "",
        thread_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = body.strip()
        if not body:
            raise ValueError("Message body is required.")
        owner = owner or "local"
        thread_id = thread_id or stable_id("thread", project, owner, platform, sender, subject)
        message_id = stable_id("message", project, owner, platform, thread_id, body, utc_now())
        summary = summarize(body)
        follow_up = bool(FOLLOW_UP_RE.search(body))
        priority = "high" if any(word in body.lower() for word in ["urgent", "asap", "deadline"]) else "normal"
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO inbox_messages (
                    id, project, platform, thread_id, sender, recipients_json,
                    subject, body, summary, priority, follow_up, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    project,
                    platform,
                    thread_id,
                    sender,
                    json.dumps(recipients or []),
                    subject,
                    body,
                    summary,
                    priority,
                    1 if follow_up else 0,
                    json.dumps(metadata or {}, sort_keys=True),
                    created_at,
                ),
            )
            connection.execute("UPDATE inbox_messages SET owner = ? WHERE id = ?", (owner, message_id))
        self.upsert_contact(sender, project=project, owner=owner, handles={platform: sender})
        self.store.add_event(
            ContextEvent(
                title=f"{platform} message: {subject or sender}",
                content=body,
                type="communication",
                source=f"communication:{platform}",
                project=project,
                metadata={"message_id": message_id, "follow_up": follow_up, "owner": owner},
            )
        )
        return self.get_message(message_id) or {}

    def get_message(self, message_id: str, *, owner: str = "") -> dict[str, Any] | None:
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT * FROM inbox_messages WHERE id = ? AND owner = ?",
                    (message_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM inbox_messages WHERE id = ?", (message_id,)).fetchone()
        return message_row(row) if row else None

    def inbox(
        self,
        *,
        project: str = "default",
        query: str = "",
        follow_up: bool | None = None,
        owner: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        pattern = f"%{query.lower()}%"
        clauses = ["project = ?", "(? = '%%' OR lower(subject) LIKE ? OR lower(body) LIKE ? OR lower(sender) LIKE ?)"]
        params: list[Any] = [project, pattern, pattern, pattern, pattern]
        if follow_up is not None:
            clauses.append("follow_up = ?")
            params.append(1 if follow_up else 0)
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM inbox_messages WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [message_row(row) for row in rows]

    def mutate_message(
        self,
        message_id: str,
        *,
        owner: str = "",
        folder: str | None = None,
        read: bool | None = None,
        answered: bool | None = None,
        spam: bool | None = None,
        archived: bool | None = None,
    ) -> dict[str, Any]:
        current = self.get_message(message_id, owner=owner)
        if current is None:
            raise ValueError(f"Unknown message '{message_id}'.")
        updates: dict[str, Any] = {}
        if folder is not None:
            clean_folder = folder.strip().lower()
            if clean_folder not in {"inbox", "archive", "sent", "trash", "spam", "custom"}:
                raise ValueError("Unsupported mailbox folder.")
            updates["folder"] = clean_folder
        for key, value in (("read", read), ("answered", answered), ("spam", spam), ("archived", archived)):
            if value is not None:
                updates[key] = 1 if value else 0
        if spam:
            updates["folder"] = "spam"
        elif archived:
            updates["folder"] = "archive"
        if updates:
            with self.store.session() as connection:
                connection.execute(
                    f"UPDATE inbox_messages SET {', '.join(f'{key} = ?' for key in updates)} WHERE id = ?",
                    [*updates.values(), message_id],
                )
        result = self.get_message(message_id, owner=owner) or {}
        self.store.add_audit_event(
            AuditEvent(actor="communication", action="message.mutate", resource=f"message:{message_id}", project=current["project"], metadata={"owner": current["owner"], "updates": updates})
        )
        return result

    def pending_approvals(self, *, project: str = "default", owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?", "status = 'pending'"]
        values: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            values.append(owner)
        values.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_send_approvals WHERE {' AND '.join(clauses)} ORDER BY created_at DESC LIMIT ?", values
            ).fetchall()
        return [dict(row) for row in rows]

    def schedule_draft(
        self,
        draft_id: str,
        *,
        send_at: str,
        approval_id: str = "",
        project: str = "default",
        owner: str = "local",
    ) -> dict[str, Any]:
        if self.get_draft(draft_id, owner=owner) is None:
            raise ValueError(f"Unknown draft '{draft_id}'.")
        parsed = datetime.fromisoformat(send_at.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        schedule_id = uuid4().hex
        now = utc_now()
        normalized = parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_scheduled_sends (id, project, owner, draft_id, approval_id, send_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (schedule_id, project, owner, draft_id, approval_id, normalized, now, now),
            )
        return self.get_scheduled_send(schedule_id)

    def get_scheduled_send(self, schedule_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM communication_scheduled_sends WHERE id = ?", (schedule_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unknown scheduled send '{schedule_id}'.")
        result = dict(row)
        result["result"] = json.loads(result.pop("result_json") or "{}")
        return result

    def scheduled_sends(self, *, project: str = "default", owner: str = "", limit: int = 100) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        values: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            values.append(owner)
        values.append(max(1, min(int(limit), 500)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT id FROM communication_scheduled_sends WHERE {' AND '.join(clauses)} ORDER BY send_at LIMIT ?", values
            ).fetchall()
        return [self.get_scheduled_send(row["id"]) for row in rows]

    def process_scheduled_sends(self, *, project: str = "default", owner: str = "", allow_external: bool = False) -> dict[str, Any]:
        now = utc_now()
        clauses = ["project = ?", "status = 'scheduled'", "send_at <= ?"]
        values: list[Any] = [project, now]
        if owner:
            clauses.append("owner = ?")
            values.append(owner)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_scheduled_sends WHERE {' AND '.join(clauses)} ORDER BY send_at", values
            ).fetchall()
        completed = []
        for row in rows:
            try:
                result = self.send_draft(row["draft_id"], approval_id=row["approval_id"], owner=row["owner"], allow_external=allow_external)
                status = "sent"
            except Exception as exc:
                result = {"error": str(exc)}
                status = "blocked"
            with self.store.session() as connection:
                connection.execute(
                    "UPDATE communication_scheduled_sends SET status = ?, result_json = ?, updated_at = ? WHERE id = ?",
                    (status, json.dumps(result, sort_keys=True), utc_now(), row["id"]),
                )
            completed.append(self.get_scheduled_send(row["id"]))
        return {"processed": completed, "count": len(completed)}

    def draft_reply(
        self,
        message_id: str,
        *,
        project: str = "default",
        owner: str = "local",
        instructions: str = "",
    ) -> dict[str, Any]:
        message = self.get_message(message_id, owner=owner if owner else "")
        if message is None:
            raise ValueError(f"Unknown message '{message_id}'.")
        body = (
            f"Hi {message['sender']},\n\n"
            f"Thanks for your message about {message['subject'] or 'this'}.\n"
            f"{instructions.strip() or 'I will follow up soon.'}\n\n"
            "Best,"
        )
        return self.create_draft(
            platform=message["platform"],
            recipients=[message["sender"]],
            subject=f"Re: {message['subject']}".strip(),
            body=body,
            project=project,
            message_id=message_id,
            owner=owner or message.get("owner", "local"),
        )

    def create_draft(
        self,
        *,
        platform: str,
        recipients: list[str],
        subject: str,
        body: str,
        project: str = "default",
        owner: str = "local",
        message_id: str = "",
    ) -> dict[str, Any]:
        owner = owner or "local"
        draft_id = stable_id("draft", project, owner, platform, subject, body, utc_now())
        now = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_drafts (
                    id, project, platform, message_id, recipients_json, subject, body, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (draft_id, project, platform, message_id, json.dumps(recipients), subject, body, "draft", now, now),
            )
            connection.execute("UPDATE communication_drafts SET owner = ? WHERE id = ?", (owner, draft_id))
        return self.get_draft(draft_id) or {}

    def get_draft(self, draft_id: str, *, owner: str = "") -> dict[str, Any] | None:
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT * FROM communication_drafts WHERE id = ? AND owner = ?",
                    (draft_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM communication_drafts WHERE id = ?", (draft_id,)).fetchone()
        return draft_row(row) if row else None

    def request_send_approval(
        self,
        draft_id: str,
        *,
        project: str = "default",
        owner: str = "local",
        reason: str = "",
    ) -> dict[str, Any]:
        if self.get_draft(draft_id, owner=owner if owner else "") is None:
            raise ValueError(f"Unknown draft '{draft_id}'.")
        approval_id = stable_id("communication-approval", project, owner, draft_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_send_approvals (
                    id, project, draft_id, reason, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (approval_id, project, draft_id, reason or "Approve communication send.", "pending", created_at),
            )
            connection.execute("UPDATE communication_send_approvals SET owner = ? WHERE id = ?", (owner or "local", approval_id))
        return {"id": approval_id, "project": project, "owner": owner or "local", "draft_id": draft_id, "status": "pending", "created_at": created_at}

    def resolve_send_approval(self, approval_id: str, *, approved: bool, owner: str = "") -> dict[str, Any]:
        status = "approved" if approved else "rejected"
        resolved_at = utc_now()
        with self.store.session() as connection:
            if owner:
                connection.execute(
                    "UPDATE communication_send_approvals SET status = ?, resolved_at = ? WHERE id = ? AND owner = ?",
                    (status, resolved_at, approval_id, owner),
                )
                row = connection.execute(
                    "SELECT * FROM communication_send_approvals WHERE id = ? AND owner = ?",
                    (approval_id, owner),
                ).fetchone()
            else:
                connection.execute(
                    "UPDATE communication_send_approvals SET status = ?, resolved_at = ? WHERE id = ?",
                    (status, resolved_at, approval_id),
                )
                row = connection.execute(
                    "SELECT * FROM communication_send_approvals WHERE id = ?",
                    (approval_id,),
                ).fetchone()
        if row is None:
            raise ValueError(f"Unknown approval '{approval_id}'.")
        return approval_row(row)

    def send_draft(
        self,
        draft_id: str,
        *,
        approval_id: str | None = None,
        allow_external: bool = False,
        owner: str = "",
    ) -> dict[str, Any]:
        draft = self.get_draft(draft_id, owner=owner)
        if draft is None:
            raise ValueError(f"Unknown draft '{draft_id}'.")
        if not self.approval_is_approved(approval_id, owner=draft["owner"]):
            approval = self.request_send_approval(draft_id, project=draft["project"], owner=draft["owner"])
            raise CommunicationApprovalRequired(json.dumps(approval, sort_keys=True))
        status = "queued_external" if allow_external else "simulated_local"
        provider_metadata: dict[str, Any] = {"allow_external": allow_external}
        if allow_external and draft["platform"] == "email":
            provider_metadata = self._send_email_via_smtp(draft)
            status = "sent_external"
        sent_id = stable_id("sent", draft_id, utc_now())
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO sent_messages (
                    id, project, draft_id, platform, recipients_json, subject, body, status, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sent_id,
                    draft["project"],
                    draft_id,
                    draft["platform"],
                    json.dumps(draft["recipients"]),
                    draft["subject"],
                    draft["body"],
                    status,
                    json.dumps(provider_metadata, sort_keys=True),
                    created_at,
                ),
            )
            connection.execute("UPDATE sent_messages SET owner = ? WHERE id = ?", (draft["owner"], sent_id))
            connection.execute(
                "UPDATE communication_drafts SET status = ?, updated_at = ? WHERE id = ?",
                ("sent", created_at, draft_id),
            )
        self.store.add_audit_event(
            AuditEvent(
                actor="communication",
                action="message.send",
                resource=f"draft:{draft_id}",
                project=draft["project"],
                status=status,
            metadata={"platform": draft["platform"], "approval_id": approval_id},
            )
        )
        return {
            "id": sent_id,
            "project": draft["project"],
            "draft_id": draft_id,
            "platform": draft["platform"],
            "status": status,
            "created_at": created_at,
        }

    def _send_email_via_smtp(self, draft: dict[str, Any]) -> dict[str, Any]:
        config = self.connector_config("email")
        host = str(config.get("smtp_host", "")).strip()
        if not host:
            raise ValueError("email connector requires smtp_host for external send")
        port = int(config.get("smtp_port") or (465 if config.get("smtp_ssl", True) else 587))
        username = str(config.get("smtp_username") or config.get("username") or "").strip()
        password = str(config.get("smtp_password") or config.get("password") or "")
        sender = str(config.get("smtp_from") or config.get("from") or username).strip()
        if not sender:
            raise ValueError("email connector requires smtp_from or username for external send")
        oauth_account_id = str(config.get("smtp_oauth_account_id") or config.get("oauth_account_id") or "").strip()
        message = EmailMessage()
        message["From"] = sender
        message["To"] = ", ".join(draft["recipients"])
        message["Subject"] = draft["subject"]
        message.set_content(draft["body"])
        timeout = int(config.get("timeout_seconds") or 20)
        use_ssl = bool(config.get("smtp_ssl", True))
        try:
            client_cls = smtplib.SMTP_SSL if use_ssl else smtplib.SMTP
            with client_cls(host, port, timeout=timeout) as smtp:
                if not use_ssl and config.get("smtp_starttls", True):
                    smtp.starttls()
                if oauth_account_id:
                    token = self._fresh_oauth_access_token(oauth_account_id, owner=draft.get("owner", ""))
                    auth_blob = base64.b64encode(self._xoauth2_payload(username or sender, token)).decode("ascii")
                    code, response = smtp.docmd("AUTH", "XOAUTH2 " + auth_blob)
                    if int(code) >= 400:
                        raise ValueError(f"SMTP XOAUTH2 rejected: {response!r}")
                elif username or password:
                    smtp.login(username, password)
                smtp.send_message(message)
        except Exception as exc:  # noqa: BLE001 - converted into provider send failure
            raise ValueError(f"SMTP send failed: {exc}") from exc
        return {
            "allow_external": True,
            "provider": "smtp",
            "host": host,
            "port": port,
            "recipients": len(draft["recipients"]),
        }

    def approval_is_approved(self, approval_id: str | None, *, owner: str = "") -> bool:
        if not approval_id:
            return False
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT status FROM communication_send_approvals WHERE id = ? AND owner = ?",
                    (approval_id, owner),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT status FROM communication_send_approvals WHERE id = ?",
                    (approval_id,),
                ).fetchone()
        return bool(row and row["status"] == "approved")

    def add_calendar_event(
        self,
        *,
        title: str,
        starts_at: str,
        ends_at: str,
        project: str = "default",
        owner: str = "local",
        calendar_id: str = "local",
        attendees: list[str] | None = None,
        source: str = "local",
    ) -> dict[str, Any]:
        owner = owner or "local"
        event_id = stable_id("calendar", project, owner, calendar_id, title, starts_at, ends_at)
        created_at = utc_now()
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO calendar_events (
                    id, project, calendar_id, title, starts_at, ends_at, attendees_json, source, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    project,
                    calendar_id,
                    title,
                    starts_at,
                    ends_at,
                    json.dumps(attendees or []),
                    source,
                    "{}",
                    created_at,
                ),
            )
            connection.execute("UPDATE calendar_events SET owner = ? WHERE id = ?", (owner, event_id))
        return {
            "id": event_id,
            "project": project,
            "calendar_id": calendar_id,
            "title": title,
            "starts_at": starts_at,
            "ends_at": ends_at,
            "attendees": attendees or [],
            "source": source,
            "created_at": created_at,
        }

    def calendar(self, *, project: str = "default", owner: str = "") -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM calendar_events WHERE {' AND '.join(clauses)} ORDER BY starts_at",
                tuple(args),
            ).fetchall()
        return [calendar_row(row) for row in rows]

    def analytics(self, *, project: str = "default", owner: str = "") -> dict[str, Any]:
        messages = self.inbox(project=project, owner=owner, limit=200)
        by_platform = Counter(message["platform"] for message in messages)
        return {
            "project": project,
            "messages": len(messages),
            "unread": sum(1 for message in messages if not message["read"]),
            "follow_ups": sum(1 for message in messages if message["follow_up"]),
            "by_platform": dict(by_platform),
        }

    def sync_connector(
        self,
        platform: str,
        *,
        project: str = "default",
        owner: str = "local",
        payload: dict[str, Any] | None = None,
        allow_external: bool = False,
    ) -> dict[str, Any]:
        """Run a local provider-payload sync worker for a configured connector."""

        platform = platform.strip().lower()
        connector = next((item for item in self.connectors() if item["id"] == platform), None)
        if connector is None:
            raise ValueError(f"Unknown connector '{platform}'.")
        project = project.strip() or "default"
        owner = owner or "local"
        payload = payload or {}
        started_at = utc_now()
        run_id = stable_id("communication-sync", project, platform, started_at, uuid4().hex)
        imported_messages = 0
        imported_contacts = 0
        imported_calendar_events = 0
        status = "completed"
        error = ""
        try:
            if platform == "email" and allow_external and not payload:
                email_counts = self._poll_email_provider(project=project, owner=owner, run_id=run_id)
                imported_messages += email_counts["imported_messages"]
                imported_contacts += email_counts["imported_contacts"]
            for contact in payload.get("contacts", []) if isinstance(payload.get("contacts"), list) else []:
                if isinstance(contact, dict):
                    self.upsert_contact(
                        str(contact.get("name", "")).strip() or str(contact.get("handle", "")).strip() or "Unknown",
                        project=project,
                        owner=owner,
                        handles=contact.get("handles") if isinstance(contact.get("handles"), dict) else {},
                        notes=str(contact.get("notes", "")),
                        metadata={"sync_run_id": run_id, **(contact.get("metadata") if isinstance(contact.get("metadata"), dict) else {})},
                    )
                    imported_contacts += 1
            for message in payload.get("messages", []) if isinstance(payload.get("messages"), list) else []:
                if isinstance(message, dict):
                    self.import_message(
                        platform=platform,
                        sender=str(message.get("sender", "")).strip() or "unknown",
                        recipients=[str(item) for item in message.get("recipients", [])] if isinstance(message.get("recipients"), list) else [],
                        subject=str(message.get("subject", "")),
                        body=str(message.get("body", "")),
                        project=project,
                        owner=owner,
                        thread_id=str(message.get("thread_id", "")),
                        metadata={"sync_run_id": run_id, **(message.get("metadata") if isinstance(message.get("metadata"), dict) else {})},
                    )
                    imported_messages += 1
            for event in payload.get("calendar_events", []) if isinstance(payload.get("calendar_events"), list) else []:
                if isinstance(event, dict):
                    self.add_calendar_event(
                        title=str(event.get("title", "")).strip() or "Synced event",
                        starts_at=str(event.get("starts_at", "")),
                        ends_at=str(event.get("ends_at", "")),
                        project=project,
                        owner=owner,
                        calendar_id=str(event.get("calendar_id", platform)).strip() or platform,
                        attendees=[str(item) for item in event.get("attendees", [])] if isinstance(event.get("attendees"), list) else [],
                        source=platform,
                    )
                    imported_calendar_events += 1
        except Exception as exc:
            status = "failed"
            error = str(exc)
        completed_at = utc_now()
        metadata = {
            "connector_enabled": connector["enabled"],
            "capabilities": connector["capabilities"],
            "privacy_boundary": (
                "Local sync consumes an explicit provider payload. External provider polling requires "
                "connector-specific credentials and allow_external."
            ),
        }
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_sync_runs (
                    id, project, platform, status, allow_external, imported_messages,
                    imported_contacts, imported_calendar_events, error, metadata_json,
                    started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    project,
                    platform,
                    status,
                    1 if allow_external else 0,
                    imported_messages,
                    imported_contacts,
                    imported_calendar_events,
                    error,
                    json.dumps(metadata, sort_keys=True),
                    started_at,
                    completed_at,
                ),
            )
            connection.execute("UPDATE communication_sync_runs SET owner = ? WHERE id = ?", (owner, run_id))
        self.store.add_audit_event(
            AuditEvent(
                actor="communication",
                action="connector.sync",
                resource=f"connector:{platform}",
                project=project,
                status=status,
                metadata={
                    "run_id": run_id,
                    "allow_external": allow_external,
                    "imported_messages": imported_messages,
                    "imported_contacts": imported_contacts,
                    "imported_calendar_events": imported_calendar_events,
                },
            )
        )
        run = self.get_sync_run(run_id) or {}
        if status == "failed":
            raise ValueError(error)
        return run

    def _poll_email_provider(self, *, project: str, owner: str, run_id: str) -> dict[str, int]:
        config = self.connector_config("email")
        host = str(config.get("imap_host", "")).strip()
        if not host:
            raise ValueError("email connector requires imap_host for external sync")
        port = int(config.get("imap_port") or (993 if config.get("imap_ssl", True) else 143))
        username = str(config.get("imap_username") or config.get("username") or "").strip()
        password = str(config.get("imap_password") or config.get("password") or "")
        oauth_account_id = str(config.get("imap_oauth_account_id") or config.get("oauth_account_id") or "").strip()
        if oauth_account_id:
            policy = self.mailbox_policy(account_id=oauth_account_id, owner=owner)
        else:
            policy = self.mailbox_policy(provider=str(config.get("provider", "generic")))
        mailbox_alias = str(config.get("imap_mailbox") or "inbox").strip().lower()
        mailbox = str(policy.get("mailboxes", {}).get(mailbox_alias) or config.get("imap_mailbox") or "INBOX")
        search_query = str(config.get("imap_search") or "UNSEEN")
        limit = max(1, min(int(config.get("imap_limit") or 25), 100))
        timeout = int(config.get("timeout_seconds") or 20)
        imported_messages = 0
        imported_contacts = 0
        try:
            client_cls = imaplib.IMAP4_SSL if config.get("imap_ssl", True) else imaplib.IMAP4
            with client_cls(host, port, timeout=timeout) as imap:
                if oauth_account_id:
                    token = self._fresh_oauth_access_token(oauth_account_id, owner=owner)
                    imap.authenticate("XOAUTH2", lambda _: self._xoauth2_payload(username, token))
                elif username or password:
                    imap.login(username, password)
                status, _ = imap.select(mailbox)
                if status != "OK":
                    raise ValueError(f"IMAP select failed for mailbox {mailbox}")
                status, data = imap.search(None, search_query)
                if status != "OK":
                    raise ValueError(f"IMAP search failed: {search_query}")
                ids = (data[0].split() if data else [])[-limit:]
                for message_id in ids:
                    status, fetch_data = imap.fetch(message_id, "(RFC822)")
                    if status != "OK":
                        continue
                    raw = next((item[1] for item in fetch_data if isinstance(item, tuple) and len(item) > 1), b"")
                    if not raw:
                        continue
                    parsed = BytesParser(policy=email_policy).parsebytes(raw)
                    sender = str(parsed.get("From", "unknown"))
                    subject = str(parsed.get("Subject", ""))
                    body = self._email_body(parsed)
                    if not body:
                        body = "(empty email body)"
                    self.import_message(
                        platform="email",
                        sender=sender,
                        recipients=[str(item) for item in parsed.get_all("To", [])],
                        subject=subject,
                        body=body,
                        project=project,
                        owner=owner,
                        thread_id=str(parsed.get("Message-ID", "")),
                        metadata={"sync_run_id": run_id, "provider": "imap", "mailbox": mailbox},
                    )
                    imported_messages += 1
                    imported_contacts += 1
        except Exception as exc:  # noqa: BLE001 - converted into sync failure
            raise ValueError(f"IMAP sync failed: {exc}") from exc
        return {"imported_messages": imported_messages, "imported_contacts": imported_contacts}

    def _email_body(self, message: EmailMessage) -> str:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_maintype() == "multipart":
                    continue
                if part.get_content_type() == "text/plain":
                    try:
                        return str(part.get_content()).strip()
                    except Exception:
                        payload = part.get_payload(decode=True) or b""
                        return payload.decode(part.get_content_charset() or "utf-8", errors="replace").strip()
            return ""
        try:
            return str(message.get_content()).strip()
        except Exception:
            payload = message.get_payload(decode=True) or b""
            return payload.decode(message.get_content_charset() or "utf-8", errors="replace").strip()

    def get_sync_run(self, run_id: str, *, owner: str = "") -> dict[str, Any] | None:
        with self.store.session() as connection:
            if owner:
                row = connection.execute(
                    "SELECT * FROM communication_sync_runs WHERE id = ? AND owner = ?",
                    (run_id, owner),
                ).fetchone()
            else:
                row = connection.execute("SELECT * FROM communication_sync_runs WHERE id = ?", (run_id,)).fetchone()
        return sync_run_row(row) if row else None

    def sync_runs(
        self,
        *,
        project: str = "default",
        platform: str = "",
        owner: str = "",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        params: list[Any] = [project]
        if platform:
            clauses.append("platform = ?")
            params.append(platform)
        if owner:
            clauses.append("owner = ?")
            params.append(owner)
        params.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_sync_runs WHERE {' AND '.join(clauses)} ORDER BY started_at DESC LIMIT ?",
                tuple(params),
            ).fetchall()
        return [sync_run_row(row) for row in rows]

    def sync_connector_resilient(
        self,
        platform: str,
        *,
        project: str = "default",
        owner: str = "local",
        payload: dict[str, Any] | None = None,
        allow_external: bool = False,
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Retry-aware, dedupe-aware wrapper around sync_connector.

        sync_connector itself is left unchanged for backward compatibility
        (existing callers/tests keep their exact semantics). This wrapper
        additionally: skips messages that were already imported (matched by
        platform/thread/sender/subject/body fingerprint), retries transient
        failures up to max_attempts, and records a dead letter instead of
        raising when every attempt fails — so one flaky or offline connector
        cannot take down a broader sync loop.
        """
        project = (project or "default").strip() or "default"
        payload = dict(payload or {})
        duplicates_skipped = 0
        messages = payload.get("messages")
        if isinstance(messages, list):
            kept = []
            for message in messages:
                if isinstance(message, dict) and self._is_duplicate_message(project, owner or "local", platform, message):
                    duplicates_skipped += 1
                    continue
                kept.append(message)
            payload["messages"] = kept

        attempts = 0
        last_error = ""
        while attempts < max(1, max_attempts):
            attempts += 1
            try:
                run = self.sync_connector(platform, project=project, owner=owner, payload=payload, allow_external=allow_external)
            except Exception as exc:  # noqa: BLE001 - any connector failure is retried, not just specific ones
                last_error = str(exc)
                continue
            run = dict(run)
            run["duplicates_skipped"] = duplicates_skipped
            run["attempts"] = attempts
            return run
        dead_letter_id = self._record_dead_letter(project, owner or "local", platform, payload, last_error, attempts)
        return {
            "status": "dead_lettered",
            "platform": platform,
            "project": project,
            "attempts": attempts,
            "duplicates_skipped": duplicates_skipped,
            "error": last_error,
            "dead_letter_id": dead_letter_id,
        }

    def _is_duplicate_message(self, project: str, owner: str, platform: str, message: dict[str, Any]) -> bool:
        sender = str(message.get("sender", "")).strip()
        subject = str(message.get("subject", ""))
        body = str(message.get("body", ""))
        thread_id = str(message.get("thread_id", ""))
        with self.store.session() as connection:
            row = connection.execute(
                """
                SELECT id FROM inbox_messages
                WHERE project = ? AND owner = ? AND platform = ? AND thread_id = ? AND sender = ? AND subject = ? AND body = ?
                LIMIT 1
                """,
                (project, owner or "local", platform, thread_id, sender, subject, body),
            ).fetchone()
        return row is not None

    def _record_dead_letter(self, project: str, owner: str, platform: str, payload: dict[str, Any], error: str, attempts: int) -> str:
        dead_letter_id = stable_id("communication-dead-letter", project, owner, platform, str(attempts), utc_now())
        with self.store.session() as connection:
            connection.execute(
                """
                INSERT INTO communication_dead_letters (id, project, platform, payload_json, error, attempts, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (dead_letter_id, project, platform, json.dumps(payload, sort_keys=True), error, attempts, utc_now()),
            )
            connection.execute("UPDATE communication_dead_letters SET owner = ? WHERE id = ?", (owner or "local", dead_letter_id))
        return dead_letter_id

    def list_dead_letters(self, *, project: str = "default", owner: str = "", limit: int = 50) -> list[dict[str, Any]]:
        clauses = ["project = ?"]
        args: list[Any] = [project or "default"]
        if owner:
            clauses.append("owner = ?")
            args.append(owner)
        args.append(max(1, min(limit, 200)))
        with self.store.session() as connection:
            rows = connection.execute(
                f"SELECT * FROM communication_dead_letters WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, rowid DESC LIMIT ?",
                tuple(args),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "project": r["project"],
                "owner": r["owner"],
                "platform": r["platform"],
                "payload": json.loads(r["payload_json"] or "{}"),
                "error": r["error"],
                "attempts": r["attempts"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]


def summarize(body: str) -> str:
    first_sentence = re.split(r"(?<=[.!?])\s+", body.strip())[0]
    return first_sentence[:220]


def stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:32]


def connector_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "platform": row["platform"],
        "label": row["label"],
        "enabled": bool(row["enabled"]),
        "config": json.loads(row["config_json"]),
        "capabilities": json.loads(row["capabilities_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def contact_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "name": row["name"],
        "handles": json.loads(row["handles_json"]),
        "notes": row["notes"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def attachment_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "message_id": row["message_id"],
        "filename": row["filename"],
        "content_type": row["content_type"],
        "size_bytes": row["size_bytes"],
        "sha256": row["sha256"],
        "local_path": row["local_path"],
        "document_id": row["document_id"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
        "created_at": row["created_at"],
    }


def message_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "platform": row["platform"],
        "thread_id": row["thread_id"],
        "sender": row["sender"],
        "recipients": json.loads(row["recipients_json"]),
        "subject": row["subject"],
        "body": row["body"],
        "summary": row["summary"],
        "priority": row["priority"],
        "follow_up": bool(row["follow_up"]),
        "read": bool(row["read"]),
        "answered": bool(row["answered"]),
        "spam": bool(row["spam"]),
        "archived": bool(row["archived"]),
        "folder": row["folder"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def draft_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "platform": row["platform"],
        "message_id": row["message_id"],
        "recipients": json.loads(row["recipients_json"]),
        "subject": row["subject"],
        "body": row["body"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def approval_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "draft_id": row["draft_id"],
        "reason": row["reason"],
        "status": row["status"],
        "created_at": row["created_at"],
        "resolved_at": row["resolved_at"],
    }


def sync_run_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "platform": row["platform"],
        "status": row["status"],
        "allow_external": bool(row["allow_external"]),
        "imported_messages": row["imported_messages"],
        "imported_contacts": row["imported_contacts"],
        "imported_calendar_events": row["imported_calendar_events"],
        "error": row["error"],
        "metadata": json.loads(row["metadata_json"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
    }


def oauth_account_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "platform": row["platform"],
        "provider": row["provider"],
        "label": row["label"],
        "client_id": row["client_id"],
        "auth_endpoint": row["auth_endpoint"],
        "token_endpoint": row["token_endpoint"],
        "redirect_uri": row["redirect_uri"],
        "scopes": json.loads(row["scopes_json"] or "[]"),
        "token_present": bool(row["access_token"]),
        "refresh_token_present": bool(row["refresh_token"]),
        "expires_at": row["expires_at"],
        "token_type": row["token_type"],
        "mailbox_policy": json.loads(row["mailbox_policy_json"] or "{}"),
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "privacy_boundary": "OAuth secrets and tokens stay in the local OpenAgent database and are never returned in API responses.",
    }


def calendar_row(row: Any) -> dict[str, Any]:
    return {
        "id": row["id"],
        "project": row["project"],
        "owner": row["owner"],
        "calendar_id": row["calendar_id"],
        "title": row["title"],
        "starts_at": row["starts_at"],
        "ends_at": row["ends_at"],
        "attendees": json.loads(row["attendees_json"]),
        "source": row["source"],
        "metadata": json.loads(row["metadata_json"]),
        "created_at": row["created_at"],
    }


def _attachment_text(filename: str, content: bytes) -> str:
    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        pieces = []
        for match in re.findall(rb"\((.*?)\)", content):
            text = match.decode("latin-1", errors="replace").strip()
            if text:
                pieces.append(text)
        return " ".join(pieces)
    if lowered.endswith((".txt", ".md", ".csv", ".json", ".html", ".xml", ".log")):
        return content.decode("utf-8", errors="replace")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return ""
