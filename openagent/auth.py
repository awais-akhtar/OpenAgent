"""Local account, session, and API token management.

Independent authentication and authorization layer for OpenAgent: local
password accounts, bearer sessions, scoped API tokens, and optional TOTP
two-factor verification. Everything is stored in the existing SQLite-backed
store and never leaves the machine.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import secrets
import struct
import time
from typing import Any
from uuid import uuid4

from .config import auth_enforcement_enabled, deployment_mode
from .models import utc_now
from .storage import OpenAgentStore


class AuthError(RuntimeError):
    """Raised when an authentication or authorization operation is invalid."""


PBKDF2_ITERATIONS = 210_000
SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days
TOTP_STEP_SECONDS = 30
TOTP_DIGITS = 6

SCOPE_CATALOG: tuple[dict[str, str], ...] = (
    {"scope": "chat:read", "description": "Read conversations and messages"},
    {"scope": "chat:write", "description": "Send messages and manage conversations"},
    {"scope": "agents:run", "description": "Run and delegate agents"},
    {"scope": "agents:admin", "description": "Manage agent permissions and approvals"},
    {"scope": "workflows:read", "description": "View workflows and their run history"},
    {"scope": "workflows:write", "description": "Create, edit, and trigger workflows"},
    {"scope": "knowledge:read", "description": "Search and browse the knowledge base"},
    {"scope": "knowledge:write", "description": "Ingest and modify knowledge documents"},
    {"scope": "semantic:read", "description": "Query semantic memory"},
    {"scope": "semantic:write", "description": "Write semantic memory records"},
    {"scope": "media:read", "description": "View media assets and analyses"},
    {"scope": "media:write", "description": "Ingest and analyze media"},
    {"scope": "voice:read", "description": "View voice personas, sessions, and turns"},
    {"scope": "voice:write", "description": "Create voice sessions and turns"},
    {"scope": "communication:read", "description": "View inbox, contacts, and connectors"},
    {"scope": "communication:write", "description": "Send messages and manage connectors"},
    {"scope": "calendar:read", "description": "View calendars, events, and reminders"},
    {"scope": "calendar:write", "description": "Create and modify calendar data"},
    {"scope": "docs:read", "description": "Read documents and version history"},
    {"scope": "docs:write", "description": "Create, edit, archive, and restore documents"},
    {"scope": "drafts:read", "description": "Read editor drafts"},
    {"scope": "drafts:write", "description": "Create and update editor drafts"},
    {"scope": "security:read", "description": "View security and privacy posture"},
    {"scope": "security:admin", "description": "Manage secrets, retention, and sandbox policy"},
    {"scope": "developer:read", "description": "View developer platform resources"},
    {"scope": "developer:admin", "description": "Manage webhooks, plugins, tools, and API keys"},
    {"scope": "local_ai:read", "description": "View local model and hardware status"},
    {"scope": "local_ai:admin", "description": "Manage local model jobs and adapters"},
    {"scope": "enterprise:read", "description": "View enterprise configuration"},
    {"scope": "enterprise:admin", "description": "Manage enterprise users, roles, and policy"},
    {"scope": "personal_os:read", "description": "View personal OS presets, drafts, notes, gallery"},
    {"scope": "personal_os:write", "description": "Create and modify personal OS resources"},
    {"scope": "vault:read", "description": "View encrypted vault status and entry metadata"},
    {"scope": "vault:admin", "description": "Create, reveal, and manage encrypted vault entries"},
    {"scope": "twin:read", "description": "View digital twin profile, patterns, and suggestions"},
    {"scope": "twin:write", "description": "Update digital twin goals and preferences"},
    {"scope": "scraper:read", "description": "View web scraper status and jobs"},
    {"scope": "scraper:write", "description": "Create web scraper jobs"},
    {"scope": "finetune:read", "description": "View fine-tuning datasets, jobs, and adapters"},
    {"scope": "finetune:write", "description": "Create fine-tuning jobs and adapters"},
    {"scope": "workspace:read", "description": "View workspace items, dashboard, and canvas"},
    {"scope": "workspace:write", "description": "Create and modify workspace items"},
    {"scope": "production:read", "description": "View production runtime status and policies"},
    {"scope": "production:admin", "description": "Manage production queue and audit anchors"},
    {"scope": "assistant:read", "description": "View assistant goals, tasks, and briefs"},
    {"scope": "assistant:write", "description": "Create assistant goals, tasks, and summaries"},
    {"scope": "integrations:read", "description": "View integration hub status and diagnostics"},
    {"scope": "integrations:admin", "description": "Manage pairing codes, preferences, and backups"},
    {"scope": "settings:read", "description": "View runtime settings and model policy"},
    {"scope": "settings:write", "description": "Change runtime settings"},
    {"scope": "phd:read", "description": "View private PhD profiles, positions, sources, and artifacts"},
    {"scope": "phd:write", "description": "Manage PhD application data, research, generation, and exports"},
)

DEFAULT_SCOPES: tuple[str, ...] = tuple(entry["scope"] for entry in SCOPE_CATALOG)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2$sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.b64encode(salt).decode("ascii"),
        base64.b64encode(derived).decode("ascii"),
    )


def _verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, algo, iterations_text, salt_b64, hash_b64 = encoded.split("$")
    except ValueError:
        return False
    if scheme != "pbkdf2":
        return False
    salt = base64.b64decode(salt_b64)
    expected = base64.b64decode(hash_b64)
    derived = hashlib.pbkdf2_hmac(algo, password.encode("utf-8"), salt, int(iterations_text))
    return hmac.compare_digest(derived, expected)


def _random_token(prefix: str) -> str:
    return f"{prefix}_{secrets.token_urlsafe(32)}"


def _fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _iso_offset(seconds: int) -> str:
    return (datetime.now(timezone.utc).replace(microsecond=0) + timedelta(seconds=seconds)).isoformat()


def _new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _totp_code(secret: str, *, at: float | None = None) -> str:
    padded = secret.upper() + "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(padded)
    counter = int((time.time() if at is None else at) // TOTP_STEP_SECONDS)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    truncated = struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF
    return str(truncated % (10 ** TOTP_DIGITS)).zfill(TOTP_DIGITS)


def _totp_matches(secret: str, code: str) -> bool:
    code = (code or "").strip()
    if not code or not secret:
        return False
    now = time.time()
    for drift in (-1, 0, 1):
        if hmac.compare_digest(_totp_code(secret, at=now + drift * TOTP_STEP_SECONDS), code):
            return True
    return False


def _new_backup_codes(count: int = 8) -> list[str]:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    codes = []
    for _ in range(count):
        raw = "".join(secrets.choice(alphabet) for _ in range(8))
        codes.append(f"{raw[:4]}-{raw[4:]}")
    return codes


class AuthManager:
    """Owns local user accounts, sessions, API tokens, and 2FA state."""

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS auth_users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL DEFAULT '',
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'owner',
                    totp_secret TEXT NOT NULL DEFAULT '',
                    totp_enabled INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS auth_backup_codes (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    used_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    ip TEXT NOT NULL DEFAULT '',
                    user_agent TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_auth_sessions_user
                    ON auth_sessions(user_id, revoked_at);

                CREATE TABLE IF NOT EXISTS auth_api_tokens (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    scopes_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL DEFAULT '',
                    revoked_at TEXT NOT NULL DEFAULT '',
                    last_used_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_auth_api_tokens_user
                    ON auth_api_tokens(user_id, revoked_at);

                CREATE TABLE IF NOT EXISTS auth_events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_auth_events_user_created
                    ON auth_events(user_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS auth_failed_logins (
                    id TEXT PRIMARY KEY,
                    identifier TEXT NOT NULL,
                    reason TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_event(self, connection: Any, user_id: str, event_type: str, detail: str = "") -> None:
        connection.execute(
            "INSERT INTO auth_events (id, user_id, event_type, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (uuid4().hex, user_id, event_type, detail, utc_now()),
        )

    def _log_failed_login(self, identifier: str, reason: str) -> None:
        # Uses its own session so the failure record is durable even though
        # the caller raises immediately afterwards (a raise inside the same
        # `with self.store.session()` block would roll the insert back).
        with self.store.session() as connection:
            connection.execute(
                "INSERT INTO auth_failed_logins (id, identifier, reason, created_at) VALUES (?, ?, ?, ?)",
                (uuid4().hex, identifier or "unknown", reason, utc_now()),
            )

    def _row_to_user(self, row: Any) -> dict[str, Any]:
        return {
            "user_id": row["id"],
            "username": row["username"],
            "email": row["email"],
            "display_name": row["display_name"],
            "role": row["role"],
            "totp_enabled": bool(row["totp_enabled"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _find_user_row(self, connection: Any, identifier: str) -> Any:
        return connection.execute(
            "SELECT * FROM auth_users WHERE username = ? OR email = ?",
            (identifier, identifier),
        ).fetchone()

    def _user_scopes(self, user_id: str) -> list[str]:
        # Local-first system: every active account currently receives the
        # full scope catalog. Fine-grained restriction is enforced at the
        # API-token level instead (see create_api_token), which is what
        # external integrations and automations should use.
        return list(DEFAULT_SCOPES)

    # ------------------------------------------------------------------
    # Account lifecycle
    # ------------------------------------------------------------------

    def register(self, username: str, email: str, password: str, display_name: str = "") -> dict[str, Any]:
        username = (username or "").strip()
        email = (email or "").strip().lower()
        if not username or not email:
            raise ValueError("username and email are required")
        if len(password or "") < 8:
            raise ValueError("password must be at least 8 characters")
        with self.store.session() as connection:
            existing_count = connection.execute("SELECT COUNT(*) AS n FROM auth_users").fetchone()["n"]
            if self._find_user_row(connection, username) or self._find_user_row(connection, email):
                raise AuthError("username or email already registered")
            user_id = uuid4().hex
            now = utc_now()
            role = "owner" if existing_count == 0 else "member"
            connection.execute(
                """
                INSERT INTO auth_users
                    (id, username, email, display_name, password_hash, role, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, username, email, display_name or username, _hash_password(password), role, now, now),
            )
            self._record_event(connection, user_id, "register", f"role={role}")
            return {
                "user_id": user_id,
                "username": username,
                "email": email,
                "display_name": display_name or username,
                "role": role,
            }

    def login(
        self,
        username_or_email: str,
        password: str,
        *,
        otp_code: str = "",
        ip: str = "",
        user_agent: str = "",
    ) -> dict[str, Any]:
        identifier = (username_or_email or "").strip()
        with self.store.session() as connection:
            row = self._find_user_row(connection, identifier)
        if row is None or not _verify_password(password, row["password_hash"]):
            self._log_failed_login(identifier, "bad_credentials")
            raise AuthError("invalid username/email or password")
        if row["status"] != "active":
            raise AuthError("account is not active")
        if bool(row["totp_enabled"]):
            second_factor_valid = _totp_matches(row["totp_secret"], otp_code)
            if not second_factor_valid:
                second_factor_valid = self.consume_recovery_code(str(row["id"]), otp_code)
            if not second_factor_valid:
                self._log_failed_login(identifier, "bad_otp")
                raise AuthError("missing or invalid two-factor or recovery code")
        return self.issue_session(str(row["id"]), ip=ip, user_agent=user_agent, event_type="login")

    def issue_session(
        self,
        user_id: str,
        *,
        ip: str = "",
        user_agent: str = "",
        event_type: str = "login",
    ) -> dict[str, Any]:
        """Issue a local bearer session after an already-verified authentication ceremony."""

        token = _random_token("oas")
        session_id = uuid4().hex
        now = utc_now()
        expires_at = _iso_offset(SESSION_TTL_SECONDS)
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM auth_users WHERE id = ?", (user_id,)).fetchone()
            if row is None or row["status"] != "active":
                raise AuthError("account is not active")
            connection.execute(
                """
                INSERT INTO auth_sessions
                    (id, user_id, token_hash, ip, user_agent, created_at, last_seen_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, user_id, _fingerprint(token), ip, user_agent, now, now, expires_at),
            )
            self._record_event(connection, user_id, event_type, ip)
            user = self._row_to_user(row)
        return {
            "session_id": session_id,
            "access_token": token,
            "refresh_token": token,
            "expires_at": expires_at,
            "user": user,
            "scopes": self._user_scopes(user_id),
        }

    def logout(self, access_token: str) -> dict[str, Any]:
        with self.store.session() as connection:
            token_hash = _fingerprint(access_token or "")
            row = connection.execute(
                "SELECT id, user_id FROM auth_sessions WHERE token_hash = ? AND revoked_at = ''",
                (token_hash,),
            ).fetchone()
            if row is None:
                return {"revoked": False}
            connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE id = ?",
                (utc_now(), row["id"]),
            )
            self._record_event(connection, row["user_id"], "logout")
            return {"revoked": True}

    def refresh_session(self, refresh_token: str) -> dict[str, Any]:
        refresh_token = (refresh_token or "").strip()
        if not refresh_token:
            raise AuthError("refresh_token is required")
        with self.store.session() as connection:
            token_hash = _fingerprint(refresh_token)
            row = connection.execute(
                """
                SELECT s.id AS session_id, s.user_id, s.expires_at, s.revoked_at, u.status
                FROM auth_sessions s
                JOIN auth_users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None or row["revoked_at"] or row["status"] != "active" or row["expires_at"] < utc_now():
                raise AuthError("invalid or expired refresh token")
            new_token = _random_token("oas")
            expires_at = _iso_offset(SESSION_TTL_SECONDS)
            connection.execute(
                "UPDATE auth_sessions SET token_hash = ?, last_seen_at = ?, expires_at = ? WHERE id = ?",
                (_fingerprint(new_token), utc_now(), expires_at, row["session_id"]),
            )
            self._record_event(connection, row["user_id"], "session_refresh", row["session_id"])
            return {
                "session_id": row["session_id"],
                "access_token": new_token,
                "refresh_token": new_token,
                "expires_at": expires_at,
                "scopes": self._user_scopes(row["user_id"]),
            }

    def current_user(self, access_token: str) -> dict[str, Any] | None:
        if not access_token:
            return None
        with self.store.session() as connection:
            token_hash = _fingerprint(access_token)
            row = connection.execute(
                """
                SELECT s.id AS session_id, s.expires_at, s.revoked_at, u.*
                FROM auth_sessions s
                JOIN auth_users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None or row["revoked_at"] or row["expires_at"] < utc_now():
                return None
            connection.execute(
                "UPDATE auth_sessions SET last_seen_at = ? WHERE id = ?",
                (utc_now(), row["session_id"]),
            )
            user = self._row_to_user(row)
            user["session_id"] = row["session_id"]
            user["scopes"] = self._user_scopes(row["id"])
            return user

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT id, ip, user_agent, created_at, last_seen_at, expires_at, revoked_at
                FROM auth_sessions WHERE user_id = ? ORDER BY created_at DESC, rowid DESC
                """,
                (user_id,),
            ).fetchall()
            now = utc_now()
            return [
                {
                    "session_id": r["id"],
                    "ip": r["ip"],
                    "user_agent": r["user_agent"],
                    "created_at": r["created_at"],
                    "last_seen_at": r["last_seen_at"],
                    "expires_at": r["expires_at"],
                    "active": not r["revoked_at"] and r["expires_at"] >= now,
                }
                for r in rows
            ]

    def revoke_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id FROM auth_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError("session not found")
            connection.execute(
                "UPDATE auth_sessions SET revoked_at = ? WHERE id = ?",
                (utc_now(), session_id),
            )
            self._record_event(connection, user_id, "session_revoke", session_id)
            return {"revoked": True, "session_id": session_id}

    # ------------------------------------------------------------------
    # Two-factor authentication
    # ------------------------------------------------------------------

    def setup_two_factor(self, user_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT username FROM auth_users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise KeyError("user not found")
            secret = _new_totp_secret()
            codes = _new_backup_codes()
            connection.execute(
                "UPDATE auth_users SET totp_secret = ?, totp_enabled = 0, updated_at = ? WHERE id = ?",
                (secret, utc_now(), user_id),
            )
            connection.execute("DELETE FROM auth_backup_codes WHERE user_id = ?", (user_id,))
            for code in codes:
                connection.execute(
                    "INSERT INTO auth_backup_codes (id, user_id, code_hash, created_at) VALUES (?, ?, ?, ?)",
                    (uuid4().hex, user_id, _fingerprint(code), utc_now()),
                )
            otpauth_url = (
                f"otpauth://totp/OpenAgent:{row['username']}?secret={secret}&issuer=OpenAgent"
                f"&period={TOTP_STEP_SECONDS}&digits={TOTP_DIGITS}"
            )
            self._record_event(connection, user_id, "2fa_setup_started")
            return {"secret": secret, "otpauth_url": otpauth_url, "backup_codes": codes}

    def rotate_recovery_codes(self, user_id: str, *, count: int = 8) -> dict[str, Any]:
        codes = _new_backup_codes(max(4, min(int(count), 20)))
        with self.store.session() as connection:
            if connection.execute("SELECT id FROM auth_users WHERE id = ?", (user_id,)).fetchone() is None:
                raise KeyError("user not found")
            connection.execute("DELETE FROM auth_backup_codes WHERE user_id = ?", (user_id,))
            now = utc_now()
            for code in codes:
                connection.execute(
                    "INSERT INTO auth_backup_codes (id, user_id, code_hash, created_at) VALUES (?, ?, ?, ?)",
                    (uuid4().hex, user_id, _fingerprint(code), now),
                )
            self._record_event(connection, user_id, "recovery_codes_rotated", f"count={len(codes)}")
        return {"user_id": user_id, "recovery_codes": codes, "count": len(codes)}

    def consume_recovery_code(self, user_id: str, code: str) -> bool:
        code = (code or "").strip().upper()
        if not code:
            return False
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id FROM auth_backup_codes WHERE user_id = ? AND code_hash = ? AND used_at = ''",
                (user_id, _fingerprint(code)),
            ).fetchone()
            if row is None:
                return False
            now = utc_now()
            connection.execute("UPDATE auth_backup_codes SET used_at = ? WHERE id = ?", (now, row["id"]))
            self._record_event(connection, user_id, "recovery_code_used", str(row["id"]))
            return True

    def verify_two_factor(self, user_id: str, otp_code: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT totp_secret FROM auth_users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise KeyError("user not found")
            if not _totp_matches(row["totp_secret"], otp_code):
                raise AuthError("invalid two-factor code")
            connection.execute(
                "UPDATE auth_users SET totp_enabled = 1, updated_at = ? WHERE id = ?",
                (utc_now(), user_id),
            )
            self._record_event(connection, user_id, "2fa_enabled")
            return {"enabled": True}

    def disable_two_factor(self, user_id: str, otp_code: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT totp_secret FROM auth_users WHERE id = ?", (user_id,)).fetchone()
            if row is None:
                raise KeyError("user not found")
            if not _totp_matches(row["totp_secret"], otp_code):
                raise AuthError("invalid two-factor code")
            connection.execute(
                "UPDATE auth_users SET totp_enabled = 0, totp_secret = '', updated_at = ? WHERE id = ?",
                (utc_now(), user_id),
            )
            self._record_event(connection, user_id, "2fa_disabled")
            return {"enabled": False}

    # ------------------------------------------------------------------
    # API tokens
    # ------------------------------------------------------------------

    def create_api_token(
        self,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
        expires_at: str = "",
    ) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("token name is required")
        requested = [s for s in (scopes or []) if s in DEFAULT_SCOPES]
        if not requested:
            requested = list(DEFAULT_SCOPES)
        token = _random_token("oat")
        token_id = uuid4().hex
        with self.store.session() as connection:
            user = connection.execute("SELECT id FROM auth_users WHERE id = ?", (user_id,)).fetchone()
            if user is None:
                raise KeyError("user not found")
            connection.execute(
                """
                INSERT INTO auth_api_tokens
                    (id, user_id, name, token_hash, scopes_json, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (token_id, user_id, name, _fingerprint(token), json.dumps(requested), utc_now(), expires_at),
            )
            self._record_event(connection, user_id, "api_token_created", name)
        return {"token_id": token_id, "token": token, "name": name, "scopes": requested}

    def list_api_tokens(self, user_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                """
                SELECT id, name, scopes_json, created_at, expires_at, revoked_at, last_used_at
                FROM auth_api_tokens WHERE user_id = ? ORDER BY created_at DESC, rowid DESC
                """,
                (user_id,),
            ).fetchall()
            return [
                {
                    "token_id": r["id"],
                    "name": r["name"],
                    "scopes": json.loads(r["scopes_json"] or "[]"),
                    "created_at": r["created_at"],
                    "expires_at": r["expires_at"],
                    "revoked": bool(r["revoked_at"]),
                    "last_used_at": r["last_used_at"],
                }
                for r in rows
            ]

    def revoke_api_token(self, user_id: str, token_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute(
                "SELECT id FROM auth_api_tokens WHERE id = ? AND user_id = ?",
                (token_id, user_id),
            ).fetchone()
            if row is None:
                raise KeyError("token not found")
            connection.execute(
                "UPDATE auth_api_tokens SET revoked_at = ? WHERE id = ?",
                (utc_now(), token_id),
            )
            self._record_event(connection, user_id, "api_token_revoked", token_id)
            return {"revoked": True, "token_id": token_id}

    def resolve_bearer(self, token: str) -> dict[str, Any] | None:
        """Resolve a session or API token into a user context with scopes."""
        if not token:
            return None
        user = self.current_user(token)
        if user is not None:
            return user
        with self.store.session() as connection:
            token_hash = _fingerprint(token)
            row = connection.execute(
                """
                SELECT t.*, u.username, u.email, u.role, u.status
                FROM auth_api_tokens t JOIN auth_users u ON u.id = t.user_id
                WHERE t.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()
            if row is None or row["revoked_at"] or row["status"] != "active":
                return None
            if row["expires_at"] and row["expires_at"] < utc_now():
                return None
            connection.execute(
                "UPDATE auth_api_tokens SET last_used_at = ? WHERE id = ?",
                (utc_now(), row["id"]),
            )
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "email": row["email"],
                "role": row["role"],
                "scopes": json.loads(row["scopes_json"] or "[]"),
                "token_id": row["id"],
            }

    def list_scopes(self) -> list[dict[str, str]]:
        return [dict(entry) for entry in SCOPE_CATALOG]

    def list_users(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT * FROM auth_users ORDER BY created_at ASC, rowid ASC"
            ).fetchall()
        return [self._row_to_user(row) for row in rows]

    def roles(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "owner",
                "description": "Full local administrator for a single-user or owned deployment.",
                "default_scopes": list(DEFAULT_SCOPES),
            },
            {
                "role": "member",
                "description": "Authenticated local user. API-token scopes should be narrowed for integrations.",
                "default_scopes": list(DEFAULT_SCOPES),
            },
        ]

    def policy(self) -> dict[str, Any]:
        mode = deployment_mode()
        return {
            "deployment_mode": mode,
            "auth_enforced": auth_enforcement_enabled(),
            "default_on_modes": ["team", "enterprise"],
            "protected_routes": [
                {"prefix": prefix, "read_scope": read_scope, "write_scope": write_scope}
                for prefix, read_scope, write_scope in PROTECTED_ROUTE_SCOPES
            ],
            "scope_count": len(SCOPE_CATALOG),
            "privacy_boundary": "Auth policy is evaluated locally from environment and SQLite account state.",
        }

    def auth_audit_summary(self, *, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(limit, 500))
        with self.store.session() as connection:
            events = connection.execute(
                "SELECT * FROM auth_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            failed = connection.execute(
                "SELECT * FROM auth_failed_logins ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "events": [
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "event_type": row["event_type"],
                    "detail": row["detail"],
                    "created_at": row["created_at"],
                }
                for row in events
            ],
            "failed_logins": [
                {
                    "id": row["id"],
                    "identifier": row["identifier"],
                    "reason": row["reason"],
                    "created_at": row["created_at"],
                }
                for row in failed
            ],
        }


# ----------------------------------------------------------------------
# Route protection policy
#
# Maps a path prefix to the (read_scope, write_scope) pair required to
# access it once auth enforcement is turned on (see
# config.auth_enforcement_enabled). GET requests require read_scope;
# POST/PUT/PATCH/DELETE require write_scope. This intentionally covers every
# route group from the authorization scope map, not just the highest-risk
# ones — enforcement itself stays opt-in (off by default) so this is safe to
# be comprehensive. Purely informational/public routes (health, static
# build info, capability listings) are left ungated on purpose.
# ----------------------------------------------------------------------

PROTECTED_ROUTE_SCOPES: tuple[tuple[str, str, str], ...] = (
    ("/api/chat/", "chat:read", "chat:write"),
    ("/api/conversations", "chat:read", "chat:write"),
    ("/api/conversation/", "chat:read", "chat:write"),
    ("/api/messages", "chat:read", "chat:write"),
    ("/api/agents", "agents:run", "agents:admin"),
    ("/api/workflows", "workflows:read", "workflows:write"),
    ("/api/workflow-runs", "workflows:read", "workflows:write"),
    ("/api/knowledge/", "knowledge:read", "knowledge:write"),
    ("/api/research/", "knowledge:read", "knowledge:write"),
    ("/api/semantic/", "semantic:read", "semantic:write"),
    ("/api/embeddings", "semantic:read", "semantic:write"),
    ("/api/media/", "media:read", "media:write"),
    ("/api/voice/", "voice:read", "voice:write"),
    ("/api/communication/", "communication:read", "communication:write"),
    ("/api/calendar/", "calendar:read", "calendar:write"),
    ("/api/document/", "docs:read", "docs:write"),
    ("/api/documents/", "docs:read", "docs:write"),
    ("/api/editor-drafts", "drafts:read", "drafts:write"),
    ("/api/security/", "security:read", "security:admin"),
    ("/api/developer/", "developer:read", "developer:admin"),
    ("/api/graphql", "developer:read", "developer:admin"),
    ("/api/local-ai/", "local_ai:read", "local_ai:admin"),
    ("/api/models", "local_ai:read", "local_ai:admin"),
    ("/api/model-policy", "local_ai:read", "settings:write"),
    ("/api/enterprise/", "enterprise:read", "enterprise:admin"),
    ("/api/personal-os/", "personal_os:read", "personal_os:write"),
    ("/api/vault/", "vault:read", "vault:admin"),
    ("/api/twin/", "twin:read", "twin:write"),
    ("/api/scraper/", "scraper:read", "scraper:write"),
    ("/api/finetune/", "finetune:read", "finetune:write"),
    ("/api/workspace/", "workspace:read", "workspace:write"),
    ("/api/production/", "production:read", "production:admin"),
    ("/api/assistant/", "assistant:read", "assistant:write"),
    ("/api/integrations/", "integrations:read", "integrations:admin"),
    ("/api/settings/", "settings:read", "settings:write"),
    ("/api/phd/", "phd:read", "phd:write"),
)


def required_scope_for_path(path: str, method: str = "GET") -> str | None:
    """Return the scope required for *path*/*method*, or None if it is not gated."""
    for prefix, read_scope, write_scope in PROTECTED_ROUTE_SCOPES:
        if path.startswith(prefix):
            return read_scope if method.upper() == "GET" else write_scope
    return None


def auth_enforced_owner(user: dict[str, Any] | None, *, fallback: str = "local") -> str:
    """Return the authenticated owner id when present, otherwise the local fallback."""

    if user and str(user.get("user_id", "")).strip():
        return str(user["user_id"])
    return fallback
