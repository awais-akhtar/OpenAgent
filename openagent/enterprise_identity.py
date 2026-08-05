"""Passkeys, recovery, device trust, and federation policy for local identity."""

from __future__ import annotations

import json
import secrets
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .auth import AuthError, AuthManager, _fingerprint, _iso_offset
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore

try:
    from webauthn import (
        base64url_to_bytes,
        generate_authentication_options,
        generate_registration_options,
        options_to_json,
        verify_authentication_response,
        verify_registration_response,
    )
    from webauthn.helpers import bytes_to_base64url
    from webauthn.helpers.structs import (
        AuthenticatorSelectionCriteria,
        PublicKeyCredentialDescriptor,
        ResidentKeyRequirement,
        UserVerificationRequirement,
    )
except ImportError:  # pragma: no cover - exercised in minimal installs.
    base64url_to_bytes = generate_authentication_options = generate_registration_options = None
    options_to_json = verify_authentication_response = verify_registration_response = None
    bytes_to_base64url = None
    AuthenticatorSelectionCriteria = PublicKeyCredentialDescriptor = None
    ResidentKeyRequirement = UserVerificationRequirement = None


class EnterpriseIdentity:
    def __init__(self, store: OpenAgentStore, auth: AuthManager) -> None:
        self.store = store
        self.auth = auth
        self.initialize()

    def initialize(self) -> None:
        with self.store.session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS identity_webauthn_challenges (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT '',
                    flow TEXT NOT NULL,
                    challenge TEXT NOT NULL,
                    rp_id TEXT NOT NULL,
                    origin TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    consumed_at TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_identity_challenges_expiry
                    ON identity_webauthn_challenges(expires_at, consumed_at);
                CREATE TABLE IF NOT EXISTS identity_passkeys (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    credential_id TEXT NOT NULL UNIQUE,
                    public_key TEXT NOT NULL,
                    sign_count INTEGER NOT NULL DEFAULT 0,
                    device_type TEXT NOT NULL DEFAULT '',
                    backed_up INTEGER NOT NULL DEFAULT 0,
                    transports_json TEXT NOT NULL DEFAULT '[]',
                    label TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_identity_passkeys_user ON identity_passkeys(user_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS identity_trusted_devices (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    trust_level TEXT NOT NULL DEFAULT 'standard',
                    created_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    revoked_at TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY(user_id) REFERENCES auth_users(id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS identity_federation_policies (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL UNIQUE,
                    issuer TEXT NOT NULL,
                    audience TEXT NOT NULL,
                    allowed_domains_json TEXT NOT NULL DEFAULT '[]',
                    required_groups_json TEXT NOT NULL DEFAULT '[]',
                    default_role TEXT NOT NULL DEFAULT 'member',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def status(self, *, user_id: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            passkeys = connection.execute("SELECT COUNT(*) AS n FROM identity_passkeys WHERE user_id = ?", (user_id,)).fetchone()["n"] if user_id else 0
            devices = connection.execute("SELECT COUNT(*) AS n FROM identity_trusted_devices WHERE user_id = ? AND revoked_at = ''", (user_id,)).fetchone()["n"] if user_id else 0
            policies = connection.execute("SELECT COUNT(*) AS n FROM identity_federation_policies WHERE enabled = 1").fetchone()["n"]
        return {"webauthn_available": generate_registration_options is not None, "passkeys": passkeys,
                "trusted_devices": devices, "federation_policies": policies,
                "privacy_boundary": "Credential public keys and device trust records stay in local SQLite. Authenticator private keys and biometrics never enter OpenAgent."}

    def registration_options(self, user_id: str, *, rp_id: str, origin: str) -> dict[str, Any]:
        self._require_webauthn()
        rp_id, origin = self._validate_relying_party(rp_id, origin)
        with self.store.session() as connection:
            user = connection.execute("SELECT * FROM auth_users WHERE id = ? AND status = 'active'", (user_id,)).fetchone()
            credentials = connection.execute("SELECT credential_id FROM identity_passkeys WHERE user_id = ?", (user_id,)).fetchall()
        if user is None:
            raise KeyError("user not found")
        challenge = secrets.token_bytes(32)
        options = generate_registration_options(
            rp_id=rp_id, rp_name="OpenAgent", user_name=str(user["username"]), user_id=bytes.fromhex(user_id),
            user_display_name=str(user["display_name"]), challenge=challenge,
            exclude_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(row["credential_id"])) for row in credentials],
            authenticator_selection=AuthenticatorSelectionCriteria(
                resident_key=ResidentKeyRequirement.PREFERRED,
                user_verification=UserVerificationRequirement.REQUIRED,
            ),
        )
        ceremony_id = self._store_challenge(user_id, "registration", bytes_to_base64url(challenge), rp_id, origin)
        return {"ceremony_id": ceremony_id, "publicKey": json.loads(options_to_json(options))}

    def finish_registration(self, user_id: str, ceremony_id: str, credential: dict[str, Any], *, label: str = "") -> dict[str, Any]:
        challenge = self._consume_challenge(ceremony_id, "registration", user_id=user_id)
        try:
            verified = verify_registration_response(
                credential=credential, expected_challenge=base64url_to_bytes(challenge["challenge"]),
                expected_rp_id=challenge["rp_id"], expected_origin=challenge["origin"], require_user_verification=True,
            )
        except Exception as exc:
            self._audit("identity.passkey.registration.denied", user_id, status="denied", metadata={"reason": type(exc).__name__})
            raise AuthError("passkey registration verification failed") from exc
        credential_id = bytes_to_base64url(verified.credential_id)
        record = {
            "id": uuid4().hex, "user_id": user_id, "credential_id": credential_id,
            "public_key": bytes_to_base64url(verified.credential_public_key), "sign_count": verified.sign_count,
            "device_type": str(verified.credential_device_type.value), "backed_up": int(verified.credential_backed_up),
            "transports": list(credential.get("response", {}).get("transports", [])), "label": label.strip() or "Passkey",
            "created_at": utc_now(), "last_used_at": "",
        }
        with self.store.session() as connection:
            connection.execute(
                """INSERT INTO identity_passkeys (id, user_id, credential_id, public_key, sign_count,
                   device_type, backed_up, transports_json, label, created_at, last_used_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record["id"], user_id, credential_id, record["public_key"], record["sign_count"], record["device_type"],
                 record["backed_up"], json.dumps(record["transports"]), record["label"], record["created_at"], ""),
            )
        self._audit("identity.passkey.register", user_id, metadata={"passkey_id": record["id"]})
        record["backed_up"] = bool(record["backed_up"])
        record.pop("public_key")
        return record

    def authentication_options(self, identifier: str, *, rp_id: str, origin: str) -> dict[str, Any]:
        self._require_webauthn()
        rp_id, origin = self._validate_relying_party(rp_id, origin)
        with self.store.session() as connection:
            user = connection.execute("SELECT id FROM auth_users WHERE (username = ? OR email = ?) AND status = 'active'", (identifier, identifier)).fetchone()
            credentials = connection.execute("SELECT credential_id, transports_json FROM identity_passkeys WHERE user_id = ?", (user["id"],)).fetchall() if user else []
        challenge = secrets.token_bytes(32)
        options = generate_authentication_options(
            rp_id=rp_id, challenge=challenge,
            allow_credentials=[PublicKeyCredentialDescriptor(id=base64url_to_bytes(row["credential_id"])) for row in credentials],
            user_verification=UserVerificationRequirement.REQUIRED,
        )
        ceremony_id = self._store_challenge(str(user["id"]) if user else "", "authentication", bytes_to_base64url(challenge), rp_id, origin)
        return {"ceremony_id": ceremony_id, "publicKey": json.loads(options_to_json(options))}

    def finish_authentication(self, ceremony_id: str, credential: dict[str, Any], *, ip: str = "", user_agent: str = "") -> dict[str, Any]:
        challenge = self._consume_challenge(ceremony_id, "authentication")
        credential_id = str(credential.get("id", ""))
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM identity_passkeys WHERE credential_id = ?", (credential_id,)).fetchone()
        if row is None or (challenge["user_id"] and challenge["user_id"] != row["user_id"]):
            self._audit("identity.passkey.authentication.denied", challenge["user_id"], status="denied", metadata={"reason": "unknown_credential"})
            raise AuthError("passkey authentication verification failed")
        try:
            verified = verify_authentication_response(
                credential=credential, expected_challenge=base64url_to_bytes(challenge["challenge"]),
                expected_rp_id=challenge["rp_id"], expected_origin=challenge["origin"],
                credential_public_key=base64url_to_bytes(row["public_key"]), credential_current_sign_count=int(row["sign_count"]),
                require_user_verification=True,
            )
        except Exception as exc:
            self._audit("identity.passkey.authentication.denied", row["user_id"], status="denied", metadata={"reason": type(exc).__name__})
            raise AuthError("passkey authentication verification failed") from exc
        with self.store.session() as connection:
            connection.execute("UPDATE identity_passkeys SET sign_count = ?, last_used_at = ? WHERE id = ?", (verified.new_sign_count, utc_now(), row["id"]))
        self._audit("identity.passkey.authentication", row["user_id"], metadata={"passkey_id": row["id"]})
        return self.auth.issue_session(str(row["user_id"]), ip=ip, user_agent=user_agent, event_type="passkey_login")

    def passkeys(self, user_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT id, credential_id, sign_count, device_type, backed_up, transports_json, label, created_at, last_used_at FROM identity_passkeys WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
        return [{**dict(row), "backed_up": bool(row["backed_up"]), "transports": json.loads(row["transports_json"])} for row in rows]

    def delete_passkey(self, user_id: str, passkey_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            cursor = connection.execute("DELETE FROM identity_passkeys WHERE id = ? AND user_id = ?", (passkey_id, user_id))
            if cursor.rowcount != 1:
                raise KeyError("passkey not found")
        self._audit("identity.passkey.delete", user_id, metadata={"passkey_id": passkey_id})
        return {"deleted": True, "id": passkey_id}

    def trust_device(self, user_id: str, *, name: str, fingerprint: str, trust_level: str = "standard", days: int = 30) -> dict[str, Any]:
        if trust_level not in {"standard", "managed", "privileged"}:
            raise ValueError("invalid trust level")
        token = f"oad_{secrets.token_urlsafe(32)}"
        now = utc_now()
        record = {"id": uuid4().hex, "user_id": user_id, "name": name.strip() or "Trusted device", "fingerprint": fingerprint.strip(),
                  "trust_level": trust_level, "created_at": now, "last_seen_at": now, "expires_at": _iso_offset(max(1, min(days, 365)) * 86400), "revoked_at": ""}
        if not record["fingerprint"]:
            raise ValueError("device fingerprint is required")
        with self.store.session() as connection:
            connection.execute("INSERT INTO identity_trusted_devices (id, user_id, name, fingerprint, token_hash, trust_level, created_at, last_seen_at, expires_at, revoked_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '')",
                (record["id"], user_id, record["name"], record["fingerprint"], _fingerprint(token), trust_level, now, now, record["expires_at"]))
        self._audit("identity.device.trust", user_id, metadata={"device_id": record["id"], "trust_level": trust_level})
        return {**record, "device_token": token}

    def verify_device(self, token: str, fingerprint: str) -> dict[str, Any] | None:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM identity_trusted_devices WHERE token_hash = ? AND fingerprint = ? AND revoked_at = '' AND expires_at >= ?", (_fingerprint(token), fingerprint, utc_now())).fetchone()
            if row is None:
                return None
            connection.execute("UPDATE identity_trusted_devices SET last_seen_at = ? WHERE id = ?", (utc_now(), row["id"]))
        return {key: row[key] for key in ("id", "user_id", "name", "fingerprint", "trust_level", "created_at", "expires_at")}

    def devices(self, user_id: str) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT id, user_id, name, fingerprint, trust_level, created_at, last_seen_at, expires_at, revoked_at FROM identity_trusted_devices WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
        return [dict(row) for row in rows]

    def revoke_device(self, user_id: str, device_id: str) -> dict[str, Any]:
        with self.store.session() as connection:
            cursor = connection.execute("UPDATE identity_trusted_devices SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at = ''", (utc_now(), device_id, user_id))
            if cursor.rowcount != 1:
                raise KeyError("device not found")
        self._audit("identity.device.revoke", user_id, metadata={"device_id": device_id})
        return {"revoked": True, "id": device_id}

    def save_federation_policy(self, *, provider: str, issuer: str, audience: str, allowed_domains: list[str], required_groups: list[str], default_role: str = "member", enabled: bool = True) -> dict[str, Any]:
        if not provider.strip() or not issuer.startswith("https://") or not audience.strip():
            raise ValueError("provider, HTTPS issuer, and audience are required")
        if default_role not in {"member", "admin", "owner"}:
            raise ValueError("invalid default role")
        record = {"id": uuid4().hex, "provider": provider.strip(), "issuer": issuer.rstrip("/"), "audience": audience.strip(),
                  "allowed_domains": sorted({item.lower().strip() for item in allowed_domains if item.strip()}),
                  "required_groups": sorted({item.strip() for item in required_groups if item.strip()}), "default_role": default_role,
                  "enabled": bool(enabled), "updated_at": utc_now()}
        with self.store.session() as connection:
            old = connection.execute("SELECT id FROM identity_federation_policies WHERE provider = ?", (record["provider"],)).fetchone()
            if old:
                record["id"] = old["id"]
            connection.execute("INSERT OR REPLACE INTO identity_federation_policies (id, provider, issuer, audience, allowed_domains_json, required_groups_json, default_role, enabled, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (record["id"], record["provider"], record["issuer"], record["audience"], json.dumps(record["allowed_domains"]), json.dumps(record["required_groups"]), default_role, int(enabled), record["updated_at"]))
        self._audit("identity.federation.policy", "", metadata={"provider": record["provider"]})
        return record

    def federation_policies(self) -> list[dict[str, Any]]:
        with self.store.session() as connection:
            rows = connection.execute("SELECT * FROM identity_federation_policies ORDER BY provider").fetchall()
        return [{**dict(row), "allowed_domains": json.loads(row["allowed_domains_json"]), "required_groups": json.loads(row["required_groups_json"]), "enabled": bool(row["enabled"])} for row in rows]

    def evaluate_federation(self, provider: str, claims: dict[str, Any]) -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM identity_federation_policies WHERE provider = ? AND enabled = 1", (provider,)).fetchone()
        reasons = []
        if row is None:
            reasons.append("policy_not_found")
        else:
            if str(claims.get("iss", "")).rstrip("/") != row["issuer"]:
                reasons.append("issuer_mismatch")
            audiences = claims.get("aud", [])
            if isinstance(audiences, str):
                audiences = [audiences]
            if row["audience"] not in audiences:
                reasons.append("audience_mismatch")
            domains = json.loads(row["allowed_domains_json"])
            email_domain = str(claims.get("email", "")).rsplit("@", 1)[-1].lower() if "@" in str(claims.get("email", "")) else ""
            if domains and email_domain not in domains:
                reasons.append("domain_denied")
            required = set(json.loads(row["required_groups_json"]))
            groups = set(str(item) for item in claims.get("groups", []) if isinstance(claims.get("groups"), list))
            if required and not required.issubset(groups):
                reasons.append("required_group_missing")
        allowed = not reasons
        self._audit("identity.federation.evaluate", str(claims.get("sub", "")), status="allowed" if allowed else "denied", metadata={"provider": provider, "reasons": reasons})
        return {"allowed": allowed, "provider": provider, "reasons": reasons, "role": row["default_role"] if allowed and row else ""}

    def _store_challenge(self, user_id: str, flow: str, challenge: str, rp_id: str, origin: str) -> str:
        ceremony_id = uuid4().hex
        with self.store.session() as connection:
            connection.execute("INSERT INTO identity_webauthn_challenges (id, user_id, flow, challenge, rp_id, origin, expires_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (ceremony_id, user_id, flow, challenge, rp_id, origin, _iso_offset(300), utc_now()))
        return ceremony_id

    def _consume_challenge(self, ceremony_id: str, flow: str, *, user_id: str = "") -> dict[str, Any]:
        with self.store.session() as connection:
            row = connection.execute("SELECT * FROM identity_webauthn_challenges WHERE id = ? AND flow = ?", (ceremony_id, flow)).fetchone()
            if row is None or row["consumed_at"] or row["expires_at"] < utc_now() or (user_id and row["user_id"] != user_id):
                raise AuthError("invalid, expired, or already-used WebAuthn ceremony")
            connection.execute("UPDATE identity_webauthn_challenges SET consumed_at = ? WHERE id = ?", (utc_now(), ceremony_id))
            return dict(row)

    @staticmethod
    def _validate_relying_party(rp_id: str, origin: str) -> tuple[str, str]:
        parsed = urlparse(origin)
        host = (parsed.hostname or "").lower()
        rp_id = rp_id.lower().strip()
        local = host in {"127.0.0.1", "localhost"}
        if parsed.scheme not in ({"http", "https"} if local else {"https"}):
            raise AuthError("WebAuthn origin must use HTTPS except on localhost")
        if not host or (host != rp_id and not host.endswith("." + rp_id)):
            raise AuthError("WebAuthn RP ID does not match the request origin")
        return rp_id, origin.rstrip("/")

    @staticmethod
    def _require_webauthn() -> None:
        if generate_registration_options is None:
            raise RuntimeError("Install OpenAgent with the security or full dependency group to enable passkeys")

    def _audit(self, action: str, user_id: str, *, status: str = "allowed", metadata: dict[str, Any]) -> None:
        self.store.add_audit_event(AuditEvent(actor="enterprise-identity", action=action, resource=f"user:{user_id or 'federation'}", project="identity", status=status, metadata=metadata))
