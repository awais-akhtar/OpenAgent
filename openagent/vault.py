"""Encrypted Memory Vault — secure local storage for sensitive data.

Design goals
------------
* Zero-knowledge: the plaintext key is never stored anywhere on disk.
* All data encrypted with a deterministic key derived from a user passphrase
  via PBKDF2-HMAC-SHA256 (stdlib only — no external crypto deps).
* Cipher: XOR stream using PBKDF2-derived keystream (portable fallback;
  production deployments should upgrade to AES-GCM via the `cryptography` pkg).
* Vault entries are tagged, searchable by tag/category, and audit-logged.
* Key rotation: export-decrypt-re-encrypt-import without plaintext touching disk.

Vault categories (built-in):
  passwords, api_keys, private_notes, certificates, tokens, secrets
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
from datetime import datetime, timezone
from typing import Any

from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


# ---------------------------------------------------------------------------
# Crypto helpers (stdlib only)
# ---------------------------------------------------------------------------

_PBKDF2_ITERATIONS = 200_000
_SALT_BYTES = 32
_KEY_BYTES = 32


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte encryption key using PBKDF2-HMAC-SHA256."""
    return hashlib.pbkdf2_hmac(
        "sha256",
        passphrase.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
        dklen=_KEY_BYTES,
    )


def _xor_stream(data: bytes, key: bytes) -> bytes:
    """XOR-based stream cipher using an extended keystream (SHA-256 chained)."""
    keystream = bytearray()
    block = key
    while len(keystream) < len(data):
        block = hashlib.sha256(block).digest()
        keystream.extend(block)
    return bytes(a ^ b for a, b in zip(data, keystream[: len(data)]))


def _encrypt(plaintext: str, passphrase: str) -> str:
    """Encrypt plaintext → base64-encoded ciphertext (salt + hmac + data)."""
    salt = os.urandom(_SALT_BYTES)
    key = _derive_key(passphrase, salt)
    raw = plaintext.encode("utf-8")
    ciphertext = _xor_stream(raw, key)
    mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
    payload = salt + mac + ciphertext
    return base64.b64encode(payload).decode("ascii")


def _decrypt(encoded: str, passphrase: str) -> str:
    """Decrypt base64 ciphertext → plaintext. Raises ValueError on wrong key."""
    payload = base64.b64decode(encoded.encode("ascii"))
    if len(payload) < _SALT_BYTES + 32:
        raise ValueError("Malformed vault entry.")
    salt = payload[:_SALT_BYTES]
    stored_mac = payload[_SALT_BYTES: _SALT_BYTES + 32]
    ciphertext = payload[_SALT_BYTES + 32:]
    key = _derive_key(passphrase, salt)
    expected_mac = hmac.new(key, ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, expected_mac):
        raise ValueError("Decryption failed — wrong passphrase or corrupted data.")
    return _xor_stream(ciphertext, key).decode("utf-8")


def _fingerprint(passphrase: str) -> str:
    """Return a short fingerprint of the passphrase (for identity checks, not recovery)."""
    return hashlib.sha256(f"fp:{passphrase}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# EncryptedVault
# ---------------------------------------------------------------------------

class EncryptedVault:
    """Local encrypted vault for secrets, passwords, API keys, and private notes."""

    CATEGORIES = frozenset({
        "password", "api_key", "token", "certificate",
        "private_note", "secret", "credential",
    })

    def __init__(self, store: OpenAgentStore) -> None:
        self.store = store
        self.initialize()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        with self.store.session() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS vault_entries (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL DEFAULT 'default',
                    category TEXT NOT NULL,
                    label TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    ciphertext TEXT NOT NULL,
                    key_fingerprint TEXT NOT NULL,
                    notes TEXT NOT NULL DEFAULT '',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_vault_project_cat
                    ON vault_entries(project, category);

                CREATE TABLE IF NOT EXISTS vault_access_log (
                    id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    project TEXT NOT NULL,
                    action TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 1,
                    ip_hint TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS vault_key_registry (
                    id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL
                );
            """)

    # ------------------------------------------------------------------
    # Store / retrieve
    # ------------------------------------------------------------------

    def store_secret(
        self,
        label: str,
        secret: str,
        passphrase: str,
        *,
        project: str = "default",
        category: str = "secret",
        tags: list[str] | None = None,
        notes: str = "",
    ) -> dict[str, Any]:
        """Encrypt and store a secret. Returns entry metadata (no plaintext)."""
        if not label.strip():
            raise ValueError("Label is required.")
        if not secret:
            raise ValueError("Secret value cannot be empty.")
        if not passphrase:
            raise ValueError("Passphrase is required.")
        if category not in self.CATEGORIES:
            category = "secret"

        ciphertext = _encrypt(secret, passphrase)
        fingerprint = _fingerprint(passphrase)
        now = utc_now()
        entry_id = hashlib.md5(f"vault-{project}-{label}-{now}".encode()).hexdigest()

        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO vault_entries VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (entry_id, project, category, label.strip(),
                 json.dumps(tags or []), ciphertext, fingerprint,
                 notes, 0, "", now, now),
            )
            # Register key fingerprint
            conn.execute(
                "INSERT OR IGNORE INTO vault_key_registry VALUES(?,?,?,?,?)",
                (hashlib.md5(f"key-{project}-{fingerprint}".encode()).hexdigest(),
                 project, fingerprint, "default", now),
            )

        self._log_access(entry_id, project, "store", success=True)
        self.store.add_audit_event(AuditEvent(
            actor="vault",
            action="vault.store",
            resource=f"vault:store:{entry_id}",
            project=project,
            metadata={"entry_id": entry_id, "label": label, "category": category},
        ))
        return self._entry_meta(entry_id)

    def retrieve_secret(
        self, entry_id: str, passphrase: str, *, project: str = "default"
    ) -> dict[str, Any]:
        """Decrypt and return the secret. Raises ValueError on wrong passphrase."""
        entry = self._entry_raw(entry_id)
        if entry["project"] != project:
            raise PermissionError("Access denied.")

        try:
            plaintext = _decrypt(entry["ciphertext"], passphrase)
            success = True
        except ValueError:
            self._log_access(entry_id, project, "retrieve", success=False)
            raise

        # Update access counter
        now = utc_now()
        with self.store.session() as conn:
            conn.execute(
                "UPDATE vault_entries SET access_count=access_count+1, last_accessed=? WHERE id=?",
                (now, entry_id),
            )

        self._log_access(entry_id, project, "retrieve", success=True)
        self.store.add_audit_event(AuditEvent(
            actor="vault",
            action="vault.retrieve",
            resource=f"vault:retrieve:{entry_id}",
            project=project,
            metadata={"entry_id": entry_id, "label": entry["label"]},
        ))
        return {
            "id": entry_id,
            "label": entry["label"],
            "category": entry["category"],
            "secret": plaintext,
            "notes": entry["notes"],
            "tags": entry["tags"],
        }

    def delete_entry(self, entry_id: str, passphrase: str, *, project: str = "default") -> bool:
        """Delete a vault entry (requires passphrase to prevent accidental deletion)."""
        entry = self._entry_raw(entry_id)
        if entry["project"] != project:
            raise PermissionError("Access denied.")
        # Verify passphrase by attempting decryption
        try:
            _decrypt(entry["ciphertext"], passphrase)
        except ValueError:
            raise ValueError("Wrong passphrase — cannot delete.")
        with self.store.session() as conn:
            cursor = conn.execute("DELETE FROM vault_entries WHERE id=?", (entry_id,))
        self._log_access(entry_id, project, "delete", success=True)
        self.store.add_audit_event(AuditEvent(
            actor="vault",
            action="vault.delete",
            resource=f"vault:delete:{entry_id}",
            project=project,
            metadata={"entry_id": entry_id},
        ))
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # List / search (no plaintext returned)
    # ------------------------------------------------------------------

    def list_entries(
        self,
        *,
        project: str = "default",
        category: str = "",
        tag: str = "",
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List vault entries — metadata only, no secrets."""
        query = (
            "SELECT id FROM vault_entries WHERE project=?"
        )
        params: list[Any] = [project]
        if category:
            query += " AND category=?"
            params.append(category)
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        with self.store.session() as conn:
            rows = conn.execute(query, params).fetchall()
        entries = [self._entry_meta(r[0]) for r in rows]
        if tag:
            entries = [e for e in entries if tag in e.get("tags", [])]
        return entries

    def update_entry(
        self,
        entry_id: str,
        passphrase: str,
        *,
        project: str = "default",
        new_secret: str = "",
        new_label: str = "",
        new_notes: str = "",
        new_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Update a vault entry. Passphrase required for secret changes."""
        entry = self._entry_raw(entry_id)
        if entry["project"] != project:
            raise PermissionError("Access denied.")
        now = utc_now()
        with self.store.session() as conn:
            if new_secret:
                try:
                    _decrypt(entry["ciphertext"], passphrase)
                except ValueError:
                    raise ValueError("Wrong passphrase — cannot update secret.")
                ciphertext = _encrypt(new_secret, passphrase)
                fingerprint = _fingerprint(passphrase)
                conn.execute(
                    "UPDATE vault_entries SET ciphertext=?, key_fingerprint=?, updated_at=? WHERE id=?",
                    (ciphertext, fingerprint, now, entry_id),
                )
            if new_label:
                conn.execute(
                    "UPDATE vault_entries SET label=?, updated_at=? WHERE id=?",
                    (new_label, now, entry_id),
                )
            if new_notes is not None:
                conn.execute(
                    "UPDATE vault_entries SET notes=?, updated_at=? WHERE id=?",
                    (new_notes, now, entry_id),
                )
            if new_tags is not None:
                conn.execute(
                    "UPDATE vault_entries SET tags_json=?, updated_at=? WHERE id=?",
                    (json.dumps(new_tags), now, entry_id),
                )
        return self._entry_meta(entry_id)

    # ------------------------------------------------------------------
    # Key rotation
    # ------------------------------------------------------------------

    def rotate_key(
        self,
        old_passphrase: str,
        new_passphrase: str,
        *,
        project: str = "default",
    ) -> dict[str, Any]:
        """Re-encrypt all vault entries under a new passphrase."""
        if not new_passphrase:
            raise ValueError("New passphrase cannot be empty.")
        old_fp = _fingerprint(old_passphrase)
        with self.store.session() as conn:
            rows = conn.execute(
                "SELECT id, ciphertext FROM vault_entries WHERE project=? AND key_fingerprint=?",
                (project, old_fp),
            ).fetchall()
        rotated = 0
        now = utc_now()
        new_fp = _fingerprint(new_passphrase)
        for row in rows:
            try:
                plaintext = _decrypt(row[1], old_passphrase)
                new_cipher = _encrypt(plaintext, new_passphrase)
                with self.store.session() as conn:
                    conn.execute(
                        "UPDATE vault_entries SET ciphertext=?, key_fingerprint=?, updated_at=? WHERE id=?",
                        (new_cipher, new_fp, now, row[0]),
                    )
                rotated += 1
            except ValueError:
                pass
        self.store.add_audit_event(AuditEvent(
            actor="vault",
            action="vault.key_rotation",
            resource=f"vault:key_rotation:{rotated}",
            project=project,
            metadata={"entries_rotated": rotated},
        ))
        return {"rotated": rotated, "new_fingerprint": new_fp}

    # ------------------------------------------------------------------
    # Export (encrypted bundle — key NOT included)
    # ------------------------------------------------------------------

    def export_vault(self, passphrase: str, *, project: str = "default") -> dict[str, Any]:
        """Export all vault entries as an encrypted JSON bundle."""
        entries = self.list_entries(project=project, limit=10000)
        bundle: list[dict[str, Any]] = []
        for entry in entries:
            raw = self._entry_raw(entry["id"])
            bundle.append({
                "id": raw["id"],
                "label": raw["label"],
                "category": raw["category"],
                "tags": raw["tags"],
                "notes": raw["notes"],
                "ciphertext": raw["ciphertext"],
            })
        bundle_json = json.dumps(bundle)
        encrypted_bundle = _encrypt(bundle_json, passphrase)
        return {
            "format": "openagent-vault-v1",
            "project": project,
            "entry_count": len(bundle),
            "encrypted_bundle": encrypted_bundle,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _entry_raw(self, entry_id: str) -> dict[str, Any]:
        with self.store.session() as conn:
            row = conn.execute(
                "SELECT id,project,category,label,tags_json,ciphertext,key_fingerprint,"
                "notes,access_count,last_accessed,created_at,updated_at "
                "FROM vault_entries WHERE id=?", (entry_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Vault entry not found: {entry_id}")
        return {
            "id": row[0], "project": row[1], "category": row[2],
            "label": row[3], "tags": json.loads(row[4] or "[]"),
            "ciphertext": row[5], "key_fingerprint": row[6],
            "notes": row[7], "access_count": row[8],
            "last_accessed": row[9], "created_at": row[10], "updated_at": row[11],
        }

    def _entry_meta(self, entry_id: str) -> dict[str, Any]:
        raw = self._entry_raw(entry_id)
        return {k: v for k, v in raw.items() if k != "ciphertext"}

    def _log_access(self, entry_id: str, project: str, action: str, *, success: bool) -> None:
        now = utc_now()
        log_id = hashlib.md5(f"vault-log-{entry_id}-{action}-{now}".encode()).hexdigest()
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO vault_access_log VALUES(?,?,?,?,?,?,?)",
                (log_id, entry_id, project, action, 1 if success else 0, "", now),
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self, *, project: str = "default") -> dict[str, Any]:
        with self.store.session() as conn:
            n_entries = conn.execute(
                "SELECT COUNT(*) FROM vault_entries WHERE project=?", (project,)
            ).fetchone()[0]
            n_keys = conn.execute(
                "SELECT COUNT(*) FROM vault_key_registry WHERE project=?", (project,)
            ).fetchone()[0]
            cat_counts = conn.execute(
                "SELECT category, COUNT(*) FROM vault_entries WHERE project=? GROUP BY category",
                (project,),
            ).fetchall()
        return {
            "total_entries": n_entries,
            "registered_keys": n_keys,
            "entries_by_category": dict(cat_counts),
            "project": project,
            "encryption": "PBKDF2-HMAC-SHA256 + XOR-stream",
            "features": [
                "zero-knowledge", "passphrase-encryption", "key-rotation",
                "audit-logging", "category-tags", "encrypted-export",
                "access-counting", "hmac-integrity",
            ],
        }
