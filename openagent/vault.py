"""Encrypted Memory Vault for local secrets and credentials.

Design goals
------------
* Zero-knowledge: the plaintext key is never stored anywhere on disk.
* New records use an authenticated AES-256-GCM envelope with a scrypt-derived
  key and a random nonce.
* Existing legacy records are decrypted only for a verified, one-way migration
  to the current envelope.
* Vault entries are tagged, searchable by tag/category, and audit-logged.
* Key rotation: export-decrypt-re-encrypt-import without plaintext touching disk.

Vault categories (built-in):
  passwords, api_keys, private_notes, certificates, tokens, secrets
"""

from __future__ import annotations

import hashlib
import json
from typing import Any
from uuid import uuid4

from .crypto import CryptoError, decrypt_legacy_vault_envelope, decrypt_text, encrypt_text, is_current_envelope
from .models import AuditEvent, utc_now
from .storage import OpenAgentStore


def _encrypt(plaintext: str, passphrase: str) -> str:
    return json.dumps(
        encrypt_text(plaintext, passphrase, purpose="openagent.vault.entry"),
        sort_keys=True,
        separators=(",", ":"),
    )


def _decrypt(encoded: str, passphrase: str) -> str:
    """Decrypt current records and legacy records retained for migration."""

    try:
        payload = json.loads(encoded)
    except (TypeError, json.JSONDecodeError):
        return decrypt_legacy_vault_envelope(encoded, passphrase)
    if not isinstance(payload, dict) or not is_current_envelope(payload):
        raise CryptoError("Unsupported vault encryption envelope.")
    return decrypt_text(payload, passphrase, purpose="openagent.vault.entry")


def _is_legacy_ciphertext(encoded: str) -> bool:
    try:
        return not is_current_envelope(json.loads(encoded))
    except (TypeError, json.JSONDecodeError):
        return True


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
        self.store.initialize()

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
        entry_id = uuid4().hex

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
                (uuid4().hex,
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
        except ValueError:
            self._log_access(entry_id, project, "retrieve", success=False)
            raise

        replacement = _encrypt(plaintext, passphrase) if _is_legacy_ciphertext(entry["ciphertext"]) else None
        now = utc_now()
        with self.store.session() as conn:
            if replacement:
                conn.execute(
                    "UPDATE vault_entries SET ciphertext=?, access_count=access_count+1, "
                    "last_accessed=?, updated_at=? WHERE id=?",
                    (replacement, now, now, entry_id),
                )
            else:
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
            metadata={"entry_id": entry_id, "label": entry["label"], "envelope_migrated": bool(replacement)},
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
        now = utc_now()
        new_fp = _fingerprint(new_passphrase)
        with self.store.session() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                "SELECT id, ciphertext FROM vault_entries WHERE project=? AND key_fingerprint=?",
                (project, old_fp),
            ).fetchall()
            replacements = []
            for row in rows:
                try:
                    replacements.append((row[0], _encrypt(_decrypt(row[1], old_passphrase), new_passphrase)))
                except ValueError as exc:
                    raise ValueError("Vault key rotation could not verify every entry; no keys were changed.") from exc
            for entry_id, ciphertext in replacements:
                conn.execute(
                    "UPDATE vault_entries SET ciphertext=?, key_fingerprint=?, updated_at=? WHERE id=?",
                    (ciphertext, new_fp, now, entry_id),
                )
        rotated = len(replacements)
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
            "format": "openagent-vault-v2",
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
        log_id = uuid4().hex
        with self.store.session() as conn:
            conn.execute(
                "INSERT INTO vault_access_log VALUES(?,?,?,?,?,?,?)",
                (log_id, entry_id, project, action, 1 if success else 0, "", now),
            )

    def list_access_log(self, *, project: str = "default", limit: int = 100) -> list[dict[str, Any]]:
        """Return vault operation records without secret values or passphrases."""
        bounded_limit = max(1, min(int(limit), 500))
        with self.store.session() as connection:
            rows = connection.execute(
                "SELECT id, entry_id, project, action, success, created_at "
                "FROM vault_access_log WHERE project=? ORDER BY created_at DESC LIMIT ?",
                (project, bounded_limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "entry_id": row["entry_id"],
                "project": row["project"],
                "action": row["action"],
                "success": bool(row["success"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

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
            "encryption": "AES-256-GCM + scrypt",
            "envelope_version": 2,
            "features": [
                "zero-knowledge", "passphrase-encryption", "key-rotation",
                "audit-logging", "category-tags", "encrypted-export",
                "access-counting", "authenticated-encryption", "legacy-migration",
            ],
        }
