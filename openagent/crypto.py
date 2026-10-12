"""Versioned authenticated encryption used by OpenAgent secret stores.

New records use AES-256-GCM with a key derived by scrypt.  The legacy readers
exist only so an authenticated old record can be decrypted and immediately
rewritten in the current format.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from typing import Any, Protocol

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


CURRENT_ENVELOPE_VERSION = 2
_KEY_BYTES = 32
_SALT_BYTES = 16
_NONCE_BYTES = 12
_SCRYPT_N = 2**15
_SCRYPT_R = 8
_SCRYPT_P = 1


class CryptoError(ValueError):
    """Raised when an encrypted record is malformed or cannot be authenticated."""


class KeyProvider(Protocol):
    """Interface for future OS keyring, KMS, or HSM-backed master keys."""

    def key(self, key_id: str, purpose: str) -> bytes:
        """Return exactly 32 bytes for the requested key and purpose."""


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: Any, field: str) -> bytes:
    try:
        return base64.b64decode(str(value).encode("ascii"), validate=True)
    except Exception as exc:
        raise CryptoError(f"Malformed encrypted envelope field: {field}") from exc


def _metadata(*, key_id: str, purpose: str) -> dict[str, Any]:
    return {
        "version": CURRENT_ENVELOPE_VERSION,
        "algorithm": "AES-256-GCM",
        "kdf": "scrypt",
        "key_id": key_id,
        "purpose": purpose,
    }


def _aad(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _derive_scrypt_key(passphrase: str, salt: bytes, *, n: int, r: int, p: int) -> bytes:
    if not passphrase:
        raise CryptoError("A passphrase is required.")
    try:
        return Scrypt(salt=salt, length=_KEY_BYTES, n=n, r=r, p=p).derive(passphrase.encode("utf-8"))
    except Exception as exc:
        raise CryptoError("Unable to derive the encryption key.") from exc


def encrypt_text(
    value: str,
    passphrase: str,
    *,
    purpose: str,
    key_id: str = "passphrase-v1",
) -> dict[str, Any]:
    """Encrypt text into a self-describing authenticated envelope."""

    salt = os.urandom(_SALT_BYTES)
    nonce = os.urandom(_NONCE_BYTES)
    metadata = _metadata(key_id=key_id, purpose=purpose)
    key = _derive_scrypt_key(passphrase, salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    ciphertext = AESGCM(key).encrypt(nonce, value.encode("utf-8"), _aad(metadata))
    return {
        **metadata,
        "scrypt": {"n": _SCRYPT_N, "r": _SCRYPT_R, "p": _SCRYPT_P},
        "salt": _b64encode(salt),
        "nonce": _b64encode(nonce),
        "ciphertext": _b64encode(ciphertext),
    }


def decrypt_text(payload: dict[str, Any], passphrase: str, *, purpose: str) -> str:
    """Authenticate and decrypt a current envelope."""

    if int(payload.get("version", 0)) != CURRENT_ENVELOPE_VERSION:
        raise CryptoError("Unsupported encrypted envelope version.")
    metadata = {
        "version": CURRENT_ENVELOPE_VERSION,
        "algorithm": str(payload.get("algorithm", "")),
        "kdf": str(payload.get("kdf", "")),
        "key_id": str(payload.get("key_id", "")),
        "purpose": str(payload.get("purpose", "")),
    }
    if metadata["algorithm"] != "AES-256-GCM" or metadata["kdf"] != "scrypt":
        raise CryptoError("Unsupported encrypted envelope algorithm.")
    if not hmac.compare_digest(metadata["purpose"], purpose):
        raise CryptoError("Encrypted record purpose does not match this operation.")
    params = payload.get("scrypt") if isinstance(payload.get("scrypt"), dict) else {}
    n = int(params.get("n", _SCRYPT_N))
    r = int(params.get("r", _SCRYPT_R))
    p = int(params.get("p", _SCRYPT_P))
    if n < 2**14 or n > 2**20 or r < 1 or r > 32 or p < 1 or p > 16:
        raise CryptoError("Encrypted envelope KDF parameters are outside the accepted range.")
    salt = _b64decode(payload.get("salt", ""), "salt")
    nonce = _b64decode(payload.get("nonce", ""), "nonce")
    ciphertext = _b64decode(payload.get("ciphertext", ""), "ciphertext")
    if len(salt) < 16 or len(nonce) != _NONCE_BYTES:
        raise CryptoError("Malformed encrypted envelope.")
    key = _derive_scrypt_key(passphrase, salt, n=n, r=r, p=p)
    try:
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _aad(metadata))
    except InvalidTag as exc:
        raise CryptoError("Decryption failed: wrong passphrase or corrupted data.") from exc
    try:
        return plaintext.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CryptoError("Decrypted record is not valid UTF-8.") from exc


def is_current_envelope(payload: object) -> bool:
    return isinstance(payload, dict) and int(payload.get("version", 0)) == CURRENT_ENVELOPE_VERSION


def decrypt_legacy_security_envelope(payload: dict[str, Any], passphrase: str) -> str:
    """Decrypt the former PBKDF2/XOR/HMAC envelope for one-way migration."""

    salt = _b64decode(payload.get("salt", ""), "salt")
    nonce = _b64decode(payload.get("nonce", ""), "nonce")
    ciphertext = _b64decode(payload.get("ciphertext", ""), "ciphertext")
    iterations = int(payload.get("iterations", 200_000))
    if iterations < 100_000 or iterations > 2_000_000:
        raise CryptoError("Legacy envelope KDF parameters are outside the accepted range.")
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations, dklen=32)
    expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, str(payload.get("tag", ""))):
        raise CryptoError("Decryption failed: wrong passphrase or corrupted data.")
    stream = bytearray()
    counter = 0
    while len(stream) < len(ciphertext):
        stream.extend(hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest())
        counter += 1
    return bytes(left ^ right for left, right in zip(ciphertext, stream)).decode("utf-8")


def decrypt_legacy_vault_envelope(encoded: str, passphrase: str) -> str:
    """Decrypt the former vault string envelope for one-way migration."""

    payload = _b64decode(encoded, "legacy ciphertext")
    if len(payload) < 64:
        raise CryptoError("Malformed legacy vault entry.")
    salt, stored_mac, ciphertext = payload[:32], payload[32:64], payload[64:]
    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, 200_000, dklen=32)
    expected = hmac.new(key, ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(stored_mac, expected):
        raise CryptoError("Decryption failed: wrong passphrase or corrupted data.")
    stream = bytearray()
    block = key
    while len(stream) < len(ciphertext):
        block = hashlib.sha256(block).digest()
        stream.extend(block)
    return bytes(left ^ right for left, right in zip(ciphertext, stream)).decode("utf-8")
