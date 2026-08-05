# Security and Privacy Hardening

Stage 9 is implemented as a local-first security and privacy layer.

## Implemented Features

- Encrypted local secret vault implementation
- Secret reveal and rotation controls
- Data retention policies
- Permission-based sharing grants
- Selective egress approval records
- Tamper-evident audit-chain sealing and verification
- Privacy transparency reports
- Sandbox policy profiles
- Security and privacy API surface

## APIs

Status and vault:

```bash
GET /api/security/status
GET /api/security/secrets
POST /api/security/secrets
POST /api/security/secrets/reveal
POST /api/security/secrets/rotate
```

Retention, sharing, and egress:

```bash
GET /api/security/retention
POST /api/security/retention
POST /api/security/retention/apply
GET /api/security/shares
POST /api/security/shares
POST /api/security/shares/resolve
GET /api/security/egress
POST /api/security/egress/request
POST /api/security/egress/resolve
```

Audit chain, reports, and sandbox profiles:

```bash
GET /api/security/audit-chain
GET /api/security/audit-chain/verify
POST /api/security/audit-chain/seal
GET /api/security/privacy-reports
POST /api/security/privacy-report
GET /api/security/sandbox-profiles
POST /api/security/sandbox-profiles
POST /api/security/sandbox/evaluate
```

## Secret Vault

Secrets are encrypted before being stored in SQLite. The stored record includes ciphertext and integrity metadata; list APIs do not return plaintext. Plaintext is returned only by the reveal endpoint when the caller provides the passphrase.

The Stage 9 implementation intentionally uses only the Python standard library so OpenAgent remains dependency-light. Stage 12 adds audited local cryptography envelopes, encrypted sync bundles, hardware key metadata, and secure enclave metadata records. Production deployments can still wire this layer to certified cryptography providers, hardware-backed key storage, or an external secret manager that remains under user control.

## Retention

Retention policies can target local operational tables such as context events, agent runs, workflow runs, and developer webhook delivery logs. Applying a policy supports dry-run mode first, then deletion when explicitly requested.

## Egress and Sharing

Egress records capture why data may leave local infrastructure, where it is going, who requested it, and whether it was approved or denied. Sharing grants capture local permission-based sharing intent and revocation state.

## Audit Chain

The audit-chain sealer hashes local audit events into a linked chain. Verification recomputes event hashes and chain links to detect missing or modified sealed audit records.

## Sandbox Profiles

Sandbox profiles define allowed permissions, network access, filesystem scopes, and environment metadata for connector and plugin execution. Stage 9 evaluates requests against these profiles; Stage 12 adds sandboxed plugin run records and local audit events for those executions.
