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

Secrets are encrypted before they are stored in SQLite. New records use the shared versioned AES-256-GCM envelope from `openagent.crypto`, with a random nonce and authenticated purpose and key metadata. List APIs do not return plaintext. The reveal endpoint returns plaintext only after the caller supplies the passphrase.

Older authenticated records remain readable so an existing installation can upgrade without losing secrets. A successful legacy read is followed by a verified rewrite in the current envelope format. Key versions are recorded for rotation, and the provider interface leaves room for an operating-system keyring, KMS, HSM, or external secret manager. These adapters are integration points, not certified deployments in this repository.

## Retention

Retention policies can target local operational tables such as context events, agent runs, workflow runs, and developer webhook delivery logs. Applying a policy supports dry-run mode first, then deletion when explicitly requested.

## Egress and Sharing

Egress records capture why data may leave local infrastructure, where it is going, who requested it, and whether it was approved or denied. Sharing grants capture local permission-based sharing intent and revocation state.

## Audit Chain

The audit-chain sealer hashes local audit events into a linked chain. Verification recomputes event hashes and chain links to detect missing or modified sealed audit records.

## Sandbox Profiles

Sandbox profiles define allowed permissions, network access, filesystem scopes, and environment metadata for connector and plugin execution. Stage 9 evaluates requests against these profiles; Stage 12 adds sandboxed plugin run records and local audit events for those executions.
