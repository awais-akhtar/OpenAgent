# Security Policy

OpenAgent is designed around local ownership, explicit permissions, and inspectable execution.

## Current Security Posture

- The development server binds to `127.0.0.1` by default.
- No remote telemetry service is configured by default. Audit records and local runtime metrics are collected on the host.
- The shared egress path requires explicit approval for public destinations. Some older network integrations have not yet been migrated; review enabled connectors and firewall policy before exposing a deployment.
- Application data is stored in the configured local data directory, with SQLite as the default system of record.
- Chat, agents, research, workflows, and multi-model collaboration use the shared model gateway. External providers are disabled unless the caller approves that boundary.
- RBAC, federation policies, OIDC/JWKS validation, passkey records, device trust, audit exports, and backup manifests are local-first.
- API key hashes, plugin manifests, MCP configuration, SDK artifacts, and webhook delivery logs remain local.
- Webhook delivery to HTTP or HTTPS targets is blocked unless a caller explicitly passes an approval flag.
- Secret vault records, retention policies, sharing grants, egress approvals, audit-chain seals, privacy reports, and sandbox policy profiles are stored locally.
- Encrypted envelopes and vault records use authenticated local encryption with integrity checks. Deployments that require certified cryptographic modules or hardware-backed keys must configure and validate those providers for their environment.
- The shared process policy bounds selected local command paths. Not every legacy subprocess integration uses it yet. Operators remain responsible for reviewing enabled permissions and third-party packages.

The authenticated container and Kubernetes manifests are starting points, not a certified multi-user deployment. A shared database backend, complete tenant isolation review, backup/restore drill, and provider-specific acceptance tests are still required. See [modernization status](docs/MODERNIZATION_STATUS.md).

## Reporting Issues

Please open a private security advisory or contact the maintainers before publicly disclosing vulnerabilities.

Include:

- Affected version or commit
- Reproduction steps
- Impact
- Suggested fix, if known

## Security Goals

- Permissioned tool execution
- Human approval checkpoints for external actions
- Auditable agent and workflow runs
- Secret management with production-grade audited encryption
- Role-based access control for multi-user deployments
- Policy management and audit export controls
- Developer API key rotation and webhook retry hardening
- Sandboxed connector and plugin execution
