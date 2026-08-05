# Operational Lifecycle Completion

This document describes the runnable lifecycle surfaces that close the ten
operational areas covered by the 2026-07-12 completion pass. Every mutating
operation writes an audit event or a durable lifecycle record.

## Research Lifecycle

- Durable sessions, worker jobs, retry limits, cancellation, checkpoints,
  sources, source graphs, branches, archive, unarchive, and delete are stored in
  SQLite.
- `GET /api/research/sessions/{id}/stream` is a real server-sent event stream.
  Checkpoint IDs are durable cursors, so a client can reconnect with
  `after_id` without losing progress.
- Privacy boundary: research stays local unless that individual session has
  `allow_external=true`. The progress stream reads local session state only.

## Image Studio

- Albums, metadata, favorites, batch tags, SHA-256 duplicate groups, local
  transforms, version history, and undo are implemented.
- Inpainting, background removal, restoration, style transfer, sharpening, and
  denoising use local image bytes and write versioned derivatives under the
  configured data directory.
- Privacy boundary: image files and derivatives remain in local media storage.

## Document Processing

- PDF page rendering, OCR overlays, annotations, versioned redaction,
  deterministic PDF generation, optional HMAC-SHA256 signatures, and document
  comparison are available through the document processing routes.
- Privacy boundary: source documents, rendered pages, overlays, and exports are
  local artifacts. A signing key is used for the current export and is not
  persisted by the export service.

## Communication Providers

- Mailbox state mutations, local attachments, scheduled sends, approval queues,
  OAuth onboarding metadata, token refresh, IMAP/SMTP sync, and provider policy
  presets are implemented for Google, Microsoft, and generic providers.
- Live certification remains gated by operator-owned OAuth applications,
  credentials, provider tenants, and explicit network approval. OpenAgent does
  not claim that Google or Microsoft has certified a deployment without that
  external evidence.
- Privacy boundary: tokens and provider configuration remain local. Network
  exchange and mailbox operations require explicit external approval.

## Task Lifecycle

- Update, delete, pause, resume, stop, revision revert, cache clearing, run
  history, dependency-cycle rejection, natural-language scheduling, and signed
  webhook registration are implemented.
- Output destinations include run history, notifications, confined local JSON
  files, and HMAC-SHA256 signed webhooks. Delivery outcomes are persisted.
- Privacy boundary: local outputs are confined to `data/task_outputs`. Remote
  webhooks require HTTPS and `allow_external=true`; loopback endpoints remain
  local.

## Skills and Marketplace

- Skills can be edited, imported over explicitly approved HTTPS, tested,
  dependency-scanned, permission-previewed, exported, and packaged as signed
  ZIP files.
- The local marketplace supports publish, signature-verified install, update,
  uninstall, and listing removal records.
- The declarative sandbox grants only explicitly approved permissions and does
  not execute package code, a shell, network calls, or filesystem writes.

## Conversation Organization

- Pinning, bulk actions, archive/unarchive, manual compaction, semantic sorting,
  imports, project folders, and branch graphs are persisted and exposed in the
  chat workspace.
- Privacy boundary: organization metadata and compacted digests stay in the
  local conversation database.

## Enterprise Identity

- WebAuthn passkey ceremonies verify origin, relying-party ID, challenge
  expiry, one-time use, signatures, and counters. Recovery codes, trusted
  devices, and federation policies are implemented with owner checks.
- Privacy boundary: authenticator private keys and biometrics never enter
  OpenAgent. Only credential public keys and trust records are stored locally.

## Model Operations

- Resumable downloads, partial-file retention, checksums, local probes,
  dependencies, tool assignments, and lifecycle states are implemented.
- OpenAI-compatible endpoints can be certified with bounded model and chat
  probes. Reports are signed locally and can be revoked by an operator.
- Privacy boundary: remote probes send only a fixed certification prompt to an
  explicitly approved HTTPS endpoint. Localhost endpoints do not cross the
  machine boundary.

## Production Proof

Run the local release proof:

```powershell
openagent-production-proof --queue-jobs 200
```

It covers migrations, expired-lease crash recovery, queue soak, provider
matrices, parser fuzzing, accessibility contracts, installer lifecycle,
multi-user isolation, HTTP acceptance, and presence of the enforced Playwright
browser gate. CI additionally runs Chromium E2E tests and fails on browser
console errors.

```powershell
npm ci
npx playwright install chromium
npm run test:e2e
```

Browser tests use an isolated `.tmp-playwright` data directory and a temporary
localhost server. They do not contact external providers.
