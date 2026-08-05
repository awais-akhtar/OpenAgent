# Production Runtime

Stage 12 adds local production-depth implementations for the remaining OpenAgent vision items.

The implementation is intentionally self-hosted and dependency-light. It does not outsource trust to cloud services. External provider polling, hosted package registries, hardware security modules, signed installer distribution, and mobile store distribution still require deployment-specific credentials and infrastructure, but the local runtime surfaces are now runnable, tested, documented, and auditable.

## Privacy Boundary

- Policy, queue, vector, semantic science, reranking, automation, packaging, marketplace, GPU planning, fine-tune metadata, proactive jobs, federated aggregation, and sync records stay in the local SQLite database.
- External model and connector boundaries are policy-gated.
- Connector polling supports local fixture adapters by default. Live adapters require `allow_external=true`.
- Crypto envelopes use local passphrase-derived encryption and HMAC integrity checks.
- OIDC/JWKS validation is local and supports HS256 JWKS plus RS256 verification with stdlib RSA math.
- Desktop and browser automation actions are recorded and sandboxed locally. OS/browser effects require explicit approval or client integration.
- Native desktop launch through `openagent-desktop` reads and writes only the configured local data directory and does not require a browser.
- Multi-device sync bundles are encrypted before transport.

## APIs

Production status:

```bash
GET /api/production/status?project=core
```

Policy engine:

```bash
POST /api/production/policies
POST /api/production/policies/evaluate
```

Queue workers:

```bash
POST /api/production/queue
POST /api/production/queue/run-next
GET /api/production/queue?project=core
```

Append-only audit anchors:

```bash
POST /api/production/audit-anchor
GET /api/production/audit-anchors?project=core
```

Production vector index:

```bash
POST /api/production/vector/indexes
POST /api/production/vector/upsert
POST /api/production/vector/search
```

Semantic science and RAG reranking:

```bash
POST /api/production/semantic/science
POST /api/production/rag/rerank
```

OIDC/JWKS validation:

```bash
POST /api/production/oidc/validate
```

Crypto envelopes and encrypted sync:

```bash
POST /api/production/crypto/encrypt
POST /api/production/crypto/decrypt
POST /api/production/sync/devices
POST /api/production/sync/bundles
```

Connectors:

```bash
POST /api/production/connectors/adapters
POST /api/production/connectors/poll
```

Automation:

```bash
POST /api/production/automation/desktop
POST /api/production/automation/browser
```

Packaging and marketplace:

```bash
POST /api/production/packages/app
POST /api/production/marketplace/agents
POST /api/production/sdk/publish
```

Native Windows desktop:

```powershell
openagent-desktop --status-json
openagent-desktop --feature-manifest-json
openagent-desktop --feature-smoke-json --project core
openagent-desktop --hard-audit-json --project hard-audit
openagent-hard-check --pretty --project hard-audit
.\installers\windows\install.ps1
```

Local AI production planning:

```bash
POST /api/production/gpu/plan
POST /api/production/finetune/jobs
```

Autonomous and federated systems:

```bash
POST /api/production/proactive/jobs
POST /api/production/proactive/run
POST /api/production/federated/rounds
```

Sandboxed plugin execution:

```bash
POST /api/production/sandbox/run
```

## Auditing

Every mutating Stage 12 action writes a `production.*` audit event. Audit anchoring writes append-only records to `production_audit_anchors` with a Merkle root and prior-anchor hash.

## What This Completes

- Production semantic model governance implementation.
- External model privacy boundary enforcement.
- Workflow and production queue scaling implementation.
- Production policy engine.
- Append-only audit anchoring.
- Production RAG reranking implementation.
- Connector adapter architecture and polling worker implementation.
- Deployment/package manifests for desktop, mobile, and PWA targets.
- Native Windows desktop command and installer assets.
- Local OIDC/JWKS validation.
- Local SDK package publishing artifacts.
- Webhook retry queue support.
- Audited crypto envelopes and encrypted sync bundles.
- Production vector database implementation.
- Semantic memory science metrics.
- OCR/vision/audio/video metadata production analysis path.
- Desktop and browser automation action runtimes.
- External/community agent marketplace records.
- End-to-end encrypted local envelopes.
- Advanced RBAC/federation policy hooks through policy and OIDC validation.
- Sandboxed plugin execution records.
- GPU planning, fine-tune and LoRA job metadata.
- Hardware security key and secure enclave metadata records.
- Proactive assistant jobs.
- Federated private aggregation.
- Multi-device encrypted sync.
