# Enterprise and Team Deployment

Stage 7 provides local enterprise controls and deployment manifests.

## Local Features

- Role-based access control
- Multi-user records
- Multi-user workspace membership
- Organization memory
- Audit export to JSONL
- Audit events for enterprise admin operations
- SSO configuration records
- SSO HS256 token validation implementation
- Backup manifests and database snapshots
- Disaster recovery plans
- Policy management and evaluation
- Kubernetes deployment manifests

## APIs

Status:

```bash
GET /api/enterprise/status
```

Users, roles, and workspaces:

```bash
GET /api/enterprise/users
POST /api/enterprise/users
GET /api/enterprise/roles
POST /api/enterprise/roles
GET /api/enterprise/workspaces
POST /api/enterprise/workspaces
POST /api/enterprise/workspace/members
POST /api/enterprise/access/check
```

Organization memory:

```bash
GET /api/enterprise/org-memory?workspace_id=...
POST /api/enterprise/org-memory
```

Policies:

```bash
GET /api/enterprise/policies
POST /api/enterprise/policies
POST /api/enterprise/policies/evaluate
```

SSO configuration:

```bash
GET /api/enterprise/sso
POST /api/enterprise/sso
POST /api/enterprise/sso/validate
```

Audit and recovery:

```bash
POST /api/enterprise/audit/export
GET /api/enterprise/backups
POST /api/enterprise/backups
GET /api/enterprise/disaster-recovery
POST /api/enterprise/disaster-recovery
```

## Kubernetes

Kubernetes manifests are in `deploy/kubernetes`.

```bash
kubectl apply -k deploy/kubernetes
```

The default manifest deploys a single OpenAgent replica with a persistent volume, health probes, resource limits, a same-namespace network policy, and an hourly backup trigger. Production deployments should add TLS ingress, private registry configuration, secret management, retention automation, and off-node encrypted backup storage.

## Safety

The enterprise layer stores configuration locally. SSO validation supports a local HS256 JWT implementation with issuer, audience, signature, `nbf`, and `exp` checks. Stage 12 adds OIDC/JWKS validation records through the production runtime; provider-specific IdP claims, federation rules, and key-rotation policy remain deployment configuration. Audit exports and backups are written under the OpenAgent data directory.
