# Privacy and Security Model

OpenAgent is privacy-first because the default behavior keeps user data under user control.

## Default Guarantees

- The local server binds to localhost unless configured otherwise.
- The current core does not call remote services.
- Data is stored locally.
- Agent outputs are generated inside the process.
- External providers must be explicit configuration, not defaults.

## Trust Boundaries

### Local Boundary

The user's device or private server. This is the default boundary.

### Self-Hosted Network Boundary

A private LAN, homelab, Kubernetes cluster, or enterprise deployment. Services inside this boundary can be trusted only according to deployment policy.

### External Provider Boundary

Any cloud model, API, email provider, messaging service, or hosted integration. Crossing this boundary requires explicit user configuration and clear disclosure.

## Permission Model

Every future action should describe:

- What data it reads
- What data it writes
- Which external systems it touches
- Whether it is reversible
- Whether it requires human approval
- What audit event it creates

## Audit Model

OpenAgent should record:

- Agent runs
- Tool calls
- Workflow executions
- Permission grants
- External data transfers
- Connector sync events
- Security-sensitive configuration changes

## Implemented Security and Privacy Controls

- Local encrypted secret vault implementation
- Secret reveal and rotation controls
- Data retention policy records and dry-run deletion
- Sharing grant records and revocation
- Egress approval records
- Audit-chain sealing and verification
- Privacy transparency reports
- Sandbox policy profiles

## Future Security Work

- Production audited encryption provider for secrets
- Encrypted memory vault
- Hardware security key support
- Sandboxed plugin runtime
- Zero-trust service boundaries
- Policy engine
- Production append-only audit anchoring
