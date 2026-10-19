# OpenAgent documentation

This directory is the canonical home for maintained project documentation.

## Navigation

- [Architecture](ARCHITECTURE.md) — system-level component boundaries.
- [Repository structure](REPOSITORY_STRUCTURE.md) — source and artifact ownership.
- [Adaptive execution engine](ADAPTIVE_EXECUTION_ENGINE.md) — engine overview.
- [Phase 1 audit](audits/phase1/README.md) — active closure status and evidence rules.
- [Release and implementation audits](audits/releases/README.md) — historical delivery records.
- [Architecture decisions](DECISIONS/) — durable design decisions.
- [Deployment](deployment.md) — operational setup.
- [Security](../SECURITY.md) — repository security policy.

Documentation that describes current behavior belongs directly under `docs/`
or in a named subject directory. Historical audit output belongs under
`docs/audits/`; images referenced by maintained documentation belong under
`docs/assets/`. Generated build output, temporary captures, model weights, and
package staging copies do not belong in the tracked documentation tree.
