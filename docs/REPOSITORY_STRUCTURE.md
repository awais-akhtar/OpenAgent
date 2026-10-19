# Repository structure and ownership

OpenAgent keeps editable product source, generated output, local runtime data,
and audit evidence in distinct ownership domains.

| Path | Ownership | Policy |
| --- | --- | --- |
| `openagent/` | Python product runtime | Canonical backend source. Domain modules expose routes through `openagent/routes/`. |
| `openagent/adaptive_engine/` | Local inference engine | Exact family adapters and shared lifecycle code. Unsupported combinations fail closed. |
| `frontend-second/` | Current React application | Canonical UI used by normal root npm commands. |
| `frontend/` | Legacy React application | Compatibility-only; change only when maintaining an explicit `*:legacy` command. |
| `tests/` | Product verification | Tests mirror public domains and use `tests/support/` for shared fixtures. |
| `docs/` | Maintained documentation | Architecture, operations, decisions, and small audit manifests. |
| `scripts/` | Repository tooling | Bounded maintenance, capture, and audit entrypoints. |
| `installers/`, `deploy/` | Distribution and operations | Installer and deployment definitions, not runtime state. |
| `web/` | Served web resources | Stable served resources only; hashed/generated bundles are ignored. |
| `.qa/` | Local QA workspace | Small reviewed evidence may be tracked; copied source trees and run output are ignored. |

## Root policy

The repository root is reserved for contributor entrypoints, packaging/build
configuration, policy files, and the active Phase 1 queue. Historical reports,
screenshots, and architecture notes belong under `docs/`.

## Generated and local data

Model weights, virtual environments, dependency directories, databases,
runtime state, build outputs, archives, screenshots, and temporary directories
are local artifacts. They are excluded through `.gitignore` and should be kept
outside the repository root when practical.

## Evidence policy

Large evidence packages are release artifacts. Track a small manifest that
records source revision, file hashes, environment, commands, results, and the
external artifact location. Never commit a complete copied frontend/backend
tree merely to prove what source was tested.
