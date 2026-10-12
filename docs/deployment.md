# Deployment and release checks

OpenAgent ships a local Docker Compose profile and Kubernetes manifests for a
single OpenAgent runtime. Both start in `team` mode with authentication enabled.
The examples are a secure baseline, not a substitute for the identity, storage,
ingress, backup, and monitoring choices required by a specific installation.

## Container build

The Dockerfile builds the Python wheel in a separate stage and installs that
wheel into a clean runtime image. The runtime includes the `media` and `security`
extras, runs as UID/GID 10001, has a health check, and writes application data
under `/data`. It does not copy the source tree into the final image. The
`.dockerignore` file keeps local data, credentials, test output, Git history,
and development dependencies out of the build context.

```text
source + web assets
        |
        v
wheel builder
        |
        v
clean Python runtime
        |
        +-- non-root process
        +-- read-only root filesystem when deployed
        +-- /data persistent volume
        +-- /tmp bounded temporary volume
```

Run the local profile after setting a long bootstrap token:

```powershell
$env:OPENAGENT_BOOTSTRAP_TOKEN = "replace-with-a-long-random-value"
docker compose up --build
```

Compose publishes only `127.0.0.1:8765`. Exposing the service through a reverse
proxy is a separate operator decision and must retain team mode, authentication,
TLS, and an explicit trusted-origin list.
The local Compose service does not restart automatically after a reboot; start
it explicitly when needed. The image gives the non-root runtime ownership of
`/data` before a named volume is mounted there.

## Kubernetes profiles

`deploy/kubernetes` is the shared Kustomize base. Two overlays are available:

- `overlays/development` uses a locally built `openagent:development` image and
  keeps the same authenticated network posture.
- `overlays/production` selects the versioned `0.13.0` image. Release operators
  should replace the version tag with the registry digest produced by their
  trusted build before applying it, or enforce immutability for release tags in
  the registry.

Render the manifests before applying them:

```bash
kubectl kustomize deploy/kubernetes/overlays/production > openagent.yaml
kubeconform -strict -summary openagent.yaml
```

The base applies default-deny ingress and egress. It permits DNS for the runtime
and narrowly permits the suspended backup job to resolve and call the OpenAgent
service. External model, mail, calendar, research, and webhook traffic remains
blocked until an operator adds destination-specific egress policy.

The backup CronJob is suspended because a backup destination and restore policy
are deployment-specific, and this release does not yet expose a dedicated
long-lived backup service-credential flow. Its Secret entry is a complete HTTP
authorization header mounted as a file; the value is not placed in process
arguments. Enable the schedule only after the deployment has an accepted scoped
credential, durable backup storage, and a completed restore drill.

The supplied persistent volume uses `ReadWriteOnce` and the Deployment uses a
single replica with `Recreate`. This is appropriate for the SQLite-backed local
runtime. Horizontal replicas require the commercial shared-state/Postgres path;
adding an HPA to this SQLite profile would not be safe.

## CI acceptance gates

The main CI workflow separates failures by responsibility:

1. Python syntax and undefined-name checks.
2. Unit and integration tests on Python 3.11, 3.12, and 3.13.
3. Focused security, route, crypto migration, and schema migration coverage.
4. Playwright browser and accessibility smoke checks.
5. Wheel and source-distribution build, metadata validation, checksum creation,
   and installation into a clean virtual environment.
6. Docker build, non-root identity check, authenticated team-mode smoke test,
   Kubernetes render/schema checks, a high-severity container scan, and an SBOM.

The separate security workflow runs Python and npm dependency audits, dependency
review on pull requests, full-history secret scanning, and CodeQL for Python and
JavaScript. Findings at the configured gate severity fail the workflow; they are
not converted into warnings.

## PyPI release path

Tags matching `v<project-version>` start the publishing workflow. The verify job
runs the test suite, builds the wheel and source distribution once, checks their
metadata, writes SHA-256 checksums, and uploads one immutable Actions artifact.
The publish job downloads that artifact, verifies the checksums, records GitHub
build provenance, and sends the same files to PyPI through Trusted Publishing.
No long-lived PyPI token is stored in the workflow.

PyPI must have a Trusted Publisher configured for this repository, the
`publish-pypi.yml` workflow, and the `pypi` environment. The workflow will fail
closed if that relationship is missing.

## Local validation

The checks that do not require container or cluster tooling can run directly:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_config -v
.\.venv\Scripts\python.exe -m unittest tests.test_commercial_security tests.test_migrations -v
npm.cmd run test:e2e
```

Docker, kubectl, kubeconform, Trivy, and Syft are required for the corresponding
local deployment checks. CI runs those checks on Linux even when the development
machine does not have the tools installed.
