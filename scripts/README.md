# Repository scripts

Scripts in this directory are maintainer entrypoints for bounded audits,
hardware smoke checks, and UI evidence capture. Product runtime code belongs
under `openagent/`, and frontend-specific build/capture tooling belongs under
`frontend-second/scripts/`.

Every new script should document its inputs, outputs, destructive behavior,
and external dependencies in its help text or module documentation.
