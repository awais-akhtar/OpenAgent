# Decision 0001: Start With a Local-First Standard-Library Core

## Status

Accepted

## Context

OpenAgent's vision is broad: agents, memory, knowledge, automation, communication, local models, and enterprise deployment. The first repository version needs to be runnable and trustworthy without requiring cloud services or complex infrastructure.

## Decision

Start with a Python standard-library local core:

- `http.server` for the local API
- SQLite for durable local storage
- Static HTML, CSS, and JavaScript for the first workspace
- Deterministic starter agents
- `unittest` for regression tests

## Consequences

The project can run immediately with Python 3.11+ and no dependency installation. This keeps the initial trust boundary small.

The core can later add FastAPI, frontend frameworks, model runtimes, background workers, and vector databases when their interfaces are clearer.

