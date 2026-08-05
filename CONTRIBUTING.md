# Contributing to OpenAgent

OpenAgent is intended to become a large, modular, privacy-first system. Contributions should keep the local-first trust boundary clear and avoid introducing hidden external dependencies.

## Development Setup

```bash
python -m openagent --data-dir ./data --port 8765
python -m unittest discover -s tests
```

## Contribution Rules

- Keep features local-only unless the user has explicitly configured an external provider.
- Prefer open standards and small, inspectable modules.
- Add tests for behavior that touches storage, permissions, agents, connectors, or user data.
- Document new privacy or security assumptions.
- Avoid telemetry, hosted callbacks, or automatic cloud access.

## Architectural Expectations

New capabilities should fit one of the documented layers:

- Local core
- Context engine
- Agent runtime
- Tool and action layer
- Knowledge and retrieval
- Model gateway
- Connectors
- User interfaces

If a new feature crosses a trust boundary, include an approval path and an audit trail.

