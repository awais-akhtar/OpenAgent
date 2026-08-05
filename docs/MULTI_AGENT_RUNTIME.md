# Multi-Agent Runtime

Stage 3 is implemented as a local-first multi-agent runtime.

## Implemented Features

- Built-in agent registry
- Agent permission grants
- Human approval checkpoints
- Agent-to-agent delegation
- Parallel agent execution
- Verification and reflection hooks
- Agent run audit logging
- Built-in agent marketplace manifests

## APIs

Checked agent run:

```bash
POST /api/agents/run-checked
{
  "agent": "planning",
  "goal": "Plan the next milestone",
  "project": "default"
}
```

Delegate:

```bash
POST /api/agents/delegate
{
  "from_agent": "executive",
  "to_agent": "research",
  "goal": "Find knowledge gaps",
  "project": "default"
}
```

Parallel run:

```bash
POST /api/agents/parallel
{
  "project": "default",
  "tasks": [
    { "agent": "planning", "goal": "Plan memory work" },
    { "agent": "security", "goal": "Review permissions" }
  ]
}
```

Permissions:

```bash
GET /api/agents/permissions?project=default
POST /api/agents/permissions/grant
```

Approvals:

```bash
GET /api/agents/approvals?project=default
POST /api/agents/approvals/request
POST /api/agents/approvals/resolve
```

Marketplace:

```bash
GET /api/agents/marketplace
```

## Notes

The runtime grants safe built-in permissions by default for local agents. Sensitive future actions should require explicit grants and human approval checkpoints before they cross the local data boundary or modify external systems.

