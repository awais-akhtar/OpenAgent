# Agent System

OpenAgent agents are specialized workers that share the unified context engine and obey the same permission model.

## Built-In Agent Direction

- Executive Agent: planning, prioritization, delegation, and milestone management
- Research Agent: local and approved external research
- Coding Agent: repository understanding, implementation, tests, and review
- Writing Agent: drafting, editing, synthesis, and publication workflows
- Planning Agent: goals, tasks, schedules, and dependencies
- Automation Agent: workflow creation and execution
- Knowledge Agent: memory search, knowledge graph, and retrieval
- Voice Agent: speech, transcription, and real-time interaction
- Communication Agent: inbox, drafting, replies, and follow-ups
- Security Agent: permissions, risk review, audit, and policy
- Operations Agent: deployment, backups, health checks, and monitoring

## Runtime Contract

An agent should receive:

- Goal
- Project or workspace scope
- Relevant context
- Permission scope
- Tool availability
- Model routing policy

An agent should return:

- User-visible response
- Proposed next actions
- Confidence
- Sources or context references
- Audit metadata

## Starter Agents

The current repository includes a local implementation for every built-in role.
Each role can return a deterministic local result, and model-backed execution
is available when a configured gateway responds:

- `executive`
- `research`
- `coding`
- `writing`
- `planning`
- `automation`
- `knowledge`
- `voice`
- `communication`
- `security`
- `operations`

They prove the local agent lifecycle without requiring a model provider. A
model-backed run still follows the provider and external-network policy.

The shared runtime also records agent manifests, memory, verification hooks,
delegation, parallel work, approval checkpoints, tool scopes, and marketplace
metadata. These records do not by themselves make an agent autonomous or grant
it access to an external service.
