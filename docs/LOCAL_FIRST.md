# Local-First Data Model

OpenAgent starts from a simple rule: user data lives locally unless the user explicitly authorizes a different boundary.

## Core Entities

### Context Event

A context event is a durable record of something the system can remember:

- Note
- Conversation turn
- Document chunk
- Task update
- Calendar event
- Contact interaction
- Agent run
- Workflow event
- Connector sync

### Memory

Memory is a higher-level interpretation built from events. Future memory types include:

- Working memory
- Long-term memory
- Semantic memory
- Episodic memory
- Procedural memory
- Knowledge graph memory

### Project

A project scopes context, files, chats, tasks, agents, and workflows.

### Agent Run

An agent run records a goal, the selected agent, the response, timestamps, and future audit metadata.

## Storage Direction

The bootstrap uses SQLite tables for events and agent runs. Future storage can add:

- SQLite FTS for local text search
- Vector indexes for embeddings
- Graph storage for entities and relationships
- Encrypted vaults for sensitive memory
- Append-only audit logs

## Synchronization Direction

Future multi-device sync should be:

- Encrypted
- Conflict-aware
- User-controlled
- Able to operate peer-to-peer or through self-hosted infrastructure
- Transparent about what leaves each device

