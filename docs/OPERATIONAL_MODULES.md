# Operational Modules

This document covers the local runtime modules added for the deeper route,
provider, model, document, image, scheduler, terminal, and MCP work.

## Route Handler Extraction

- Handler registry: `GET /api/routes/handlers`
- Code: `openagent/routes/handlers/*`
- Boundary: the handler registry reports local route ownership metadata only.
  It does not execute actions or call providers.

## Model Cookbook

- Routes:
  - `GET /api/model-cookbook/hosts`
  - `POST /api/model-cookbook/hosts`
  - `GET /api/model-cookbook/cached`
  - `POST /api/model-cookbook/serve`
  - `POST /api/model-cookbook/adopt`
  - `POST /api/model-cookbook/stop/{id}`
  - `GET /api/model-cookbook/logs/{id}`
  - `GET /api/model-cookbook/hardware-fit`
- Boundary: cached model discovery reads local model folders. Serve sessions are
  planned by default; process launch requires an explicit `allow_process` flag
  and uses an allow-listed runner with `shell=False`.

## Deep Document Processing

- Routes:
  - `POST /api/documents/import-pdf`
  - `POST /api/documents/import`
  - `POST /api/document/{id}/extract-pdf-text`
  - `POST /api/document/{id}/render-preview`
  - `POST /api/documents/export-zip`
- Boundary: uploads, extracted text, previews, and ZIP exports stay inside local
  SQLite/editor storage and the local document-processing artifact directory.

## Image Studio

- Routes:
  - `POST /api/gallery/upload`
  - `GET /api/gallery/library`
  - `GET /api/gallery/{id}/operations`
  - `POST /api/gallery/{id}/rotate`
  - `POST /api/gallery/{id}/replace`
  - `POST /api/gallery/upscale`
- Boundary: image assets and derivatives stay under the local `image_studio`
  media directory. Operations are audit-recorded local artifact actions.

## Scheduler And Notifications

- Routes:
  - `GET /api/tasks`
  - `POST /api/tasks`
  - `PATCH /api/tasks/{id}`
  - `POST /api/tasks/{id}/pause`
  - `POST /api/tasks/{id}/resume`
  - `POST /api/tasks/run-due`
  - `GET /api/tasks/notifications`
  - `POST /api/tasks/notifications`
- Boundary: scheduled tasks, runs, and notifications are local SQLite records.
  The durable worker can process due tasks through the `tasks.run_due` job type.

## Safe Terminal Streaming

- Routes:
  - `GET /api/terminal/runs`
  - `GET /api/terminal/runs/{id}`
  - `POST /api/terminal/stream`
- Boundary: terminal execution uses `shell=False`, denies shell metacharacters,
  requires approval for non-default commands, enforces a workspace/data
  directory `cwd` boundary, redacts secret-like output, and stores logs locally.

## MCP Runtime

- Routes:
  - `GET /api/mcp/servers`
  - `POST /api/mcp/servers`
  - `POST /api/mcp/servers/{id}/reconnect`
  - `GET /api/mcp/tools`
  - `PATCH /api/mcp/tools/{id}`
  - `GET /api/mcp/permissions`
- Boundary: MCP server, tool, and permission records are local configuration
  records. The runtime does not launch untrusted MCP processes by default.

## Agent Bridge

- Routes:
  - `GET /api/agent-bridge/capabilities`
  - `GET /api/agent-bridge/tasks`
  - `POST /api/agent-bridge/tasks`
  - `GET /api/agent-bridge/memory`
  - `POST /api/agent-bridge/memory`
  - `GET /api/agent-bridge/calendar/events`
  - `POST /api/agent-bridge/calendar/events`
  - `GET /api/agent-bridge/documents`
  - `POST /api/agent-bridge/documents`
- Boundary: the bridge reads and writes local OpenAgent memory, calendar,
  document, and bridge-task records only. External agents should use this
  namespace instead of broad internal APIs.

## Verification

Current verification commands:

```powershell
python -m compileall openagent
node --check web\app.js
python -m unittest tests.test_model_cookbook tests.test_provider_certification tests.test_deep_operational_modules tests.test_route_handler_registry
python -m unittest discover -s tests
```
