# Workspace and Personal Assistant Layer

Stage 10 adds local workspace objects, conversation management, assistant planning, and richer knowledge/workflow utilities.

## Workspace Objects

Supported workspace item kinds:

- `folder`
- `document`
- `note`
- `wiki_page`
- `dashboard`
- `research_canvas`
- `whiteboard`
- `mind_map`
- `template`

Create a wiki page:

```bash
curl -X POST http://127.0.0.1:8765/api/workspace/items \
  -H "Content-Type: application/json" \
  -d "{\"kind\":\"wiki_page\",\"title\":\"Project Wiki\",\"project\":\"core\",\"content\":\"Local project knowledge.\"}"
```

Search workspace items:

```bash
curl "http://127.0.0.1:8765/api/workspace/items?project=core&q=project"
```

Instantiate a template:

```bash
curl -X POST http://127.0.0.1:8765/api/workspace/templates/instantiate \
  -H "Content-Type: application/json" \
  -d "{\"template_id\":\"TEMPLATE_ID\",\"title\":\"Weekly Brief\",\"project\":\"core\",\"variables\":{\"name\":\"OpenAgent\"}}"
```

## Chat Management

OpenAgent now supports:

- Conversation search
- Conversation branching
- Local JSON export
- Local import
- Conversation replay

Branch a conversation:

```bash
curl -X POST http://127.0.0.1:8765/api/chat/branch \
  -H "Content-Type: application/json" \
  -d "{\"conversation_id\":\"CONVERSATION_ID\",\"at_message_id\":\"MESSAGE_ID\",\"title\":\"New branch\"}"
```

Export a conversation:

```bash
curl "http://127.0.0.1:8765/api/chat/export?conversation_id=CONVERSATION_ID"
```

## Personal Assistant

The local assistant layer supports:

- User preference and fact profiles
- Tasks
- Goals
- Daily brief
- Meeting brief
- Meeting summary
- Travel planning
- CRM relationship, lead, opportunity, and contact intelligence records
- Productivity coaching

Create a goal:

```bash
curl -X POST http://127.0.0.1:8765/api/assistant/goals \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Launch OpenAgent\",\"project\":\"core\",\"description\":\"Ship the local-first foundation.\"}"
```

Create a task:

```bash
curl -X POST http://127.0.0.1:8765/api/assistant/tasks \
  -H "Content-Type: application/json" \
  -d "{\"title\":\"Prepare release checklist\",\"project\":\"core\",\"priority\":\"high\"}"
```

Get the daily brief:

```bash
curl "http://127.0.0.1:8765/api/assistant/daily-brief?project=core"
```

## Personal OS Surface

OpenAgent also includes a local Personal OS surface for day-to-day assistant features that should not require external services:

- Prompt/model presets
- Reusable signatures
- Versioned drafts
- Notes with tags, status, and due dates
- Gallery metadata for local media or document references
- Blind response comparison and voting
- Local export/import bundle

Save a chat preset:

```bash
curl -X POST http://127.0.0.1:8765/api/personal-os/presets \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"name\":\"Research local\",\"kind\":\"research\",\"instructions\":\"Use local knowledge first.\",\"model\":\"phi4-mini\"}"
```

Create a versioned draft:

```bash
curl -X POST http://127.0.0.1:8765/api/personal-os/drafts \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"title\":\"Launch note\",\"content\":\"First draft\"}"
```

Create a note:

```bash
curl -X POST http://127.0.0.1:8765/api/personal-os/notes \
  -H "Content-Type: application/json" \
  -d "{\"project\":\"core\",\"title\":\"Prepare release\",\"body\":\"Check local model readiness\",\"tags\":[\"release\"],\"due_at\":\"2026-07-09\"}"
```

Export the local bundle:

```bash
curl "http://127.0.0.1:8765/api/personal-os/export?project=core"
```

Privacy boundary: Personal OS data is written only to the configured local SQLite database. Actions emit local audit events and do not call model providers or external networks.

## Knowledge Upgrades

Stage 10 adds:

- Automatic document tags
- Timeline generation
- Hybrid lexical/vector retrieval
- Source-attributed local synthesis

Run synthesis:

```bash
curl -X POST http://127.0.0.1:8765/api/knowledge/synthesize \
  -H "Content-Type: application/json" \
  -d "{\"query\":\"private AI memory\",\"project\":\"core\"}"
```

## Workflow Data Actions

New workflow actions:

- `database_query`
- `spreadsheet_read`
- `spreadsheet_write`
- `report_generate`
- `monitor_check`
- `alert`

Network actions are still blocked unless explicitly approved. Database actions are read-only and limited to `SELECT` queries.
