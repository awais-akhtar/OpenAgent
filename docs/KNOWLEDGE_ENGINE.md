# Knowledge Engine

Stage 2 is implemented as a local-first knowledge engine.

## Implemented Features

- Document and text ingestion
- Semantic chunking
- SQLite FTS search with fallback search
- Local vector adapter using deterministic hash embeddings
- Source attribution on chunks
- Entity extraction
- Relationship graph extraction
- Automatic document tags
- Timeline generation
- Hybrid lexical/vector retrieval
- Source-attributed local synthesis
- Memory compression summaries
- Context replay snapshots

## APIs

Ingest text:

```bash
POST /api/knowledge/ingest
{
  "title": "Project notes",
  "content": "OpenAgent stores private knowledge locally.",
  "project": "default"
}
```

Search:

```bash
GET /api/knowledge/search?project=default&q=private
```

Semantic search:

```bash
POST /api/knowledge/semantic-search
{
  "project": "default",
  "query": "private memory"
}
```

Hybrid search:

```bash
POST /api/knowledge/hybrid-search
{
  "project": "default",
  "query": "private memory"
}
```

Tags and timeline:

```bash
GET /api/knowledge/tags?project=default
GET /api/knowledge/timeline?project=default
```

Synthesize:

```bash
POST /api/knowledge/synthesize
{
  "project": "default",
  "query": "private memory"
}
```

Graph:

```bash
GET /api/knowledge/graph?project=default
```

Compress memory:

```bash
POST /api/knowledge/compress
{
  "project": "default"
}
```

Create replay:

```bash
POST /api/knowledge/replay
{
  "project": "default",
  "label": "Planning replay",
  "query": "roadmap"
}
```

## Notes

The vector adapter is intentionally local and deterministic. It provides a private retrieval interface now and can be swapped for stronger embedding backends later without changing the API boundary.
