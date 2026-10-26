# Knowledge System Archive Policy

## Status

**Product Decision: Document archive is NOT implemented as a primary feature.**

Document-level archive/restore is intentionally unavailable in the current product.

## Current State

### What IS Available
- **Collection-level archive/restore**: Entire semantic collections can be archived and restored
- **Source deletion**: Individual indexed sources can be deleted
- **Metadata operations**: Source title, snippets, and metadata can be edited

### What is NOT Available
- **Document archive**: Individual documents cannot be archived (disabled)
- **Document restore**: Archived documents cannot be restored
- **Selective document recovery**: Only full collection recovery is possible

## Design Rationale

### Why Not Document Archive?

1. **Complexity**: Document-level archive adds persistence, retrieval, and search indexing complexity
2. **Unclear Semantics**: Documents are indexed sources - archive meaning is ambiguous
3. **Collection-Level Suffices**: Most archive/restore scenarios work at collection level
4. **Minimal User Demand**: Current product focuses on source management, not individual doc lifecycle

### When Would You Need It?

You might consider implementing document archive if:
- Users need to selectively recover deleted sources from a collection
- Fine-grained document lifecycle management becomes a requirement
- Regulatory compliance requires granular document retention

## Current Architecture

### Collection Archive/Restore (Available)
```python
# Archive entire collection
knowledge.archive_collection(collection_id, project)
# Returns: collection marked as archived, sources hidden from search

# Restore entire collection  
knowledge.restore_collection(collection_id, project)
# Returns: collection unhidden, sources re-indexed for search
```

### Source Operations (Available)
```python
# Delete source (permanent)
knowledge.delete_source(source_id, project)

# List sources in collection
sources = knowledge.list_sources(collection_id, project)

# Get source detail
source = knowledge.get_source(source_id, project)
```

## If You Need Document Archive

Should document archive become a requirement, here's the implementation path:

### 1. Database Schema Addition
```sql
ALTER TABLE knowledge_sources ADD COLUMN archived_at TEXT;
ALTER TABLE knowledge_sources ADD COLUMN archived_by TEXT;
ALTER TABLE knowledge_sources ADD COLUMN restore_lineage TEXT;
```

### 2. Archive Operation
```python
def archive_document(
    source_id: str,
    project: str,
    *,
    reason: str = "user-initiated",
) -> dict[str, Any]:
    """Archive an individual source document.
    
    Marks document as archived but retains full metadata and content
    for restoration.
    """
    # Validate ownership
    # Record archive metadata
    # Remove from active search/index
    # Return archive confirmation
```

### 3. Restore Operation
```python
def restore_document(
    source_id: str,
    project: str,
) -> dict[str, Any]:
    """Restore previously archived source document.
    
    Restores to original collection with full metadata intact.
    """
    # Validate archive status
    # Re-index for search
    # Restore search visibility
    # Return restore confirmation
```

### 4. Search/List Modifications
```python
# List archived sources in collection
sources = knowledge.list_sources(
    collection_id, 
    project, 
    include_archived=True
)

# Search excludes archived by default
results = knowledge.search(
    query, 
    project, 
    include_archived=False  # Default: only active docs
)
```

### 5. Test Coverage Needed
```python
class KnowledgeDocumentArchiveTests(TestCase):
    def test_archive_document_removes_from_search()
    def test_restore_document_returns_to_search()
    def test_archive_preserves_metadata()
    def test_cross_user_archive_denied()
    def test_list_sources_excludes_archived_by_default()
    def test_include_archived_parameter()
```

## Product Decision Alternatives

### Option A: Implement Document Archive (Rejected)
- Pros: Finer-grained control, regulatory compliance
- Cons: Complexity, code maintenance, UI space
- Status: Not current priority

### Option B: Collection-Only Archive (Current)
- Pros: Simpler model, covers 80% of use cases
- Cons: All-or-nothing recovery
- Status: Current implementation

### Option C: Soft Delete (Not Implemented)
- Pros: Data safety, recoverability
- Cons: Search performance impact
- Status: Intentionally avoided

## Recommendation

**Keep current collection-level design.** If users need document-level archive:
1. Implement it as an optional extension
2. Keep collection archive as the default
3. Clearly document both options
4. Measure user demand before adding complexity

## References

- [Knowledge Routes](../openagent/routes/knowledge_routes.py)
- [Knowledge Service](../openagent/knowledge.py)
- [Semantic Routes](../openagent/routes/semantic_routes.py)
- Related tests: `tests/test_knowledge_routes.py`
