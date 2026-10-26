# Document Archive Policy for OpenAgent

## Overview
This document defines the contract for indexed-document archival in OpenAgent Knowledge management, addressing Gap G14.

## Scope
- **Product Context**: OpenAgent Knowledge module with document indexing
- **Current Status**: Semantic collection archive is implemented; indexed-document archive is NOT implemented
- **Distinction**: Indexed documents are separate from semantic vector collections
- **Status**: Product decision for this additional operation

## Principles

### 1. When NOT to Implement
Document archive should NOT be implemented unless ALL of the following are true:

✅ **Required Condition 1**: Canonical user control explicitly calls for it
  - The control must be in approved app design
  - Not inferred from backend capability availability
  - Must be explicitly in INTERACTION_STATE_CATALOG.md

✅ **Required Condition 2**: Clear user need
  - Workflow analysis shows users want to archive documents
  - Not just "because the backend can"
  - Workflow evidence in user research

✅ **Required Condition 3**: Distinct from collection archive
  - Cannot be implemented as general collection deletion
  - Must preserve retrieval exclusion semantics
  - Different from vector metadata handling

### 2. Indexed Document vs Collection Archive

#### Indexed Documents
- Individual documents stored in knowledge base
- Content is indexed for search
- Each document has metadata (author, date, source)
- Archive = remove from search index
- Restore = re-index the document

#### Semantic Collections
- Vector embeddings of documents
- Used for semantic search/RAG
- Already has archive/restore implementation
- Separate from indexed document lifecycle

### 3. Archive/Restore Lineage

If implemented, must define:

```
Archive Workflow:
  Document → (explicit user action) → Archived State
  - Metadata preserved
  - Content stored
  - Index removed
  - Search unavailable
  - Restore path exists

Restore Workflow:
  Archived Document → (explicit user action) → Active State
  - Re-index content
  - Search available
  - Metadata unchanged
  - Audit log shows restore
  - No data loss
```

## Implementation Status

### Current (G14 - Defined)
- ✅ Indexed document CRUD exists
- ✅ Semantic collection archive exists
- ✅ Document archive is NOT wired
- ✅ No canonical UI control exists
- ❌ Document archive implementation (by design)

### NOT Done - Reasons
1. **No Canonical Control**: App design doesn't call for document archiving
2. **Complexity**: Archive/restore requires careful indexing coordination
3. **Semantic Distinction**: Separate from collection archive
4. **Product Decision**: Waiting for explicit user workflow requirement

## Architecture

### Current Knowledge Structure
```
knowledge/
  ├── documents/        # Indexed documents
  │   └── {id}         # Individual document
  │       ├── content  # Raw/indexed content
  │       ├── metadata # Author, date, source
  │       └── vectors  # Embeddings
  │
  └── collections/      # Semantic collections
      └── {id}         # Collection of vectors
          ├── metadata # Collection info
          └── vectors  # Vector store
```

### If Archive Added (Hypothetical)
```
knowledge/
  ├── documents/
  │   ├── active/      # Currently indexed
  │   └── archived/    # Removed from index
  │       └── {id}     # Archived document
  │           ├── content
  │           ├── metadata
  │           └── archive_date
  │
  └── collections/
      ├── active/
      └── archived/
```

## Required Decisions Before Implementation

### Decision 1: Scope
- Archive individual documents? Documents and collections? Everything?
- What about embedded images/attachments?

### Decision 2: Retention
- Archived documents retained permanently?
- Auto-delete after time period?
- User-initiated deletion with confirmation?

### Decision 3: Search Behavior
- Search excludes archived? Includes with flag?
- Separate archived search interface?

### Decision 4: Permissions
- Who can archive? (owner, project admin, any user?)
- Who can restore? (same as archive?)
- Who can permanently delete?

## Testing Plan

If implemented, must include:

```python
def test_document_archive_excludes_from_search():
    """Archived documents not in normal search"""
    
def test_document_restore_re_indexes():
    """Restored documents reappear in search"""
    
def test_archive_preserves_metadata():
    """Archive doesn't lose document info"""
    
def test_cross_owner_archive_denied():
    """Owner A cannot archive Owner B's documents"""
    
def test_archive_history_logged():
    """Audit trail shows all archive/restore ops"""
```

## Related Policies

- **G11**: Legacy Attribution Policy (owner scoping)
- **G10**: Fact Provenance Policy (metadata preservation)
- **Collection Archive**: Existing semantic archive implementation

## References
- Gap ID: G14
- Status: Product decision required
- Implementation: openagent/knowledge.py
- Routes: openagent/routes/knowledge_routes.py
- Frontend: src/screens/WorkspaceScreens.tsx::KnowledgeScreen
