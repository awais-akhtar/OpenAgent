# G23: Backend-to-Frontend Placement Review (B01-B09)

## Candidates Review

| ID | Capability | Current Routes | Recommendation |
|----|-----------|-----------------|-----------------|
| B01 | Personal OS authoring/import/compare | POST signatures, drafts, gallery, import, compare/* | Add to Personal OS contextual actions (with approval) |
| B02 | Gallery advanced edits | POST metadata, batch-tags, transform | Add to Image Studio after G19 verified |
| B03 | Contacts import | POST import, GET search | Add to Communications Contacts import dialog |
| B04 | MCP/skills lifecycle | /api/mcp/*, /api/skills/* | Keep restricted - no new general controls |
| B05 | Advanced document processing | /api/document/* processing operations, /api/rag/* | Keep specialized - no general UI |
| B06 | Scheduler task controls | scheduler routes | Calendar/Automations contextual diagnostics |
| B07 | Conversation organization | /api/conversations/* organization/branch/export | Chat history/detail menus |
| B08 | Model Cookbook adoption | serve/adopt/session contracts | Cookbook with explicit G20 fix |
| B09 | Enterprise/operator/internal | Enterprise/identity/production/worker routes | Keep restricted - admin only |

## Policy Decisions

### Don't Expose (Keep Internal)
- B04: MCP/skills command execution (security risk)
- B05: Advanced RAG operations (specialized)
- B09: Enterprise administration (single-owner not applicable)

### Add in Phase 2 (After Core Stabilizes)
- B02: Gallery advanced edits (after image edit verified)
- B03: Contacts import (clear workflow exists)
- B06: Scheduler diagnostics (bounded read-only)

### Already Wired or Not Applicable
- B01: Personal OS - add edit/import UI when ready
- B07: Chat organization - mostly wired
- B08: Model Cookbook - already covered

## Implementation Criteria

Before adding any B-candidate UI:
1. Backend routes tested and verified
2. Security scopes validated
3. Error handling complete
4. Approval workflow clear (if needed)
5. No unvetted external calls

## Recommendation

**Accept B01, B02, B03, B06, B07, B08 pending review.**
**Reject B04, B05, B09 as not user-facing.**
