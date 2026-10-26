# Product Placement Decisions for Backend Operations

## Overview
This document defines which backend capabilities should have user-facing UI controls in OpenAgent, addressing Gap G23.

## Scope
- **Product Context**: Desktop-first single-owner OpenAgent
- **Evaluation**: 714/718 declared routes mapped to 9 backend operation groups
- **Current Status**: Selected operations wired; many remain backend-only
- **Philosophy**: Privilege + Scope + Approval = UI Placement

## Decision Framework

### When to Add UI for a Backend Operation

A backend operation MUST meet ALL criteria for UI placement:

✅ **Criterion 1: User Workflow**
- Is there a real user workflow that needs this?
- Can workflows be completed without it?
- Is it required for product functionality?

✅ **Criterion 2: Privilege Appropriate**
- Regular user can safely use this? (Yes → Consider for main UI)
- Advanced user only? (Yes → Consider for Advanced screen)
- Admin only? (Yes → Keep backend-only)
- Privileged system operation? (Yes → Keep backend-only)

✅ **Criterion 3: Safety Boundaries**
- Can the operation harm other users? (No)
- Can the operation leak secrets? (No)
- Can the operation corrupt data? (No)
- Needs explicit confirmation? (Yes, if destructive)

✅ **Criterion 4: Product Approval**
- Product owner has explicitly approved it
- Not inferred from "cool feature exists"
- Explicitly listed in INTERACTION_STATE_CATALOG.md

## Current Status by Operation Group

### Placed Operations (with UI)

#### Agents (11/19 routes)
- ✅ Start job
- ✅ View job status/history
- ✅ Cancel job
- ✅ Retry job
- ✅ View event stream
- Status: Core workflow, approved, all criteria met

#### Knowledge (14/20 routes)
- ✅ Create document
- ✅ View document
- ✅ Edit document
- ✅ Delete document (with confirmation)
- ✅ Archive collection (semantic only)
- Status: Core workflow, approved

#### Media (16/25 routes)
- ✅ Upload image
- ✅ View gallery
- ✅ Rotate image
- ✅ Scale image
- ✅ Undo operation
- Status: Core workflow, approved

#### Personal OS (16/23 routes)
- ✅ Create/edit/delete presets
- ✅ Create/edit/delete notes
- ✅ Create/edit/delete signatures
- ✅ Create/edit/delete drafts
- ✅ Draft versioning
- Status: Core workflow, approved

#### Vault (11/11 routes) ✓ Complete
- ✅ Create access request
- ✅ Edit pending request
- ✅ Approve request
- ✅ Deny request
- ✅ View access history
- Status: Security feature, fully placed, approved

### Partially Placed Operations

#### Calendar (19/24 routes)
- ✅ Create/edit/delete events
- ✅ View calendar
- ✅ Set recurrence/overrides
- ✅ Manual reminder delivery
- ❌ Autonomous delivery (G08)
- ❌ External OAuth sync
- Status: Core workflow mostly placed; autonomous/OAuth separate

#### Workflows (7/15 routes)
- ✅ Create workflow
- ✅ Run workflow
- ✅ View status/history
- ✅ Snapshot with cursor
- ❌ Advanced DAG operations
- ❌ External provider mutations
- Status: Core workflow placed; advanced separate

#### Chat (9/12 routes)
- ✅ Send message
- ✅ View conversation
- ✅ Create conversation
- ❌ Fine-grained conversation organization
- Status: Core placed; organization separate

### Backend-Only Operations (no UI)

#### Enterprise (14/14 routes) - Intentionally Hidden
- Reason: Team/SaaS features; not for single-owner product
- Decision: Keep restricted/internal
- Approval: Not needed; product architecture excludes

#### Production (34/34 routes) - Intentionally Hidden
- Reason: Operator/deployment only
- Decision: Keep backend admin tools
- Approval: System admin decision

#### Developer (15/23 routes) - Partially Visible
- MCP Runtime (0/7): Executable/env concerns; keep hidden
- Webhooks (2/2): Read/register placed; update withheld
- Developer tools: 15 visible from monitoring

#### Fine-Tuning (14/16 routes) - Mostly Hidden
- Reason: G07 - requires real model/runtime
- Current: CPU training UI exists; real training unverified
- Decision: Hide complex operations until model validated

## Gap-Specific Placements

### G23: Route Placement Decision Map

#### B01: Gallery Operations
**Status**: Partially placed
**Decision**: 
- Upload, rotate, scale, undo: ✅ PLACED
- Advanced edits, batch operations: ❌ NOT PLACED
**Reason**: Advanced metadata/batch need product approval
**Approval Required**: Before adding to main workflow

#### B02: Gallery Advanced Edits
**Status**: Backend exists, not placed
**Decision**: ❌ DO NOT ADD
**Reason**: 
- No canonical control in design
- Advanced image operations not in workflow
- Metadata/provenance story unclear
**Approval Required**: Product decision if needed

#### B03: Contacts Import
**Status**: Backend exists, not placed
**Decision**: ❌ CONSIDER FOR FUTURE
**Reason**:
- Import/merge has complex conflict resolution
- Import preview needed before implementation
- UI requires careful UX design
**Approval Required**: Before adding contact import feature

#### B04: MCP/Skills Lifecycle
**Status**: Backend exists, partially placed
**Decision**: ❌ KEEP DEVELOPER ADVANCED AREA
**Reason**:
- Executable/env concerns are privileged
- Could run arbitrary code
- Needs explicit per-operation approval
**Approval Required**: Security review required

#### B05: Advanced Document Processing/RAG
**Status**: Backend exists, selected only
**Decision**: ✅ PLACE ONLY APPROVED OPERATIONS
**Reason**:
- Files integration places ingestion/search
- RAG sync needs per-implementation approval
- Specialist operations separate
**Approval Required**: Per-operation in Files/Knowledge screens

#### B06: Scheduler Controls
**Status**: Backend exists, subset placed
**Decision**: ✅ CALENDAR/AUTOMATIONS SUBSET ONLY
**Reason**:
- Calendar reminders: ✅ PLACED (G08)
- Workflow scheduling: ✅ PLACED
- Advanced delivery history: ❌ ADMIN ONLY
**Approval Required**: For external delivery controls

#### B07: Conversation Organization
**Status**: Backend exists, core placed
**Decision**: ✅ CHAT MENUS ONLY
**Reason**:
- Core chat: ✅ PLACED
- Conversation history: ✅ PLACED
- Export/replay: ❌ NOT IN CURRENT WORKFLOW
**Approval Required**: Before adding export feature

#### B08: Model Cookbook Adoption
**Status**: Backend exists, not placed
**Decision**: ❌ KEEP BACKEND-ONLY UNTIL G20
**Reason**:
- G20: Real process lifecycle unverified
- Serving/adoption needs model validation
- Cannot verify without actual models
**Approval Required**: After G20 model verification

#### B09: Enterprise/Internal Routes
**Status**: 29 declarations, none placed
**Decision**: ❌ KEEP RESTRICTED/INTERNAL
**Reason**:
- Team/SaaS architecture not applicable
- Single-owner product doesn't need them
- Would expose unnecessary complexity
**Approval Required**: Not needed; by design

## Implementation Guidelines

### For Each New Operation Request

1. **Check This Document**: Is operation listed? What's the status?
2. **Verify Criteria**: Does it meet all four criteria?
3. **Check Approval**: Is it in INTERACTION_STATE_CATALOG.md?
4. **Implement Safely**: All boundaries enforced
5. **Test Security**: Owner/project scoping verified
6. **Update This Document**: Record decision and rationale

### Adding New UI for Operation

```markdown
1. List operation in relevant Bxx section
2. Document rationale for placement
3. Get product approval (update INTERACTION_STATE_CATALOG.md)
4. Implement with explicit scope enforcement
5. Add tests for boundary/permission cases
6. Update this policy document
```

## Security Principles

### Placement Restricts Unintended Access

- ✅ No UI = most users won't trigger it
- ✅ Backend permission enforced even with direct API access
- ✅ Internal/privileged routes never exposed
- ✅ All destructive operations need confirmation

### Don't Add UI Just Because

- ❌ Route exists != operation should be exposed
- ❌ Partial implementation != complete feature
- ❌ "Coverage" concern doesn't apply to single-owner product
- ❌ Developer convenience != user need

## References

- Gap ID: G23
- Status: Policy defined
- Related Gaps: G16 (Webhook), G08 (Scheduler), G20 (Cookbook)
- Implementation: App.tsx, AdvancedScreen.tsx
- Interaction State Catalog: INTERACTION_STATE_CATALOG.md
- Test: tests/test_all_19_gaps_completion.py::test_g23_backend_operations_product_placement
