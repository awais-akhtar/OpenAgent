# Research Legacy Attribution Policy

## Overview

This document defines the ownership and attribution policy for research sessions in OpenAgent, particularly for legacy records created before owner attribution was enforced.

## Current Status

As of October 26, 2026, the OpenAgent research system supports full owner attribution:
- All newly created research sessions are associated with an owner
- Owner verification is enforced on all access patterns
- Legacy records with missing owners are intentionally hidden from authenticated users

## Ownership Model

### Current Sessions (Post-Attribution)
- Created with explicit `owner` parameter
- Owner ID matches authenticated user in routes
- Required for all new sessions
- Enforced in routes.py before persistence

### Legacy Sessions (Pre-Attribution)
- Created with empty/NULL owner field
- Hidden from authenticated list operations
- Inaccessible via owner-filtered queries
- Preserved for audit/recovery purposes only

## Access Control

### Authenticated Access (Default)
When `OPENAGENT_AUTH_ENFORCED=1`:
```python
# Owned sessions - visible and accessible
lifecycle.list(project="default", owner="user-id")  # Returns [owned sessions]

# Legacy sessions - completely hidden
lifecycle.list(project="default", owner="")  # Returns only truly ownerless records
lifecycle.get(session_id, owner="any-user")  # 404 if session has no owner
```

### Unauthenticated Access (Development)
When `OPENAGENT_AUTH_ENFORCED=0` (single_user mode):
```python
# All sessions visible, including legacy
lifecycle.list(project="default")  # Returns all sessions
lifecycle.get(session_id)  # Returns session regardless of owner
```

## Intentional Design Decisions

### 1. No Automatic Migration
Legacy records are NOT automatically migrated to the current user.

**Rationale:**
- Prevents accidental data ownership changes
- Preserves audit trail of original creation
- Allows explicit, auditable migration if needed
- Maintains data integrity in production

### 2. Explicit Access Denial
Legacy records return 404 (not found) rather than 403 (forbidden).

**Rationale:**
- Consistent API behavior
- No information leak about legacy data existence
- Forces explicit migration decision at application level
- Simplifies permission model

### 3. Preserved Data Integrity
Legacy records are kept intact, not deleted.

**Rationale:**
- Enables future recovery/migration if policies change
- Supports audit requirements
- Allows ad-hoc queries when auth is disabled
- No permanent data loss

## Migration Scenarios

### Scenario 1: Development Mode Migration
If migrating from single_user mode to authenticated mode:

```python
# 1. Enumerate legacy records with auth disabled
legacy_records = lifecycle.list(project="default", owner="")

# 2. Review and approve ownership assignment
# (Manual decision: which user owns these records)

# 3. Enable auth and verify legacy records are hidden
# (No automatic migration happens)

# 4. (Optional) Create new owned copies if needed
for legacy in legacy_records:
    new_session = lifecycle.create(
        legacy["query"],
        project=legacy["project"],
        owner="assigned-user-id"
    )
    # Copy checkpoints and sources if needed
```

### Scenario 2: Single-User Desktop Mode
In single_user deployment mode:

```
OPENAGENT_DEPLOYMENT_MODE=single_user
OPENAGENT_AUTH_ENFORCED=0
```

Legacy records remain visible because:
- No distinction between users
- All records effectively belong to the single owner
- No authentication enforced
- Suitable for personal/desktop use

### Scenario 3: Team Mode with Mixed Records
In team/multi-user deployments:

```python
# Each user sees only their own sessions
alice_sessions = lifecycle.list(project="default", owner="alice-id")
bob_sessions = lifecycle.list(project="default", owner="bob-id")
# Neither alice nor bob can see legacy records

# Admin can query all legacy records only with auth disabled
# or through direct database access (not recommended)
```

## Testing

### Owner Verification Tests
See `tests/test_research_owner_access.py`:

```python
def test_enforced_research_reads_and_mutations_are_owner_bound():
    """
    Validates that:
    - Owned sessions are accessible only to owner
    - Legacy records with empty owner are hidden from authenticated users
    - Cross-user access attempts return 404
    - Owner filtering is additive (AND, not OR)
    """
```

### Key Test Cases
1. **Legacy Record Visibility**
   - Legacy records (owner="") not returned in authenticated list
   - Legacy records return 404 on direct access by any authenticated user
   - Legacy records accessible when auth is disabled

2. **Owner Binding**
   - Sessions created by user are only accessible by that user
   - Owner cannot be changed after creation
   - Branching preserves owner from parent

3. **Cross-User Denial**
   - Different users see different session lists
   - Cross-user mutations attempt 404
   - Cross-user reads return 404

### Test Execution
```bash
# Run owner access tests
.venv\Scripts\python.exe -m unittest tests.test_research_owner_access -v

# Run complete research test suite
.venv\Scripts\python.exe -m unittest tests.test_research_lifecycle -v
```

## Database Schema

Research sessions table includes owner field:
```sql
CREATE TABLE research_sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    owner TEXT,  -- NULL for legacy records
    query TEXT NOT NULL,
    status TEXT,
    created_at TEXT,
    updated_at TEXT,
    -- ... other fields
);
```

Key behaviors:
- `owner = NULL` or `owner = ''` identifies legacy records
- Index on (project, owner) for efficient filtering
- No constraints preventing NULL owner (backward compatibility)

## Route Implementation

Research routes enforce owner verification:
```python
# From research_routes.py
def get_session(session_id: str, owner: str):
    session = lifecycle.get(session_id, owner=owner)
    if session is None:
        raise HTTPError(404, "Session not found")
    return session
```

This ensures:
- All read operations filter by owner
- All write operations verify ownership first
- No legacy records accessible through authenticated routes

## Future Enhancement: Explicit Migration

Potential future feature (not currently implemented):

```python
# Hypothetical: Explicit migration endpoint
def migrate_legacy_session(session_id: str, new_owner: str):
    """
    Requires:
    - Admin scope
    - Audit logging
    - User consent (by email/notification)
    - Explicit session ID list
    
    Returns:
    - Success/failure status
    - Audit trail entry
    """
```

This would require:
1. Product decision on data ownership
2. User notification mechanism
3. Explicit approval workflow
4. Comprehensive audit logging
5. Rollback capability

## Backward Compatibility

### Single-User Mode
Legacy records remain fully accessible in single_user deployment.

### Upgrade Path
```
1. Running single_user mode (auth disabled)
   - Legacy records visible and functional
   
2. Upgrade to team/enterprise mode
   - Enable authentication
   - Legacy records become hidden
   - Existing functionality unchanged for owned sessions
   
3. (Optional) Explicit migration
   - Enumerate legacy records with auth disabled
   - Create owned copies
   - Update application references
```

## Compliance

This design aligns with:
- **Data Privacy**: No automatic data reassignment
- **Audit Requirements**: All ownership changes logged
- **Security**: Owner verification on all access
- **User Control**: Explicit decision required for legacy data

## Related Documentation

- [Auth and Permissions](./AUTH.md) - Authentication enforcement
- [Research Lifecycle](../openagent/research_lifecycle.py) - Implementation
- [Route Security](../openagent/routes/research_routes.py) - Route-level enforcement

## Questions and Clarifications

**Q: Will legacy records be deleted?**
A: No. Records are preserved to support future migration or recovery.

**Q: Can I access legacy records?**
A: Only when auth is disabled (single_user mode). This is intentional to prevent data leakage.

**Q: How do I migrate legacy records to owned sessions?**
A: Disable auth, enumerate legacy records, create new owned copies, verify, then enable auth.

**Q: Why not automatically assign to current user?**
A: This could cause data ownership issues and privacy violations. Explicit decision required.

**Q: Can admins override owner filtering?**
A: No. Owner filtering is enforced consistently. Use direct database access only for administration.
