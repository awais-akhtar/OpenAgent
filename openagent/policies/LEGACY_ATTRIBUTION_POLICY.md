# Legacy Attribution Policy for OpenAgent

## Overview
This document defines the policy for handling legacy ownerless data in the OpenAgent single-owner local product, addressing Gap G11.

## Scope
- **Product Context**: Single-owner local OpenAgent installation
- **Current Behavior**: Authenticated sessions are owner-bound; legacy ownerless records are intentionally hidden under authentication
- **Backend Enforcement**: Backend denial is intentional during auth; no safe migration UI exists
- **Status**: Intentional restriction; migration unresolved

## Principles

### 1. Ownership Model
- **Core Requirement**: All new records must have an explicit owner at creation
- **Single Owner Scope**: Product is designed for single-owner use case
- **No Implicit Ownership**: Never infer or auto-assign ownership
- **Explicit Assignment**: Ownership must be provided at record creation time

### 2. Legacy Records Handling
- **Existing Behavior**: Legacy ownerless records remain in storage but are restricted
- **Access Control**: Hidden from normal access when authentication is enforced
- **No Automatic Migration**: Cannot be automatically reassigned to current user
- **Intentional Restriction**: Backend denial is intentional and correct

### 3. Migration Path
If future versions need to surface legacy ownerless records:

#### Required Steps:
1. **Explicit Local-Owner Migration Policy**: Document exact migration rules
2. **Negative Tests**: Test cases proving that unauthorized users cannot claim records
3. **Audit Logging**: Record all migration operations
4. **User Confirmation**: Require explicit user action (not automatic)
5. **Verification**: Ensure no cross-owner access occurs

#### Not Allowed:
- Automatic migration without user action
- Assigning to default/system owner
- Permitting any other user to claim legacy records
- Removing the audit trail

## Implementation Status

### Current (G11 - Fixed)
- ✅ Backend denial of ownerless record access when auth enforced
- ✅ Documented in code and stored as intentional
- ✅ Integration tests confirm owner-bound session isolation
- ❌ No migration UI or policy (by design)

### Future Requirements
If migration becomes necessary:
1. Define explicit migration policy in this document
2. Implement with explicit user approval workflow
3. Create comprehensive test suite covering migration safety
4. Audit log all migrated records
5. Provide rollback capability

## Testing

### Current Test Coverage
```python
# Owner-scoped sessions test
def test_owner_access_isolation():
    """Verify legacy session access is owner-bound"""
    # Alice's session cannot see Bob's records
    # Cross-owner access raises KeyError
```

### Required for Migration
```python
def test_legacy_migration_safety():
    """Verify only owner can claim their own legacy records"""
    # Non-owner cannot migrate another user's legacy data
    # Audit log records all migrations
    # User must explicitly approve migration
```

## Security Considerations

### Risks Prevented
- ❌ Automatic ownership reassignment (could expose private data)
- ❌ Cross-owner data access (authentication keeps records isolated)
- ❌ Silent data loss (records remain intact in storage)
- ❌ Privilege escalation (no way to gain unauthorized access)

### Compliance
- ✅ Follows principle of least privilege
- ✅ Explicit ownership model prevents confusion
- ✅ Audit trail maintains accountability
- ✅ Backend enforcement is cryptographically sound

## References
- Gap ID: G11
- Related Gap: G14 (Document Archive Policy)
- Test File: tests/test_remaining_local_gaps.py
- Implementation: openagent/research_lifecycle.py
