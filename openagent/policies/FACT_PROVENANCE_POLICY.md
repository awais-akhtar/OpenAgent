# Fact Provenance and Source Linkage Policy

## Overview
This document defines the policy for tracking fact sources and lineage in the OpenAgent Digital Twin, addressing Gap G10.

## Scope
- **Product Context**: Personal OS Facts, Profile Data, and Inferred Preferences
- **Current Status**: Topic→preference association exists; arbitrary facts have no source linkage
- **Limited Attribution**: Only specified UI fields have source attribution wired
- **Status**: Documentation added; general fact provenance unavailable

## Principles

### 1. Fact Definition
A **fact** is any persistent user data:
```
- Profile fields (name, email, preferences)
- Inferred preferences (from behavior)
- Learning outcomes (from interactions)
- Personal context (from Twin)
- Consolidation results (from memory)
```

### 2. Provenance Definition
**Provenance** = Source + Timestamp + Authority
```
{
  "fact_id": "uuid",
  "fact_text": "User prefers async communication",
  "source_id": "uuid",
  "source_type": "user_input|inference|integration",
  "source_url": "optional",
  "created_at": "ISO timestamp",
  "created_by": "user|system|provider_id",
  "confidence": 0.0-1.0,  # For inferred facts
  "can_retract": true|false
}
```

### 3. Source Types

#### User Input
- User explicitly entered the fact
- Example: "I prefer email to Slack"
- Can be edited by user
- Can be deleted by user
- Authority: User

#### Inference
- Derived from user behavior
- Example: "Schedules meetings 2pm-4pm block"
- Cannot be directly edited
- Can be rejected/overridden
- Authority: System

#### Integration
- Learned from connected service
- Example: "Gmail timezone = America/New_York"
- Synced from external source
- Can be overridden locally
- Authority: Integration

## Current Implementation Status

### What Exists (G10 - Partial)
- ✅ Topic→preference association schema defined
- ✅ Limited UI fields have source attribution
- ✅ Digital Twin tracks source for some fields
- ✅ Documentation in code/docstrings

### What's Missing (G10 - Not Done)
- ❌ General fact/source field persistence
- ❌ Fact-level source linkage in database
- ❌ Edit/retract lineage implementation
- ❌ Retraction audit trail
- ❌ Source inference provenance (confidence scoring)

## Architecture

### If Full Provenance Added

#### Database Schema
```sql
CREATE TABLE user_facts (
  id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  project TEXT NOT NULL,
  category TEXT NOT NULL,  -- profile|preference|behavior|context
  fact_text TEXT NOT NULL,
  source_id TEXT NOT NULL REFERENCES fact_sources(id),
  created_at TEXT NOT NULL,
  created_by TEXT NOT NULL,
  confidence REAL DEFAULT 1.0,  -- 0.0-1.0 for inferred
  active BOOLEAN DEFAULT true,
  FOREIGN KEY (user_id, project) REFERENCES users(id, project)
);

CREATE TABLE fact_sources (
  id TEXT PRIMARY KEY,
  fact_id TEXT NOT NULL REFERENCES user_facts(id),
  source_type TEXT NOT NULL,  -- user_input|inference|integration
  source_name TEXT,  -- "Gmail sync", "User feedback", etc.
  source_url TEXT,   -- optional provider link
  created_at TEXT NOT NULL,
  metadata JSON,     -- provider-specific info
  FOREIGN KEY (fact_id) REFERENCES user_facts(id) ON DELETE CASCADE
);

CREATE TABLE fact_retractions (
  id TEXT PRIMARY KEY,
  fact_id TEXT NOT NULL REFERENCES user_facts(id),
  retracted_at TEXT NOT NULL,
  retracted_by TEXT NOT NULL,
  reason TEXT,
  restoration_path TEXT,  -- how to restore
  FOREIGN KEY (fact_id) REFERENCES user_facts(id) ON DELETE CASCADE
);
```

#### API Contract
```python
class FactProvenanceService:
    def record_fact(
        self,
        user_id: str,
        category: str,  # profile|preference|behavior|context
        fact_text: str,
        source_type: str,  # user_input|inference|integration
        source_name: str = "",
        confidence: float = 1.0,
        source_url: str = ""
    ) -> dict[str, Any]:
        """Record a fact with full provenance"""
        
    def get_fact_sources(
        self,
        fact_id: str,
        user_id: str
    ) -> list[dict[str, Any]]:
        """Get all sources for a fact"""
        
    def retract_fact(
        self,
        fact_id: str,
        user_id: str,
        reason: str = ""
    ) -> dict[str, Any]:
        """Retract a fact, recording why"""
        
    def restore_fact(
        self,
        fact_id: str,
        user_id: str
    ) -> dict[str, Any]:
        """Restore previously retracted fact"""
        
    def consolidate_with_provenance(
        self,
        facts: list[dict],
        consolidation_rule: str
    ) -> dict[str, Any]:
        """Consolidate facts preserving provenance"""
```

## Decision Tree: When to Add Provenance

### If User Asked for It
✅ Proceed with implementation

### If It's a Core Workflow
✅ Consider for core implementation

### If It's "Nice to Have"
Check ALL conditions:
1. **User Need**: Real workflow requires it (YES → implement)
2. **Privacy Risk**: Could expose private inference? (NO → safe to implement)
3. **Complexity**: Major implementation effort? (NO → implement)
4. **Scope Creep**: Changes Core Twin behavior? (NO → implement)

**Must Check**: Does NOT automatically need:
- ❌ Every inferred field having confidence scores
- ❌ Complete historical lineage for all facts
- ❌ Complex rollback mechanisms
- ❌ Granular source attribution UI

## Current Placement Decision: G10

### Status: NOT IMPLEMENTED BY DESIGN

### Rationale:
1. **Limited UI Scope**: Only specified fields need UI attribution
2. **Privacy First**: General provenance could expose inference reasoning
3. **Complexity**: Full provenance requires significant schema changes
4. **User Need Unclear**: No specific workflow requires complete provenance
5. **Integration Risk**: Could complicate learning/inference

### If Product Requires Full Provenance:

#### Steps Required:
1. **Product Approval**: Add to INTERACTION_STATE_CATALOG.md
2. **Privacy Review**: Ensure no sensitive inference exposed
3. **Database Migration**: Plan schema addition
4. **UI Design**: Approve source attribution display
5. **Implementation**: Add FactProvenanceService
6. **Testing**: Comprehensive test coverage

#### Test Requirements:
```python
def test_fact_provenance_recording():
    """Facts preserve source information"""
    
def test_inferred_facts_have_confidence():
    """System-inferred facts include confidence"""
    
def test_fact_retraction_creates_audit():
    """Fact retraction creates audit trail"""
    
def test_cross_user_fact_isolation():
    """User A cannot see User B's facts"""
    
def test_source_linkage_edit_chain():
    """Editing fact creates new version"""
```

## Security Considerations

### Risks of Full Provenance
- **Inference Exposure**: Could reveal how much system knows about user
- **Privacy Leak**: Source URLs might reveal third-party integrations
- **Data Inference**: Complete lineage could reveal learning patterns

### Mitigations
- Only expose sources user explicitly approved
- Hide confidence/probability for sensitive inferences
- Limit source URL exposure (normalize URLs)
- Require explicit user action for sensitive facts

## Related Policies

- **G11**: Legacy Attribution Policy (user ownership)
- **G14**: Document Archive Policy (metadata preservation)
- **Personal OS**: Personal record lifecycle

## Implementation Responsibility

- **Current Owner**: DigitalTwin.list_sources() has basic structure
- **Location**: openagent/digital_twin.py
- **Routes**: openagent/routes/twin_routes.py
- **Frontend**: src/screens/AdvancedScreen.tsx::TwinOperations/Sources

## References

- Gap ID: G10
- Status: Proposed work defined, not implemented
- Related Gaps: G11 (attribution), G14 (archive)
- Test: tests/test_all_19_gaps_completion.py::test_g10_fact_provenance_field_linkage
