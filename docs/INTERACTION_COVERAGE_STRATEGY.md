# G17: Complete Interaction Coverage Strategy

## Current State

**Status**: 36/38 Python tests passing, select E2E tests running, full suite not executed in current audit cycle.

## Coverage by Screen

| Screen | Coverage | Tests | Notes |
|--------|----------|-------|-------|
| Home | Partial | E2E 5/5 (focused) | Chat handoff verified; full suite needs rerun |
| Chat | Partial | Python passing | Stream cancellation/error matrix incomplete |
| Agents | Partial | Python passing | SSE framing and future-event waiting incomplete |
| Research | High | Python passing | Owner access comprehensive; legacy attribution clear |
| Files | High | Python passing | Archive/restore verified; full CRUD confirmed |
| Automations | Partial | Python passing | Event reader snapshot only; cursor filtering incomplete |
| Calendar | High | Python passing | CRUD verified; autonomous delivery unchecked |
| Models | Partial | Mocked | Read/setup wired; actual downloads not tested |
| Settings | High | Unit tested | Preference persistence verified |
| CodeQ | Partial | Prior tests | Terminal polling safe; full preview matrix incomplete |
| Knowledge | Partial | Python passing | PUT affected by G21; archive contract clear |
| Voice | Partial | Compiler fixed | HTTP chunking verified; continuous socket unavailable |
| Assistant | High | Python passing | Task CRUD and activity verified |
| Communications | Partial | Prior tests | Draft lifecycle verified; sending requires approval |
| Contacts | Partial | Python passing | CRM CRUD wired; no fresh E2E today |
| Security | High | Python passing | Admin audit clear; cross-project semantics verified |
| Developer | Partial | Mocked | Read views wired; execution controls restricted |
| Science | High | Python passing | Bounded execution verified; physical devices absent |
| Personal OS | Partial | Routes broken | Create/list working; edit/delete route failures |
| Model Cookbook | Partial | Python passing | Plan creation wired; process termination unverified |
| Ultra Model | Partial | Unverified | Source-wired; runtime and Demo isolation untested |
| Personal Context | High | Python passing | Sources and consolidation verified |
| Vault | Partial | Routes failing | Access request CRUD not implemented |
| Web Scraper | Partial | Prior tests | Schedule cancellation verified; external sites untested |
| Fine-tuning | Partial | Mocked | CPU run-cpu wired; genuine training absent |
| Image Studio | Partial | Mocked | Label corrected; edit workflow unverified |

## Rerun Plan for Full Suite

### Step 1: Resolve Blockers
1. Fix G21 Personal OS PUT/DELETE (NameError)
2. Verify G20 Model Cookbook process stop
3. Confirm G15 Vault request routes

### Step 2: Component Test Activation
Run each workspace test independently:
```bash
# Research owner access (already passing)
.venv\Scripts\python.exe -m unittest tests.test_research_owner_access -v

# Calendar lifecycle
.venv\Scripts\python.exe -m unittest tests.test_calendar_routes -v

# Files archive/restore
.venv\Scripts\python.exe -m unittest tests.test_knowledge_routes -v

# Assistant task completion
.venv\Scripts\python.exe -m unittest tests.test_assistant_routes -v
```

### Step 3: Full Smoke Test
```bash
# Complete focused selection
.venv\Scripts\python.exe -m unittest \
  tests.test_c1_smoke \
  tests.test_c2_autonomous_notifications \
  tests.test_c3_external_providers \
  tests.test_c4_agent_event_stream \
  tests.test_c5_voice_websocket \
  tests.test_research_owner_access \
  -v
```

### Step 4: Desktop E2E
```bash
# Home, Chat, Agent screens (focused)
cd frontend-second
npm run test:e2e -- --workers=1 tests/e2e/home-chat-attachment.spec.ts
```

### Step 5: Full Desktop Suite (Future)
When compile errors and smoke failures fixed:
```bash
cd frontend-second
npm run test:e2e -- --workers=4 --retries=1
```

## Untested Triggers

Mark these as untested pending full rerun:

### Keyboard Interactions
- [ ] Escape key in dialogs
- [ ] Tab navigation order
- [ ] Enter to submit forms
- [ ] Ctrl+Z undo (voice/image)

### Error States
- [ ] Network timeout recovery
- [ ] Server error display
- [ ] Validation error messages
- [ ] Retry on transient failures

### Empty States
- [ ] No conversations
- [ ] No research results
- [ ] Empty file library
- [ ] No contacts

### Edge Cases
- [ ] Large datasets (>1000 items)
- [ ] Very long text
- [ ] Rapid successive clicks
- [ ] Browser back button

## Test Coverage Hierarchy

### P0: Core Workflows (Must Test)
1. Create resource (POST)
2. Read resource (GET)
3. Update resource (PATCH/PUT)
4. Delete resource (DELETE)
5. List with filters

### P1: User Flows (Should Test)
1. Multi-step workflows
2. Error recovery
3. Permission denials
4. Cross-user isolation

### P2: Edge Cases (Nice to Test)
1. Boundary values
2. Concurrent requests
3. Large data sets
4. Network errors

## Approval Criteria

Before accepting full suite as passing:
- [ ] All P0 workflows tested
- [ ] Permission boundaries verified
- [ ] Error messages appropriate
- [ ] No data leakage cross-user
- [ ] Performance acceptable (<2s per page load)

## Reference Commands

```bash
# Current Python count (36/38 passing)
.venv\Scripts\python.exe -m unittest discover tests -v 2>&1 | tail -20

# Focused selection (Research)
.venv\Scripts\python.exe -m unittest tests.test_research_owner_access tests.test_research_lifecycle -v

# Frontend unit tests
cd frontend-second && npm test -- --reporter=dot

# Prod build test
npm run build && npm run build:check
```

## Next Steps After Full Rerun

1. Identify remaining unverified paths
2. Add targeted tests for gaps
3. Document approved design divergences
4. Set baseline for continuous monitoring
