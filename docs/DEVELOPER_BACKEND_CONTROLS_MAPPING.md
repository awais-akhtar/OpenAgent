# Developer Backend Controls - UI Placement Mapping

## Status

**Gap G16: Backend controls exist but lack safe UI for some operations.**

This document maps backend developer operations to their UI placement and safety requirements.

## Backend Operations Inventory

Routes below are the ones `openagent.routes.route_contracts()` actually
declares. Operations with no declared route are listed separately as not
implemented, so this table cannot drift back into describing routes that do not
exist.

| Operation | Declared route | Current UI | Scope | Safety | Status |
|-----------|----------------|-----------|-------|--------|--------|
| API Key list | GET /api/developer/api-keys | Keys panel | developer:read | Safe | Route declared |
| API Key create | POST /api/developer/api-keys | Create button | developer:write | Safe | Route declared |
| API Key verify | POST /api/developer/api-keys/verify | (none) | developer:write | Safe | Route declared, no UI |
| Webhook list | GET /api/developer/webhooks | Webhooks panel | developer:read | Safe | Route declared |
| Webhook register | POST /api/developer/webhooks | Register dialog | developer:write | Requires URL | Route declared |
| Webhook update | PUT /api/developer/webhooks/{id} | (none) | developer:write | URL validation | Route declared, no UI |
| Webhook delete | DELETE /api/developer/webhooks/{id} | Delete action | developer:write | Safe | Route declared |
| Webhook dispatch | POST /api/developer/webhooks/dispatch | Test button | developer:write | Safe | Route declared |
| Webhook deliveries | GET /api/developer/webhook-deliveries | History panel | developer:read | Safe | Route declared |
| MCP server list | GET /api/mcp/servers | MCP panel | developer:read | Safe | Route declared |
| MCP server add | POST /api/mcp/servers | (none) | developer:admin | Requires validation | Route declared, no UI |
| MCP server reconnect | POST /api/mcp/servers/{id}/reconnect | (none) | developer:admin | Safe | Route declared, no UI |
| MCP tool list | GET /api/mcp/tools | MCP panel | developer:read | Safe | Route declared |
| MCP tool update | PATCH /api/mcp/tools/{id} | (none) | developer:admin | Requires validation | Route declared, no UI |
| MCP tool invoke | POST /api/mcp/tools/{id}/invoke | (none) | developer:admin | Requires approval | Route declared, restricted |
| MCP permissions | GET /api/mcp/permissions | (none) | developer:read | Safe | Route declared, no UI |

"Route declared" means the HTTP contract exists; it is not a claim that a
control has been exercised end to end.

### Not implemented (no declared route)

These were previously listed above as wired or partial, which was incorrect —
no such route is declared:

- API key revoke (`DELETE .../api-keys/{id}`) and rotation (`POST .../api-keys/{id}/rotate`)
- MCP enable / disable toggles
- MCP per-server environment variable editing

Deciding whether to implement these is open; see Implementation Priority below.

## UI Placement Plan

### Tier 1: Currently Wired (No Action)
- API key list/create
- Webhook list/register/delete/dispatch
- Webhook delivery history
- MCP server and tool listing

### Tier 2: Partial Wiring (Complete)

#### Webhook Update Endpoint
- **Route**: PUT /api/developer/webhooks/{id}
- **Parameters**: url, events, active, retry_policy
- **Safety**: URL must be validated against egress policy
- **UI**: Add "Edit webhook" dialog in webhooks panel
- **Tests**: URL validation, egress enforcement, schema validation

#### API Key Rotation
- **Route**: not implemented; would need a new declaration
- **Parameters**: (none)
- **Safety**: Safe, returns new key
- **UI**: Add "Rotate" button in keys list
- **Tests**: Verify old key invalidation, new key generation

### Tier 3: Requires Approval (Restrict)

#### MCP Execution Controls
- **Route**: POST /api/mcp/tools/{id}/invoke
- **Parameters**: command, arguments, timeout
- **Safety**: Very high - executes arbitrary code
- **UI**: Developer advanced section only, requires approval
- **Approval**: User must explicitly approve each command execution
- **Tests**: Command validation, timeout enforcement, error isolation

#### MCP Environment Variables
- **Route**: not implemented; deliberately undeclared
- **Parameters**: environment variables
- **Safety**: Critical - can pass secrets, affect all commands
- **UI**: **Do not expose** in current UI
- **Alternative**: Use credential management system
- **Future**: Admin-only environment management with audit logging

## Implementation Priority

### Phase 1 (Immediate)
1. Complete webhook update endpoint UI
2. Add API key rotation UI
3. Run full test suite for both

### Phase 2 (Medium-term)
1. Add MCP command execution approval workflow
2. Document MCP sandbox limitations
3. Comprehensive execution logging

### Phase 3 (Future)
1. Environment variable management system
2. Admin audit dashboard for privileged operations
3. Multi-user approval workflows

## Safety Requirements by Operation

### URL-Based Operations (Webhooks)
- Must validate against egress policy
- Must support localhost and private networks
- Must require HTTPS for external URLs
- Must log all external deliveries

### MCP Operations
- Commands limited by allowlist
- Environment isolated per MCP instance
- Execution timeout enforced
- Failures don't cascade to other MCPs
- No network access by default

### Credential Operations
- API keys never exposed in UI (only masked)
- Rotation invalidates old keys immediately
- Audit log for all key operations
- Separate storage for secrets

## Test Coverage Checklist

### Webhook Update
- [ ] Update URL to valid external domain
- [ ] Update URL to localhost (allowed)
- [ ] Update URL to private IP (allowed)
- [ ] Reject invalid URL format
- [ ] Validate event types
- [ ] Verify retry policy constraints
- [ ] Confirm egress policy enforcement
- [ ] Test cross-owner denial

### API Key Rotation
- [ ] Rotate generates new key
- [ ] Old key invalid after rotation
- [ ] New key has same scopes
- [ ] Audit event logged
- [ ] User cannot rotate others' keys

### MCP Execution
- [ ] Command execution succeeds
- [ ] Timeout enforced
- [ ] Invalid command rejected
- [ ] Failure doesn't affect other MCPs
- [ ] Output captured completely
- [ ] Environment properly isolated
- [ ] Requires approval (when implemented)

## Security Considerations

### What We're NOT Doing
- Exposing MCP environment editor to users
- Allowing arbitrary command execution
- Shared credentials across users
- Unrestricted outbound network access

### What We ARE Doing
- Approvals for high-impact operations
- Audit logging for all operations
- Scope-based access control
- Resource limits on execution

## Decision: Don't Expose These
1. **MCP environment variables** - Use credential system instead
2. **Direct command execution** - Use pre-validated workflows
3. **Network socket binding** - Keep restricted to internal calls

## References

- [Developer Routes](../openagent/routes/developer_routes.py)
- [MCP Runtime Routes](../openagent/routes/mcp_runtime_routes.py) (declares the `/api/mcp/*` group)
- [Related Tests](../tests/test_developer_security_routes.py)
