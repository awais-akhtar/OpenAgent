# Developer Backend Controls - UI Placement Mapping

## Status

**Gap G16: Backend controls exist but lack safe UI for some operations.**

This document maps backend developer operations to their UI placement and safety requirements.

## Backend Operations Inventory

| Operation | Current Route | Current UI | Scope | Safety | Status |
|-----------|---------------|-----------|-------|--------|--------|
| API Key list | GET /api/developer/keys | Keys panel | developer:read | Safe | Wired |
| API Key create | POST /api/developer/keys | Create button | developer:write | Safe | Wired |
| API Key revoke | DELETE /api/developer/keys/{id} | Revoke action | developer:write | Safe | Wired |
| API Key rotation | POST /api/developer/keys/{id}/rotate | Rotate button | developer:write | Safe | Partially wired |
| Webhook list | GET /api/developer/webhooks | Webhooks panel | developer:read | Safe | Wired |
| Webhook register | POST /api/developer/webhooks | Register dialog | developer:write | Requires URL | Wired |
| Webhook update | PATCH /api/developer/webhooks/{id} | (Missing) | developer:write | URL validation | **Not wired** |
| Webhook delete | DELETE /api/developer/webhooks/{id} | Delete action | developer:write | Safe | Wired |
| Webhook test | POST /api/developer/webhooks/{id}/test | Test button | developer:write | Safe | Wired |
| Webhook history | GET /api/developer/webhooks/{id}/delivery-history | History panel | developer:read | Safe | Wired |
| MCP list | GET /api/mcp-runtime/configs | MCP panel | developer:read | Safe | Wired (read-only) |
| MCP enable | POST /api/mcp-runtime/{id}/enable | (Limited) | developer:write | Requires validation | **Partial** |
| MCP disable | POST /api/mcp-runtime/{id}/disable | (Limited) | developer:write | Safe | Partial |
| MCP execute | POST /api/mcp-runtime/{id}/execute | (Limited) | developer:write | Requires approval | **Restricted** |
| MCP env vars | PATCH /api/mcp-runtime/{id}/environment | (Missing) | developer:admin | Secrets handling | **Not wired** |

## UI Placement Plan

### Tier 1: Currently Wired (No Action)
- API key CRUD operations
- Webhook list/register/delete/test
- Webhook delivery history
- MCP list and basic enable/disable

### Tier 2: Partial Wiring (Complete)

#### Webhook Update Endpoint
- **Route**: PATCH /api/developer/webhooks/{id}
- **Parameters**: url, events, active, retry_policy
- **Safety**: URL must be validated against egress policy
- **UI**: Add "Edit webhook" dialog in webhooks panel
- **Tests**: URL validation, egress enforcement, schema validation

#### API Key Rotation
- **Route**: POST /api/developer/keys/{id}/rotate
- **Parameters**: (none)
- **Safety**: Safe, returns new key
- **UI**: Add "Rotate" button in keys list
- **Tests**: Verify old key invalidation, new key generation

### Tier 3: Requires Approval (Restrict)

#### MCP Execution Controls
- **Route**: POST /api/mcp-runtime/{id}/execute
- **Parameters**: command, arguments, timeout
- **Safety**: Very high - executes arbitrary code
- **UI**: Developer advanced section only, requires approval
- **Approval**: User must explicitly approve each command execution
- **Tests**: Command validation, timeout enforcement, error isolation

#### MCP Environment Variables
- **Route**: PATCH /api/mcp-runtime/{id}/environment
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
- [MCP Runtime Routes](../openagent/routes/mcp_runtime_routes.py)
- [Related Tests](../tests/test_developer_security_routes.py)
