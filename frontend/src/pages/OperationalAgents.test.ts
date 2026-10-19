import { describe, expect, it } from 'vitest';
import { agentSections, approvalContext, permissionContext } from './OperationalSpecialists';

describe('assistant registry composition', () => {
  it('keeps the roster and approval queue primary with marketplace and permissions under Manage', () => {
    expect(agentSections.primary).toEqual(['agents', 'approvals']);
    expect(agentSections.secondary).toEqual(['marketplace', 'permissions']);
  });

  it('surfaces approval requester, reason and project context', () => {
    expect(approvalContext({ approval_id: 'a1', action: 'Run an agent', requested_by: 'planner', reason: 'Read project docs', project: 'team' }, 'default')).toEqual({
      title: 'Run an agent', requester: 'planner', reason: 'Read project docs', project: 'team',
    });
    expect(approvalContext({ action: 'Review access' }, 'default').project).toBe('default');
  });

  it('makes assistant permission and scope explicit for the active project', () => {
    expect(permissionContext({ agent_name: 'Researcher', scope: 'files:read', project: 'lab' }, 'default')).toEqual({
      agent: 'Researcher', scope: 'files:read', project: 'lab',
    });
    expect(permissionContext({ scope: 'agents:delegate' }, 'default').project).toBe('default');
  });
});
