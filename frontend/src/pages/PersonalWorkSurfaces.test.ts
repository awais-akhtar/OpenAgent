import { describe, expect, it } from 'vitest';
import { actionBody, projectScope, scopeReadPath } from './PersonalWorkSurfaces';
import { resourcePageConfigs } from './ResourcePages';

describe('personal AionUI work surfaces', () => {
  it('uses URL project scope before device preference and safely encodes it in resource reads', () => {
    expect(projectScope('http://localhost/?project=team%2Fone', 'other')).toBe('team/one');
    expect(projectScope('http://localhost/', 'other')).toBe('other');
    expect(scopeReadPath('/api/assistant/tasks?project={project}&limit=100', 'team/one')).toBe('/api/assistant/tasks?project=team%2Fone&limit=100');
  });
  it('builds only configured assistant, calendar and twin mutations', () => {
    const assistant = resourcePageConfigs.assistant.actions!;
    const task = assistant.find(a => a.label === 'Add a task')!;
    expect(task.path({})).toBe('/api/assistant/tasks');
    expect(actionBody(task, { title: 'Prepare', priority: '' })).toMatchObject({ title: 'Prepare', project: 'default' });
    const calendar = resourcePageConfigs.calendar.actions!.find(a => a.label === 'Create a calendar event')!;
    expect(actionBody(calendar, { title: 'Review', starts_at: '2026-10-20T10:00', ends_at: '2026-10-20T11:00' })).toMatchObject({ title: 'Review', project: 'default' });
    const twin = resourcePageConfigs.twin.actions!.find(a => a.label === 'Add a personal goal')!;
    expect(twin.path({})).toBe('/api/twin/goals');
  });
  it('keeps communication delivery approval gated and external sending disabled', () => {
    const actions = resourcePageConfigs.communications.actions!;
    expect(actions.find(a => a.label === 'Request send approval')?.path({})).toBe('/api/communication/send/request-approval');
    const send = actions.find(a => a.label === 'Send an approved draft')!;
    expect(send.confirmText).toBeTruthy();
    expect(actionBody(send, { draft_id: 'd1', approval_id: 'a1' })).toMatchObject({ draft_id: 'd1', approval_id: 'a1', allow_external: false });
    expect(actionBody(actions.find(a => a.label === 'Save a local draft')!, { recipients: 'a@example.test', body: 'Draft' })).toMatchObject({ project: 'default' });
  });
});
