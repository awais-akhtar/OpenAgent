import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn(); const post = vi.fn(); const put = vi.fn(); const del = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post, put, delete: del },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => { const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : []; return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : []; },
}));

describe('Aion-style agent and workflow workspaces', () => {
  let host: HTMLDivElement; let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset(); post.mockReset().mockResolvedValue({ status: 'accepted' }); put.mockReset().mockResolvedValue({}); del.mockReset().mockResolvedValue({ deleted: true });
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/agents/marketplace')) return { agents: [{ id: 'market-planner', name: 'Planning assistant', status: 'available' }] };
      if (path.startsWith('/api/agents/approvals')) return { approvals: [{ id: 'approval-7', action: 'write file', status: 'pending', reason: 'Generate the requested report', project: 'default' }] };
      if (path.startsWith('/api/agents/permissions')) return { permissions: [{ id: 'perm-1', agent: 'research', scope: 'files:read', status: 'granted' }] };
      if (path.startsWith('/api/agents')) return { agents: [{ id: 'research', name: 'Research assistant', status: 'available', description: 'Local research support' }] };
      if (path.startsWith('/api/workflows/builder')) return { triggers: { manual: { label: 'Manual' }, scheduled: { label: 'Scheduled', fields: ['interval_seconds'] }, event: { label: 'Event', fields: ['event_name'] } }, actions: { run_agent: { label: 'Run Agent', category: 'agent' }, file_write: { label: 'Write File', category: 'file' } }, safety: { network_default: false } };
      if (path.startsWith('/api/workflow-runs')) return { runs: [{ id: 'run-1', workflow_id: 'wf-1', status: 'completed', created_at: '2026-10-13T12:00:00Z', completed_at: '2026-10-13T12:00:02Z', result: { dry_run: true, steps: [{ action: 'run_agent', status: 'planned', goal: 'Summarize', run: { id: 'run-agent-1' } }] } }] };
      if (path.startsWith('/api/workflows')) return { workflows: [{ id: 'wf-1', name: 'Summarize project', status: 'enabled', enabled: true, trigger_type: 'manual', trigger: { enabled: true, trigger_type: 'manual', config: {} }, description: 'Build a project summary', steps: [{ action: 'run_agent', goal: 'Summarize' }] }] };
      return {};
    });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

  it('stages approval decisions and submits them once after explicit confirmation', async () => {
    const { AgentsWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AgentsWorkspace)));
    expect(get).toHaveBeenCalledWith('/api/agents?project=default', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/agents/approvals?status=pending&project=default', expect.any(AbortSignal));
    expect(host.textContent).toContain('Research assistant');
    const approvals = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Approvals'));
    await act(async () => approvals?.click());
    expect(host.textContent).toContain('write file');
    const approve = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Approve'));
    await act(async () => approve?.click());
    expect(post).not.toHaveBeenCalledWith('/api/agents/approvals/resolve', expect.anything());
    const review = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Submit');
    await act(async () => review?.click());
    expect(post).not.toHaveBeenCalledWith('/api/agents/approvals/resolve', expect.anything());
    expect(document.body.textContent).toContain('Approve this pending request?');
    const confirm = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Confirm');
    await act(async () => confirm?.click());
    expect(post).toHaveBeenCalledWith('/api/agents/approvals/resolve', { approval_id: 'approval-7', approved: true });
    expect(post.mock.calls.filter(([path]) => path === '/api/agents/approvals/resolve')).toHaveLength(1);
  });

  it('uses builder-supported step metadata and confirms live workflow execution', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    expect(get).toHaveBeenCalledWith('/api/workflows?project=default', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/workflow-runs?limit=60&project=default', expect.any(AbortSignal));
    expect(host.textContent).toContain('Summarize project');
    expect(host.querySelector('.oa-ops-list-head')).toBeNull();
    expect(host.querySelector('.specialist-context')).toBeNull();
    const live = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Live run'));
    await act(async () => live?.click());
    const confirm = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Review live run');
    await act(async () => confirm?.click());
    expect(document.body.textContent).toContain('Start a live run?');
    const final = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Confirm run');
    await act(async () => final?.click());
    expect(post).toHaveBeenCalledWith('/api/workflows/run', { workflow_id: 'wf-1', dry_run: false, allow_network: false, project: 'default' });
  });

  it('shows server-backed run labels, timestamps, and step outcomes', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    const history = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Run history'));
    await act(async () => history?.click());
    expect(host.querySelector('h2')?.textContent).toBe('Summarize project');
    expect(host.textContent).toContain('Dry run');
    expect(host.textContent).toContain('Run Agent');
    expect(host.textContent).toContain('planned');
    expect(host.textContent).toContain('13 Oct 2026');
  });

  it('creates scheduled workflows with multiple ordered steps and real trigger configuration', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    const create = host.querySelector<HTMLButtonElement>('button[aria-label="New workflow"]');
    await act(async () => create?.click());
    const setInput = async (element: HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement, value: string) => act(async () => {
      const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(element), 'value')?.set;
      setter?.call(element, value); element.dispatchEvent(new Event('input', { bubbles: true })); element.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(document.querySelector('.oa-ops-dialog')).toBeNull();
    const named = document.querySelector<HTMLInputElement>('.oa-workflow-builder input[required]');
    expect(named).toBeTruthy(); if (named) await setInput(named, 'Scheduled report');
    const trigger = document.querySelector<HTMLSelectElement>('select[aria-label="Workflow trigger"]');
    expect(trigger).toBeTruthy(); if (trigger) await setInput(trigger, 'scheduled');
    const interval = document.querySelector<HTMLInputElement>('input[aria-label="Repeat every seconds"]');
    expect(interval).toBeTruthy(); if (interval) await setInput(interval, '900');
    const firstGoal = document.querySelector<HTMLInputElement>('input[aria-label="Step 1 Goal"]');
    expect(firstGoal).toBeTruthy(); if (firstGoal) await setInput(firstGoal, 'Summarize changes');
    const add = [...document.querySelectorAll('button')].find(button => button.textContent?.includes('Add step'));
    await act(async () => add?.click());
    const secondAction = document.querySelector<HTMLSelectElement>('select[aria-label="Step 2 action"]');
    expect(secondAction).toBeTruthy(); if (secondAction) await setInput(secondAction, 'file_write');
    const secondPath = document.querySelector<HTMLInputElement>('input[aria-label="Step 2 Project file path"]');
    const secondContent = document.querySelector<HTMLInputElement>('input[aria-label="Step 2 File content"]');
    expect(secondPath).toBeTruthy(); expect(secondContent).toBeTruthy();
    if (secondPath) await setInput(secondPath, 'reports/summary.md');
    if (secondContent) await setInput(secondContent, 'Scheduled summary');
    const submit = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Create workflow');
    await act(async () => submit?.click());
    expect(post).toHaveBeenCalledWith('/api/workflows', {
      name: 'Scheduled report', description: '', project: 'default', trigger_type: 'scheduled', trigger: { interval_seconds: 900 },
      steps: [{ action: 'run_agent', goal: 'Summarize changes' }, { action: 'file_write', path: 'reports/summary.md', content: 'Scheduled summary' }],
    });
  });

  it('uses workflow API enable and delete contracts only after explicit confirmation', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    const disable = [...host.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Disable');
    await act(async () => disable?.click());
    const review = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Save changes');
    await act(async () => review?.click());
    expect(put).not.toHaveBeenCalledWith('/api/workflows/wf-1', expect.objectContaining({ enabled: false }));
    const confirm = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Disable workflow');
    await act(async () => confirm?.click());
    expect(put).toHaveBeenCalledWith('/api/workflows/wf-1', { enabled: false, project: 'default' });
    const remove = [...host.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Delete');
    await act(async () => remove?.click());
    const reviewDelete = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Save changes');
    await act(async () => reviewDelete?.click());
    expect(del).not.toHaveBeenCalled();
    const finalDelete = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Delete workflow');
    await act(async () => finalDelete?.click());
    expect(del).toHaveBeenCalledWith('/api/workflows/wf-1');
  });

  it('validates workflow names, supported triggers, schedule bounds and required step inputs', async () => {
    const { validateWorkflowDraft, workflowStepPayload } = await import('./OperationalSpecialists');
    const actions = { run_agent: { label: 'Run Agent' }, file_write: { label: 'Write File' }, condition: { label: 'Condition' }, spreadsheet_write: { label: 'Write CSV Spreadsheet' } };
    const triggers = { manual: {}, scheduled: {}, event: {} };
    expect(validateWorkflowDraft('', 'manual', {}, [{ action: 'run_agent', values: { goal: 'Summarize' } }], actions, triggers)).toBe('Enter a workflow name.');
    expect(validateWorkflowDraft('Weekly', 'cron', {}, [{ action: 'run_agent', values: { goal: 'Summarize' } }], actions, triggers)).toBe('Choose a supported trigger.');
    expect(validateWorkflowDraft('Weekly', 'scheduled', { interval_seconds: '30' }, [{ action: 'run_agent', values: { goal: 'Summarize' } }], actions, triggers)).toContain('at least 60 seconds');
    expect(validateWorkflowDraft('Weekly', 'manual', {}, [{ action: 'file_write', values: { path: '', content: '' } }], actions, triggers)).toContain('project file path');
    expect(validateWorkflowDraft('Weekly', 'manual', {}, [{ action: 'run_agent', values: { goal: 'Summarize' } }], actions, triggers)).toBe('');
    expect(validateWorkflowDraft('Weekly', 'manual', {}, [{ action: 'condition', values: { path: 'steps.1.status', op: 'equals' } }], actions, triggers)).toContain('expected value');
    expect(validateWorkflowDraft('Weekly', 'manual', {}, [{ action: 'spreadsheet_write', values: { path: 'out.csv', csv: 'name,value\none,1' } }], actions, triggers)).toBe('');
    expect(workflowStepPayload({ action: 'condition', values: { path: 'steps.1.status', op: 'equals', value: 'completed' } })).toEqual({ action: 'condition', condition: { path: 'steps.1.status', op: 'equals', value: 'completed' } });
    expect(workflowStepPayload({ action: 'spreadsheet_write', values: { path: 'out.csv', csv: 'name,value\n"one, two",1' } })).toEqual({ action: 'spreadsheet_write', path: 'out.csv', rows: [{ name: 'one, two', value: '1' }] });
  });

  it('blocks workflow creation and explains a missing required step field', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    const create = host.querySelector<HTMLButtonElement>('button[aria-label="New workflow"]');
    await act(async () => create?.click());
    const name = document.querySelector<HTMLInputElement>('.oa-workflow-builder input[required]');
    await act(async () => { if (name) { const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set; setter?.call(name, 'My workflow'); name.dispatchEvent(new Event('input', { bubbles: true })); } });
    const form = document.querySelector<HTMLFormElement>('.oa-workflow-form');
    await act(async () => form?.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true })));
    expect(document.body.textContent).toContain('Enter goal for step 1.');
    expect(post).not.toHaveBeenCalledWith('/api/workflows', expect.anything());
  });
});
