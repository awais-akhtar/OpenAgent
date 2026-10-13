import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn(); const post = vi.fn(); const put = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post, put },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => { const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : []; return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : []; },
}));

describe('Aion-style agent and workflow workspaces', () => {
  let host: HTMLDivElement; let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset(); post.mockReset().mockResolvedValue({ status: 'accepted' }); put.mockReset().mockResolvedValue({});
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/agents/marketplace')) return { agents: [{ id: 'market-planner', name: 'Planning assistant', status: 'available' }] };
      if (path.startsWith('/api/agents/approvals')) return { approvals: [{ id: 'approval-7', action: 'write file', status: 'pending', reason: 'Generate the requested report', project: 'default' }] };
      if (path.startsWith('/api/agents/permissions')) return { permissions: [{ id: 'perm-1', agent: 'research', scope: 'files:read', status: 'granted' }] };
      if (path.startsWith('/api/agents')) return { agents: [{ id: 'research', name: 'Research assistant', status: 'available', description: 'Local research support' }] };
      if (path.startsWith('/api/workflows/builder')) return { triggers: { manual: { label: 'Manual' } }, actions: { run_agent: { label: 'Run Agent', category: 'agent' }, file_write: { label: 'Write File', category: 'file' } }, safety: { network_default: false } };
      if (path.startsWith('/api/workflow-runs')) return { runs: [{ id: 'run-1', workflow_id: 'wf-1', status: 'completed', created_at: '2026-10-13' }] };
      if (path.startsWith('/api/workflows')) return { workflows: [{ id: 'wf-1', name: 'Summarize project', status: 'enabled', description: 'Build a project summary', steps: [{ action: 'run_agent', goal: 'Summarize' }] }] };
      return {};
    });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

  it('loads project-scoped agent sources and approves an existing server approval', async () => {
    const { AgentsWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AgentsWorkspace)));
    expect(get).toHaveBeenCalledWith('/api/agents?project=default', expect.any(AbortSignal));
    expect(host.textContent).toContain('Research assistant');
    const approvals = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Approvals'));
    await act(async () => approvals?.click());
    expect(host.textContent).toContain('write file');
    const approve = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Approve'));
    await act(async () => approve?.click());
    expect(post).toHaveBeenCalledWith('/api/agents/approvals/resolve', { approval_id: 'approval-7', approved: true });
  });

  it('uses builder-supported step metadata and confirms live workflow execution', async () => {
    const { AutomationWorkspace } = await import('./OperationalSpecialists');
    await act(async () => root.render(createElement(AutomationWorkspace)));
    expect(get).toHaveBeenCalledWith('/api/workflows?project=default', expect.any(AbortSignal));
    expect(host.textContent).toContain('Summarize project');
    const live = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Live run'));
    await act(async () => live?.click());
    const confirm = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Review live run');
    await act(async () => confirm?.click());
    expect(document.body.textContent).toContain('Start a live workflow run?');
    const final = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Confirm run');
    await act(async () => final?.click());
    expect(post).toHaveBeenCalledWith('/api/workflows/run', { workflow_id: 'wf-1', dry_run: false, allow_network: false, project: 'default' });
  });
});
