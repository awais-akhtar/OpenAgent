import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get },
  ApiError: class ApiError extends Error { constructor(message: string, readonly status: number) { super(message); } },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('Activity view', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset();
    get.mockImplementation(async (path: string) => {
      if (path === '/api/health') return { status: 'ok', version: '1.2' };
      if (path === '/api/system/metrics') return { cpu: { usage_percent: 18, count: 8 }, memory: { used_percent: 42, used_gb: 6, total_gb: 16 } };
      if (path.startsWith('/api/modules/health')) return { modules: [{ id: 'chat', status: 'ok' }] };
      if (path === '/api/inference/status') return { status: 'ready', model: 'local-small' };
      if (path === '/api/local-ai/status') return { hardware: { system: 'Linux', release: '6.1', processor: 'x86', cpu_count: 8 } };
      if (path.startsWith('/api/audit')) return { events: [{ id: 'audit-1', title: 'Workspace opened', resource_type: 'workspace', status: 'ok', created_at: '2026-10-13T10:00:00Z' }] };
      if (path.startsWith('/api/agents/approvals')) return { approvals: [{ id: 'approval-1', status: 'pending' }] };
      if (path.startsWith('/api/worker/jobs')) return { jobs: [{ id: 'job-1', kind: 'index', state: 'complete', updated_at: '2026-10-13T10:01:00Z' }] };
      return {};
    });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove();
  });

  it('loads the existing project-scoped status and activity contracts and renders semantic activity', async () => {
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'project one/α' })); });
    expect(get).toHaveBeenCalledWith('/api/health', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/system/metrics', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/modules/health?project=project%20one%2F%CE%B1', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/inference/status', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/local-ai/status', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/audit?project=project%20one%2F%CE%B1&limit=8', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/agents/approvals?project=project%20one%2F%CE%B1&status=pending', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/worker/jobs?project=project%20one%2F%CE%B1&limit=8', expect.any(AbortSignal));
    expect(host.querySelector('h1')?.textContent).toBe('Activity');
    expect(host.textContent).toContain('Workspace opened');
    expect(host.textContent).toContain('local-small');
    expect(host.textContent).toContain('Pending');
    expect(host.textContent).toContain('index');
    expect(host.querySelector('.dash-tile, .dash-summary, .dash-lede')).toBeNull();
  });

  it('keeps manual refresh and displays an empty event state without fabricated records', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/audit') ? { events: [] } : path.startsWith('/api/worker/jobs') ? { jobs: [] } : {});
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(host.textContent).toContain('No recent audit events.');
    expect(host.textContent).toContain('No worker runs for this project.');
    const initialReads = get.mock.calls.length;
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Refresh activity"]')?.click(); });
    expect(get.mock.calls.length).toBe(initialReads + 8);
  });

  it('shows an authorization recovery message for a denied health read', async () => {
    const { ApiError } = await import('../lib/react/api');
    get.mockImplementation(async (path: string) => {
      if (path === '/api/health') throw new ApiError('Forbidden', 403);
      return {};
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(host.textContent).toContain('Your account cannot read this workspace activity.');
    expect(host.querySelector('[role="alert"]')).toBeTruthy();
  });
});
