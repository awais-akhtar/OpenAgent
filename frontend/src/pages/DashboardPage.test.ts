import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>(done => { resolve = done; });
  return { promise, resolve };
};
const snapshotValue = (path: string, marker: string) => {
  if (path === '/api/health') return { status: 'ok' };
  if (path.startsWith('/api/conversations/organized')) return { conversations: [{ id: marker, title: marker }] };
  if (path === '/api/audit' || path.startsWith('/api/audit?')) return { events: [{ id: marker, title: marker, status: 'ok' }] };
  if (path.startsWith('/api/worker/jobs')) return { jobs: [{ id: marker, kind: marker, state: 'done' }] };
  if (path === '/api/system/metrics') return { cpu: { usage_percent: 10 }, memory: { used_percent: 20 } };
  if (path.startsWith('/api/modules/health')) return { modules: [{ id: marker, status: 'ok' }] };
  if (path === '/api/local-ai/status') return { hardware: { system: marker } };
  if (path === '/api/inference/status') return { status: marker };
  return {};
};
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
      if (path.startsWith('/api/conversations/organized')) return { conversations: [{ id: 'conversation-1', title: 'Draft launch plan', updated_at: '2026-10-13T10:02:00Z' }] };
      if (path === '/api/agents') return { agents: [{ id: 'writer', name: 'Writer', description: 'Drafts documents' }] };
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
    expect(get).toHaveBeenCalledWith('/api/conversations/organized?project=project%20one%2F%CE%B1&limit=5&sort=pinned&include_archived=false&archived_only=false', expect.any(AbortSignal));
    expect(get).toHaveBeenCalledWith('/api/agents', expect.any(AbortSignal));
    expect(host.querySelector('h1')?.textContent).toBe('Home');
    expect(host.textContent).toContain('Draft launch plan');
    expect(host.textContent).toContain('Workspace opened');
    expect(host.textContent).toContain('local-small');
    expect(host.textContent).toContain('Pending');
      expect(host.querySelector('#home-today-title')?.textContent).toContain('Pending approvals');
    expect(host.querySelector('.home-rail-section')?.getAttribute('aria-labelledby')).toBe('home-today-title');
    expect(host.textContent).toContain('Writer');
    expect(host.textContent).toContain('index');
    expect(host.textContent!.indexOf('Recent activity')).toBeLessThan(host.textContent!.indexOf('System status'));
    expect(host.querySelector('.dash-tile, .dash-summary, .dash-lede')).toBeNull();
    expect(host.querySelector('.activity-details')?.hasAttribute('open')).toBe(false);
  });

  it('keeps manual refresh and displays an empty event state without fabricated records', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/audit') ? { events: [] } : path.startsWith('/api/worker/jobs') ? { jobs: [] } : {});
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(host.querySelector('#activity-events-title')).toBeNull();
    expect(host.querySelector('#activity-runs-title')).toBeNull();
    expect(host.textContent).not.toContain('No recent audit events.');
    expect(host.textContent).not.toContain('No worker runs for this project.');
    expect(host.querySelector('.activity-details')).toBeNull();
    const initialReads = get.mock.calls.length;
    await act(async () => { host.querySelector<HTMLButtonElement>('button[aria-label="Refresh activity"]')?.click(); });
    expect(get.mock.calls.length).toBe(initialReads + 10);
  });

  it('offers functional workspace shortcuts and opens a real recent conversation', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/conversations/organized')
      ? { conversations: [{ id: 'saved chat/α', title: 'Planning notes' }] }
      : {});
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'project one' })); });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.home-start h2')?.textContent).toBe('Welcome to OpenAgent');
    expect(host.querySelector('.home-start p')?.textContent).toBe('What would you like to work on?');
    expect(host.querySelector('#home-prompt-input')).toBeTruthy();
    expect(host.querySelector('.activity-details')).toBeNull();
    await act(async () => { host.querySelector<HTMLButtonElement>('.home-open-conversation')?.click(); });
    expect(new URL(location.href).searchParams.get('view')).toBe('chat');
    expect(new URL(location.href).searchParams.get('project')).toBe('project one');
    expect(new URL(location.href).searchParams.get('conversation_id')).toBe('saved chat/α');
  });

  it('does not present failed activity reads as a confirmed empty workspace', async () => {
    const { ApiError } = await import('../lib/react/api');
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/audit') || path.startsWith('/api/worker/jobs') || path.startsWith('/api/conversations/organized')) throw new ApiError('Unavailable', 503);
      return {};
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('Some workspace information could not be loaded.');
    expect(host.querySelector('.home-empty')).toBeNull();
  });

  it('shows an authorization recovery message for a denied health read', async () => {
    const { ApiError } = await import('../lib/react/api');
    get.mockImplementation(async (path: string) => {
      if (path === '/api/health') throw new ApiError('Forbidden', 403);
      return {};
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Your account cannot read this workspace activity.');
    expect(host.querySelector('[role="alert"]')).toBeTruthy();
  });

  it('translates API health states and omits unknown module placeholders', async () => {
    get.mockImplementation(async (path: string) => {
      if (path === '/api/health') return { status: 'ok' };
      if (path.startsWith('/api/modules/health')) return { modules: [
        { id: 'chat', status: 'ok' }, { id: 'pending-module', status: 'unknown' },
      ] };
      if (path.startsWith('/api/audit')) return { events: [] };
      if (path.startsWith('/api/worker/jobs')) return { jobs: [] };
      return {};
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(host.querySelector('.activity-state')?.textContent).toContain('Operational');
    await act(async () => { host.querySelector<HTMLElement>('.activity-details summary')?.click(); });
    expect(host.textContent).toContain('Available');
    expect(host.textContent).not.toContain('unknown');
    expect(host.textContent).not.toContain('pending-module');
  });

  it('humanizes audit event names, groups repeated activity, and hides opaque resource IDs', async () => {
    const opaqueId = '98f5b99e-0c39-4d80-90ce-9387342691dd';
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/audit')) return { events: [
        { id: 'a1', action: 'voice.persona.upsert', resource_type: 'voice_persona', resource_id: opaqueId, created_at: '2026-10-13T10:00:00Z', status: 'ok' },
        { id: 'a2', action: 'voice.persona.upsert', resource_type: 'voice_persona', resource_id: '1ac8d3c1-3970-4a9b-aa02-859c0e9b72f0', created_at: '2026-10-13T09:59:00Z', status: 'ok' },
        { id: 'a3', action: 'model.complete', resource_type: 'model', resource_id: opaqueId, created_at: '2026-10-13T09:58:00Z', status: 'success' },
      ] };
      if (path.startsWith('/api/worker/jobs')) return { jobs: [] };
      return {};
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    const rows = [...host.querySelectorAll('.activity-section[aria-labelledby="activity-events-title"] .activity-list li')];
    expect(rows).toHaveLength(2);
    expect(host.textContent).toContain('Voice persona updated');
    expect(host.textContent).toContain('Model completed');
    expect(host.textContent).toContain('2 similar events');
    expect(host.textContent).not.toContain('voice.persona.upsert');
    expect(host.textContent).not.toContain('98f5b99e-0c39-4d80-90ce-9387342691dd');
  });

  it('deduplicates interval and visibility refreshes while the initial snapshot is pending', async () => {
    vi.useFakeTimers();
    const priorVisibility = Object.getOwnPropertyDescriptor(document, 'visibilityState');
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'visible' });
    const pending = Array.from({ length: 10 }, () => deferred<unknown>());
    // Use call order, since the initial snapshot invokes all eight reads synchronously.
    let callIndex = 0;
    get.mockImplementation(() => pending[callIndex++ % 9].promise);
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'p' })); });
    expect(get).toHaveBeenCalledTimes(10);
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    await act(async () => { await vi.advanceTimersByTimeAsync(30_000); });
    expect(get).toHaveBeenCalledTimes(10);
    pending.forEach((item, index) => item.resolve(snapshotValue(String(get.mock.calls[index][0]), 'current')));
    await act(async () => { await Promise.resolve(); });
    Object.defineProperty(document, 'visibilityState', priorVisibility ?? { configurable: true, value: 'prerender' });
    vi.useRealTimers();
  });

  it('ignores a delayed response from the previous project after project scope changes', async () => {
    const oldReads = Array.from({ length: 10 }, () => deferred<unknown>());
    const newReads = Array.from({ length: 10 }, () => deferred<unknown>());
    let callIndex = 0;
    get.mockImplementation(() => {
      const index = callIndex++;
      return index < 10 ? oldReads[index].promise : newReads[index - 10].promise;
    });
    const { DashboardPage } = await import('./DashboardPage');
    await act(async () => { root.render(createElement(DashboardPage, { project: 'old' })); });
    await act(async () => { root.render(createElement(DashboardPage, { project: 'new' })); });
    newReads.forEach((item, index) => item.resolve(snapshotValue(String(get.mock.calls[index + 10][0]), 'new-project')));
    await act(async () => { await Promise.resolve(); });
    expect(host.textContent).toContain('new-project');
    oldReads.forEach((item, index) => item.resolve(snapshotValue(String(get.mock.calls[index][0]), 'old-project')));
    await act(async () => { await Promise.resolve(); });
    expect(host.textContent).toContain('new-project');
    expect(host.textContent).not.toContain('old-project');
  });
});
