import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
const getEvents = vi.fn();

vi.mock('../lib/react/api', () => ({
  api: { get, post },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key: string) => {
    const items = value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined;
    return Array.isArray(items) ? items : [];
  },
}));
vi.mock('../lib/react/stream', () => ({ getEvents }));

describe('ResearchPage API behavior', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset(); post.mockReset(); getEvents.mockReset();
    localStorage.setItem('oa-project', 'research-test');
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/research/status')) return { status: 'available' };
      if (path.startsWith('/api/research/sessions?')) return { sessions: [{ id: 'session-1', query: 'Find local evidence', status: 'running' }] };
      if (path.endsWith('/checkpoints')) return { checkpoints: [{ id: 8, stage: 'collect', progress: 35, message: 'Collected a source' }] };
      if (path.endsWith('/report')) return { report: '', sources: [{ id: 'source-1', title: 'A source', url: 'javascript:alert(1)', snippet: 'Evidence' }] };
      if (path.endsWith('/source-graph')) return { nodes: [{ id: 'n1' }], edges: [{ from: 'session-1', to: 'source-1', relation: 'uses' }] };
      if (path === '/api/research/sessions/session-1') return { session: { id: 'session-1', query: 'Find local evidence', status: 'running' } };
      return {};
    });
    getEvents.mockImplementation(async function* () { throw new Error('test disconnect'); });
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); localStorage.clear(); });

  it('loads actual session/checkpoint/report/graph contracts and resumes events after its persisted checkpoint cursor', async () => {
    const { ResearchPage } = await import('./ResearchPage');
    await act(async () => { root.render(createElement(ResearchPage)); });
    const sessionButton = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Find local evidence'));
    expect(sessionButton).toBeTruthy();
    await act(async () => { sessionButton?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/checkpoints');
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/report');
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/source-graph');
    expect(getEvents).toHaveBeenCalledWith('/api/research/sessions/session-1/stream?after_id=8&timeout=15', expect.any(AbortSignal));
    expect(host.textContent).toContain('Collected a source');
    expect(host.textContent).toContain('Evidence');
    expect(host.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(host.textContent).toContain('Reconnect to progress');
  });
});
