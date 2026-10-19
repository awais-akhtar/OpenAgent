import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
const getEvents = vi.fn();

vi.mock('../lib/react/api', () => ({
  api: { get, post },
  ApiError: class ApiError extends Error { constructor(message: string, readonly status: number) { super(message); } },
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
    await act(async () => { root.render(createElement(ResearchPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.research-workspace.has-session')).toBeTruthy();
    expect([...host.querySelectorAll('button')].filter(button => button.textContent?.includes('Archived')).length).toBe(1);
    expect(host.querySelectorAll('.research-workspace > section, .research-workspace > aside')).toHaveLength(2);
    expect(host.querySelector('.research-sources')).toBeTruthy();
    expect(host.querySelector<HTMLSelectElement>('[aria-label="Recent research"]')?.value).toBe('session-1');
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/checkpoints');
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/report');
    expect(get).toHaveBeenCalledWith('/api/research/sessions/session-1/source-graph');
    expect(getEvents).toHaveBeenCalledWith('/api/research/sessions/session-1/stream?after_id=8&timeout=15', expect.any(AbortSignal));
    expect(host.textContent).toContain('Evidence');
    await act(async () => host.querySelector<HTMLButtonElement>('.research-inspector-tabs [role="tab"][aria-selected="false"]')?.click());
    expect(host.querySelector('.research-activity-list')?.textContent).toContain('Collected a source');
    await act(async () => host.querySelector<HTMLButtonElement>('.research-inspector-tabs [role="tab"][aria-selected="false"]')?.click());
    expect([...host.querySelectorAll('details summary')].some(summary => summary.textContent?.includes('Provenance details'))).toBe(true);
    expect(host.querySelector('details.research-actions')?.textContent).toContain('Deleting removes it permanently.');
    expect(host.querySelector('a[href^="javascript:"]')).toBeNull();
    expect(host.textContent).toContain('Reconnect to progress');
  });

  it('uses the full investigation canvas when the project has no recent sessions', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/research/sessions?') ? { sessions: [] } : {});
    const { ResearchPage } = await import('./ResearchPage');
    await act(async () => { root.render(createElement(ResearchPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.research-workspace.no-history')).toBeTruthy();
    expect(host.querySelector('.research-sessions')).toBeNull();
    expect(host.querySelector('.research-main')).toBeTruthy();
    expect(host.textContent).not.toContain('No research sessions for this project.');
    expect(host.textContent).toContain('Archived');
    expect([...host.querySelectorAll('button')].some(button => button.textContent === 'Start research')).toBe(true);
    expect(host.textContent).toContain('When enabled, research may use sources outside this workspace.');
  });

  it('keeps the latest selected session when an earlier session detail request resolves last', async () => {
    let releaseOlder: ((value: unknown) => void) | undefined;
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/research/sessions?')) return { sessions: [{ id: 'older', query: 'Older request' }, { id: 'newer', query: 'Newer request' }] };
      if (path === '/api/research/sessions/older') return await new Promise(resolve => { releaseOlder = resolve; });
      if (path === '/api/research/sessions/newer') return { session: { id: 'newer', query: 'Newer request', status: 'completed' } };
      if (path.endsWith('/checkpoints')) return { checkpoints: [] };
      if (path.endsWith('/report')) return { report: 'Newer report' };
      if (path.endsWith('/source-graph')) return { nodes: [], edges: [] };
      return {};
    });
    const { ResearchPage } = await import('./ResearchPage');
    await act(async () => { root.render(createElement(ResearchPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    const historyPicker = host.querySelector<HTMLSelectElement>('[aria-label="Recent research"]')!;
    const setPicker = (value: string) => { const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value')?.set; setter?.call(historyPicker, value); historyPicker.dispatchEvent(new Event('change', { bubbles: true })); };
    await act(async () => { setPicker('older'); });
    await act(async () => { setPicker('newer'); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Newer report');
    await act(async () => { releaseOlder?.({ session: { id: 'older', query: 'Older request', status: 'completed' } }); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Newer report');
    expect(host.textContent).not.toContain('Older report');
  });

  it('renders the returned report as safe Markdown with validated citation links', async () => {
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/research/sessions?')) return { sessions: [{ id: 'done', query: 'Summarize local evidence', status: 'completed' }] };
      if (path === '/api/research/sessions/done') return { session: { id: 'done', query: 'Summarize local evidence', status: 'completed' } };
      if (path.endsWith('/checkpoints')) return { checkpoints: [] };
      if (path.endsWith('/report')) return { report: '# Findings\nA result with [an unsafe link](javascript:alert(1)) and [a citation](https://example.com/source).' };
      if (path.endsWith('/source-graph')) return { edges: [] };
      return {};
    });
    const { ResearchPage } = await import('./ResearchPage');
    await act(async () => { root.render(createElement(ResearchPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    const session = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Summarize local evidence'));
    await act(async () => { session?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.research-report h3')?.textContent).toBe('Findings');
    expect(host.querySelector('.research-report a[href^="javascript:"]')).toBeNull();
    expect(host.querySelector('.research-report a[href="https://example.com/source"]')?.getAttribute('rel')).toBe('noopener noreferrer');
  });
});
