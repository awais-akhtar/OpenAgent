import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post },
  ApiError: class ApiError extends Error { constructor(message: string, readonly status: number) { super(message); } },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('task-focused Knowledge and Voice pages', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset().mockImplementation(async (path: string) => {
      if (path.startsWith('/api/knowledge/status')) return { documents: 2, chunks: 8, fts_available: true };
      if (path.startsWith('/api/knowledge/documents')) return { documents: [{ id: 'doc-1', title: 'Project brief', chunk_count: 4, source_path: 'C:/private/workspace/brief.md' }] };
      if (path.startsWith('/api/voice/status')) return { stt: { engines: [] } };
      if (path.startsWith('/api/voice/personas')) return { personas: [{ id: 'calm', name: 'Calm', language: 'en' }] };
      if (path.startsWith('/api/voice/sessions')) return { sessions: [] };
      return {};
    });
    post.mockReset().mockResolvedValue({ session: { id: 'voice-1' } });
    history.replaceState({}, '', '/?view=knowledge&project=focused-test');
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); history.replaceState({}, '', '/'); });

  it('keeps Knowledge search as the primary task and moves library import to its own view', async () => {
    post.mockReset()
      .mockResolvedValueOnce({ results: [{ id: 'result-1', title: 'First source', content: 'Matched content' }] })
      .mockRejectedValueOnce(new Error('Search unavailable'));
    const { KnowledgePage } = await import('./KnowledgePage');
    await act(async () => { root.render(createElement(KnowledgePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.knowledge-search-workspace')).toBeTruthy();
    expect(host.textContent).toContain('2 sources');
    expect(host.textContent).not.toContain('Project library');
    expect(host.querySelector('pre')).toBeNull();
    expect([...host.querySelectorAll('[role="tab"]')].map(tab => tab.textContent)).toEqual(['Search', 'Library', 'Graph', 'Timeline']);
    const query = host.querySelector<HTMLTextAreaElement>('.knowledge-query textarea');
    await act(async () => { if (query) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(query, 'deployment decisions'); query.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { host.querySelector<HTMLButtonElement>('.knowledge-query button.primary')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/knowledge/hybrid-search', { query: 'deployment decisions', project: 'focused-test', limit: 10 });
    expect(host.textContent).toContain('First source');
    expect(host.querySelector('[aria-label="Selected source details"]')).toBeTruthy();
    expect(host.textContent).toContain('Matched content');
    await act(async () => { if (query) { const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')?.set; setter?.call(query, 'different query'); query.dispatchEvent(new Event('input', { bubbles: true })); } });
    await act(async () => { host.querySelector<HTMLButtonElement>('.knowledge-query button.primary')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).not.toContain('First source');
    expect(host.textContent).toContain('Search unavailable');
    await act(async () => { host.querySelector<HTMLButtonElement>('[role="tab"][aria-controls="knowledge-panel-Library"]')?.click(); });
    expect(host.querySelector('input[type="file"]')).toBeTruthy();
  });

  it('shows useful source metadata without exposing a host filesystem path', async () => {
    const { KnowledgePage } = await import('./KnowledgePage');
    await act(async () => { root.render(createElement(KnowledgePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => host.querySelector<HTMLButtonElement>('[role="tab"][aria-controls="knowledge-panel-Library"]')?.click());
    const source = host.querySelector<HTMLButtonElement>('.knowledge-document');
    expect(source?.textContent).toContain('Project brief');
    await act(async () => source?.click());
    expect(host.textContent).toContain('Indexed passages');
    expect(host.textContent).toContain('Project brief');
    expect(host.textContent).not.toContain('C:/private/workspace/brief.md');
  });

  it('renders a permission-specific recovery state when the source index is forbidden', async () => {
    const { ApiError } = await import('../lib/react/api');
    get.mockImplementation(async (path: string) => path.startsWith('/api/knowledge/status') ? Promise.reject(new ApiError('Forbidden', 403)) : {});
    const { KnowledgePage } = await import('./KnowledgePage');
    await act(async () => { root.render(createElement(KnowledgePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('do not have permission');
  });

  it('does not show fabricated zero counts while the index status request fails', async () => {
    get.mockImplementation(async (path: string) => path.startsWith('/api/knowledge/status') ? Promise.reject(new Error('Status unavailable')) : {});
    const { KnowledgePage } = await import('./KnowledgePage');
    await act(async () => { root.render(createElement(KnowledgePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.knowledge-index-line')).toBeNull();
    expect(host.textContent).toContain('Status unavailable');
  });

  it('keeps Voice idle until explicit session/recording actions and hides persona creation in its disclosure', async () => {
    const { VoicePage } = await import('./VoicePage');
    await act(async () => { root.render(createElement(VoicePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.voice-workspace')).toBeTruthy();
    expect(host.textContent).toContain('Push to talk');
    expect(host.querySelector('.voice-persona-manager')?.hasAttribute('open')).toBe(false);
    expect(host.querySelector('.voice-session-meta')).toBeNull();
    expect(post).not.toHaveBeenCalledWith('/api/voice/sessions', expect.anything());
    const createSession = [...host.querySelectorAll('button')].find(button => button.textContent === 'New');
    await act(async () => { createSession?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(post).toHaveBeenCalledWith('/api/voice/sessions', expect.objectContaining({ project: 'focused-test', mode: 'push_to_talk', always_listening: false, metadata: { client: 'openagent-console', explicit_microphone_start: true } }));
  });

  it('keeps the newest selected Voice session when an older transcript request returns late', async () => {
    let releaseSlow: ((value: unknown) => void) | undefined;
    get.mockImplementation(async (path: string) => {
      if (path.startsWith('/api/voice/status')) return { stt: { engines: [] } };
      if (path.startsWith('/api/voice/personas')) return { personas: [] };
      if (path.startsWith('/api/voice/sessions')) return { sessions: [{ id: 'slow' }, { id: 'fast' }] };
      if (path.includes('session_id=slow')) return await new Promise(resolve => { releaseSlow = resolve; });
      if (path.includes('session_id=fast')) return { turns: [{ id: 'fast-turn', role: 'user', text: 'Fast transcript' }] };
      return {};
    });
    const { VoicePage } = await import('./VoicePage');
    await act(async () => { root.render(createElement(VoicePage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    const sessions = [...host.querySelectorAll<HTMLButtonElement>('.voice-sessions .list button')];
    await act(async () => { sessions[0]?.click(); });
    await act(async () => { sessions[1]?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Fast transcript');
    await act(async () => { releaseSlow?.({ turns: [{ id: 'slow-turn', role: 'user', text: 'Stale transcript' }] }); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Fast transcript');
    expect(host.textContent).not.toContain('Stale transcript');
    expect(host.querySelector('.voice-transcript')).toBeNull();
  });
});
