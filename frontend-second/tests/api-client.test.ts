import { afterEach, describe, expect, it, vi } from 'vitest';
import { apiGet, apiPost, apiStreamGet, apiStreamPost, getApiToken, setApiToken } from '../src/api/client';
import { useWorkspaceState } from '../src/state';

afterEach(() => {
  vi.restoreAllMocks();
  setApiToken('');
  useWorkspaceState.getState().setDataMode('live');
});

describe('OpenAgent API client', () => {
  it('uses the same-origin API path and legacy-compatible response shape', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ success: true, data: { status: 'ok' }, status: 'ok' }), { status: 200 }));
    const health = await apiGet('/api/health');
    expect(health.status).toBe('ok');
    expect(fetch).toHaveBeenCalledWith('/api/health', expect.objectContaining({ method: 'GET', credentials: 'same-origin' }));
    expect(new Headers(fetch.mock.calls[0][1]?.headers).get('X-Response-Envelope')).toBe('legacy');
  });

  it('adds an explicitly saved tab-scoped bearer token to backend requests', async () => {
    setApiToken('oa_test_token');
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ agents: [] }), { status: 200 }));
    await apiGet('/api/agents');
    expect(getApiToken()).toBe('oa_test_token');
    expect(new Headers(fetch.mock.calls[0][1]?.headers).get('Authorization')).toBe('Bearer oa_test_token');
  });

  it('sends JSON and dispatches real events from a backend SSE stream', async () => {
    setApiToken('oa_chat_stream_token');
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('data: {"type":"token","text":"hello"}\n\ndata: {"type":"done","done":true}\n\n', { status: 200, headers: { 'Content-Type': 'text/event-stream' } }));
    const events: unknown[] = [];
    await apiStreamPost('/api/chat/stream', { message: 'hello' }, event => events.push(event));
    expect(fetch).toHaveBeenCalledWith('/api/chat/stream', expect.objectContaining({ method: 'POST', body: JSON.stringify({ message: 'hello' }) }));
    const headers = new Headers(fetch.mock.calls[0][1]?.headers);
    expect(headers.get('Authorization')).toBe('Bearer oa_chat_stream_token');
    expect(headers.get('Accept')).toBe('text/event-stream');
    expect(events).toEqual([{ type: 'token', text: 'hello' }, { type: 'done', done: true }]);
  });

  it('uses the shared auth and mode guard for backend SSE GET streams', async () => {
    setApiToken('oa_stream_token');
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('data: {"type":"stream.closed","sequence":2}\n\n', { status: 200, headers: { 'Content-Type': 'text/event-stream' } }));
    const events: unknown[] = [];
    await apiStreamGet('/api/codebase/runs/r1/events/stream?after=1', event => events.push(event));
    expect(fetch).toHaveBeenCalledWith('/api/codebase/runs/r1/events/stream?after=1', expect.objectContaining({ method: 'GET', credentials: 'same-origin' }));
    const headers = new Headers(fetch.mock.calls[0][1]?.headers);
    expect(headers.get('Authorization')).toBe('Bearer oa_stream_token');
    expect(headers.get('Accept')).toBe('text/event-stream');
    expect(headers.get('X-Response-Envelope')).toBe('legacy');
    expect(events).toEqual([{ type: 'stream.closed', sequence: 2 }]);
  });

  it('blocks all backend HTTP access while local demo mode is selected', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    localStorage.setItem('oa-second-data-mode', 'demo');
    const fetch = vi.spyOn(globalThis, 'fetch');
    await expect(apiGet('/api/health')).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiStreamPost('/api/chat/stream', { message: 'example' }, () => undefined)).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiStreamGet('/api/research/sessions/s1/stream', () => undefined)).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    expect(fetch).not.toHaveBeenCalled();
    localStorage.setItem('oa-second-data-mode', 'live');
    useWorkspaceState.getState().setDataMode('live');
  });

  it('fails closed for JSON, form and SSE requests when Demo is only held in memory', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    localStorage.removeItem('oa-second-data-mode');
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new DOMException('Storage disabled', 'SecurityError'); });
    const fetch = vi.spyOn(globalThis, 'fetch');

    await expect(apiGet('/api/health')).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiPost('/api/settings', { theme: 'dark' })).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiPost('/api/files/upload', new FormData())).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiStreamPost('/api/chat/stream', { message: 'example' }, () => undefined)).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    await expect(apiStreamGet('/api/research/sessions/s1/stream', () => undefined)).rejects.toMatchObject({ code: 'demo_mode_request_blocked' });
    expect(fetch).not.toHaveBeenCalled();
  });
});
