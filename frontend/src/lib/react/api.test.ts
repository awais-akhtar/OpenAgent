import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiError, api, record, records } from './api';

afterEach(() => { vi.restoreAllMocks(); sessionStorage.clear(); });

describe('React same-origin API client', () => {
  it('sends credentialed same-origin JSON with the saved bearer token', async () => {
    sessionStorage.setItem('oa-auth-token', 'session-token');
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"ok":true}', { headers: { 'content-type': 'application/json' } }));
    await expect(api.post('/api/example', { value: 7 })).resolves.toEqual({ ok: true });
    const [path, init] = fetch.mock.calls[0];
    expect(path).toBe('/api/example');
    expect(init?.credentials).toBe('same-origin');
    expect(new Headers(init?.headers).get('authorization')).toBe('Bearer session-token');
    expect(new Headers(init?.headers).get('content-type')).toContain('application/json');
    expect(init?.body).toBe('{"value":7}');
  });

  it('rejects absolute, protocol-relative, and non-API paths before fetch', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    await expect(api.get('https://example.test/api/health')).rejects.toThrow(/same-origin/);
    await expect(api.get('//example.test/api/health')).rejects.toThrow(/same-origin/);
    await expect(api.get('/health')).rejects.toThrow(/same-origin/);
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([
    [401, { error: { message: 'Sign in required' } }, 'Sign in required'],
    [403, { error: 'Permission denied' }, 'Permission denied'],
    [429, { message: 'Rate limit reached' }, 'Rate limit reached'],
    [500, 'backend unavailable', 'backend unavailable'],
  ])('preserves HTTP %s status and useful response message', async (status, payload, expected) => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(typeof payload === 'string' ? payload : JSON.stringify(payload), {
      status: Number(status), statusText: 'Request failed',
      headers: { 'content-type': typeof payload === 'string' ? 'text/plain' : 'application/json', 'x-request-id': 'req-test' },
    }));
    await expect(api.get('/api/private')).rejects.toMatchObject({ name: 'ApiError', status, requestId: 'req-test', message: expected });
  });

  it('does not retry mutating requests and leaves abort semantics to fetch', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('{"error":"busy"}', { status: 503, headers: { 'content-type': 'application/json' } }));
    await expect(api.post('/api/action', {})).rejects.toBeInstanceOf(ApiError);
    expect(fetch).toHaveBeenCalledTimes(1);
    const controller = new AbortController(); controller.abort();
    fetch.mockRejectedValueOnce(new DOMException('Aborted', 'AbortError'));
    await expect(api.get('/api/slow', controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('retries one transient safe read, honors Retry-After, and never retries a final server error', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response('{"error":"busy"}', { status: 503, headers: { 'content-type': 'application/json', 'retry-after': '0' } }))
      .mockResolvedValueOnce(new Response('{"ok":true}', { headers: { 'content-type': 'application/json' } }));
    await expect(api.get('/api/read')).resolves.toEqual({ ok: true });
    expect(fetch).toHaveBeenCalledTimes(2);
    fetch.mockReset().mockResolvedValue(new Response('busy', { status: 503, headers: { 'content-type': 'text/plain', 'retry-after': '0' } }));
    await expect(api.get('/api/read')).rejects.toMatchObject({ status: 503 });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('normalizes only object records and named arrays', () => {
    expect(record(null)).toEqual({});
    expect(records({ rows: [{ id: 'a' }, null, 4] }, 'rows')).toEqual([{ id: 'a' }]);
    expect(records({ rows: [] }, 'other')).toEqual([]);
  });
});
