import { afterEach, describe, expect, it, vi } from 'vitest';
import { getEvents, postEvents } from './stream';

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); sessionStorage.clear(); });

async function collect(input: AsyncIterable<unknown>) { const events = []; for await (const item of input) events.push(item); return events; }

describe('React POST SSE client', () => {
  it('decodes split multibyte text, multiple frames and a final unterminated frame', async () => {
    const encoder = new TextEncoder();
    const bytes = encoder.encode('id: 4\nevent: chunk\ndata: caf\ndata: é\n\ndata: {"type":"done"}');
    const split = bytes.findIndex(byte => byte === 0xc3) + 1;
    const chunks = [bytes.slice(0, split), bytes.slice(split, split + 3), bytes.slice(split + 3)];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new ReadableStream({ pull(controller) { const next = chunks.shift(); if (next) controller.enqueue(next); else controller.close(); } }), { headers: { 'content-type': 'text/event-stream' } })));
    await expect(collect(postEvents('/api/chat/stream', { message: 'test' }, new AbortController().signal))).resolves.toEqual([
      { event: 'chunk', data: 'caf\né', id: '4' }, { event: 'message', data: '{"type":"done"}', id: undefined },
    ]);
  });

  it('sends JSON POST with the same-origin session token', async () => {
    sessionStorage.setItem('oa-auth-token', 'stream-token');
    const fetch = vi.fn().mockResolvedValue(new Response(new ReadableStream({ start(controller) { controller.close(); } })));
    vi.stubGlobal('fetch', fetch);
    await collect(postEvents('/api/chat/stream', { message: 'hello' }, new AbortController().signal));
    const [path, init] = fetch.mock.calls[0];
    expect(path).toBe('/api/chat/stream'); expect(init.method).toBe('POST'); expect(init.credentials).toBe('same-origin');
    expect(new Headers(init.headers).get('authorization')).toBe('Bearer stream-token');
    expect(init.body).toBe('{"message":"hello"}');
  });

  it('rejects non-API paths and unsuccessful server responses', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    await expect(collect(postEvents('https://example.test/api/chat/stream', {}, new AbortController().signal))).rejects.toThrow(/same-origin/);
    expect(fetch).not.toHaveBeenCalled();
    fetch.mockResolvedValueOnce(new Response('forbidden', { status: 403 }));
    await expect(collect(postEvents('/api/chat/stream', {}, new AbortController().signal))).rejects.toThrow('Stream request failed (403).');
  });

  it('propagates cancellation into fetch', async () => {
    const controller = new AbortController();
    const fetch = vi.fn().mockImplementation((_path, init) => new Promise((_resolve, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')), { once: true })));
    vi.stubGlobal('fetch', fetch);
    const pending = collect(postEvents('/api/chat/stream', {}, controller.signal));
    controller.abort();
    await expect(pending).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch.mock.calls[0][1].signal).toBe(controller.signal);
  });
});

describe('React GET SSE client', () => {
  it('uses a same-origin authenticated GET and parses streamed run events', async () => {
    sessionStorage.setItem('oa-auth-token', 'event-token');
    const fetch = vi.fn().mockResolvedValue(new Response('data: {"type":"agent.started","sequence":1}\n\ndata: {"type":"stream.closed","sequence":2}\n\n', { headers: { 'content-type': 'text/event-stream' } }));
    vi.stubGlobal('fetch', fetch);
    await expect(collect(getEvents('/api/codebase/runs/run-1/events/stream?after=0&timeout_seconds=15', new AbortController().signal))).resolves.toEqual([
      { event: 'message', data: '{"type":"agent.started","sequence":1}', id: undefined },
      { event: 'message', data: '{"type":"stream.closed","sequence":2}', id: undefined },
    ]);
    const [path, init] = fetch.mock.calls[0];
    expect(path).toContain('/api/codebase/runs/run-1/events/stream');
    expect(init.method).toBeUndefined(); expect(init.credentials).toBe('same-origin');
    expect(new Headers(init.headers).get('authorization')).toBe('Bearer event-token');
  });
});
