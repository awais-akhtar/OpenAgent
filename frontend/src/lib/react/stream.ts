export type SseEvent = { event: string; data: string; id?: string };

function validatePath(path: string) {
  if (!path.startsWith('/api/') || path.startsWith('//') || /[\r\n]/.test(path)) throw new TypeError('Only same-origin API paths are allowed.');
}
function headers() {
  const result = new Headers({ Accept: 'text/event-stream' });
  try { const token = sessionStorage.getItem('oa-auth-token'); if (token) result.set('Authorization', `Bearer ${token}`); } catch { /* Storage is optional. */ }
  return result;
}
async function* readEvents(response: Response): AsyncGenerator<SseEvent> {
  if (!response.ok) throw new Error(`Stream request failed (${response.status}).`);
  if (!response.body) throw new Error('Streaming is unavailable in this browser.');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const parse = (frame: string): SseEvent | null => {
    let event = 'message'; let id: string | undefined; const data: string[] = [];
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('id:')) id = line.slice(3).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    return data.length ? { event, data: data.join('\n'), id } : null;
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let boundary: number;
      while ((boundary = buffer.search(/\r?\n\r?\n/)) >= 0) {
        const frame = buffer.slice(0, boundary); const separator = buffer.slice(boundary).match(/^\r?\n\r?\n/)?.[0] ?? '\n\n'; buffer = buffer.slice(boundary + separator.length);
        const event = parse(frame); if (event) yield event;
      }
      if (done) break;
    }
    if (buffer.trim()) { const event = parse(buffer); if (event) yield event; }
  } finally { reader.releaseLock(); }
}

export async function* getEvents(path: string, signal: AbortSignal): AsyncGenerator<SseEvent> {
  validatePath(path);
  const response = await fetch(path, { headers: headers(), credentials: 'same-origin', signal });
  yield* readEvents(response);
}

export async function* postEvents(path: string, payload: unknown, signal: AbortSignal): AsyncGenerator<SseEvent> {
  validatePath(path);
  const requestHeaders = headers(); requestHeaders.set('Content-Type', 'application/json');
  const response = await fetch(path, { method: 'POST', headers: requestHeaders, body: JSON.stringify(payload), credentials: 'same-origin', signal });
  yield* readEvents(response);
}
