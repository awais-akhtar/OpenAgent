export class ApiError extends Error { constructor(message: string, readonly status: number, readonly requestId?: string) { super(message); this.name = 'ApiError'; } }
export function record(value: unknown): Record<string, unknown> { return value && typeof value === 'object' ? value as Record<string, unknown> : {}; }
export function records(value: unknown, key?: string): Record<string, unknown>[] { const rows = Array.isArray(value) ? value : key && Array.isArray(record(value)[key]) ? record(value)[key] as unknown[] : []; return rows.filter(row => row && typeof row === 'object') as Record<string, unknown>[]; }
function message(data: unknown, fallback: string) { if (typeof data === 'string' && data.trim()) return data.trim().slice(0, 500); const root = record(data); const error = record(root.error); return typeof root.error === 'string' ? root.error : typeof error.message === 'string' ? error.message : typeof root.message === 'string' ? root.message : fallback; }
type ApiFunction = {
  <T>(path: string, init?: RequestInit): Promise<T>;
  get<T>(path: string, signal?: AbortSignal): Promise<T>;
  post<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T>;
  put<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T>;
  patch<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T>;
  delete<T>(path: string, signal?: AbortSignal): Promise<T>;
};
const SAFE_GET_TIMEOUT_MS = 45_000;
const RETRYABLE_STATUS = new Set([429, 502, 503, 504]);
const MAX_RETRY_AFTER_MS = 3_000;
function retryDelay(response: Response | undefined, attempt: number): number {
  const raw = response?.headers.get('retry-after');
  if (raw) {
    const seconds = Number(raw);
    const until = Number.isFinite(seconds) ? seconds * 1_000 : Date.parse(raw) - Date.now();
    if (Number.isFinite(until) && until > 0) return Math.min(until, MAX_RETRY_AFTER_MS);
  }
  return Math.min(250 * 2 ** attempt, MAX_RETRY_AFTER_MS);
}
const wait = (ms: number, signal: AbortSignal): Promise<void> => new Promise((resolve, reject) => {
  if (signal.aborted) return reject(signal.reason ?? new DOMException('Aborted', 'AbortError'));
  const timer = window.setTimeout(() => { signal.removeEventListener('abort', abort); resolve(); }, ms);
  const abort = () => { window.clearTimeout(timer); signal.removeEventListener('abort', abort); reject(signal.reason ?? new DOMException('Aborted', 'AbortError')); };
  signal.addEventListener('abort', abort, { once: true });
});
const send = async <T,>(path: string, init: RequestInit = {}): Promise<T> => {
  if (!path.startsWith('/api/') || path.startsWith('//') || /[\r\n]/.test(path)) throw new TypeError('Only same-origin API paths are allowed.');
  const headers = new Headers(init.headers); headers.set('Accept', headers.get('Accept') || 'application/json');
  try { const token = sessionStorage.getItem('oa-auth-token'); if (token && !headers.has('Authorization')) headers.set('Authorization', `Bearer ${token}`); } catch {}
  let body = init.body;
  if (body !== undefined && !(body instanceof FormData) && !(body instanceof Blob) && typeof body !== 'string') { headers.set('Content-Type', 'application/json'); body = JSON.stringify(body); }
  const method = (init.method || (body === undefined ? 'GET' : 'POST')).toUpperCase();
  const isSafeRead = method === 'GET' && body === undefined;
  const timeout = isSafeRead ? AbortSignal.timeout(SAFE_GET_TIMEOUT_MS) : undefined;
  const signal = init.signal && timeout ? AbortSignal.any([init.signal, timeout]) : timeout || init.signal;
  let response: Response | undefined;
  for (let attempt = 0; ; attempt += 1) {
    try {
      response = await fetch(path, { ...init, signal, headers, body, credentials: 'same-origin' });
    } catch (error) {
      if (!isSafeRead || attempt >= 1 || signal?.aborted) {
        if (timeout?.aborted && !init.signal?.aborted) throw new ApiError('The local service read timed out. Try again.', 408);
        throw error;
      }
      await wait(retryDelay(undefined, attempt), init.signal ?? new AbortController().signal);
      continue;
    }
    if (!isSafeRead || !RETRYABLE_STATUS.has(response.status) || attempt >= 1) break;
    await wait(retryDelay(response, attempt), init.signal ?? new AbortController().signal);
  }
  if (!response) throw new ApiError('The local service did not return a response.', 502);
  const data = response.status === 204 ? undefined : response.headers.get('content-type')?.includes('application/json') ? await response.json().catch(() => null) : await response.text();
  if (!response.ok) throw new ApiError(message(data, response.statusText || 'Local service error'), response.status, response.headers.get('x-request-id') || undefined);
  return data as T;
};
export const api: ApiFunction = Object.assign(send, {
  get: <T,>(path: string, signal?: AbortSignal) => send<T>(path, { signal }),
  post: <T,>(path: string, body?: unknown, signal?: AbortSignal) => send<T>(path, { method: 'POST', body: body as BodyInit, signal }),
  put: <T,>(path: string, body?: unknown, signal?: AbortSignal) => send<T>(path, { method: 'PUT', body: body as BodyInit, signal }),
  patch: <T,>(path: string, body?: unknown, signal?: AbortSignal) => send<T>(path, { method: 'PATCH', body: body as BodyInit, signal }),
  delete: <T,>(path: string, signal?: AbortSignal) => send<T>(path, { method: 'DELETE', signal }),
});
