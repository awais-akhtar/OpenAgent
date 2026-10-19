import { useWorkspaceState } from '../state';

export type JsonRecord = Record<string, unknown>;

const configuredOrigin = (import.meta.env.VITE_OPENAGENT_API_ORIGIN as string | undefined)?.trim().replace(/\/$/, '') ?? '';
const TOKEN_KEY = 'oa-second-api-token';

export class ApiError extends Error {
  readonly status: number;
  readonly code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

function assertLiveRequestsAllowed(method: string): void {
  // Demo/live isolation is a product boundary, not a development-only guard.
  // Keep this check in production builds too so an accidental adapter fallback
  // cannot send fixture-mode actions to the real Python backend.
  // The Zustand mode is authoritative for the current session. Persistence is
  // best-effort, so a blocked localStorage must not turn an in-memory Demo
  // selection into permission to contact the backend. The persisted value is
  // also checked fail-closed in case it could not be cleared while switching.
  let persistedDemo = false;
  try { persistedDemo = localStorage.getItem('oa-second-data-mode') === 'demo'; } catch { /* in-memory mode remains authoritative */ }
  const dataMode = useWorkspaceState.getState().dataMode;
  if (import.meta.env.PROD) {
    // Production is Live-only. Ignore any stale persisted development value;
    // the state initializer and setter both enforce Live in production.
    if (dataMode === 'demo') {
      throw new ApiError(`This ${method.toUpperCase()} request is unavailable in local demo mode. Switch to Live mode to use the backend.`, 409, 'demo_mode_request_blocked');
    }
    return;
  }
  if (dataMode === 'demo' || persistedDemo) {
    throw new ApiError(`This ${method.toUpperCase()} request is unavailable in local demo mode. Switch to Live mode to use the backend.`, 409, 'demo_mode_request_blocked');
  }
}

export function apiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const normalized = path.startsWith('/') ? path : `/${path}`;
  return `${configuredOrigin}${normalized}`;
}

export function getApiToken(): string {
  try { return sessionStorage.getItem(TOKEN_KEY) ?? ''; } catch { return ''; }
}

export function setApiToken(token: string): void {
  try {
    if (token.trim()) sessionStorage.setItem(TOKEN_KEY, token.trim());
    else sessionStorage.removeItem(TOKEN_KEY);
  } catch { /* API requests can still use unauthenticated local mode. */ }
  window.dispatchEvent(new CustomEvent('openagent-api-credentials-changed'));
}

function requestHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (!headers.has('Accept')) headers.set('Accept', 'application/json');
  headers.set('X-Response-Envelope', 'legacy');
  const token = getApiToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  return headers;
}

async function readJson(response: Response): Promise<JsonRecord> {
  let payload: unknown;
  try { payload = await response.json(); }
  catch { throw new ApiError(`The backend returned an unreadable response (${response.status}).`, response.status); }
  const data = payload && typeof payload === 'object' ? payload as JsonRecord : {};
  if (!response.ok) {
    const error = data.error;
    const message = typeof error === 'string'
      ? error
      : error && typeof error === 'object' && 'message' in error
        ? String((error as JsonRecord).message)
        : `Request failed (${response.status}).`;
    const code = error && typeof error === 'object' && 'code' in error ? String((error as JsonRecord).code) : undefined;
    throw new ApiError(message, response.status, code);
  }
  // Also tolerate the v2 envelope when deployed with an older proxy or host.
  if (data.success === true && data.data && typeof data.data === 'object') {
    return { ...(data.data as JsonRecord), ...Object.fromEntries(Object.entries(data).filter(([key]) => !['data', 'success', 'error', 'meta'].includes(key))) };
  }
  return data;
}

export async function apiRequest<T extends JsonRecord = JsonRecord>(
  path: string,
  options: { method?: string; body?: unknown; signal?: AbortSignal; headers?: HeadersInit } = {},
): Promise<T> {
  assertLiveRequestsAllowed(options.method ?? 'GET');
  const headers = requestHeaders(options.headers);
  let body: BodyInit | undefined;
  if (options.body instanceof FormData || options.body instanceof Blob) body = options.body;
  else if (options.body !== undefined) {
    headers.set('Content-Type', 'application/json');
    body = JSON.stringify(options.body);
  }
  const response = await fetch(apiUrl(path), {
    method: options.method ?? 'GET', body, signal: options.signal, headers,
    credentials: 'same-origin',
  });
  return await readJson(response) as T;
}

export const apiGet = <T extends JsonRecord = JsonRecord>(path: string, signal?: AbortSignal) => apiRequest<T>(path, { signal });
export const apiPost = <T extends JsonRecord = JsonRecord>(path: string, body: unknown, signal?: AbortSignal) => apiRequest<T>(path, { method: 'POST', body, signal });
export const apiPut = <T extends JsonRecord = JsonRecord>(path: string, body: unknown, signal?: AbortSignal) => apiRequest<T>(path, { method: 'PUT', body, signal });
export const apiPatch = <T extends JsonRecord = JsonRecord>(path: string, body: unknown, signal?: AbortSignal) => apiRequest<T>(path, { method: 'PATCH', body, signal });
export const apiDelete = <T extends JsonRecord = JsonRecord>(path: string, signal?: AbortSignal) => apiRequest<T>(path, { method: 'DELETE', signal });

export type AdvancedRoute = 'voice' | 'assistant' | 'communications' | 'security' | 'developer' | 'engine' | 'science' | 'twin' | 'vault' | 'scraper' | 'finetune';
export type AdvancedTab = string;

/**
 * Read endpoints currently implemented by OpenAgent for each advanced view.
 * `null` deliberately means that the backend has no list/read endpoint for
 * that tab; callers must render an unavailable state rather than invent data.
 */
export function advancedScreenEndpoints(route: AdvancedRoute, tab: AdvancedTab): Record<string, string> | null {
  const routes: Record<AdvancedRoute, Record<string, Record<string, string> | null>> = {
    voice: {
      Recordings: { status: '/api/voice/status', sessions: '/api/voice/sessions', personas: '/api/voice/personas' },
      Transcripts: { status: '/api/voice/status', sessions: '/api/voice/sessions', personas: '/api/voice/personas' },
      'Voice settings': { status: '/api/voice/status', personas: '/api/voice/personas' },
    },
    assistant: {
      Today: { tasks: '/api/assistant/tasks', brief: '/api/assistant/daily-brief' },
      Tasks: { tasks: '/api/assistant/tasks', goals: '/api/assistant/goals' },
      Goals: { goals: '/api/assistant/goals' },
      'Daily brief': { brief: '/api/assistant/daily-brief' },
    },
    communications: {
      Inbox: { messages: '/api/communication/inbox', connectors: '/api/communication/connectors' },
      Contacts: { contacts: '/api/communication/contacts?project=default' },
      Calendar: { events: '/api/communication/calendar?project=default' },
      Sent: { sent: '/api/communication/sent?project=default&limit=100' },
      Drafts: { drafts: '/api/communication/drafts?project=default&limit=100', approvals: '/api/communication/approvals?project=default&limit=100' },
      Scheduled: { scheduled: '/api/communication/scheduled?project=default&limit=100' },
      'Connected accounts': { connectors: '/api/communication/connectors', accounts: '/api/communication/oauth/accounts?project=default', providers: '/api/communication/oauth/providers', sync_runs: '/api/communication/sync-runs?project=default&limit=50' },
    },
    security: {
      Overview: { status: '/api/security/status' },
      Access: { status: '/api/security/status', grants: '/api/security/shares', egress: '/api/security/egress' },
      Sessions: { sessions: '/api/auth/sessions' },
      // GET /api/audit requires security:admin but can return cross-project
      // events when the project filter is omitted. Keep the list unavailable
      // until this UI verifies the intended project scope and authorization.
      Audit: null,
    },
    developer: {
      Connections: { status: '/api/developer/status', plugins: '/api/developer/plugins', extensions: '/api/developer/extensions' },
      'API keys': { keys: '/api/developer/api-keys', scopes: '/api/auth/scopes' },
      Webhooks: { webhooks: '/api/developer/webhooks', deliveries: '/api/developer/webhook-deliveries' },
      MCP: { manifest: '/api/developer/mcp', servers: '/api/developer/mcp/servers' },
      Activity: { deliveries: '/api/developer/webhook-deliveries', tools: '/api/developer/tools' },
    },
    engine: {
      Overview: { status: '/api/adaptive-engine/status', compute: '/api/compute/status', direct: '/api/adaptive-engine/direct/status' },
      Hardware: { hardware: '/api/adaptive-engine/hardware' },
      Jobs: { runs: '/api/adaptive-engine/orchestrations' },
      Plans: { plans: '/api/adaptive-engine/plans' },
    },
    science: {
      Projects: null,
      Experiments: { experiments: '/api/science/experiments?project=default' },
      Data: { datasets: '/api/science/materials/datasets', memory: '/api/science/memory' },
      Reports: null,
    },
    twin: {
      Profile: { status: '/api/twin/status', profile: '/api/twin/profile' },
      Preferences: { patterns: '/api/twin/patterns', suggestions: '/api/twin/suggestions' },
      Sources: { sources: '/api/twin/sources?project=default&limit=100' },
      'Data controls': { status: '/api/twin/status', profile: '/api/twin/profile', goals: '/api/twin/goals' },
    },
    vault: {
      Secrets: { status: '/api/vault/status', entries: '/api/vault/entries' },
      Access: { access_log: '/api/vault/access-log?project=default&limit=100' },
      Audit: { access_log: '/api/vault/access-log?project=default&limit=100' },
    },
    scraper: {
      'New extraction': { status: '/api/scraper/status' },
      Jobs: { jobs: '/api/scraper/jobs', status: '/api/scraper/status' },
      Results: { results: '/api/scraper/history?project=default&limit=100' },
      Rules: { status: '/api/scraper/status' },
    },
    finetune: {
      Datasets: { datasets: '/api/finetune/datasets', status: '/api/finetune/status' },
      Jobs: { jobs: '/api/finetune/jobs', status: '/api/finetune/status' },
      Checkpoints: null,
      Results: { adapters: '/api/finetune/adapters', status: '/api/finetune/status' },
    },
  };
  return routes[route][tab] ?? null;
}

export async function fetchAdvancedScreenData(
  route: AdvancedRoute,
  tab: AdvancedTab,
  signal?: AbortSignal,
): Promise<Record<string, JsonRecord>> {
  const endpoints = advancedScreenEndpoints(route, tab);
  if (!endpoints) return {};
  const rows = await Promise.all(Object.entries(endpoints).map(async ([key, path]) => [key, await apiGet(path, signal)] as const));
  return Object.fromEntries(rows);
}

export type StreamEvent = JsonRecord;

/** Read an authenticated backend SSE GET stream through the same mode guard as JSON requests. */
export async function apiStreamGet(
  path: string,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  assertLiveRequestsAllowed('GET');
  const headers = requestHeaders();
  headers.set('Accept', 'text/event-stream');
  const response = await fetch(apiUrl(path), { method: 'GET', headers, signal, credentials: 'same-origin' });
  if (!response.ok) { await readJson(response); return; }
  if (!response.body) throw new ApiError('The backend did not provide a readable event stream.', response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const frames = pending.split(/\r?\n\r?\n/);
      pending = frames.pop() ?? '';
      for (const frame of frames) {
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (!data) continue;
        let event: StreamEvent;
        try { event = JSON.parse(data) as StreamEvent; }
        catch { throw new ApiError('The backend sent an invalid event frame.', response.status); }
        onEvent(event);
        if (event.type === 'done' || event.type === 'stream.closed' || event.done === true) return;
        if (event.type === 'error') throw new ApiError(String(event.error ?? 'The backend stream failed.'), response.status || 500);
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

/** Read newline-delimited SSE frames from the backend without faking progress. */
export async function apiStreamPost(
  path: string,
  body: unknown,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  assertLiveRequestsAllowed('POST');
  const headers = requestHeaders({ 'Content-Type': 'application/json', Accept: 'text/event-stream' });
  const response = await fetch(apiUrl(path), { method: 'POST', headers, body: JSON.stringify(body), signal, credentials: 'same-origin' });
  if (!response.ok) { await readJson(response); return; }
  if (!response.body) throw new ApiError('The backend did not provide a readable event stream.', response.status);

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      pending += decoder.decode(value, { stream: !done });
      const frames = pending.split(/\r?\n\r?\n/);
      pending = frames.pop() ?? '';
      for (const frame of frames) {
        const data = frame.split(/\r?\n/).filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
        if (!data) continue;
        try {
          const event = JSON.parse(data) as StreamEvent;
          onEvent(event);
          if (event.type === 'error') throw new ApiError(String(event.error ?? 'The backend stream failed.'), response.status || 500);
        } catch (error) {
          if (error instanceof ApiError) throw error;
          throw new ApiError('The backend sent an invalid event frame.', response.status);
        }
      }
      if (done) break;
    }
  } finally {
    reader.releaseLock();
  }
}

export function apiConnectionMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Sign-in is required by the backend. Add a bearer token in Settings to continue.';
    if (error.status === 403) return 'This account does not have permission for this action.';
    return error.message;
  }
  if (error instanceof TypeError) return 'Cannot reach the OpenAgent backend. Start it at the configured API address.';
  return error instanceof Error ? error.message : 'The backend request failed.';
}
