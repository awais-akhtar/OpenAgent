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

export async function apiGetBlob(path: string, signal?: AbortSignal): Promise<Blob> {
  assertLiveRequestsAllowed('GET');
  const headers = requestHeaders();
  headers.set('Accept', '*/*');
  const response = await fetch(apiUrl(path), { method: 'GET', signal, headers, credentials: 'same-origin' });
  if (!response.ok) { await readJson(response); throw new ApiError(`Request failed (${response.status}).`, response.status); }
  return response.blob();
}

export type AdvancedRoute = 'voice' | 'assistant' | 'communications' | 'security' | 'developer' | 'engine' | 'science' | 'twin' | 'vault' | 'scraper' | 'finetune' | 'gallery' | 'cookbook' | 'ultramodel' | 'personalos' | 'g25advanced' | 'voiceconv' | 'modelserve' | 'factprovenance';
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
      Today: { tasks: '/api/assistant/tasks', brief: '/api/assistant/daily-brief', activity: '/api/assistant/activity?project=default&limit=20' },
      Tasks: { tasks: '/api/assistant/tasks', goals: '/api/assistant/goals' },
      Goals: { goals: '/api/assistant/goals' },
      'Daily brief': { brief: '/api/assistant/daily-brief' },
    },
    communications: {
      Inbox: { messages: '/api/communication/inbox', connectors: '/api/communication/connectors' },
      Contacts: { contacts: '/api/communication/contacts?project=default', crm_contacts: '/api/contacts' },
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
      // The endpoint requires security:admin. Always constrain the UI query
      // to its explicit local project; never issue an unscoped audit request.
      Audit: { events: '/api/audit?project=default&limit=100' },
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
      Reports: { executions: '/api/science/executions?project=default' },
    },
    twin: {
      Profile: { status: '/api/twin/status', profile: '/api/twin/profile' },
      Preferences: { patterns: '/api/twin/patterns', suggestions: '/api/twin/suggestions' },
      Sources: { sources: '/api/twin/sources?project=default&limit=100' },
      'Data controls': { status: '/api/twin/status', profile: '/api/twin/profile', goals: '/api/twin/goals' },
    },
    vault: {
      Secrets: { status: '/api/vault/status', entries: '/api/vault/entries' },
      Access: { requests: '/api/vault/access-requests?project=default&limit=100', entries: '/api/vault/entries?project=default' },
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
      Jobs: { jobs: '/api/finetune/jobs', status: '/api/finetune/status', templates: '/api/finetune/templates', localModels: '/api/local-ai/models?project=default', localJobs: '/api/local-ai/jobs?project=default' },
      Checkpoints: { checkpoints: '/api/finetune/checkpoints?project=default&limit=100' },
      Results: { adapters: '/api/finetune/adapters', status: '/api/finetune/status' },
    },
    gallery: {
      Library: { images: '/api/gallery/library?project=default', albums: '/api/gallery/albums?project=default' },
      Albums: { albums: '/api/gallery/albums?project=default' },
      Upload: null,
      Operations: null,
    },
    cookbook: {
      Hosts: { hosts: '/api/model-cookbook/hosts', sessions: '/api/model-cookbook/sessions?project=default', hardware: '/api/model-cookbook/hardware-fit' },
      Cached: { cached: '/api/model-cookbook/cached' },
      Certifications: { certifications: '/api/model-cookbook/certifications' },
      Logs: null,
    },
    ultramodel: {
      Status: { status: '/api/ultra-model/status', sessions: '/api/ultra-model/sessions' },
      Manifests: { manifests: '/api/ultra-model/manifests' },
      Plans: { plans: '/api/ultra-model/plans' },
      Sessions: { sessions: '/api/ultra-model/sessions' },
    },
    personalos: {
      Status: { status: '/api/personal-os/status', presets: '/api/personal-os/presets' },
      Presets: { presets: '/api/personal-os/presets' },
      Notes: { notes: '/api/personal-os/notes' },
      Gallery: { gallery: '/api/personal-os/gallery' },
      Signatures: { signatures: '/api/personal-os/signatures' },
      Drafts: { drafts: '/api/personal-os/drafts' },
    },
    g25advanced: {
      Documents: { documents: '/api/g25/documents?project=default&owner=default' },
      'Knowledge graph': null,
      RAG: { documents: '/api/g25/documents?project=default&owner=default' },
      Reasoning: null,
      Memory: null,
    },
    voiceconv: {
      Conversations: { conversations: '/api/voice/conversations?project=default&owner=default' },
    },
    modelserve: {
      'Serving sessions': { sessions: '/api/model-serve/sessions?project=default&owner=default' },
      'Ultra models': { models: '/api/ultra-models/models?project=default&owner=default' },
      'Ultra sessions': { sessions: '/api/ultra-models/sessions?project=default&owner=default' },
      Readiness: { readiness: '/api/ultra-models/readiness?project=default&owner=default' },
    },
    factprovenance: {
      // Fact provenance routes are scoped by project only; they take no owner.
      Facts: {
        facts: '/api/fact-provenance/facts?project=default&limit=100&active_only=true',
        statistics: '/api/fact-provenance/statistics?project=default',
      },
      'All facts': {
        facts: '/api/fact-provenance/facts?project=default&limit=100&active_only=false',
        statistics: '/api/fact-provenance/statistics?project=default',
      },
      'Record a fact': null,
      Statistics: { statistics: '/api/fact-provenance/statistics?project=default' },
    },
  };
  return routes[route]?.[tab] ?? null;
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

// ── Communication permanent delete ─────────────────────────────────────────
export const deleteMessage = (messageId: string, project = 'default') =>
  apiDelete(`/api/communication/messages/${encodeURIComponent(messageId)}?project=${encodeURIComponent(project)}`);

// ── Calendar provider lifecycle ────────────────────────────────────────────
export const deleteCalendarProvider = (providerId: string) =>
  apiDelete(`/api/calendar/providers/${encodeURIComponent(providerId)}`);

export const reauthCalendarProvider = (
  providerId: string,
  body: { account_ref?: string; config?: Record<string, unknown> },
) => apiPost(`/api/calendar/providers/${encodeURIComponent(providerId)}/reauth`, body);

// ── Agent job events ───────────────────────────────────────────────────────
export const getAgentJobEvents = (jobId: string, after = '', limit = 100) =>
  apiGet(
    `/api/agents/jobs/${encodeURIComponent(jobId)}/events?limit=${limit}${after ? `&after=${encodeURIComponent(after)}` : ''}`,
  );

// ── Browser voice streaming ────────────────────────────────────────────────
export const startVoiceStream = (body: {
  project?: string;
  voice_session_id?: string;
  audio_format?: string;
  language?: string;
  stt_engine?: string;
}) => apiPost('/api/voice/stream/start', body);

export const pushVoiceStreamChunk = (body: {
  stream_id: string;
  audio_base64: string;
  sequence: number;
}) => apiPost('/api/voice/stream/chunk', body);

export const finishVoiceStream = (body: {
  stream_id: string;
  speaker?: string;
  transcript_hint?: string;
  allow_fallback_stt?: boolean;
}) => apiPost('/api/voice/stream/finish', body);

// ── Project scope helpers ──────────────────────────────────────────────────
/** Build the `project`/`owner` query string every scoped backend GET expects. */
function scopeQuery(project = 'default', owner = 'default', extra: Record<string, string> = {}): string {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(extra)) params.set(key, value);
  params.set('project', project);
  params.set('owner', owner);
  return params.toString();
}

const seg = (value: string) => encodeURIComponent(value);

// ── G25 advanced: documents, knowledge graph, RAG, reasoning, memory ───────
export const uploadG25Document = (body: { filename: string; content: string; project?: string }) =>
  apiPost('/api/g25/documents/upload', { project: 'default', ...body });

export const listG25Documents = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/documents?${scopeQuery(project, owner)}`, signal);

export const getG25Document = (documentId: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/documents/${seg(documentId)}`, signal);

export const analyzeG25Document = (documentId: string, project = 'default') =>
  apiPost(`/api/g25/documents/${seg(documentId)}/analyze`, { project });

export const createG25GraphNode = (body: { label: string; properties?: JsonRecord; project?: string }) =>
  apiPost('/api/g25/graph/nodes', { project: 'default', properties: {}, ...body });

export const createG25GraphEdge = (body: { source_id: string; target_id: string; relationship: string; project?: string }) =>
  apiPost('/api/g25/graph/edges', { project: 'default', ...body });

export const queryG25Graph = (q: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/graph/query?${scopeQuery(project, owner, { q })}`, signal);

export const findG25GraphPaths = (start: string, end: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/graph/paths?${scopeQuery(project, owner, { start, end })}`, signal);

// The backend chunks the stored document itself, so only doc_id is sent.
export const indexG25Rag = (body: { doc_id: string; project?: string }) =>
  apiPost('/api/g25/rag/index', { project: 'default', ...body });

export const searchG25Rag = (body: { query: string; project?: string }) =>
  apiPost('/api/g25/rag/search', { project: 'default', ...body });

export const getG25RagContext = (docId: string, q: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/rag/context?${scopeQuery(project, owner, { doc_id: docId, q })}`, signal);

export const createG25ReasoningChain = (body: { topic: string; project?: string }) =>
  apiPost('/api/g25/reasoning/chains', { project: 'default', ...body });

export const analyzeG25Reasoning = (body: { text: string; project?: string }) =>
  apiPost('/api/g25/reasoning/analyze', { project: 'default', ...body });

export const getG25ReasoningChain = (chainId: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/reasoning/chains/${seg(chainId)}`, signal);

export const createG25MemorySession = (project = 'default') =>
  apiPost('/api/g25/memory/sessions', { project });

export const storeG25Memory = (body: { session_id: string; role: string; content: string; project?: string }) =>
  apiPost('/api/g25/memory/store', { project: 'default', ...body });

export const getG25MemoryHistory = (sessionId: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/g25/memory/history?${scopeQuery(project, owner, { session_id: sessionId })}`, signal);

// ── Voice conversations ────────────────────────────────────────────────────
export const createVoiceConversation = (project = 'default') =>
  apiPost('/api/voice/conversations', { project });

export const listVoiceConversations = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/voice/conversations?${scopeQuery(project, owner)}`, signal);

export const getVoiceConversation = (conversationId: string, signal?: AbortSignal) =>
  apiGet(`/api/voice/conversations/${seg(conversationId)}`, signal);

export const sendVoiceConversationAudio = (conversationId: string, body: { audio: string; project?: string }) =>
  apiPost(`/api/voice/conversations/${seg(conversationId)}/audio`, { project: 'default', ...body });

export const respondVoiceConversation = (conversationId: string, body: { text: string; project?: string }) =>
  apiPost(`/api/voice/conversations/${seg(conversationId)}/respond`, { project: 'default', ...body });

export const runVoiceFullConversation = (conversationId: string, body: { audio: string; response_text: string; project?: string }) =>
  apiPost(`/api/voice/conversations/${seg(conversationId)}/full-conversation`, { project: 'default', ...body });

export const closeVoiceConversation = (conversationId: string, project = 'default') =>
  apiPost(`/api/voice/conversations/${seg(conversationId)}/close`, { project });

// ── Model serving ──────────────────────────────────────────────────────────
export const createModelServePlan = (body: { model_id: string; model_path: string; host: string; port: number; project?: string }) =>
  apiPost('/api/model-serve/plans', { project: 'default', ...body });

export const startModelServeSession = (sessionId: string, project = 'default') =>
  apiPost(`/api/model-serve/sessions/${seg(sessionId)}/start`, { project });

export const stopModelServeSession = (sessionId: string, project = 'default') =>
  apiPost(`/api/model-serve/sessions/${seg(sessionId)}/stop`, { project });

export const getModelServeSession = (sessionId: string, signal?: AbortSignal) =>
  apiGet(`/api/model-serve/sessions/${seg(sessionId)}`, signal);

export const listModelServeSessions = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/model-serve/sessions?${scopeQuery(project, owner)}`, signal);

// ── Ultra models ───────────────────────────────────────────────────────────
export const listUltraModels = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/ultra-models/models?${scopeQuery(project, owner)}`, signal);

export const downloadUltraModel = (modelId: string, project = 'default') =>
  apiPost(`/api/ultra-models/models/${seg(modelId)}/download`, { project });

export const createUltraModelSession = (body: { model_id: string; name: string; project?: string }) =>
  apiPost('/api/ultra-models/sessions', { project: 'default', ...body });

export const launchUltraModelSession = (sessionId: string, project = 'default') =>
  apiPost(`/api/ultra-models/sessions/${seg(sessionId)}/launch`, { project });

export const stopUltraModelSession = (sessionId: string, project = 'default') =>
  apiPost(`/api/ultra-models/sessions/${seg(sessionId)}/stop`, { project });

export const reloadUltraModelSession = (sessionId: string, project = 'default') =>
  apiPost(`/api/ultra-models/sessions/${seg(sessionId)}/reload`, { project });

export const getUltraModelSession = (sessionId: string, signal?: AbortSignal) =>
  apiGet(`/api/ultra-models/sessions/${seg(sessionId)}`, signal);

export const listUltraModelSessions = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/ultra-models/sessions?${scopeQuery(project, owner)}`, signal);

export const getUltraModelReadiness = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/ultra-models/readiness?${scopeQuery(project, owner)}`, signal);

// ── Developer webhook and API-key lifecycle ────────────────────────────────
// The backend declares PUT/DELETE on /api/developer/webhooks/{id}; it omits
// only the keys sent, so an omitted field is left as the backend has it.
// No secret value is ever returned to or sent from these helpers except the
// key the caller explicitly passes to the verify route, which the backend
// checks and never echoes back.
export type DeveloperWebhookUpdate = {
  event?: string;
  target_url?: string;
  enabled?: boolean;
  secret_hint?: string;
  metadata?: JsonRecord;
};

export const updateDeveloperWebhook = (webhookId: string, body: DeveloperWebhookUpdate) =>
  apiPut(`/api/developer/webhooks/${seg(webhookId)}`, body);

export const deleteDeveloperWebhook = (webhookId: string) =>
  apiDelete(`/api/developer/webhooks/${seg(webhookId)}`);

export const dispatchDeveloperWebhook = (body: { event: string; payload?: JsonRecord; allow_network?: boolean }) =>
  apiPost('/api/developer/webhooks/dispatch', { payload: {}, allow_network: false, ...body });

export const verifyDeveloperApiKey = (body: { key: string; scope?: string }) =>
  apiPost('/api/developer/api-keys/verify', body);

// ── Image Studio advanced gallery operations ───────────────────────────────
export const getGalleryAssetOperations = (assetId: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/gallery/${seg(assetId)}/operations?${scopeQuery(project, owner)}`, signal);

export const getGalleryAssetVersions = (assetId: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/gallery/${seg(assetId)}/versions?${scopeQuery(project, owner)}`, signal);

export const editGalleryAssetMetadata = (assetId: string, body: {
  filename?: string;
  tags?: string[];
  metadata?: JsonRecord;
  favorite?: boolean;
  project?: string;
  owner?: string;
}) => apiPost(`/api/gallery/${seg(assetId)}/metadata`, { project: 'default', ...body });

export const batchTagGalleryAssets = (body: {
  asset_ids: string[];
  tags: string[];
  replace?: boolean;
  project?: string;
  owner?: string;
}) => apiPost('/api/gallery/batch-tags', { project: 'default', replace: false, ...body });

/** `operation` must be one the studio supports; `inpaint` additionally needs a mask and is not offered here. */
export const transformGalleryAsset = (assetId: string, body: {
  operation: string;
  params?: JsonRecord;
  project?: string;
  owner?: string;
}) => apiPost(`/api/gallery/${seg(assetId)}/transform`, { project: 'default', params: {}, ...body });

// ── Personal OS gallery, compare and bundle transfer ───────────────────────
export const listPersonalOsGallery = (project?: string, owner?: string, tag = '', signal?: AbortSignal) =>
  apiGet(`/api/personal-os/gallery?${scopeQuery(project, owner, tag ? { tag } : {})}`, signal);

export const addPersonalOsGalleryAsset = (body: {
  title: string;
  media_type?: string;
  source_path?: string;
  content?: string;
  tags?: string[];
  metadata?: JsonRecord;
  project?: string;
}) => apiPost('/api/personal-os/gallery', { project: 'default', ...body });

export const setPersonalOsGalleryFavorite = (assetId: string, favorite: boolean, project = 'default') =>
  apiPost('/api/personal-os/gallery/favorite', { asset_id: assetId, favorite, project });

export const listPersonalOsCompares = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/personal-os/compare?${scopeQuery(project, owner)}`, signal);

export const startPersonalOsCompare = (body: {
  prompt: string;
  left_output: string;
  right_output: string;
  left_label?: string;
  right_label?: string;
  project?: string;
}) => apiPost('/api/personal-os/compare/start', { project: 'default', ...body });

export const votePersonalOsCompare = (body: { session_id: string; vote: string; project?: string }) =>
  apiPost('/api/personal-os/compare/vote', { project: 'default', ...body });

export const exportPersonalOsBundle = (project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/personal-os/export?${scopeQuery(project, owner)}`, signal);

export const importPersonalOsBundle = (bundle: JsonRecord, project = 'default') =>
  apiPost('/api/personal-os/import', { bundle, project });

export const calibrateUltraModel = (modelId: string, project?: string, owner?: string, signal?: AbortSignal) =>
  apiGet(`/api/ultra-models/models/${seg(modelId)}/calibrate?${scopeQuery(project, owner)}`, signal);

// ── Fact provenance (personal context fact sourcing and retraction) ────────
// The backend scopes these routes by `project` only; there is no `owner`
// parameter on any fact-provenance route, so scopeQuery is deliberately not
// used here. Retraction is non-destructive: `retract` records a reason and
// `restore` reverses it. Neither route deletes a fact.
export type FactProvenanceSourceInput = {
  source_type: string;
  source_reference: string;
  confidence?: number;
};

const factProvenanceQuery = (project = 'default', extra: Record<string, string> = {}): string => {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(extra)) params.set(key, value);
  params.set('project', project);
  return params.toString();
};

/** GET /api/fact-provenance/facts — list facts with their source metadata. */
export const listProvenanceFacts = (
  project = 'default',
  options: { limit?: number; activeOnly?: boolean } = {},
  signal?: AbortSignal,
) => apiGet(`/api/fact-provenance/facts?${factProvenanceQuery(project, {
  limit: String(options.limit ?? 100),
  active_only: options.activeOnly === false ? 'false' : 'true',
})}`, signal);

/** GET /api/fact-provenance/facts/{id} — one fact with sources and retraction history. */
export const getProvenanceFact = (factId: string, project = 'default', signal?: AbortSignal) =>
  apiGet(`/api/fact-provenance/facts/${seg(factId)}?${factProvenanceQuery(project)}`, signal);

/** GET /api/fact-provenance/search — substring search over recorded facts. */
export const searchProvenanceFacts = (q: string, project = 'default', limit = 50, signal?: AbortSignal) =>
  apiGet(`/api/fact-provenance/search?${factProvenanceQuery(project, { q, limit: String(limit) })}`, signal);

/** GET /api/fact-provenance/statistics — fact, source and retraction counts. */
export const getProvenanceStatistics = (project = 'default', signal?: AbortSignal) =>
  apiGet(`/api/fact-provenance/statistics?${factProvenanceQuery(project)}`, signal);

/** POST /api/fact-provenance/facts — record a fact with its provenance sources. */
export const createProvenanceFact = (body: {
  fact: string;
  fact_type?: string;
  confidence?: number;
  sources?: FactProvenanceSourceInput[];
  project?: string;
}) => apiPost('/api/fact-provenance/facts', { project: 'default', ...body });

/**
 * POST /api/fact-provenance/facts/{id}/retract — mark a fact as retracted.
 * The fact and its sources are retained; `reason` is required by the backend.
 */
export const retractProvenanceFact = (factId: string, body: {
  reason: string;
  retracted_by?: string;
  evidence?: string;
  project?: string;
}) => apiPost(`/api/fact-provenance/facts/${seg(factId)}/retract`, { project: 'default', ...body });

/** POST /api/fact-provenance/facts/{id}/restore — reverse a retraction. */
export const restoreProvenanceFact = (factId: string, project = 'default') =>
  apiPost(`/api/fact-provenance/facts/${seg(factId)}/restore`, { project });

export function apiConnectionMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 401) return 'Sign-in is required by the backend. Add a bearer token in Settings to continue.';
    if (error.status === 403) return 'This account does not have permission for this action.';
    return error.message;
  }
  if (error instanceof TypeError) return 'Cannot reach the OpenAgent backend. Start it at the configured API address.';
  return error instanceof Error ? error.message : 'The backend request failed.';
}
