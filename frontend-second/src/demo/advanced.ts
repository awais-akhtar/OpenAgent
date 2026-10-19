import { demoProvider } from './provider';
import type { DemoRecord } from './types';

type Data = Record<string, Record<string, unknown>>;
const routeKey: Record<string, string> = { voice: 'voice', assistant: 'assistant', communications: 'communications', security: 'security', developer: 'developer', engine: 'engine', science: 'science', twin: 'twin', vault: 'vault', scraper: 'scraper', finetune: 'finetune' };
const tabKey: Record<string, Record<string, string>> = {
  voice: { Recordings: 'recordings', Transcripts: 'recordings', 'Voice settings': 'settings' }, assistant: { Today: 'tasks', Tasks: 'tasks', Goals: 'goals', 'Daily brief': 'tasks' },
  communications: { Inbox: 'messages', Contacts: 'contacts', Calendar: 'events', Sent: 'messages', Drafts: 'messages', Scheduled: 'scheduled', 'Connected accounts': 'accounts' },
  security: { Overview: 'policies', Access: 'policies', Sessions: 'sessions', Audit: 'audit', Privacy: 'policies', Retention: 'policies', Secrets: 'policies', Sandbox: 'policies' },
  developer: { Connections: 'connections', 'API keys': 'keys', Webhooks: 'webhooks', MCP: 'mcp', Activity: 'connections' }, engine: { Overview: 'jobs', Hardware: 'hardware', Jobs: 'jobs', Plans: 'plans' },
  science: { Projects: 'projects', Experiments: 'experiments', Data: 'datasets', Reports: 'reports' }, twin: { Profile: 'preferences', Preferences: 'preferences', Sources: 'sources', 'Data controls': 'history' },
  vault: { Secrets: 'secrets', Access: 'requests', Audit: 'audit' }, scraper: { 'New extraction': 'jobs', Jobs: 'jobs', Results: 'results', Rules: 'rules' }, finetune: { Datasets: 'datasets', Jobs: 'jobs', Checkpoints: 'checkpoints', Results: 'results' },
};
const arr = (x: unknown) => Array.isArray(x) ? x.filter((v): v is DemoRecord => !!v && typeof v === 'object') : [];
const rename = (rows: DemoRecord[], mapping: Record<string, string>) => rows.map(row => Object.fromEntries(Object.entries(row).map(([key, value]) => [mapping[key] ?? key, value])) as DemoRecord);
async function logLocalActivity(route: string, name: string, item: string, status: string): Promise<void> {
  const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  await demoProvider.recordActivity(route, { id: `${route}-activity-${eventId}`, name, item, route, status, time: new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }), recorded_at: new Date().toISOString() });
}

/** Convert canonical HTML fixtures to the response view models consumed by AdvancedScreen. */
export async function advancedDemoScreen(route: string, tab: string): Promise<Data> {
  const all = await demoProvider.get('collections.advanced');
  const raw = (all[routeKey[route] ?? route] ?? {}) as Record<string, unknown>;
  const rows = (key: string) => arr(raw[key]);
  const wrap = (key: string, values: DemoRecord[]) => ({ [key]: values });
  const status = { status: 'demo', project: 'Personal Workspace', runtime: 'Local demo fixtures', microphone_control: 'Demo recording input', privacy_boundary: 'Sample data stored locally in this browser only.', total_entries: rows('secrets').length };
  if (route === 'voice') return { sessions: wrap('sessions', rename(rows('recordings'), { owner: 'project' }).map(r => ({ ...r, status: String(r.status).toLowerCase() === 'transcribed' ? 'completed' : r.status, mode: 'transcription', language: 'en' }))), recordings: wrap('recordings', rows('recordings')), transcripts: wrap('transcripts', [{ id: 'demo-transcript-1', name: 'Product planning sync', status: 'Sample transcript' }]), personas: wrap('personas', [{ id: 'demo-voice', name: 'Example voice' }]), status, turns: wrap('turns', [{ id: 'turn-1', speaker: 'user', text: 'Review onboarding feedback and release blockers.', role: 'user', is_final: true, created_at: 'Today' }]) };
  if (route === 'assistant') {
    const tasks = rename(rows('tasks'), { name: 'title', due: 'due_at', owner: 'project' });
    const goals = rename(rows('goals'), { name: 'title', owner: 'project' });
    return { tasks: wrap('tasks', tasks), goals: wrap('goals', goals), activity: wrap('events', await demoProvider.getActivity('assistant')), brief: { open_tasks: tasks.filter(item => String(item.status).toLowerCase() !== 'done'), active_goals: goals.filter(item => String(item.status).toLowerCase() !== 'done'), recommended_focus: tasks, priority_count: tasks.filter(item => ['high', 'urgent'].includes(String(item.priority).toLowerCase()) && String(item.status).toLowerCase() !== 'done').length, project: 'Personal Workspace' }, status };
  }
  if (route === 'communications') {
    const messages = rename(rows('messages'), { name: 'subject', description: 'body', owner: 'sender', time: 'created_at' });
    return { messages: wrap('messages', messages), threads: wrap('threads', messages), drafts: wrap('drafts', messages.filter(x => String(x.status).toLowerCase() === 'draft')), inbox: wrap('messages', messages.filter(x => String(x.status).toLowerCase() === 'inbox')), sent: wrap('messages', messages.filter(x => String(x.status).toLowerCase() === 'sent')), contacts: wrap('contacts', rows('contacts')), events: wrap('events', rows('events')), scheduled: wrap('scheduled', rows('scheduled')), accounts: wrap('accounts', rows('accounts')), connectors: wrap('connectors', [{ id: 'email', label: 'Email', enabled: false, capabilities: ['read'] }]), providers: wrap('providers', []), sync_runs: wrap('runs', []), status };
  }
  if (route === 'security') return { policies: wrap('policies', rows('policies')), status, sessions: wrap('sessions', rows('sessions')), events: wrap('events', rows('audit')), grants: wrap('grants', rows('policies')), egress: wrap('egress', []), reports: wrap('reports', []), retention: wrap('retention', []), secrets: wrap('secrets', []), profiles: wrap('profiles', []) };
  if (route === 'developer') return { status, status_data: status, api_keys: wrap('api_keys', rows('keys')), scopes: wrap('scopes', [{ id: 'read', name: 'read' }]), webhooks: wrap('webhooks', rows('webhooks')), deliveries: wrap('deliveries', []), servers: wrap('servers', rows('mcp')), plugins: wrap('plugins', rows('connections')), packages: wrap('packages', []), connections: wrap('connections', rows('connections')) };
  if (route === 'engine') return { status, hardware: wrap('hardware', rows('hardware')), jobs: wrap('jobs', rows('jobs')), runs: wrap('runs', rows('jobs')), plans: wrap('plans', rows('plans')) };
  if (route === 'science') return { status, projects: wrap('projects', rows('projects')), experiments: wrap('experiments', rows('experiments')), datasets: wrap('datasets', rows('datasets')), reports: wrap('reports', rows('reports')), catalog: wrap('datasets', rows('datasets')) };
  if (route === 'twin') return { status, profile: { facts: rows('preferences') }, goals: wrap('goals', rows('history')), suggestions: wrap('suggestions', rows('preferences')), preferences: wrap('preferences', rows('preferences')), sources: wrap('sources', rows('sources')), history: wrap('history', rows('history')) };
  if (route === 'vault') return {
    status,
    entries: wrap('entries', rename(rows('secrets'), { name: 'label', scope: 'category' })),
    requests: wrap('requests', rows('requests')),
    audit: wrap('audit', rows('audit')),
    access_log: wrap('access_log', rows(tab === 'Audit' ? 'audit' : 'requests')),
  };
  if (route === 'scraper') return { status, jobs: wrap('jobs', rename(rows('jobs'), { name: 'url', description: 'description' })), results: wrap('results', rows('results')), rules: wrap('rules', rows('rules')) };
  if (route === 'finetune') return { status, datasets: wrap('datasets', rename(rows('datasets'), { description: 'format', count: 'example_count' })), jobs: wrap('jobs', rename(rows('jobs'), { description: 'method', loss: 'progress' })), checkpoints: wrap('checkpoints', rows('checkpoints')), adapters: wrap('adapters', rows('results')), activity: wrap('events', await demoProvider.getActivity('finetune')), templates: wrap('templates', [{ id: 'demo-template', name: 'Instruction tuning' }]), localModels: wrap('models', [{ id: 'demo-model', name: 'Example base model', status: 'installed' }]), localJobs: wrap('jobs', []) };
  const key = tabKey[route]?.[tab];
  return { status, ...(key ? wrap(key, rows(key)) : {}) };
}

/** Local-only prototype operations; the record stays in this browser's isolated fixture store. */
export async function advancedDemoAction(path: string, body: unknown): Promise<Record<string, unknown>> {
  const payload = (body && typeof body === 'object' ? body : {}) as Record<string, unknown>;
  const route = path.includes('/voice/') ? 'voice' : path.includes('/assistant/') ? 'assistant' : path.includes('/communication/') ? 'communications' : path.includes('/security/') ? 'security' : path.includes('/vault/') ? 'vault' : path.includes('/finetune/') ? 'finetune' : path.includes('/scraper/') ? 'scraper' : path.includes('/science/') ? 'science' : path.includes('/twin/') ? 'twin' : path.includes('/local-ai/') ? 'finetune' : 'developer';
  const coll = (path.includes('/voice/') ? 'recordings' : path.includes('/assistant/tasks') ? 'tasks' : path.includes('/assistant/goals') ? 'goals' : path.includes('/communication/') ? (path.includes('draft') || path.includes('send') ? 'messages' : 'messages') : path.includes('/vault/') ? (path.includes('store') ? 'secrets' : 'audit') : path.includes('/finetune/datasets') ? 'datasets' : path.includes('/finetune/jobs') ? 'jobs' : path.includes('/finetune/adapters') ? 'results' : path.includes('/finetune/examples') ? 'datasets' : path.includes('/scraper/') ? 'jobs' : path.includes('/science/') ? 'datasets' : path.includes('/twin/') ? (path.endsWith('/goals') ? 'history' : 'preferences') : 'connections');
  // Credentials and passphrases are request-only, even in isolated demo mode.
  const safePayload = Object.fromEntries(Object.entries(payload).filter(([key]) => !/(secret|passphrase|token|credential|private.?key|ciphertext|password|^value$)/i.test(key)));
  const title = String(payload.title ?? payload.name ?? payload.label ?? payload.text ?? payload.context ?? payload.dataset_id ?? payload.filename ?? 'Demo action');
  let activityItem = title;
  const id = `demo-${Date.now().toString(36)}`;
  const record: DemoRecord = { ...safePayload, id, name: title, title, status: route === 'voice' && path.endsWith('/stt') ? 'Transcribed · demo' : path.includes('complete') ? 'Done' : path.includes('cancel') ? 'Cancelled' : path.includes('run') ? 'Completed' : 'Created locally', updated: 'Just now' };
  if (route === 'voice' && path.includes('/recordings/update')) {
    const recordingId = String(payload.recording_id ?? '');
    const updated = await demoProvider.updateCollection<DemoRecord>(`collections.advanced.voice.recordings`, current => current.map(item => item.id === recordingId
      ? { ...item, name: String(payload.name ?? item.name), description: String(payload.description ?? item.description ?? ''), duration: String(payload.duration ?? item.duration ?? ''), status: String(payload.status ?? item.status ?? 'Ready'), updated: 'Just now' }
      : item));
    const recording = updated.find(item => item.id === recordingId) ?? null;
    await logLocalActivity('voice', String(payload.status ?? 'Running') === 'Completed' ? 'Completed recording preview' : 'Updated recording', String(recording?.name ?? 'Voice recording'), 'Saved locally · demo');
    return { recording, demo: true };
  }
  if (route === 'voice' && path.includes('/recordings/delete')) {
    const recordingId = String(payload.recording_id ?? '');
    const voice = await demoProvider.get('collections.advanced.voice');
    const recordings = arr(voice.recordings);
    const removed = recordings.find(item => String(item.id ?? '') === recordingId);
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.voice.recordings`, current => current.filter(item => item.id !== recordingId));
    await logLocalActivity('voice', 'Deleted recording', String(removed?.name ?? 'Voice recording'), 'Deleted locally · demo');
    return { deleted: true, recording_id: recordingId, demo: true };
  }
  if (route === 'voice' && path.includes('/recordings/duplicate')) {
    const recordingId = String(payload.recording_id ?? '');
    const voice = await demoProvider.get('collections.advanced.voice');
    const rows = Array.isArray(voice.recordings) ? voice.recordings.filter((item): item is DemoRecord => !!item && typeof item === 'object') : [];
    const original = rows.find(item => item.id === recordingId);
    if (!original) throw new Error('The selected demo recording no longer exists.');
    const duplicate = { ...original, id, name: `${String(original.name ?? 'Recording')} (copy)`, status: 'Transcribed', updated: 'Just now' };
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.voice.recordings`, current => [duplicate, ...current]);
    await logLocalActivity('voice', 'Duplicated recording', String(duplicate.name), 'Saved locally · demo');
    return { recording: duplicate, demo: true };
  }
  if (route === 'voice' && path.includes('/recordings/create')) {
    const created: DemoRecord = { ...record, status: 'Ready', owner: 'Workspace', duration: String(payload.duration ?? '00:00') };
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.voice.recordings`, current => [created, ...current]);
    await logLocalActivity('voice', 'Created recording', String(created.name), 'Saved locally · demo');
    return { recording: created, demo: true };
  }
  if (path.endsWith('/complete')) {
    const taskId = String(payload.task_id ?? '');
    if (route === 'assistant') {
      const assistant = await demoProvider.get('collections.advanced.assistant');
      const tasks = arr(assistant.tasks);
      const task = tasks.find(item => String(item.id ?? '') === taskId);
      activityItem = String(task?.title ?? task?.name ?? activityItem);
    }
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.${coll}`, current => current.map(item => item.id === taskId ? { ...item, status: 'Done' } : item));
  } else if (path.includes('/close') || path.includes('/cancel')) {
    const target = String(payload.session_id ?? payload.job_id ?? '');
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.${coll}`, current => current.map(item => item.id === target ? { ...item, status: 'Cancelled' } : item));
  } else if (path.includes('/retrieve')) {
    return { secret: { secret: 'Demo-only placeholder value (no credential is stored)' }, demo: true };
  } else if (path.includes('/export')) {
    const advanced = await demoProvider.get('collections.advanced');
    const entries = arr((advanced.vault as Record<string, unknown> | undefined)?.secrets);
    return { entry_count: entries.length, entries: entries.map(({ name, status }) => ({ name, status })), demo: true };
  } else if (path.includes('/run-due') || path.includes('/process')) {
    return { status: 'completed', processed: 0, demo: true };
  } else if (route === 'vault' && path.includes('/store')) {
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.secrets`, current => [{ ...record, demo_value_placeholder: true }, ...current]);
  } else if (route === 'vault' && path.includes('/rotate-key')) {
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.audit`, current => [{ ...record, name: 'Demo passphrase rotation', status: 'Simulated locally' }, ...current]);
  } else {
    await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.${coll}`, current => [record, ...current]);
  }
  if (route === 'assistant') {
    const action = path.endsWith('/complete') ? 'Task completed' : path.endsWith('/goals') ? 'Goal created' : 'Task created';
    const eventId = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    await demoProvider.recordActivity('assistant', { id: `assistant-activity-${eventId}`, name: action, item: activityItem, route: 'assistant', status: path.endsWith('/complete') ? 'Completed locally · demo' : 'Saved locally · demo', time: new Date().toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' }), recorded_at: new Date().toISOString() });
  } else if (['voice', 'twin', 'developer', 'engine', 'science', 'scraper'].includes(route)) {
    const action = path.endsWith('/stt') ? 'Transcribed example recording' : route === 'twin' ? path.includes('/preferences') ? 'Added personal context example' : path.includes('/sources') ? 'Added personal context source example' : 'Updated personal context example' : path.endsWith('/complete') ? 'Completed local operation' : path.includes('/cancel') || path.includes('/close') ? 'Cancelled local operation' : path.includes('/run') ? 'Started simulated operation' : path.includes('/update') ? 'Updated example record' : 'Created example record';
    const status = action.includes('simulated') || action.includes('operation') || path.endsWith('/stt') ? 'Simulated locally · demo' : 'Saved locally · demo';
    await logLocalActivity(route, action, route === 'twin' ? coll : title, status);
  }
  return { [path.includes('/voice/') ? 'session' : path.includes('task') ? 'task' : path.includes('goal') ? 'goal' : path.includes('dataset') ? 'dataset' : path.includes('job') ? 'job' : path.includes('adapter') ? 'adapter' : 'result']: record, demo: true };
}

/** Restore a single advanced workspace collection from the canonical fixture file. */
export async function resetAdvancedDemoRoute(route: string): Promise<void> {
  const response = await fetch('/demo/reference-fixtures.json');
  if (!response.ok) throw new Error('The approved demo fixtures could not be loaded.');
  const file = await response.json() as { data?: { advanced?: Record<string, Record<string, unknown>> } };
  const seed = file.data?.advanced?.[route];
  if (!seed) throw new Error(`No canonical demo fixture is available for ${route}.`);
  for (const [key, value] of Object.entries(seed)) {
    if (Array.isArray(value)) await demoProvider.updateCollection<DemoRecord>(`collections.advanced.${route}.${key}`, () => arr(value));
  }
  await demoProvider.resetActivity(route);
}
