import { demoProvider } from './provider';

/** Adapter boundary: demo mode is local-only; live mode continues to use the existing HTTP client. */
export async function demoAwareRequest<T extends Record<string, unknown>>(
  mode: 'demo' | 'live',
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE',
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  if (mode !== 'demo') throw new Error('Live requests must be sent through the authenticated backend client.');
  if (method === 'GET') return await demoProvider.get(path, signal) as T;
  if (method === 'DELETE') return await demoProvider.mutate(path, { deleted: true }) as T;
  return await demoProvider.mutate(path, body) as T;
}

export async function demoQuery<T extends Record<string, unknown>>(
  mode: 'demo' | 'live',
  path: string,
  live: (signal?: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (mode !== 'demo') return live(signal);
  if (path.startsWith('shell.') && path !== 'shell') return await demoProvider.get(path) as T;
  return await demoAwareRequest<T>(mode, 'GET', path, undefined, signal);
}

export async function demoCollectionQuery<T extends Record<string, unknown>>(
  mode: 'demo' | 'live',
  route: string,
  tab: string,
  live: (signal?: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (mode !== 'demo') return live(signal);
  const fixtureRoot = await demoProvider.get('');
  const collections = await demoProvider.collections();
  const rows = collections[route] as unknown as Record<string, unknown> | undefined;
  const collectionForTab: Record<string, Record<string, string>> = {
    knowledge: { Search: 'documents', Collections: 'collections', Ingestion: 'documents', 'Retrieval details': 'documents' },
    voice: { Recordings: 'recordings', Transcripts: 'recordings', 'Voice settings': 'settings' },
    assistant: { Today: 'tasks', Tasks: 'tasks', Goals: 'goals', 'Daily brief': 'tasks' },
    communications: { Inbox: 'messages', Contacts: 'contacts', Calendar: 'events', Sent: 'messages', Drafts: 'messages', Scheduled: 'scheduled', 'Connected accounts': 'accounts' },
    security: { Overview: 'policies', Access: 'policies', Sessions: 'sessions', Audit: 'audit' },
    developer: { Connections: 'connections', 'API keys': 'keys', Webhooks: 'webhooks', MCP: 'mcp', Activity: 'connections' },
    engine: { Overview: 'jobs', Hardware: 'hardware', Jobs: 'jobs', Plans: 'plans' },
    science: { Projects: 'projects', Experiments: 'experiments', Data: 'datasets', Reports: 'reports' },
    twin: { Profile: 'preferences', Preferences: 'preferences', Sources: 'sources', 'Data controls': 'history' },
    vault: { Secrets: 'secrets', Access: 'requests', Audit: 'audit' },
    scraper: { 'New extraction': 'jobs', Jobs: 'jobs', Results: 'results', Rules: 'rules' },
    finetune: { Datasets: 'datasets', Jobs: 'jobs', Checkpoints: 'checkpoints', Results: 'results' },
  };
  const collection = collectionForTab[route]?.[tab];
  const values = collection && rows ? rows[collection] : [];
  const list = Array.isArray(values) ? values : [];
  const shell = (fixtureRoot.shell ?? {}) as Record<string, unknown>;
  if (route === 'agents') return { agents: Array.isArray(shell.agents) ? shell.agents : [], runs: Array.isArray(shell.runs) ? shell.runs : [] } as unknown as T;
  if (route === 'automations') return { workflows: Array.isArray(shell.automations) ? shell.automations : [], automations: Array.isArray(shell.automations) ? shell.automations : [], runs: Array.isArray(shell.runs) ? shell.runs : [], actions: { noop: { label: 'Prepare example output' } } } as unknown as T;
  if (route === 'models') return { providers: Array.isArray(shell.models) ? shell.models : [], models: shell.models ?? [], jobs: [], adapters: [], status: { status: 'demo', project: 'Personal Workspace' } } as unknown as T;
  if (route === 'knowledge') return { documents: list, collections: (rows?.collections as unknown[]) ?? [] } as unknown as T;
  const key = route === 'voice' ? 'sessions' : route === 'assistant' ? tab === 'Goals' || tab === 'Data controls' ? 'goals' : 'tasks' : route === 'communications' ? tab === 'Inbox' ? 'messages' : tab === 'Contacts' ? 'contacts' : tab === 'Calendar' ? 'events' : tab === 'Connected accounts' ? 'connectors' : tab.toLowerCase() : route === 'security' ? tab === 'Audit' ? 'events' : tab === 'Sessions' ? 'sessions' : 'policies' : route === 'developer' ? tab === 'API keys' ? 'api_keys' : tab === 'Webhooks' ? 'webhooks' : tab === 'Activity' ? 'deliveries' : tab === 'MCP' ? 'servers' : 'plugins' : route === 'engine' ? tab === 'Hardware' ? 'hardware' : tab === 'Plans' ? 'plans' : 'runs' : route === 'science' ? tab === 'Data' ? 'datasets' : tab.toLowerCase() : route === 'twin' ? tab === 'Sources' ? 'sources' : tab === 'Preferences' ? 'suggestions' : 'profile' : route === 'vault' ? tab === 'Secrets' ? 'entries' : 'access_log' : route === 'scraper' ? tab === 'Results' ? 'results' : 'jobs' : route === 'finetune' ? tab === 'Checkpoints' ? 'checkpoints' : tab === 'Results' ? 'adapters' : tab.toLowerCase() : 'items';
  const status = route === 'voice' ? { status: 'demo', microphone_available: false, provider: 'Local example' } : { status: 'demo', project: 'Personal Workspace' };
  return { [key]: list, status, personas: route === 'voice' ? [{ id: 'demo', name: 'Example voice', available: false }] : [], suggestions: list, connectors: list, access_log: list } as unknown as T;
}

export async function demoCoreQuery<T extends Record<string, unknown>>(
  mode: 'demo' | 'live',
  key: string,
  live: (signal?: AbortSignal) => Promise<T>,
  signal?: AbortSignal,
): Promise<T> {
  if (mode !== 'demo') return live(signal);
  const fixture = key === 'shell' ? await demoProvider.get('shell') : await demoProvider.get(key);
  return (fixture && typeof fixture === 'object' ? fixture : {}) as T;
}
