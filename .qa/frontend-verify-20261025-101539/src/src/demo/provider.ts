import type { DemoRecord } from './types';

/** Isolated fixture adapter for the approved offline reference. Nothing here calls backend services. */
export const demoProvider = {
  async get(path: string, signal?: AbortSignal): Promise<Record<string, unknown>> {
    if (signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError');
    const data = await loadFixtures();
    if (!path) return data;
    if (path === 'shell') return (data.shell ?? {}) as Record<string, unknown>;
    const key = path.split('.');
    let value: unknown = data;
    for (const part of key) {
      const current = value as Record<string, unknown> | undefined;
      if (current && Object.hasOwn(current, part)) value = current[part];
      else if (part === 'research' || part === 'researchReport' || part === 'researchCheckpoints' || part === 'researchGraph') {
        value = (data[part] as Record<string, unknown> | undefined) ?? {};
      } else value = undefined;
    }
    return value && typeof value === 'object' ? value as Record<string, unknown> : {};
  },
  async mutate<T extends Record<string, unknown>>(path: string, body?: unknown): Promise<T> {
    const data = await loadFixtures();
    const key = path.split('.');
    const leaf = key.pop()!;
    let parent: Record<string, unknown> = data;
    for (const part of key) {
      if (!parent[part] || typeof parent[part] !== 'object') parent[part] = {};
      parent = parent[part] as Record<string, unknown>;
    }
    const previous = parent[leaf] ?? {};
    const next = typeof body === 'object' && body !== null ? { ...(previous as object), ...(body as object) } : previous;
    parent[leaf] = next;
    persistFixtures(data);
    return next as T;
  },
  async collections(): Promise<Record<string, DemoRecord[]>> {
    const data = await loadFixtures();
    return ((data.collections as unknown as { advanced?: Record<string, DemoRecord[]> })?.advanced ?? {}) as Record<string, DemoRecord[]>;
  },
  /** Replace one root shell collection in isolated local demo storage. */
  async updateShellCollection<T>(name: string, update: (current: T[]) => T[]): Promise<T[]> {
    const data = await loadFixtures();
    const shell = (data.shell ?? {}) as Record<string, unknown>;
    const current = Array.isArray(shell[name]) ? shell[name] as T[] : [];
    const next = update(structuredClone(current));
    shell[name] = next;
    data.shell = shell;
    persistFixtures(data);
    return structuredClone(next);
  },
  /** Update an array fixture by dotted path while keeping the demo mutation boundary local. */
  async updateCollection<T>(path: string, update: (current: T[]) => T[]): Promise<T[]> {
    const data = await loadFixtures();
    const parts = path.split('.').filter(Boolean);
    const leaf = parts.pop();
    if (!leaf) throw new Error('A demo collection path is required.');
    let parent: Record<string, unknown> = data;
    for (const part of parts) {
      const child = parent[part];
      if (!child || typeof child !== 'object' || Array.isArray(child)) throw new Error(`Demo fixture path not found: ${path}`);
      parent = child as Record<string, unknown>;
    }
    const value = parent[leaf];
    if (!Array.isArray(value)) throw new Error(`Demo fixture collection not found: ${path}`);
    const next = update(structuredClone(value) as T[]);
    parent[leaf] = next;
    persistFixtures(data);
    return structuredClone(next);
  },
  /** Record an action that actually happened in this browser's isolated demo workspace. */
  async recordActivity(route: string, event: DemoRecord): Promise<void> {
    const data = await loadFixtures();
    const activity = (data.localActivity && typeof data.localActivity === 'object' ? data.localActivity : {}) as Record<string, unknown>;
    const current = Array.isArray(activity[route]) ? activity[route] as DemoRecord[] : [];
    activity[route] = [structuredClone(event), ...current].slice(0, 50);
    data.localActivity = activity;
    persistFixtures(data);
  },
  async getActivity(route: string): Promise<DemoRecord[]> {
    const data = await loadFixtures();
    const activity = data.localActivity && typeof data.localActivity === 'object' ? data.localActivity as Record<string, unknown> : {};
    return structuredClone(Array.isArray(activity[route]) ? activity[route] as DemoRecord[] : []);
  },
  /** Clear only one workspace's actual local Demo activity during that workspace reset. */
  async resetActivity(route: string): Promise<void> {
    const data = await loadFixtures();
    const activity = (data.localActivity && typeof data.localActivity === 'object' ? data.localActivity : {}) as Record<string, unknown>;
    delete activity[route];
    data.localActivity = activity;
    persistFixtures(data);
  },
  async reset(): Promise<void> {
    localStorage.removeItem(STORAGE_KEY);
    cachedData = undefined;
    await loadFixtures();
  },
};

const STORAGE_KEY = 'oa-second-isolated-demo-v1';
type FixtureFile = { sourceSha256: string; data: Record<string, unknown> };
let cachedData: Record<string, unknown> | undefined;

function normalize(input: Record<string, unknown>): Record<string, unknown> {
  return {
    shell: input.approvedShellFixtures ?? input.topLevel ?? {},
    collections: { advanced: input.advanced ?? {} },
    coreFixtures: input.coreFixtures ?? {},
    provenance: { sourceSha256: 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1' },
  };
}

async function loadFixtures(): Promise<Record<string, unknown>> {
  if (cachedData) return structuredClone(cachedData);
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      const parsed = JSON.parse(stored) as Record<string, unknown>;
      if (parsed.version === 1 && parsed.data && typeof parsed.data === 'object') {
        cachedData = parsed.data as Record<string, unknown>;
        return structuredClone(cachedData);
      }
    }
  } catch { /* Start with deterministic fixtures when local storage is unavailable. */ }
  const response = await fetch('/demo/reference-fixtures.json');
  if (!response.ok) throw new Error('The approved reference demo fixture could not be loaded.');
  const file = await response.json() as FixtureFile;
  if (file.sourceSha256 !== 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1') {
    throw new Error('Demo data does not match the approved frontend reference.');
  }
  const shellResponse = await fetch('/demo/approved-shell-fixtures.json');
  if (!shellResponse.ok) throw new Error('Approved HTML demo records could not be loaded.');
  const shellFile = await shellResponse.json() as { sourceSha256?: string; data?: Record<string, unknown> };
  if (shellFile.sourceSha256 !== file.sourceSha256 || !shellFile.data) throw new Error('Approved HTML demo records do not match the source design.');
  const wrapper = file.data as { topLevel?: Record<string, unknown>; advanced?: Record<string, unknown>; approvedShellFixtures?: Record<string, unknown> };
  cachedData = normalize({ ...wrapper, approvedShellFixtures: shellFile.data });
  return structuredClone(cachedData);
}

function persistFixtures(data: Record<string, unknown>) {
  cachedData = structuredClone(data);
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, data: cachedData })); } catch { /* Session remains usable without persistence. */ }
}
