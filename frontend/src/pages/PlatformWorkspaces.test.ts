import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { resourcePageConfigs } from './ResourcePages';
import { specialistFacts } from '../features/specialists/SpecialistInspector';

const apiMocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() }));
vi.mock('../lib/react/api', () => ({
  api: apiMocks,
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => { const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : []; return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : []; },
}));

const ids = ['developer', 'engine', 'finetune', 'science', 'scraper'] as const;
describe('dedicated platform workspaces', () => {
  it('covers the platform inventory surfaces and keeps their API contracts', () => {
    expect(ids).toHaveLength(5);
    for (const id of ids) {
      expect(resourcePageConfigs[id].reads.length).toBeGreaterThan(0);
      expect(resourcePageConfigs[id].reads.every(read => read.path.startsWith('/api/'))).toBe(true);
    }
    expect(resourcePageConfigs.developer.reads.map(x => x.path)).toContain('/api/developer/api-keys');
    expect(resourcePageConfigs.engine.reads.map(x => x.path)).toContain('/api/adaptive-engine/hardware?project={project}');
    expect(resourcePageConfigs.finetune.reads.map(x => x.path)).toContain('/api/finetune/jobs?project={project}');
    expect(resourcePageConfigs.science.actions?.[1].confirmText).toBeTruthy();
    expect(resourcePageConfigs.scraper.actions?.[1].payload({ url: 'https://example.org', allow_external: 'false' })).toMatchObject({ respect_robots: true, ingest: true, allow_external: false });
  });

  it('uses the allowlisted scalar summary rather than exposing arbitrary nested payloads', () => {
    expect(specialistFacts({ name: 'Local model', status: 'ready', settings: { password: 'secret' }, raw: { x: 1 } })).toEqual([['name', 'Local model'], ['status', 'ready']]);
  });

  describe('task-led defaults', () => {
    let host: HTMLDivElement; let root: Root;
    beforeEach(() => {
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
      localStorage.clear(); history.replaceState({}, '', '/');
      apiMocks.get.mockReset().mockImplementation(async (path: string) => {
        if (path.includes('/adaptive-engine/status')) return { status: 'ready' };
        if (path.includes('/adaptive-engine/hardware')) return { profile: { platform: 'Linux', cpu_threads: 16, devices: [{ id: 'gpu-1', name: 'NVIDIA RTX 5000', available: true }] } };
        if (path.includes('/adaptive-engine/plans')) return { plans: [{ id: 'plan-1', name: 'Local inference' }] };
        if (path.includes('/adaptive-engine/benchmarks')) return { benchmarks: [{ id: 'bench-1', device_name: 'NVIDIA RTX 5000', status: 'complete' }] };
        if (path.includes('/finetune/status')) return { status: 'ready' };
        if (path.includes('/finetune/datasets')) return { datasets: [{ id: 'data-1', name: 'Support examples' }] };
        if (path.includes('/finetune/jobs')) return { jobs: [{ id: 'job-1', base_model: 'local-3b', state: 'queued' }] };
        if (path.includes('/finetune/adapters')) return { adapters: [{ id: 'adapter-1', name: 'Support adapter', status: 'registered' }] };
        if (path.includes('/finetune/templates')) return { templates: [{ id: 'recipe-1', name: 'LoRA starter' }] };
        if (path.includes('/science/status')) return { status: 'ready' };
        if (path.includes('/science/tools')) return { tools: [{ id: 'tool-1', name: 'Materials solver', status: 'available' }] };
        if (path.includes('/science/materials/catalog')) return { datasets: [{ dataset_id: 'mat-1', name: 'Crystal properties', license: 'CC-BY-4.0' }] };
        if (path.includes('/science/materials/datasets')) return { datasets: [{ id: 'mat-local', name: 'Local measurements' }] };
        if (path.includes('/science/memory')) return { memory: [{ id: 'note-1', name: 'Lattice note' }] };
        if (path.includes('/scraper/status')) return { status: 'ready' };
        if (path.includes('/scraper/jobs')) return { jobs: [{ id: 'scrape-1', label: 'Docs refresh', url: 'https://example.org/docs', next_run_at: 'Tomorrow' }] };
        if (path.includes('/developer/status')) return { status: 'ready' };
        if (path.includes('/developer/tools')) return { tools: [{ id: 'tool-1', name: 'Repo search' }] };
        if (path.includes('/developer/plugins')) return { plugins: [{ id: 'plugin-1', name: 'Local runner' }] };
        if (path.includes('/developer/mcp/servers')) return { servers: [{ id: 'server-1', name: 'Workspace MCP' }] };
        if (path.includes('/developer/webhooks')) return { webhooks: [] };
        if (path.includes('/developer/api-keys')) return { api_keys: [] };
        return {};
      });
      host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    });
    afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

    it.each([
      ['engine', 'Local inference capacity', 'NVIDIA RTX 5000', 'Benchmark a device'],
      ['finetune', 'Dataset to adapter', 'Support examples', 'Create a training job'],
      ['science', 'Choose data and run an analysis', 'Crystal properties', 'Analyze a catalog dataset'],
      ['scraper', 'Scheduled scraping', 'Docs refresh', 'Schedule a scrape'],
      ['developer', 'Integrations and credentials', 'Repo search', 'Create scoped API key'],
    ] as const)('%s opens on its user workflow rather than endpoint navigation', async (id, heading, recordName, primaryAction) => {
      const { PlatformWorkspaces } = await import('./PlatformWorkspaces');
      await act(async () => root.render(createElement(PlatformWorkspaces, { id })));
      expect(host.textContent).toContain(heading);
      expect(host.textContent).toContain(recordName);
      if (id === 'engine') expect(host.textContent).toContain('16 CPU threads');
      expect([...host.querySelectorAll<HTMLButtonElement>('.platform-workflow-action')].some(button => button.textContent?.includes(primaryAction))).toBe(true);
      const technical = host.querySelector<HTMLDetailsElement>('.platform-technical');
      expect(technical?.open).toBe(false);
      expect(host.textContent).not.toMatch(/\btrue\b|\bfalse\b/);
      expect(apiMocks.get).toHaveBeenCalledTimes(resourcePageConfigs[id].reads.length);
      expect(host.querySelector('.platform-technical[open] .platform-nav')).toBeNull();
    });

    it('opens a compact selected-record inspector from the runtime workbench list', async () => {
      const { PlatformWorkspaces } = await import('./PlatformWorkspaces');
      await act(async () => root.render(createElement(PlatformWorkspaces, { id: 'engine' })));
      const device = [...host.querySelectorAll<HTMLButtonElement>('.platform-workflow-list button')].find(button => button.textContent?.includes('NVIDIA RTX 5000'));
      expect(device).toBeTruthy();
      await act(async () => device?.click());
      expect(host.querySelector('[aria-label="Hardware inventory selected details"]')?.textContent).toContain('NVIDIA RTX 5000');
      expect(host.querySelector('[aria-label="Hardware inventory selected details"]')?.textContent).not.toMatch(/\btrue\b/);
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Close selected details"]')?.click());
      expect(host.querySelector('[aria-label="Hardware inventory selected details"]')).toBeNull();
    });

    it('opens the endpoint inspector on demand and keeps API collections reachable', async () => {
      const { PlatformWorkspaces } = await import('./PlatformWorkspaces');
      await act(async () => root.render(createElement(PlatformWorkspaces, { id: 'engine' })));
      await act(async () => host.querySelector<HTMLElement>('.platform-technical > summary')?.click());
      const hardwareButton = [...host.querySelectorAll<HTMLButtonElement>('.platform-nav button')].find(button => button.textContent?.includes('Hardware inventory'));
      await act(async () => hardwareButton?.click());
      expect(host.querySelector('.platform-main')).toBeTruthy();
      expect(host.textContent).toContain('NVIDIA RTX 5000');
    });

    it('populates the engine benchmark device selector from the hardware profile response', async () => {
      const { PlatformWorkspaces } = await import('./PlatformWorkspaces');
      await act(async () => root.render(createElement(PlatformWorkspaces, { id: 'engine' })));
      const benchmark = [...host.querySelectorAll<HTMLButtonElement>('.platform-workflow-action')].find(button => button.textContent?.includes('Benchmark a device'));
      await act(async () => benchmark?.click());
      const deviceSelector = document.querySelector<HTMLSelectElement>('.platform-dialog select');
      expect(deviceSelector).toBeTruthy();
      expect([...deviceSelector?.options ?? []].map(option => option.textContent)).toContain('NVIDIA RTX 5000');
    });
  });

  describe('one-time developer API key', () => {
    let host: HTMLDivElement; let root: Root;
    beforeEach(() => {
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
      localStorage.clear(); history.replaceState({}, '', '/');
      apiMocks.get.mockReset().mockImplementation(async (path: string) => path === '/api/developer/api-keys' ? { api_keys: [] } : path === '/api/developer/webhooks' ? { webhooks: [] } : path === '/api/developer/tools' ? { tools: [] } : path === '/api/developer/plugins' ? { plugins: [] } : path === '/api/developer/mcp/servers' ? { servers: [] } : path === '/api/developer/extensions' ? { packages: [] } : path === '/api/developer/sdk' ? { artifacts: [] } : { api_keys: 0, webhooks: 0 });
      apiMocks.post.mockReset().mockResolvedValue({ api_key: { id: 'key-1', label: 'CLI', key: 'oa_dev_one-time-secret' } });
      host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    });
    afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

    it('shows the generated key once, supports conceal/copy, and clears it without persistence', async () => {
      const { PlatformWorkspaces } = await import('./PlatformWorkspaces');
      await act(async () => root.render(createElement(PlatformWorkspaces, { id: 'developer' })));
      const open = [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Create scoped API key'));
      await act(async () => open?.click());
      const labelInput = document.querySelector<HTMLInputElement>('.platform-dialog input[required]');
      expect(labelInput).toBeTruthy();
      if (labelInput) await act(async () => { const setter = Object.getOwnPropertyDescriptor(Object.getPrototypeOf(labelInput), 'value')?.set; setter?.call(labelInput, 'CLI'); labelInput.dispatchEvent(new Event('input', { bubbles: true })); labelInput.dispatchEvent(new Event('change', { bubbles: true })); });
      const review = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Review operation');
      await act(async () => review?.click());
      const confirm = [...document.querySelectorAll('button')].find(button => button.textContent?.trim() === 'Confirm and continue');
      await act(async () => confirm?.click());
      expect(apiMocks.post).toHaveBeenCalledWith('/api/developer/api-keys', { label: 'CLI', scopes: [] });
      const keyOutput = host.querySelector('output[aria-label="Generated API key"]');
      expect(keyOutput?.textContent).toBe('oa_dev_one-time-secret');
      expect(localStorage.getItem('oa-dev-api-key')).toBeNull();

      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Conceal key"]')?.click());
      expect(host.querySelector('output[aria-label="Generated API key concealed"]')?.textContent).not.toContain('oa_dev_one-time-secret');
      const clipboard = { writeText: vi.fn().mockResolvedValue(undefined) };
      Object.defineProperty(navigator, 'clipboard', { configurable: true, value: clipboard });
      await act(async () => [...host.querySelectorAll('button')].find(button => button.textContent?.includes('Copy key'))?.click());
      expect(clipboard.writeText).toHaveBeenCalledWith('oa_dev_one-time-secret');
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Dismiss and clear generated key"]')?.click());
      expect(host.querySelector('[aria-label="One-time API key"]')).toBeNull();
      expect(localStorage.getItem('oa-dev-api-key')).toBeNull();
    });
  });
});
