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

const ids = ['developer', 'engine', 'finetune', 'phd', 'science', 'scraper'] as const;
describe('dedicated platform workspaces', () => {
  it('covers the six inventory-driven surfaces and keeps their API contracts', () => {
    expect(ids).toHaveLength(6);
    for (const id of ids) {
      expect(resourcePageConfigs[id].reads.length).toBeGreaterThan(0);
      expect(resourcePageConfigs[id].reads.every(read => read.path.startsWith('/api/'))).toBe(true);
    }
    expect(resourcePageConfigs.developer.reads.map(x => x.path)).toContain('/api/developer/api-keys');
    expect(resourcePageConfigs.engine.reads.map(x => x.path)).toContain('/api/adaptive-engine/hardware?project={project}');
    expect(resourcePageConfigs.finetune.reads.map(x => x.path)).toContain('/api/finetune/jobs?project={project}');
    expect(resourcePageConfigs.phd.actions?.[1].payload({ command: 'draft' })).toMatchObject({ allow_external_model: false, allow_external_research: false });
    expect(resourcePageConfigs.science.actions?.[1].confirmText).toBeTruthy();
    expect(resourcePageConfigs.scraper.actions?.[1].payload({ url: 'https://example.org', allow_external: 'false' })).toMatchObject({ respect_robots: true, ingest: true, allow_external: false });
  });

  it('uses the allowlisted scalar summary rather than exposing arbitrary nested payloads', () => {
    expect(specialistFacts({ name: 'Local model', status: 'ready', settings: { password: 'secret' }, raw: { x: 1 } })).toEqual([['name', 'Local model'], ['status', 'ready']]);
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
