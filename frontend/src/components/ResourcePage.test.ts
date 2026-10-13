import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const collection = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(collection) ? collection.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('resource workspace', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset().mockImplementation(async (path: string) => path === '/api/workflows?project=default&limit=100'
      ? { workflows: [{ id: 'wf-weekly', name: 'Weekly summary', status: 'enabled' }] }
      : {});
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

  it('keeps specialist actions tucked into one menu and resolves workflow IDs from loaded records', async () => {
    const { ResourcePage } = await import('./ResourcePage');
    const { resourcePageConfigs } = await import('../pages/ResourcePages');
    const config = resourcePageConfigs.automation;
    await act(async () => { root.render(createElement(ResourcePage, config)); await new Promise(resolve => setTimeout(resolve, 0)); });

    const menu = host.querySelector('details.specialist-action-menu') as HTMLDetailsElement;
    expect(menu.open).toBe(false);
    expect(menu.querySelectorAll('button')).toHaveLength(4);
    await act(async () => menu.querySelector('summary')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(menu.open).toBe(true);
    const dryRun = [...menu.querySelectorAll('button')].find(button => button.textContent?.includes('Run workflow in dry-run mode'));
    await act(async () => dryRun?.click());
    const select = document.body.querySelector('select[name="workflow_id"]') as HTMLSelectElement;
    expect(select).toBeTruthy();
    expect([...select.options].map(option => option.textContent)).toContain('Weekly summary · enabled');
    expect([...select.options].map(option => option.value)).toContain('wf-weekly');
  });
});
