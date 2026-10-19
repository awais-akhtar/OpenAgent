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
    expect(menu.querySelectorAll('button')).toHaveLength(3);
    expect(menu.textContent).not.toContain('Steps as JSON');
    expect(menu.textContent).toContain('Run workflow in dry-run mode');
    await act(async () => menu.querySelector('summary')?.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(menu.open).toBe(true);
    const dryRun = [...menu.querySelectorAll('button')].find(button => button.textContent?.includes('Run workflow in dry-run mode'));
    await act(async () => dryRun?.click());
    const select = document.body.querySelector('select[name="workflow_id"]') as HTMLSelectElement;
    expect(select).toBeTruthy();
    expect([...select.options].map(option => option.textContent)).toContain('Weekly summary · enabled');
    expect([...select.options].map(option => option.value)).toContain('wf-weekly');
  });

  it('supports arrow navigation and roving focus across the shared resource tabs', async () => {
    const { ResourcePage } = await import('./ResourcePage');
    const { resourcePageConfigs } = await import('../pages/ResourcePages');
    await act(async () => { root.render(createElement(ResourcePage, resourcePageConfigs.automation)); await new Promise(resolve => setTimeout(resolve, 0)); });
    const tabs = [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    expect(tabs.length).toBeGreaterThan(1);
    expect(tabs[0].tabIndex).toBe(0);
    expect(tabs[1].tabIndex).toBe(-1);
    await act(async () => { tabs[0].dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })); });
    expect(tabs[1].getAttribute('aria-selected')).toBe('true');
    expect(tabs[1].tabIndex).toBe(0);
    expect(document.activeElement).toBe(tabs[1]);
  });

  it('starts on the configured work surface and exposes its contextual primary action', async () => {
    const { ResourcePage } = await import('./ResourcePage');
    const { resourcePageConfigs } = await import('../pages/ResourcePages');
    await act(async () => { root.render(createElement(ResourcePage, { ...resourcePageConfigs.automation, primaryRead: 'Workflow library', primaryAction: 'Run workflow in dry-run mode' })); await new Promise(resolve => setTimeout(resolve, 0)); });
    const tabs = [...host.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    expect(tabs[0].getAttribute('aria-selected')).toBe('true');
    expect(tabs[0].textContent).toContain('Workflow library');
    const primary = host.querySelector<HTMLButtonElement>('.resource-primary-action');
    expect(primary?.textContent).toContain('Run workflow in dry-run mode');
    await act(async () => primary?.click());
    expect(document.body.querySelector('[role="dialog"] select[name="workflow_id"]')).toBeTruthy();
  });
});
