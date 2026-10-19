import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { filterVaultEntries, vaultEntryMetadata } from './VaultPage';

const apiMocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../lib/react/api', () => ({
  api: apiMocks,
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => {
    const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : [];
    return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : [];
  },
}));

describe('masked Vault entry presentation', () => {
  it('projects only safe metadata for the entry list', () => {
    const metadata = vaultEntryMetadata({ id: 'entry-1', label: 'Deployment key', category: 'api_key', tags: ['prod'], secret: 'must-not-render', passphrase: 'also-private' });
    expect(metadata).toEqual({ id: 'entry-1', label: 'Deployment key', category: 'api_key', tags: ['prod'] });
    expect(JSON.stringify(metadata)).not.toContain('must-not-render');
    expect(JSON.stringify(metadata)).not.toContain('also-private');
  });

  it('filters by label, category and tags without requiring the secret value', () => {
    const rows = [{ id: 'one', label: 'Deployment', category: 'api_key', tags: ['production'], secret: 'hidden' }, { id: 'two', label: 'Notes', category: 'private_note' }];
    expect(filterVaultEntries(rows, 'prod')).toEqual([rows[0]]);
    expect(filterVaultEntries(rows, 'notes')).toEqual([rows[1]]);
    expect(filterVaultEntries(rows, '')).toHaveLength(2);
  });
});

describe('Vault empty and reveal states', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    history.replaceState({}, '', '/?project=vault-test');
    localStorage.clear();
    apiMocks.get.mockReset().mockImplementation(async (path: string) => path.includes('/entries') ? { entries: [] } : { total_entries: 0 });
    apiMocks.post.mockReset().mockResolvedValue({ secret: 'sensitive-value' });
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove(); localStorage.clear(); history.replaceState({}, '', '/');
  });

  it('keeps the empty vault focused on one add action without irrelevant filters', async () => {
    const { VaultPage } = await import('./VaultPage');
    await act(async () => { root.render(createElement(VaultPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.textContent).toContain('Your vault is empty');
    expect([...host.querySelectorAll('button')].filter(button => button.textContent?.includes('Add')).map(button => button.textContent?.trim())).toEqual(['Add your first entry']);
    expect(host.querySelector('[aria-label="Filter vault entries"]')).toBeNull();
    expect(host.querySelector('[aria-label="Filter category"]')).toBeNull();
  });

  it('conceals a revealed secret as soon as the page is hidden', async () => {
    apiMocks.get.mockImplementation(async (path: string) => path.includes('/entries') ? { entries: [{ id: 'entry-1', label: 'Deployment key', category: 'api_key' }] } : { total_entries: 1 });
    const { VaultPage } = await import('./VaultPage');
    await act(async () => { root.render(createElement(VaultPage)); await new Promise(resolve => setTimeout(resolve, 0)); });
    await act(async () => { host.querySelector<HTMLButtonElement>('.vault-reveal-trigger')?.click(); });
    const passphrase = host.querySelector<HTMLInputElement>('.vault-reveal-form input');
    expect(passphrase).toBeTruthy();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(passphrase, 'correct horse battery staple');
      passphrase?.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => { host.querySelector<HTMLButtonElement>('.vault-reveal-form button.primary')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
    expect(host.querySelector('.vault-revealed code')?.textContent).toBe('sensitive-value');
    Object.defineProperty(document, 'visibilityState', { configurable: true, value: 'hidden' });
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); });
    expect(host.querySelector('.vault-revealed')).toBeNull();
  });
});
