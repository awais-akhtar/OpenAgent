import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';

const get = vi.fn();
const post = vi.fn();
vi.mock('../lib/react/api', () => ({
  api: { get, post },
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key: string) => {
    const rows = value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : undefined;
    return Array.isArray(rows) ? rows : [];
  },
}));

describe('SecurityPage presentation and sensitive actions', () => {
  let host: HTMLDivElement;
  let root: Root;
  beforeEach(() => {
    (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    get.mockReset().mockImplementation(async (path: string) => {
      if (path === '/api/security/status') return { network_default: false, audit_chain_ok: true, secrets: 1 };
      if (path === '/api/security/audit-chain/verify') return { ok: true, sealed: 2 };
      if (path === '/api/security/secrets') return { secrets: [{ id: 'secret-1', name: 'Deployment key', status: 'active', updated_at: 'today' }] };
      if (path === '/api/audit?limit=80') return { events: [] };
      return {};
    });
    post.mockReset().mockResolvedValue({ secret: { value: 'briefly-visible' } });
    vi.stubGlobal('confirm', vi.fn(() => true));
    host = document.createElement('div'); document.body.append(host); root = createRoot(host);
  });
  afterEach(async () => {
    await act(async () => root.unmount());
    host.remove(); vi.unstubAllGlobals();
  });
  async function renderSecurity() {
    const { SecurityPage } = await import('./SecurityPage');
    await act(async () => { root.render(createElement(SecurityPage)); await Promise.resolve(); });
  }
  async function open(section: string) {
    await act(async () => {
      const groups: Record<string, string> = { 'Egress requests': 'Access', Shares: 'Access', Audit: 'Activity', Reports: 'Advanced', Retention: 'Advanced', Sandbox: 'Advanced' };
      const group = groups[section];
      if (group) [...host.querySelectorAll('button')].find(button => button.textContent?.trim() === group)?.click();
      [...host.querySelectorAll('button')].find(button => button.textContent?.trim() === section)?.click();
    });
  }

  it('keeps the overview concise and does not repeat the secret count', async () => {
    await renderSecurity();
    expect(host.textContent).toContain('Audit chain verified');
    expect(host.textContent).not.toContain('Trust / privacy posture');
    expect(host.textContent).not.toContain('Secret records');
    expect(host.textContent).not.toContain('The server remains authoritative');
    expect(host.textContent).not.toContain('Verify chain');
    expect(host.textContent).not.toContain('Generate report');
    expect(host.querySelector('.security-nav-item[aria-current="page"]')?.textContent).toBe('Overview');
    expect(host.textContent).toContain('External access review');
    expect(host.textContent).toContain('Recent security activity');
    expect(host.textContent).toContain('No external access requests need review.');
    expect(host.textContent).toContain('No recent security events.');
  });

  it('uses existing security data to surface reviews and recent audit entries with section links', async () => {
    get.mockImplementation(async (path: string) => {
      if (path === '/api/security/status') return { network_default: false, audit_chain_ok: true };
      if (path === '/api/security/egress') return { requests: [{ id: 'egress-1', destination: 'api.example.test', purpose: 'Send a report', resource_type: 'report', status: 'pending' }] };
      if (path === '/api/audit?limit=80') return { events: [{ id: 'audit-1', action: 'Secret access', actor: 'A user', created_at: '2026-10-13T10:00:00Z' }] };
      if (path === '/api/security/audit-chain/verify') return { ok: true, sealed: 2 };
      return {};
    });
    await renderSecurity();
    expect(host.textContent).toContain('api.example.test');
    expect(host.textContent).toContain('Pending review');
    expect(host.textContent).toContain('Secret access');
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Review')?.click(); });
    expect(host.textContent).toContain('Approve');
    await open('Overview');
    await act(async () => { [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'View audit log')?.click(); });
    expect(host.textContent).toContain('Chain integrity');
  });

  it('humanizes audit actions and suppresses opaque resource and actor identifiers', async () => {
    const opaqueId = '98f5b99e-0c39-4d80-90ce-9387342691dd';
    get.mockImplementation(async (path: string) => {
      if (path === '/api/security/status') return { audit_chain_ok: true };
      if (path === '/api/audit?limit=80') return { events: [
        { id: 'event-1', action: 'voice.persona.upsert', resource_type: 'voice_persona', resource: opaqueId, actor: opaqueId, created_at: '2026-10-13T10:00:00Z', status: 'ok' },
        { id: 'event-2', action: 'voice.persona.upsert', resource_type: 'voice_persona', resource: '1ac8d3c1-3970-4a9b-aa02-859c0e9b72f0', actor: opaqueId, created_at: '2026-10-13T09:59:00Z', status: 'ok' },
      ] };
      if (path === '/api/security/audit-chain/verify') return { ok: true, sealed: 2 };
      return {};
    });
    await renderSecurity();
    expect(host.textContent).toContain('Voice persona updated');
    const overviewRows = host.querySelectorAll('.security-overview-card[aria-labelledby="security-recent-title"] .record-row');
    expect(overviewRows).toHaveLength(1);
    expect(overviewRows[0].textContent).toContain('2 similar events');
    expect(host.textContent).not.toContain('voice.persona.upsert');
    expect(host.textContent).not.toContain(opaqueId);
    await open('Audit');
    expect(host.textContent).toContain('Voice persona updated');
    expect(host.textContent).toContain('Voice persona · Completed');
    expect(host.querySelectorAll('.security-layout > .panel .panel-body.list .record-row')).toHaveLength(2);
    expect(host.textContent).not.toContain(opaqueId);
  });

  it('keeps audit integrity and privacy report actions in their related sections', async () => {
    await renderSecurity();
    expect(host.textContent).not.toContain('Verify chain');
    expect(host.textContent).not.toContain('Generate report');
    expect(host.textContent).toContain('Advanced');
    await open('Audit');
    expect(host.textContent).toContain('Verify chain');
    expect(host.textContent).toContain('Seal events');
    expect(host.textContent).not.toContain('Generate report');
    await open('Reports');
    expect(host.textContent).toContain('Generate report');
    expect(host.textContent).not.toContain('Seal events');
  });

  it('keeps common access and activity sections visible while grouping diagnostics under Advanced', async () => {
    await renderSecurity();
    const primary = [...host.querySelectorAll('.security-nav-item')].map(button => button.textContent?.trim());
    expect(primary).toEqual(['Overview', 'Access', 'Secrets', 'Activity', 'Advanced']);
    expect(host.textContent).not.toContain('Sandbox');
    await open('Advanced');
    expect(host.textContent).toContain('Sandbox');
    expect(host.textContent).toContain('Retention');
  });

  it('shows a compact purposeful empty state for audit history', async () => {
    await renderSecurity(); await open('Audit');
    expect(host.textContent).toContain('No audit events have been recorded.');
    expect(host.querySelector('.security-empty')).toBeTruthy();
    expect(host.textContent).not.toContain('No summary fields were returned');
  });

  it('retains passphrase confirmation, reveal API, and timed masked-secret behavior', async () => {
    await renderSecurity(); await open('Secrets');
    expect(host.textContent).toContain('Deployment key');
    const passphrase = host.querySelector<HTMLInputElement>('input[type="password"]');
    expect(passphrase).toBeTruthy();
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(passphrase, 'step-up'); passphrase?.dispatchEvent(new Event('input', { bubbles: true }));
    });
    const reveal = [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Reveal for 15s');
    expect(reveal).toBeTruthy();
    await act(async () => { reveal?.click(); await Promise.resolve(); });
    expect(confirm).toHaveBeenCalledWith(expect.stringContaining('recorded in the audit log'));
    expect(post).toHaveBeenCalledWith('/api/security/secrets/reveal', { secret_id: 'secret-1', passphrase: 'step-up' });
    expect(host.textContent).toContain('briefly-visible');
  });

  it('conceals revealed values and clears step-up material when leaving Secrets', async () => {
    await renderSecurity(); await open('Secrets');
    const passphrase = host.querySelector<HTMLInputElement>('input[type="password"]');
    await act(async () => {
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      setter?.call(passphrase, 'step-up'); passphrase?.dispatchEvent(new Event('input', { bubbles: true }));
    });
    await act(async () => {
      [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Reveal for 15s')?.click();
      await Promise.resolve();
    });
    expect(host.textContent).toContain('briefly-visible');
    await open('Audit');
    expect(host.textContent).not.toContain('briefly-visible');
    await open('Secrets');
    expect(host.textContent).not.toContain('briefly-visible');
    expect(host.querySelector<HTMLInputElement>('input[type="password"]')?.value).toBe('');
  });

  it('preserves the last successful records and surfaces partial refresh failures', async () => {
    await renderSecurity();
    get.mockImplementation(async (path: string) => {
      if (path === '/api/security/secrets') throw new Error('Temporary secret service outage');
      return {};
    });
    await act(async () => {
      [...host.querySelectorAll<HTMLButtonElement>('button')].find(button => button.textContent === 'Refresh')?.click();
      await Promise.resolve();
    });
    await open('Secrets');
    expect(host.textContent).toContain('Deployment key');
    expect(host.textContent).toContain('Some security records could not be refreshed. Showing the last available data.');
  });
});
