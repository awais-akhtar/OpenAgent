import { describe, expect, it } from 'vitest';
import {
  accountSummaryFacts, auditChainSummary, backupListFacts, backupPreviewFacts,
  modelPolicyFacts, privacyReportSummary, sandboxEvaluationSummary, securityPostureFacts,
} from './securitySettingsSummary';

describe('curated Security and Settings projections', () => {
  it('projects service security status fields only', () => {
    expect(securityPostureFacts({ network_default: false, audit_chain_ok: true, egress_requests: 2, secrets: 1, ignored_payload: { token: 'secret' } })).toEqual([
      { label: 'Network default', value: 'Denied' }, { label: 'Audit chain', value: 'Verified' },
      { label: 'Egress requests', value: '2' }, { label: 'Secret records', value: '1' },
    ]);
  });

  it('summarizes chain health and issue types without exposing hashes or event identifiers', () => {
    const facts = auditChainSummary({ ok: false, sealed: 7, head: 'hash-value', issues: [{ audit_event_id: 'event-secret', issue: 'event_hash_mismatch' }] });
    expect(facts).toEqual([
      { label: 'Verification', value: 'Needs review' }, { label: 'Sealed events', value: '7' },
      { label: 'Integrity issues', value: '1' }, { label: 'Issue types', value: 'event_hash_mismatch' },
    ]);
    expect(JSON.stringify(facts)).not.toContain('hash-value');
    expect(JSON.stringify(facts)).not.toContain('event-secret');
  });

  it('projects privacy reports into counts and posture, omitting secret names and raw inventory keys', () => {
    const facts = privacyReportSummary({
      generated_at: '2026-10-13T00:00:00Z', network_default: false,
      secrets: { count: 2, names: ['private-a', 'private-b'] }, retention_policies: [{ id: '1' }],
      share_grants: [], egress_requests: [{ id: 'x' }], sandbox_profiles: [],
      audit_chain: { ok: true, head: 'hash' }, data_inventory: { events: 3, conversations: 4 },
    });
    expect(facts).toContainEqual({ label: 'Secret records', value: '2' });
    expect(facts).toContainEqual({ label: 'Tracked data records', value: '7' });
    expect(JSON.stringify(facts)).not.toContain('private-a');
    expect(JSON.stringify(facts)).not.toContain('hash');
  });

  it('keeps sandbox decision fields while omitting filesystem, environment and nested profile data', () => {
    const facts = sandboxEvaluationSummary({
      allowed: false, requested_network: true, requested_permissions: ['filesystem.read'],
      denied: [{ type: 'network', values: ['external'] }],
      profile: { name: 'restricted', filesystem: { read: ['/secret'] }, env: { TOKEN: 'secret' } },
    });
    expect(facts).toContainEqual({ label: 'Decision', value: 'Denied' });
    expect(facts).toContainEqual({ label: 'Denied requirements', value: 'network' });
    expect(JSON.stringify(facts)).not.toContain('/secret');
    expect(JSON.stringify(facts)).not.toContain('TOKEN');
  });

  it('summarizes model routing choices, not provider configuration objects', () => {
    const facts = modelPolicyFacts({ routing_policy: 'local-first', fallback_policy: 'local', routes: { 'local-first': ['provider-a'], offline: ['local'], cloud: ['provider-secret'] }, fallbacks: { local: ['local'] } }, { network_default: false });
    expect(facts.find(item => item.label === 'Available routing modes')?.value).toBe('local-first, offline');
    expect(JSON.stringify(facts)).not.toContain('provider-secret');
  });

  it('uses the safe account allowlist and backup preview/list projections', () => {
    const user = accountSummaryFacts({ username: 'alice', role: 'owner', status: 'active', totp_enabled: true, api_token: 'never-show' });
    expect(user.map(item => item.label)).toEqual(['Username', 'Role', 'Account status', 'Two-factor authentication']);
    expect(JSON.stringify(user)).not.toContain('never-show');
    expect(backupPreviewFacts({ status: 'ready', archive: 'C:\\private\\archive.zip', manifest: { created_at: 'today', project: 'p', include_media: true }, conflicts: [], will_write: false, media_entries: 2 })).toContainEqual({ label: 'Will change current data', value: 'No' });
    const backup = backupListFacts({ path: 'C:\\private\\archive.zip', size_bytes: 1048576, manifest: { created_at: 'today', include_media: false } });
    expect(JSON.stringify(backup)).not.toContain('private');
  });
});
