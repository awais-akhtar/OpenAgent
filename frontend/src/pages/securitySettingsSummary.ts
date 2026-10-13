import { record } from '../lib/react/api';

export type SummaryFact = { label: string; value: string };

const show = (value: unknown, fallback = 'Not reported') => value == null || value === '' ? fallback : String(value);
const count = (value: unknown) => Array.isArray(value) ? value.length : typeof value === 'number' ? value : undefined;
const bool = (value: unknown) => typeof value === 'boolean' ? value ? 'Allowed' : 'Denied' : undefined;

export function securityPostureFacts(value: unknown): SummaryFact[] {
  const status = record(value);
  const network = bool(status.network_default);
  const chain = bool(status.audit_chain_ok);
  return [
    ...(network ? [{ label: 'Network default', value: network }] : []),
    ...(chain ? [{ label: 'Audit chain', value: chain === 'Allowed' ? 'Verified' : 'Needs review' }] : []),
    ...(['egress_requests', 'sealed_audit_events', 'secrets', 'retention_policies', 'share_grants', 'privacy_reports', 'sandbox_profiles'] as const)
      .filter(key => count(status[key]) !== undefined)
      .map(key => ({ label: ({ egress_requests: 'Egress requests', sealed_audit_events: 'Sealed audit events', secrets: 'Secret records', retention_policies: 'Retention policies', share_grants: 'Share grants', privacy_reports: 'Privacy reports', sandbox_profiles: 'Sandbox profiles' } as const)[key], value: String(count(status[key])) })),
  ];
}

export function auditChainSummary(value: unknown): SummaryFact[] {
  const chain = record(value);
  const issues = Array.isArray(chain.issues) ? chain.issues.filter(item => item && typeof item === 'object').map(item => record(item)) : [];
  return [
    { label: 'Verification', value: chain.ok === true ? 'Verified' : chain.ok === false ? 'Needs review' : 'Not reported' },
    { label: 'Sealed events', value: show(chain.sealed, '0') },
    { label: 'Integrity issues', value: String(issues.length) },
    ...(issues.length ? [{ label: 'Issue types', value: [...new Set(issues.map(item => show(item.issue, 'Unknown issue')))].join(', ') }] : []),
  ];
}

export function sandboxEvaluationSummary(value: unknown): SummaryFact[] {
  const result = record(value);
  const profile = record(result.profile);
  const denied = Array.isArray(result.denied) ? result.denied.filter(item => item && typeof item === 'object').map(item => record(item)) : [];
  const permissions = Array.isArray(result.requested_permissions) ? result.requested_permissions.filter(item => typeof item === 'string') as string[] : [];
  return [
    { label: 'Decision', value: result.allowed === true ? 'Allowed' : result.allowed === false ? 'Denied' : 'Not reported' },
    ...(profile.name != null ? [{ label: 'Profile', value: String(profile.name) }] : []),
    { label: 'Requested network', value: result.requested_network === true ? 'Yes' : result.requested_network === false ? 'No' : 'Not reported' },
    { label: 'Requested permissions', value: permissions.length ? permissions.join(', ') : 'None' },
    { label: 'Denied requirements', value: denied.length ? denied.map(item => show(item.type, 'Unknown')).join(', ') : 'None' },
  ];
}

export function privacyReportSummary(value: unknown): SummaryFact[] {
  const report = record(value);
  const secretSection = record(report.secrets);
  const chain = record(report.audit_chain);
  const inventory = record(report.data_inventory);
  const entries = Object.entries(inventory).filter(([, item]) => typeof item === 'number');
  return [
    ...(report.generated_at != null ? [{ label: 'Generated', value: String(report.generated_at) }] : []),
    ...(report.network_default != null ? [{ label: 'Network default', value: report.network_default === false ? 'Denied' : report.network_default === true ? 'Allowed' : 'Not reported' }] : []),
    ...(secretSection.count != null ? [{ label: 'Secret records', value: String(secretSection.count) }] : []),
    ...(Array.isArray(report.retention_policies) ? [{ label: 'Retention policies', value: String(report.retention_policies.length) }] : []),
    ...(Array.isArray(report.share_grants) ? [{ label: 'Share grants', value: String(report.share_grants.length) }] : []),
    ...(Array.isArray(report.egress_requests) ? [{ label: 'Egress requests', value: String(report.egress_requests.length) }] : []),
    ...(Array.isArray(report.sandbox_profiles) ? [{ label: 'Sandbox profiles', value: String(report.sandbox_profiles.length) }] : []),
    ...(chain.ok != null ? [{ label: 'Audit chain', value: chain.ok === true ? 'Verified' : 'Needs review' }] : []),
    ...(entries.length ? [{ label: 'Tracked data records', value: String(entries.reduce((sum, [, item]) => sum + Number(item), 0)) }] : []),
  ];
}

export function modelPolicyFacts(policyValue: unknown, securityValue: unknown): SummaryFact[] {
  const policy = record(policyValue);
  const security = record(securityValue);
  const routeNames = Object.keys(record(policy.routes)).filter(name => ['local-first', 'local-server-first', 'offline', 'native-publisher', 'cloud-allowed'].includes(name));
  const fallbackNames = Object.keys(record(policy.fallbacks)).filter(name => ['none', 'local'].includes(name));
  return [
    { label: 'Routing policy', value: show(policy.routing_policy) },
    { label: 'Fallback policy', value: show(policy.fallback_policy) },
    { label: 'Available routing modes', value: routeNames.length ? routeNames.join(', ') : 'Not reported' },
    { label: 'Available fallback modes', value: fallbackNames.length ? fallbackNames.join(', ') : 'Not reported' },
    { label: 'Network default', value: security.network_default === false ? 'Denied' : security.network_default === true ? 'Allowed' : 'Not reported' },
    { label: 'External approval rule', value: Array.isArray(security.external_requires_explicit_approval) ? `${security.external_requires_explicit_approval.length} provider(s) require approval` : show(security.external_requires_explicit_approval) },
  ];
}

export function accountSummaryFacts(userValue: unknown): SummaryFact[] {
  const user = record(userValue);
  const allowed = [
    ['display_name', 'Display name'], ['username', 'Username'], ['email', 'Email'], ['role', 'Role'],
    ['status', 'Account status'], ['totp_enabled', 'Two-factor authentication'], ['created_at', 'Created'],
  ] as const;
  return allowed.filter(([key]) => user[key] != null).map(([key, label]) => ({
    label,
    value: key === 'totp_enabled' ? user[key] === true ? 'Enabled' : 'Disabled' : String(user[key]),
  }));
}

export function backupPreviewFacts(value: unknown): SummaryFact[] {
  const preview = record(value);
  const manifest = record(preview.manifest);
  const conflicts = Array.isArray(preview.conflicts) ? preview.conflicts.filter(item => typeof item === 'string') as string[] : [];
  const facts: SummaryFact[] = [];
  if (preview.restored === true) facts.push({ label: 'Restore', value: 'Completed' });
  else facts.push({ label: 'Archive status', value: show(preview.status) });
  if (manifest.created_at != null) facts.push({ label: 'Backup date', value: String(manifest.created_at) });
  if (manifest.project != null) facts.push({ label: 'Project', value: String(manifest.project) });
  if (manifest.include_media != null) facts.push({ label: 'Media included', value: manifest.include_media === true ? 'Yes' : 'No' });
  if (preview.media_entries != null) facts.push({ label: 'Media files', value: String(preview.media_entries) });
  if (preview.will_write != null) facts.push({ label: 'Will change current data', value: preview.will_write === true ? 'Yes' : 'No' });
  if (conflicts.length) facts.push({ label: 'Restore checks', value: conflicts.map(item => item.replaceAll('_', ' ')).join(', ') });
  if (preview.safety_copy != null) facts.push({ label: 'Safety copy', value: preview.safety_copy ? 'Created' : 'Not created' });
  return facts;
}

export function backupListFacts(value: unknown): SummaryFact[] {
  const backup = record(value);
  const manifest = record(backup.manifest);
  return [
    ...(manifest.created_at != null ? [{ label: 'Created', value: String(manifest.created_at) }] : []),
    ...(backup.size_bytes != null ? [{ label: 'Size', value: `${(Number(backup.size_bytes) / 1024 / 1024).toFixed(1)} MB` }] : []),
    ...(manifest.include_media != null ? [{ label: 'Media', value: manifest.include_media === true ? 'Included' : 'Not included' }] : []),
  ];
}
