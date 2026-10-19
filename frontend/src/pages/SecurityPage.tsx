import { useEffect, useRef, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { auditEventContext, auditEventOutcome, auditEventTitle } from '../lib/react/auditPresentation';
import { PageFrame } from '../components/PageFrame';
import { auditChainSummary, privacyReportSummary, sandboxEvaluationSummary, securityPostureFacts } from './securitySettingsSummary';
import './security-page.css';

type Row = Record<string, unknown>;
function SummaryFacts({ facts }: { facts: Array<{ label: string; value: string }> }) {
  return facts.length ? <dl className="facts">{facts.map(fact => <div key={fact.label}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}</dl> : null;
}
function SecurityEmpty({ children }: { children: string }) { return <p className="security-empty">{children}</p>; }
const sectionGroups = {
  Overview: ['Overview'],
  Access: ['Egress requests', 'Shares'],
  Secrets: ['Secrets'],
  Activity: ['Audit'],
  Advanced: ['Reports', 'Retention', 'Sandbox'],
} as const;
type Section = typeof sectionGroups[keyof typeof sectionGroups][number];
type SectionGroup = keyof typeof sectionGroups;
const sectionGroup = (section: Section): SectionGroup =>
  (Object.entries(sectionGroups).find(([, children]) => (children as readonly string[]).includes(section))?.[0] ?? 'Overview') as SectionGroup;
const value = (row: Row, key: string, fallback = '—') => row[key] == null || row[key] === '' ? fallback : String(row[key]);
const pending = (row: Row) => ['pending', 'requested', 'awaiting_approval'].includes(value(row, 'status').toLowerCase());
const readableStatus = (row: Row) => pending(row) ? 'Pending review' : value(row, 'status', 'Status unavailable').replaceAll('_', ' ').replace(/\b\w/g, character => character.toUpperCase());

export function SecurityPage() {
  const [data, setData] = useState<Record<string, unknown>>({});
  const [section, setSection] = useState<Section>('Overview');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [purpose, setPurpose] = useState('');
  const [destination, setDestination] = useState('');
  const [resource, setResource] = useState('');
  const [retentionType, setRetentionType] = useState('context_events');
  const [retentionDays, setRetentionDays] = useState('365');
  const [shareType, setShareType] = useState('document');
  const [shareId, setShareId] = useState('');
  const [grantee, setGrantee] = useState('');
  const [secretName, setSecretName] = useState('');
  const [secretValue, setSecretValue] = useState('');
  const [secretPassphrase, setSecretPassphrase] = useState('');
  const [stepUp, setStepUp] = useState<Record<string, string>>({});
  const [rotateValues, setRotateValues] = useState<Record<string, string>>({});
  const [revealed, setRevealed] = useState<Record<string, string>>({});
  const revealTimers = useRef(new Map<string, number>());
  const activeRefresh = useRef<AbortController | null>(null);
  const refreshSequence = useRef(0);
  const [sandboxId, setSandboxId] = useState('');
  const [permissionText, setPermissionText] = useState('');
  const [network, setNetwork] = useState(false);
  const [sandboxResult, setSandboxResult] = useState<Row>({});
  const [profileName, setProfileName] = useState('');
  const [profilePermissions, setProfilePermissions] = useState('');
  const [profileNetwork, setProfileNetwork] = useState(false);
  const [lastAction, setLastAction] = useState<Row>({});
  const [dryRun, setDryRun] = useState(true);

  async function refresh() {
    activeRefresh.current?.abort();
    const controller = new AbortController();
    activeRefresh.current = controller;
    const requestId = ++refreshSequence.current;
    setRefreshing(true); setError('');
    const paths = [
      ['status', '/api/security/status'], ['egress', '/api/security/egress'], ['retention', '/api/security/retention'],
      ['secrets', '/api/security/secrets'], ['shares', '/api/security/shares'], ['reports', '/api/security/privacy-reports'],
      ['audit', '/api/audit?limit=80'], ['chain', '/api/security/audit-chain/verify'], ['profiles', '/api/security/sandbox-profiles'],
    ] as const;
    try {
      const result = await Promise.allSettled(paths.map(([, path]) => api.get<unknown>(path, controller.signal)));
      if (controller.signal.aborted || requestId !== refreshSequence.current) return;
      setData(previous => {
        const next = { ...previous };
        result.forEach((item, i) => { if (item.status === 'fulfilled') next[paths[i][0]] = item.value; });
        return next;
      });
      const failures = result.filter((item): item is PromiseRejectedResult => item.status === 'rejected');
      if (failures.length) {
        const statusFailure = result[0].status === 'rejected' ? result[0].reason : undefined;
        setError(statusFailure instanceof Error
          ? statusFailure.message
          : 'Some security records could not be refreshed. Showing the last available data.');
      }
    } finally {
      if (requestId === refreshSequence.current) {
        if (activeRefresh.current === controller) activeRefresh.current = null;
        setLoading(false);
        setRefreshing(false);
      }
    }
  }
  useEffect(() => {
    void refresh();
    return () => {
      refreshSequence.current += 1;
      activeRefresh.current?.abort();
      activeRefresh.current = null;
    };
  }, []);
  useEffect(() => {
    if (section !== 'Secrets') {
      for (const timer of revealTimers.current.values()) window.clearTimeout(timer);
      revealTimers.current.clear();
      setRevealed({});
      setStepUp({});
      setRotateValues({});
    }
  }, [section]);
  useEffect(() => () => { for (const timer of revealTimers.current.values()) window.clearTimeout(timer); revealTimers.current.clear(); }, []);
  const array = (key: string, child: string) => records(data[key], child);
  const status = record(data.status);
  const chain = record(data.chain);
  const pendingEgress = array('egress', 'requests').filter(pending);
  const recentAuditGroups = new Map<string, { event: Row; title: string; context?: string; outcome?: string; count: number }>();
  for (const event of array('audit', 'events')) {
    const title = auditEventTitle(event);
    const context = auditEventContext(event);
    const outcome = auditEventOutcome(event);
    const key = [title, context ?? '', outcome ?? ''].join('\u0000');
    const existing = recentAuditGroups.get(key);
    if (existing) existing.count += 1;
    else recentAuditGroups.set(key, { event, title, ...(context ? { context } : {}), ...(outcome ? { outcome } : {}), count: 1 });
  }
  const recentAudit = [...recentAuditGroups.values()].slice(0, 5);

  async function act(path: string, body?: unknown, confirmation?: string) {
    if (confirmation && !confirm(confirmation)) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>(path, body));
      setLastAction(result); setNotice('Security settings updated.'); await refresh(); return result;
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Security request failed.'); return undefined; }
    finally { setBusy(false); }
  }
  async function submitEgress(event: React.FormEvent) {
    event.preventDefault();
    const result = await act('/api/security/egress/request', { requester: 'user', purpose: purpose.trim(), destination: destination.trim(), resource_type: resource.trim() || 'unknown', resource_id: '', risk_level: 'medium', metadata: {} });
    if (result) { setPurpose(''); setDestination(''); setResource(''); }
  }
  async function storeSecret(event: React.FormEvent) {
    event.preventDefault();
    const result = await act('/api/security/secrets', { name: secretName.trim(), value: secretValue, passphrase: secretPassphrase, description: '' });
    setSecretValue(''); setSecretPassphrase(''); if (result) setSecretName('');
  }
  async function revealSecret(secretId: string) {
    const passphrase = stepUp[secretId] ?? '';
    if (!passphrase || !confirm('Reveal this secret value briefly? The event is recorded in the audit log.')) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>('/api/security/secrets/reveal', { secret_id: secretId, passphrase }));
      const secret = record(result.secret);
      if (secret.value != null) {
        setRevealed(current => ({ ...current, [secretId]: String(secret.value) })); setStepUp(current => ({ ...current, [secretId]: '' }));
        const previous = revealTimers.current.get(secretId); if (previous) window.clearTimeout(previous);
        revealTimers.current.set(secretId, window.setTimeout(() => { revealTimers.current.delete(secretId); setRevealed(current => { const next = { ...current }; delete next[secretId]; return next; }); }, 15000));
        setNotice('Secret revealed briefly. The server recorded this access.'); await refresh();
      }
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Secret reveal failed.'); }
    finally { setBusy(false); }
  }
  async function rotateSecret(secretId: string, name: string) {
    const passphrase = stepUp[secretId] ?? '';
    if (!passphrase || !confirm(`Replace the stored value for “${name}”? The server records this action.`)) return;
    const replacement = rotateValues[secretId] ?? '';
    if (!replacement) return;
    const result = await act('/api/security/secrets/rotate', { secret_id: secretId, value: replacement, passphrase });
    setRotateValues(current => ({ ...current, [secretId]: '' }));
    if (result) { setStepUp(current => ({ ...current, [secretId]: '' })); setRevealed(current => { const next = { ...current }; delete next[secretId]; return next; }); }
  }
  async function createRetention(event: React.FormEvent) {
    event.preventDefault();
    await act('/api/security/retention', { resource_type: retentionType, retention_days: Number(retentionDays), action: 'delete', status: 'active', metadata: {} });
  }
  async function createShare(event: React.FormEvent) {
    event.preventDefault();
    const result = await act('/api/security/shares', { resource_type: shareType, resource_id: shareId.trim(), grantee: grantee.trim(), scope: 'read', expires_at: '', metadata: {} });
    if (result) { setShareId(''); setGrantee(''); }
  }
  async function evaluateSandbox(event: React.FormEvent) {
    event.preventDefault();
    const result = await act('/api/security/sandbox/evaluate', { profile_id: sandboxId, permissions: permissionText.split(/[\s,]+/).filter(Boolean), network });
    if (result) setSandboxResult(result);
  }
  async function saveSandboxProfile(event: React.FormEvent) {
    event.preventDefault();
    if (!confirm(`Create sandbox profile “${profileName.trim()}” with ${profileNetwork ? 'network access enabled' : 'network access denied'}? The profile is saved by the service.`)) return;
    const result = await act('/api/security/sandbox-profiles', { name: profileName.trim(), permissions: profilePermissions.split(/[\s,]+/).filter(Boolean), network_allowed: profileNetwork, filesystem: {}, env: {}, metadata: {} });
    if (result) { setProfileName(''); setProfilePermissions(''); setProfileNetwork(false); }
  }

  return <PageFrame eyebrow="" title="Security" description="" actions={<button className="btn" onClick={() => void refresh()} disabled={loading || refreshing || busy}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    {loading ? <p className="empty">Reading security records…</p> : <>
      <section className="panel panel-body security-overview"><div className="security-posture"><span className={`security-indicator ${chain.ok === false ? 'is-warning' : chain.ok === true ? 'is-good' : ''}`} aria-hidden="true"/><div><h2>{chain.ok === true ? 'Audit chain verified' : chain.ok === false ? 'Audit chain needs review' : 'Security posture'}</h2><SummaryFacts facts={securityPostureFacts(status).filter(fact => !['Secret records', 'Audit chain'].includes(fact.label))}/></div></div></section>
      <div className="security-workbench">
      <nav className="security-nav" aria-label="Security sections">{Object.keys(sectionGroups).map(item => {
        const group = item as SectionGroup;
        return <button className="security-nav-item" key={group} onClick={() => setSection(sectionGroups[group][0])} aria-current={sectionGroup(section) === group ? 'page' : undefined}>{group}</button>;
      })}</nav>
      <div className="security-workbench-content">
      {sectionGroups[sectionGroup(section)].length > 1 && <nav className="security-subnav" aria-label={`${sectionGroup(section)} options`}>{sectionGroups[sectionGroup(section)].map(item => <button className="security-subnav-item" key={item} onClick={() => setSection(item)} aria-current={section === item ? 'page' : undefined}>{item}</button>)}</nav>}
      {section === 'Overview' && <div className="security-overview-grid">
        <section className="panel security-overview-card" aria-labelledby="security-review-title">
          <div className="panel-head"><h2 id="security-review-title">External access review</h2><span>{pendingEgress.length} pending</span></div>
          <div className="panel-body list">
            {pendingEgress.slice(0, 4).map((request, index) => <article className="record-row" key={value(request, 'id', String(index))}>
              <div><strong>{value(request, 'destination', 'Destination not reported')}</strong><p>{value(request, 'purpose', 'Purpose not reported')}</p><small>{value(request, 'resource_type', 'Resource')} · {readableStatus(request)}</small></div>
              <button className="btn" onClick={() => setSection('Egress requests')}>Review</button>
            </article>)}
            {pendingEgress.length === 0 && <SecurityEmpty>No external access requests need review.</SecurityEmpty>}
            <div className="security-overview-link"><button className="btn" onClick={() => setSection('Egress requests')}>Open requests</button></div>
          </div>
        </section>
        <section className="panel security-overview-card" aria-labelledby="security-recent-title">
          <div className="panel-head"><h2 id="security-recent-title">Recent security activity</h2><span>{recentAudit.length} recent</span></div>
          <div className="panel-body list">
            {recentAudit.map(({ event, title, context, outcome, count }, index) => <article className="record-row" key={`${title}-${context ?? ''}-${outcome ?? ''}-${index}`}>
              <div><strong>{title}</strong><p>{[context, outcome, count > 1 ? `${count} similar events` : undefined].filter(Boolean).join(' · ')}</p></div>
              <time>{value(event, 'created_at')}</time>
            </article>)}
            {recentAudit.length === 0 && <SecurityEmpty>No recent security events.</SecurityEmpty>}
            <div className="security-overview-link"><button className="btn" onClick={() => setSection('Audit')}>View audit log</button></div>
          </div>
        </section>
      </div>}
      {section === 'Egress requests' && <div className="security-layout"><section className="panel"><div className="panel-head"><h2>Request external access</h2></div><form className="panel-body stack" onSubmit={submitEgress}><p className="tiny">Submitting creates a review request. It does not send data or approve access.</p><label>Purpose<input className="field" value={purpose} onChange={event => setPurpose(event.target.value)} required /></label><label>Destination<input className="field" value={destination} onChange={event => setDestination(event.target.value)} required /></label><label>Resource type<input className="field" value={resource} onChange={event => setResource(event.target.value)} /></label><button className="btn primary" disabled={busy}>Submit for review</button></form></section><section className="panel"><div className="panel-head"><h2>Recorded requests</h2><span>{array('egress', 'requests').length}</span></div><div className="panel-body list">{array('egress', 'requests').map(request => <article key={value(request, 'id')} className="record-row"><div><strong>{value(request, 'destination')}</strong><p>{value(request, 'purpose')}</p><small>{value(request, 'resource_type')} · {value(request, 'created_at')}</small></div><div><span>{value(request, 'status')}</span>{pending(request) && <div className="toolbar"><button className="btn" disabled={busy} onClick={() => void act('/api/security/egress/resolve', { request_id: value(request, 'id', ''), approved: true, approved_by: 'user' }, `Approve access to ${value(request, 'destination')}? The server will record this decision.`)}>Approve</button><button className="btn danger" disabled={busy} onClick={() => void act('/api/security/egress/resolve', { request_id: value(request, 'id', ''), approved: false, approved_by: 'user' }, `Deny access to ${value(request, 'destination')}?`)}>Deny</button></div>}</div></article>)}{!array('egress', 'requests').length && <SecurityEmpty>No external access requests.</SecurityEmpty>}</div></section></div>}
      {section === 'Audit' && <div className="security-layout"><section className="panel"><div className="panel-head"><h2>Audit events</h2><div className="toolbar"><button className="btn" disabled={busy} onClick={() => void act('/api/security/audit-chain/verify')}>Verify chain</button><button className="btn danger" disabled={busy} onClick={() => void act('/api/security/audit-chain/seal', {}, 'Seal the current audit events? This writes an integrity record.')}>Seal events</button></div></div><div className="panel-body list">{array('audit', 'events').map((event, index) => <article className="record-row" key={value(event, 'id', String(index))}><div><strong>{auditEventTitle(event)}</strong><p>{[auditEventContext(event), auditEventOutcome(event)].filter(Boolean).join(' · ')}</p></div><time>{value(event, 'created_at')}</time></article>)}{!array('audit', 'events').length && <SecurityEmpty>No audit events have been recorded.</SecurityEmpty>}</div></section><section className="panel"><div className="panel-head"><h2>Chain integrity</h2></div><div className="panel-body"><SummaryFacts facts={auditChainSummary(chain)}/></div></section></div>}
      {section === 'Reports' && <section className="panel"><div className="panel-head"><h2>Privacy reports</h2><button className="btn primary" disabled={busy} onClick={() => void act('/api/security/privacy-report', {})}>Generate report</button></div>{lastAction.report != null && <div className="panel-body"><details open><summary>Generated privacy report</summary><SummaryFacts facts={privacyReportSummary(lastAction.report)}/></details></div>}<div className="panel-body list">{array('reports', 'reports').map((item, i) => { const report = record(item.report); return <details className="record-row" key={value(item, 'id', String(i))}><summary>{value(item, 'created_at', value(report, 'generated_at', 'Privacy report'))}</summary><SummaryFacts facts={privacyReportSummary(report)}/></details>; })}{!array('reports', 'reports').length && lastAction.report == null && <SecurityEmpty>No privacy reports yet. Generate one to create it.</SecurityEmpty>}</div></section>}
      {section === 'Retention' && <div className="security-layout"><section className="panel"><div className="panel-head"><h2>Retention policies</h2></div><div className="panel-body list">{array('retention', 'policies').map(policy => <article className="record-row" key={value(policy, 'id')}><div><strong>{value(policy, 'resource_type')}</strong><p>{value(policy, 'retention_days')} days · {value(policy, 'action')} · {value(policy, 'status')}</p></div><button className="btn danger" disabled={busy || value(policy, 'status') !== 'active'} onClick={() => void act('/api/security/retention/apply', { policy_id: value(policy, 'id', ''), dry_run: dryRun }, dryRun ? `Preview retention for ${value(policy, 'resource_type')}? No data will be deleted.` : `Apply retention to ${value(policy, 'resource_type')} and permanently delete matching old data?`)}>{dryRun ? 'Preview' : 'Apply deletion'}</button></article>)}{!array('retention', 'policies').length && <SecurityEmpty>No retention policies configured.</SecurityEmpty>}</div><div className="panel-body"><label className="record-row">Run in preview mode<input type="checkbox" checked={dryRun} onChange={event => setDryRun(event.target.checked)} /></label>{lastAction.policy_id != null && <p>Last result · {value(lastAction, 'planned', '—')} planned · {value(lastAction, 'deleted', '—')} deleted · cutoff {value(lastAction, 'cutoff')}</p>}</div></section><form className="panel panel-body stack" onSubmit={createRetention}><h2>Create or update retention policy</h2><label>Record category<select className="field" value={retentionType} onChange={event => setRetentionType(event.target.value)}>{['context_events', 'agent_runs', 'workflow_runs', 'developer_webhook_deliveries'].map(item => <option key={item}>{item}</option>)}</select></label><label>Keep for days<input className="field" type="number" min="0" value={retentionDays} onChange={event => setRetentionDays(event.target.value)} required /></label><small className="tiny">Backend supports delete actions for the listed categories. Applying with preview off destroys matching records.</small><button className="btn primary" disabled={busy}>Save policy</button></form></div>}
      {section === 'Shares' && <div className="security-layout"><section className="panel"><div className="panel-head"><h2>Share grants</h2></div><div className="panel-body list">{array('shares', 'grants').map(grant => <article className="record-row" key={value(grant, 'id')}><div><strong>{value(grant, 'resource_type')} · {value(grant, 'resource_id')}</strong><p>Grantee {value(grant, 'grantee')} · {value(grant, 'scope')} · {value(grant, 'expires_at', 'No expiry reported')}</p><small>{value(grant, 'status')}</small></div>{value(grant, 'status') === 'active' && <button className="btn danger" disabled={busy} onClick={() => void act('/api/security/shares/resolve', { grant_id: value(grant, 'id', ''), status: 'revoked' }, `Revoke the read grant for ${value(grant, 'grantee')}?`)}>Revoke</button>}</article>)}{!array('shares', 'grants').length && <SecurityEmpty>No active share grants.</SecurityEmpty>}</div></section><form className="panel panel-body stack" onSubmit={createShare}><h2>Create read grant</h2><label>Resource type<input className="field" value={shareType} onChange={event => setShareType(event.target.value)} required /></label><label>Resource ID<input className="field" value={shareId} onChange={event => setShareId(event.target.value)} required /></label><label>Grantee<input className="field" value={grantee} onChange={event => setGrantee(event.target.value)} required /></label><small className="tiny">The route supports read scope; no external share is sent by this action.</small><button className="btn primary" disabled={busy}>Create grant</button></form></div>}
      {section === 'Secrets' && <div className="security-layout security-secrets"><section className="panel"><div className="panel-head"><h2>Secret records</h2></div><div className="panel-body list">{array('secrets', 'secrets').map(secret => { const id = value(secret, 'id', ''); return <article className="security-secret panel panel-body" key={id}><div className="record-row"><div><strong>{value(secret, 'name')}</strong>{value(secret, 'description', '') && <p>{value(secret, 'description')}</p>}<small>{value(secret, 'status')} · updated {value(secret, 'updated_at')}</small></div><span aria-label="Secret value masked">••••••••</span></div><label className="stack">Passphrase<input className="field" type="password" autoComplete="current-password" value={stepUp[id] ?? ''} onChange={event => setStepUp(current => ({ ...current, [id]: event.target.value }))} /></label>{revealed[id] == null && <label className="stack">Replacement value<input className="field" type="password" autoComplete="new-password" value={rotateValues[id] ?? ''} onChange={event => setRotateValues(current => ({ ...current, [id]: event.target.value }))} /></label>}<div className="toolbar">{revealed[id] != null ? <><code style={{ overflowWrap: 'anywhere' }}>{revealed[id]}</code><button className="btn" onClick={() => { const timer = revealTimers.current.get(id); if (timer) window.clearTimeout(timer); revealTimers.current.delete(id); setRevealed(current => { const next = { ...current }; delete next[id]; return next; }); }}>Conceal</button></> : <button className="btn" disabled={busy || !stepUp[id]} onClick={() => void revealSecret(id)}>Reveal for 15s</button>}<button className="btn danger" disabled={busy || !stepUp[id] || !rotateValues[id]} onClick={() => void rotateSecret(id, value(secret, 'name'))}>Rotate value</button></div></article>; })}{!array('secrets', 'secrets').length && <SecurityEmpty>No secrets stored.</SecurityEmpty>}</div></section><form className="panel panel-body stack" onSubmit={storeSecret}><h2>Store a secret</h2><label>Name<input className="field" value={secretName} onChange={event => setSecretName(event.target.value)} required autoComplete="off" /></label><label>Value<input className="field" type="password" value={secretValue} onChange={event => setSecretValue(event.target.value)} required autoComplete="new-password" /></label><label>Encryption passphrase<input className="field" type="password" value={secretPassphrase} onChange={event => setSecretPassphrase(event.target.value)} required autoComplete="current-password" /></label><button className="btn primary" disabled={busy}>Encrypt and store</button></form></div>}
      {section === 'Sandbox' && <div className="security-layout"><section className="panel"><div className="panel-head"><h2>Configured profiles</h2></div><div className="panel-body list">{array('profiles', 'profiles').map(profile => <button key={value(profile, 'id')} onClick={() => setSandboxId(value(profile, 'id', ''))}><strong>{value(profile, 'name')}</strong><small>{Array.isArray(profile.permissions) ? profile.permissions.map(String).join(', ') : 'No permissions'} · network {profile.network_allowed === true ? 'allowed' : 'denied'}</small></button>)}{!array('profiles', 'profiles').length && <SecurityEmpty>No sandbox profiles configured.</SecurityEmpty>}</div></section><form className="panel panel-body stack" onSubmit={saveSandboxProfile}><h2>Create a profile</h2><label>Name<input className="field" value={profileName} onChange={event => setProfileName(event.target.value)} required /></label><label>Allowed permission scopes<textarea className="field" value={profilePermissions} onChange={event => setProfilePermissions(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="record-row">Allow network access<input type="checkbox" checked={profileNetwork} onChange={event => setProfileNetwork(event.target.checked)} /></label><small className="tiny">This stores policy only; it does not create or certify an operating-system sandbox.</small><button className="btn" disabled={busy}>Save profile</button></form><form className="panel panel-body stack" onSubmit={evaluateSandbox}><h2>Evaluate requested access</h2><label>Profile<select className="field" value={sandboxId} onChange={event => setSandboxId(event.target.value)} required><option value="">Choose a profile</option>{array('profiles', 'profiles').map(profile => <option key={value(profile, 'id')} value={value(profile, 'id')}>{value(profile, 'name')}</option>)}</select></label><label>Requested permissions<input className="field" value={permissionText} onChange={event => setPermissionText(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="record-row">Request network access<input type="checkbox" checked={network} onChange={event => setNetwork(event.target.checked)} /></label><button className="btn primary" disabled={busy || !sandboxId}>Evaluate only</button>{Object.keys(sandboxResult).length > 0 && <div className="notice"><strong>{sandboxResult.allowed === true ? 'Request allowed by this profile' : 'Request denied by this profile'}</strong><SummaryFacts facts={sandboxEvaluationSummary(sandboxResult)}/></div>}</form></div>}
      </div>
      </div>
    </>}
  </PageFrame>;
}
