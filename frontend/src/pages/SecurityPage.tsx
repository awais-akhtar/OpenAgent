import { useEffect, useRef, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';

type Row = Record<string, unknown>;
const sections = ['Overview', 'Egress requests', 'Audit', 'Reports', 'Retention', 'Shares', 'Secrets', 'Sandbox'] as const;
type Section = typeof sections[number];
const value = (row: Row, key: string, fallback = '—') => row[key] == null || row[key] === '' ? fallback : String(row[key]);
const pending = (row: Row) => ['pending', 'requested', 'awaiting_approval'].includes(value(row, 'status').toLowerCase());

export function SecurityPage() {
  const [data, setData] = useState<Record<string, unknown>>({});
  const [section, setSection] = useState<Section>('Overview');
  const [loading, setLoading] = useState(true);
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
    setLoading(true); setError('');
    const paths = [
      ['status', '/api/security/status'], ['egress', '/api/security/egress'], ['retention', '/api/security/retention'],
      ['secrets', '/api/security/secrets'], ['shares', '/api/security/shares'], ['reports', '/api/security/privacy-reports'],
      ['audit', '/api/audit?limit=80'], ['chain', '/api/security/audit-chain/verify'], ['profiles', '/api/security/sandbox-profiles'],
    ] as const;
    const result = await Promise.allSettled(paths.map(([, path]) => api.get<unknown>(path)));
    const next: Record<string, unknown> = {};
    result.forEach((item, i) => { if (item.status === 'fulfilled') next[paths[i][0]] = item.value; else if (i === 0) setError(item.reason instanceof Error ? item.reason.message : 'Security status unavailable.'); });
    setData(next); setLoading(false);
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => () => { for (const timer of revealTimers.current.values()) window.clearTimeout(timer); revealTimers.current.clear(); }, []);
  const array = (key: string, child: string) => records(data[key], child);
  const status = record(data.status);
  const chain = record(data.chain);

  async function act(path: string, body?: unknown, confirmation?: string) {
    if (confirmation && !confirm(confirmation)) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const result = record(await api.post<unknown>(path, body));
      setLastAction(result); setNotice('The local security service returned a completed response.'); await refresh(); return result;
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

  return <PageFrame eyebrow="Trust / privacy posture" title="Security & privacy" description="Inspect service policy, recorded access decisions, retention behavior, secrets and audit evidence." actions={<button className="btn" onClick={() => void refresh()} disabled={loading || busy}>Refresh evidence</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    {loading ? <p className="empty">Reading security records…</p> : <>
      <section className="panel panel-body"><p className="eyebrow">Service reported posture</p><h2>{chain.ok === true ? 'Audit chain verified' : chain.ok === false ? 'Audit chain needs review' : 'Policy status available'}</h2><p>Network default: {status.network_default === false ? 'denied by default' : status.network_default === true ? 'allowed by default' : 'not reported'} · Audit events sealed: {value(status, 'sealed_audit_events', 'not reported')}.</p><div className="row">{[['Egress requests', array('egress', 'requests').length], ['Masked secret records', array('secrets', 'secrets').length], ['Retention policies', array('retention', 'policies').length], ['Share grants', array('shares', 'grants').length]].map(([label, count]) => <span key={String(label)}><strong>{String(count)}</strong><small style={{ display: 'block' }}>{String(label)}</small></span>)}</div><div className="toolbar" style={{ marginTop: 14 }}><button className="btn" disabled={busy} onClick={() => void act('/api/security/audit-chain/verify')}>Verify audit chain</button><button className="btn" disabled={busy} onClick={() => void act('/api/security/privacy-report', {})}>Generate privacy report</button><button className="btn" disabled={busy} onClick={() => void act('/api/security/audit-chain/seal', {}, 'Seal the current audit events? This writes an integrity record.')}>Seal audit events</button></div>{lastAction.report != null && <details open><summary>Generated privacy report</summary><pre>{JSON.stringify(lastAction.report, null, 2)}</pre></details>}</section>
      <nav className="toolbar" aria-label="Security sections" style={{ margin: '16px 0', flexWrap: 'wrap' }}>{sections.map(item => <button className={`btn ${section === item ? 'primary' : ''}`} key={item} onClick={() => setSection(item)} aria-current={section === item ? 'page' : undefined}>{item}</button>)}</nav>
      {section === 'Egress requests' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Request external access</h2></div><form className="panel-body stack" onSubmit={submitEgress}><p className="tiny">Submitting creates a review request. It does not send data or approve access.</p><label>Purpose<input className="field" value={purpose} onChange={event => setPurpose(event.target.value)} required /></label><label>Destination<input className="field" value={destination} onChange={event => setDestination(event.target.value)} required /></label><label>Resource type<input className="field" value={resource} onChange={event => setResource(event.target.value)} /></label><button className="btn primary" disabled={busy}>Submit for review</button></form></section><section className="panel"><div className="panel-head"><h2>Recorded requests</h2></div><div className="panel-body list">{array('egress', 'requests').map(request => <article key={value(request, 'id')} className="record-row"><div><strong>{value(request, 'destination')}</strong><p>{value(request, 'purpose')}</p><small>{value(request, 'resource_type')} · {value(request, 'created_at')}</small></div><div><span>{value(request, 'status')}</span>{pending(request) && <div className="toolbar"><button className="btn" disabled={busy} onClick={() => void act('/api/security/egress/resolve', { request_id: value(request, 'id', ''), approved: true, approved_by: 'user' }, `Approve access to ${value(request, 'destination')}? The server will record this decision.`)}>Approve</button><button className="btn danger" disabled={busy} onClick={() => void act('/api/security/egress/resolve', { request_id: value(request, 'id', ''), approved: false, approved_by: 'user' }, `Deny access to ${value(request, 'destination')}?`)}>Deny</button></div>}</div></article>)}</div></section></div>}
      {section === 'Audit' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Audit events</h2><span>{array('audit', 'events').length} records</span></div><div className="panel-body list">{array('audit', 'events').map((event, index) => <article className="record-row" key={value(event, 'id', String(index))}><div><strong>{value(event, 'action', value(event, 'event_type', 'Audit event'))}</strong><p>{value(event, 'resource', value(event, 'actor'))} · {value(event, 'project')}</p></div><time>{value(event, 'created_at')}</time></article>)}</div></section><section className="panel"><div className="panel-head"><h2>Audit chain verification</h2></div><div className="panel-body"><strong>{chain.ok === true ? 'Verified' : chain.ok === false ? 'Verification failed' : 'Unknown'}</strong><pre>{JSON.stringify(chain, null, 2)}</pre></div></section></div>}
      {section === 'Reports' && <section className="panel"><div className="panel-head"><h2>Privacy reports</h2></div><div className="panel-body list">{array('reports', 'reports').map((report, i) => <details className="record-row" key={value(report, 'id', String(i))}><summary>{value(report, 'generated_at', value(report, 'created_at', 'Privacy report'))}</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>)}{!array('reports', 'reports').length && <p className="empty">No saved reports returned. Use Generate privacy report to create one.</p>}</div></section>}
      {section === 'Retention' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Retention policies</h2></div><div className="panel-body list">{array('retention', 'policies').map(policy => <article className="record-row" key={value(policy, 'id')}><div><strong>{value(policy, 'resource_type')}</strong><p>{value(policy, 'retention_days')} days · {value(policy, 'action')} · {value(policy, 'status')}</p></div><button className="btn danger" disabled={busy || value(policy, 'status') !== 'active'} onClick={() => void act('/api/security/retention/apply', { policy_id: value(policy, 'id', ''), dry_run: dryRun }, dryRun ? `Preview retention for ${value(policy, 'resource_type')}? No data will be deleted.` : `Apply retention to ${value(policy, 'resource_type')} and permanently delete matching old data?`)}>{dryRun ? 'Preview' : 'Apply deletion'}</button></article>)}</div><div className="panel-body"><label className="record-row">Run in preview mode<input type="checkbox" checked={dryRun} onChange={event => setDryRun(event.target.checked)} /></label>{lastAction.policy_id != null && <p>Last result · {value(lastAction, 'planned', '—')} planned · {value(lastAction, 'deleted', '—')} deleted · cutoff {value(lastAction, 'cutoff')}</p>}</div></section><form className="panel panel-body stack" onSubmit={createRetention}><h2>Create or update retention policy</h2><label>Record category<select className="field" value={retentionType} onChange={event => setRetentionType(event.target.value)}>{['context_events', 'agent_runs', 'workflow_runs', 'developer_webhook_deliveries'].map(item => <option key={item}>{item}</option>)}</select></label><label>Keep for days<input className="field" type="number" min="0" value={retentionDays} onChange={event => setRetentionDays(event.target.value)} required /></label><small className="tiny">Backend supports delete actions for the listed categories. Applying with preview off destroys matching records.</small><button className="btn primary" disabled={busy}>Save policy</button></form></div>}
      {section === 'Shares' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Share grants</h2></div><div className="panel-body list">{array('shares', 'grants').map(grant => <article className="record-row" key={value(grant, 'id')}><div><strong>{value(grant, 'resource_type')} · {value(grant, 'resource_id')}</strong><p>Grantee {value(grant, 'grantee')} · {value(grant, 'scope')} · {value(grant, 'expires_at', 'No expiry reported')}</p><small>{value(grant, 'status')}</small></div>{value(grant, 'status') === 'active' && <button className="btn danger" disabled={busy} onClick={() => void act('/api/security/shares/resolve', { grant_id: value(grant, 'id', ''), status: 'revoked' }, `Revoke the read grant for ${value(grant, 'grantee')}?`)}>Revoke</button>}</article>)}</div></section><form className="panel panel-body stack" onSubmit={createShare}><h2>Create read grant</h2><label>Resource type<input className="field" value={shareType} onChange={event => setShareType(event.target.value)} required /></label><label>Resource ID<input className="field" value={shareId} onChange={event => setShareId(event.target.value)} required /></label><label>Grantee<input className="field" value={grantee} onChange={event => setGrantee(event.target.value)} required /></label><small className="tiny">The route supports read scope; no external share is sent by this action.</small><button className="btn primary" disabled={busy}>Create grant</button></form></div>}
      {section === 'Secrets' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Protected secret records</h2><span>Values masked</span></div><div className="panel-body list">{array('secrets', 'secrets').map(secret => { const id = value(secret, 'id', ''); return <article className="panel panel-body" key={id}><div className="record-row"><div><strong>{value(secret, 'name')}</strong><p>{value(secret, 'description')}</p><small>{value(secret, 'status')} · updated {value(secret, 'updated_at')}</small></div><span aria-label="Secret value masked">••••••••</span></div><label className="stack">Passphrase for reveal or rotation<input className="field" type="password" autoComplete="current-password" value={stepUp[id] ?? ''} onChange={event => setStepUp(current => ({ ...current, [id]: event.target.value }))} /></label>{revealed[id] == null && <label className="stack">Replacement value<input className="field" type="password" autoComplete="new-password" value={rotateValues[id] ?? ''} onChange={event => setRotateValues(current => ({ ...current, [id]: event.target.value }))} /></label>}<div className="toolbar">{revealed[id] != null ? <><code style={{ overflowWrap: 'anywhere' }}>{revealed[id]}</code><button className="btn" onClick={() => { const timer = revealTimers.current.get(id); if (timer) window.clearTimeout(timer); revealTimers.current.delete(id); setRevealed(current => { const next = { ...current }; delete next[id]; return next; }); }}>Conceal</button></> : <button className="btn" disabled={busy || !stepUp[id]} onClick={() => void revealSecret(id)}>Reveal for 15s</button>}<button className="btn danger" disabled={busy || !stepUp[id] || !rotateValues[id]} onClick={() => void rotateSecret(id, value(secret, 'name'))}>Rotate value</button></div></article>; })}</div></section><form className="panel panel-body stack" onSubmit={storeSecret}><h2>Store encrypted secret</h2><label>Name<input className="field" value={secretName} onChange={event => setSecretName(event.target.value)} required autoComplete="off" /></label><label>Value<input className="field" type="password" value={secretValue} onChange={event => setSecretValue(event.target.value)} required autoComplete="new-password" /></label><label>Encryption passphrase<input className="field" type="password" value={secretPassphrase} onChange={event => setSecretPassphrase(event.target.value)} required autoComplete="current-password" /></label><button className="btn primary" disabled={busy}>Encrypt and store</button><small className="tiny">Secret values and passphrases are sent only in POST requests, cleared after acknowledgement, and never included in list summaries.</small></form></div>}
      {section === 'Sandbox' && <div className="workspace"><section className="panel"><div className="panel-head"><h2>Configured profiles</h2></div><div className="panel-body list">{array('profiles', 'profiles').map(profile => <button key={value(profile, 'id')} onClick={() => setSandboxId(value(profile, 'id', ''))}><strong>{value(profile, 'name')}</strong><small>{Array.isArray(profile.permissions) ? profile.permissions.map(String).join(', ') : 'No permissions'} · network {profile.network_allowed === true ? 'allowed' : 'denied'}</small></button>)}{!array('profiles', 'profiles').length && <p className="empty">No sandbox profiles configured.</p>}</div></section><form className="panel panel-body stack" onSubmit={saveSandboxProfile}><h2>Create a profile</h2><label>Name<input className="field" value={profileName} onChange={event => setProfileName(event.target.value)} required /></label><label>Allowed permission scopes<textarea className="field" value={profilePermissions} onChange={event => setProfilePermissions(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="record-row">Allow network access<input type="checkbox" checked={profileNetwork} onChange={event => setProfileNetwork(event.target.checked)} /></label><small className="tiny">This stores policy only; it does not create or certify an operating-system sandbox.</small><button className="btn" disabled={busy}>Save profile</button></form><form className="panel panel-body stack" onSubmit={evaluateSandbox}><h2>Evaluate requested access</h2><label>Profile<select className="field" value={sandboxId} onChange={event => setSandboxId(event.target.value)} required><option value="">Choose a profile</option>{array('profiles', 'profiles').map(profile => <option key={value(profile, 'id')} value={value(profile, 'id')}>{value(profile, 'name')}</option>)}</select></label><label>Requested permissions<input className="field" value={permissionText} onChange={event => setPermissionText(event.target.value)} placeholder="filesystem.read, process.spawn" /></label><label className="record-row">Request network access<input type="checkbox" checked={network} onChange={event => setNetwork(event.target.checked)} /></label><button className="btn primary" disabled={busy || !sandboxId}>Evaluate only</button>{Object.keys(sandboxResult).length > 0 && <div className="notice"><strong>{sandboxResult.allowed === true ? 'Request allowed by this profile' : 'Request denied by this profile'}</strong><pre>{JSON.stringify(sandboxResult, null, 2)}</pre></div>}</form></div>}
      {section === 'Overview' && <section className="panel panel-body"><h2>Enforcement evidence</h2><p>The server remains authoritative for authorization and network egress. This console displays recorded decisions and calls the service for sensitive operations.</p><pre>{JSON.stringify(status, null, 2)}</pre></section>}
    </>}
  </PageFrame>;
}
