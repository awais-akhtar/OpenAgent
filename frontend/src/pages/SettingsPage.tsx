import { useEffect, useState } from 'react';
import { api, record } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';

type Row = Record<string, unknown>;
type Field = { label: string; help: string; key: string; kind?: string; options?: string[] };
const groups: Record<string, Field[]> = {
  runtime: [
    { label: 'Ollama endpoint', help: 'Local Ollama server URL.', key: 'ollama_url' },
    { label: 'Ollama default model', help: 'Provider fallback model.', key: 'ollama_model' },
    { label: 'llama.cpp endpoint', help: 'Configured OpenAI-compatible local server URL.', key: 'llamacpp_url' },
    { label: 'llama.cpp model', help: 'Model identifier sent to that endpoint.', key: 'llamacpp_model' },
    { label: 'Quantization mode', help: 'Runtime planning preference.', key: 'quantization_mode', options: ['auto', 'none', 'Q4_K_M', 'Q5_K_M', 'Q8_0'] },
    { label: 'Target RAM (GB)', help: 'Optional memory target used by runtime planning.', key: 'target_ram_gb', kind: 'number' },
    { label: 'Model timeout (seconds)', help: 'Request timeout used by the local gateway.', key: 'model_timeout_seconds', kind: 'number' },
    { label: 'OpenVINO model directory', help: 'Local path configured for the direct OpenVINO provider.', key: 'openvino_model_dir' },
    { label: 'OpenVINO device', help: 'Requested OpenVINO device; availability is separately probed.', key: 'openvino_device' },
    { label: 'OpenVINO model ID', help: 'Identifier exposed by the direct provider.', key: 'openvino_model_id' },
    { label: 'OLMoE model directory', help: 'Local model storage path.', key: 'olmoe_model_dir' },
    { label: 'OLMoE manifest', help: 'Path to the runtime manifest.', key: 'olmoe_manifest' },
    { label: 'OLMoE I/O mode', help: 'Validated by the backend: buffered, mmap, direct, or uring.', key: 'olmoe_io_mode', options: ['buffered', 'mmap', 'direct', 'uring'] },
    { label: 'OLMoE storage roots', help: 'Configured local storage roots.', key: 'olmoe_storage_roots' },
    { label: 'OLMoE kernel', help: 'Preferred kernel name, if configured.', key: 'olmoe_kernel' },
    ...(['olmoe_expert_cache_gb', 'olmoe_accelerator_cache_gb'].map(key => ({ label: key.replaceAll('_', ' '), help: 'Runtime cache size in GB.', key, kind: 'number' }))),
    ...(['olmoe_context', 'olmoe_prefetch_workers', 'olmoe_prefetch_depth'].map(key => ({ label: key.replaceAll('_', ' '), help: key === 'olmoe_prefetch_depth' ? 'Must be at least the worker count.' : 'Validated by the local model runtime.', key, kind: 'number' }))),
  ],
  models: [
    { label: 'Chat model', help: 'Default chat role model.', key: 'chat_model' },
    { label: 'Code model', help: 'Default coding role model.', key: 'code_model' },
    { label: 'Research model', help: 'Default research role model.', key: 'research_model' },
    { label: 'Voice model', help: 'Configured voice engine identifier.', key: 'voice_model' },
    { label: 'Speed model', help: 'Fast response role model.', key: 'speed_model' },
    { label: 'Low-resource code model', help: 'Coding fallback for constrained hosts.', key: 'low_resource_code_model' },
    { label: 'Extraction model', help: 'Default structured extraction model.', key: 'extraction_model' },
    { label: 'High-context research model', help: 'Research model selected for larger contexts.', key: 'high_context_research_model' },
  ],
};

export function SettingsPage() {
  const [settings, setSettings] = useState<Row>({});
  const [baseline, setBaseline] = useState<Row>({});
  const [policy, setPolicy] = useState<Row>({});
  const [security, setSecurity] = useState<Row>({});
  const [user, setUser] = useState<Row>({});
  const [backups, setBackups] = useState<Row[]>([]);
  const [backupPath, setBackupPath] = useState('');
  const [backupPreview, setBackupPreview] = useState<Row>({});
  const [section, setSection] = useState('runtime');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  async function refresh() {
    setLoading(true); setError('');
    const settled = await Promise.allSettled([api.get<unknown>('/api/settings/runtime'), api.get<unknown>('/api/model-policy'), api.get<unknown>('/api/security/status'), api.get<unknown>('/api/auth/me'), api.get<unknown>('/api/backups')]);
    if (settled[0].status === 'fulfilled') { const response = record(settled[0].value); const current = record(response.settings ?? response); setSettings({ ...current }); setBaseline({ ...current }); }
    else setError(settled[0].reason instanceof Error ? settled[0].reason.message : 'Runtime settings could not be read.');
    if (settled[1].status === 'fulfilled') setPolicy(record(settled[1].value));
    if (settled[2].status === 'fulfilled') setSecurity(record(settled[2].value));
    if (settled[3].status === 'fulfilled') setUser(record(settled[3].value).user ? record(record(settled[3].value).user) : {});
    if (settled[4].status === 'fulfilled') setBackups(Array.isArray(record(settled[4].value).backups) ? record(settled[4].value).backups as Row[] : []);
    setLoading(false);
  }
  useEffect(() => { void refresh(); }, []);

  async function save() {
    setSaving(true); setError(''); setMessage('');
    const payload: Row = {};
    for (const field of [...groups.runtime, ...groups.models]) payload[field.key] = settings[field.key] ?? '';
    try {
      const response = record(await api.post<unknown>('/api/settings/runtime', { settings: payload }));
      const confirmed = record(response.settings);
      if (!Object.keys(confirmed).length) throw new Error('The service returned no saved runtime settings.');
      setSettings({ ...confirmed }); setBaseline({ ...confirmed }); setMessage('Saved runtime settings confirmed by the local service.');
      const latestPolicy = record(await api.get<unknown>('/api/model-policy')); setPolicy(latestPolicy);
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Settings could not be saved.'); }
    finally { setSaving(false); }
  }
  async function createBackup() {
    setSaving(true); setError(''); setMessage('');
    try { const result = record(await api.post<unknown>('/api/backups/create', { project: new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default', include_media: true })); const backup = record(result.backup); setBackupPath(String(backup.archive ?? backup.path ?? '')); setMessage(`Local backup created at ${String(backup.archive ?? backup.path ?? 'the configured data directory')}.`); const latest = record(await api.get<unknown>('/api/backups')); setBackups(Array.isArray(latest.backups) ? latest.backups as Row[] : []); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Backup could not be created.'); }
    finally { setSaving(false); }
  }
  async function previewRestore() {
    setSaving(true); setError(''); setMessage('');
    try { const result = record(await api.post<unknown>('/api/backups/restore-dry-run', { path: backupPath.trim() })); setBackupPreview(result); setMessage(result.status === 'ready' ? 'Restore preview is ready; current data has not been changed.' : 'Restore preview reported that this archive is not ready.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Backup preview failed.'); }
    finally { setSaving(false); }
  }
  async function restoreBackup() {
    if (!confirm('Restore this backup over the current OpenAgent database? This replaces current data; the service attempts to create a safety copy first.')) return;
    if (!confirm('Confirm database restore. Continue only if you have selected and reviewed the correct backup archive.')) return;
    setSaving(true); setError(''); setMessage('');
    try { const result = record(await api.post<unknown>('/api/backups/restore', { path: backupPath.trim(), confirm: true })); setBackupPreview(result); setMessage(result.restored === true ? `Database restored. Safety copy: ${String(result.safety_copy || 'not reported')}.` : 'Restore response received.'); }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'Backup restore failed.'); }
    finally { setSaving(false); }
  }
  const dirty = [...groups.runtime, ...groups.models].some(field => String(settings[field.key] ?? '') !== String(baseline[field.key] ?? ''));

  return <PageFrame eyebrow="Preferences / configuration" title="Settings" description="Configure provider endpoints and role defaults through the local runtime API." actions={<button className="btn" onClick={() => void refresh()} disabled={loading || saving}>Refresh from service</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{message && <p className="notice success" role="status">{message}</p>}
    {loading ? <p className="empty">Reading runtime configuration…</p> : <div className="workspace settings-workspace">
      <nav className="panel panel-body stack settings-sections" aria-label="Settings sections">{[['runtime', 'Runtime'], ['models', 'Model defaults'], ['privacy', 'Privacy & routing'], ['account', 'Account'], ['backups', 'Backups']].map(([id, label]) => <button key={id} className="btn" onClick={() => { setSection(id); setMessage(''); }} aria-current={section === id ? 'page' : undefined}>{label}</button>)}</nav>
      <section className="panel panel-body">
        {section === 'privacy' ? <><h2>Server policy and privacy</h2><p>These values are read from the Python service. This page does not change or locally simulate access policy.</p><dl className="facts">{[['Routing policy', policy.routing_policy], ['Fallback policy', policy.fallback_policy], ['Network default', security.network_default], ['External access requires explicit approval', security.external_requires_explicit_approval]].map(([label, value]) => <div key={String(label)}><dt>{String(label)}</dt><dd>{value == null ? 'Not reported' : Array.isArray(value) ? value.map(String).join(', ') : typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl><details><summary>Provider routing policy</summary><pre>{JSON.stringify(policy.routes ?? {}, null, 2)}</pre><p className="tiny">Runtime settings change provider defaults. The service enforces egress authorization for each request.</p></details><details><summary>Privacy posture reported by service</summary><pre>{JSON.stringify(security, null, 2)}</pre></details></> : section === 'account' ? <><h2>Current account</h2><p>The identity response is read from the authenticated Python service session.</p>{Object.keys(user).length ? <dl className="facts">{Object.entries(user).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{value == null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl> : <p className="empty">No authenticated account identity was returned. Authentication controls are managed by the Python service.</p>}</> : section === 'backups' ? <><h2>Local backups</h2><p>Backups are created in the Python service’s local data directory and include the database and media when enabled.</p><div className="toolbar"><button className="btn primary" disabled={saving} onClick={() => void createBackup()}>{saving ? 'Working…' : 'Create backup'}</button></div><div className="list">{backups.map((backup, index) => <button key={String(backup.path ?? backup.archive ?? index)} onClick={() => setBackupPath(String(backup.path ?? backup.archive ?? ''))}><strong>{String(backup.path ?? backup.archive ?? 'Backup')}</strong><small>{String(backup.created_at ?? backup.size_bytes ?? '')} · click to select</small></button>)}</div><label>Backup archive path<input className="field" value={backupPath} onChange={event => { setBackupPath(event.target.value); setBackupPreview({}); }} /></label><div className="toolbar"><button className="btn" disabled={saving || !backupPath.trim()} onClick={() => void previewRestore()}>Preview restore</button><button className="btn danger" disabled={saving || !backupPath.trim() || backupPreview.status !== 'ready'} onClick={() => void restoreBackup()}>Restore database</button></div>{Object.keys(backupPreview).length > 0 && <details open><summary>Restore preview/result</summary><pre>{JSON.stringify(backupPreview, null, 2)}</pre></details>}<p className="tiny">Restore preview is read-only. Applying restore replaces the current database; the backend creates a safety copy when possible.</p></> : <>
          <h2>{section === 'models' ? 'Role model defaults' : 'Runtime endpoints and local inference'}</h2><p className="tiny">{Object.keys(settings).length} settings returned by the server. Save sends these values to the local runtime and the server applies field-specific validation and reset behavior.</p>
          <div className="two-col">{groups[section].map(field => <label key={field.key} className="stack"><strong>{field.label}</strong>{field.options ? <select className="field" value={String(settings[field.key] ?? '')} onChange={event => setSettings(current => ({ ...current, [field.key]: event.target.value }))}><option value="">Keep current value</option>{field.options.map(option => <option key={option} value={option}>{option}</option>)}</select> : <input className="field" type={field.kind ?? 'text'} min={field.kind === 'number' ? 0 : undefined} step={field.kind === 'number' ? 'any' : undefined} value={String(settings[field.key] ?? '')} onChange={event => setSettings(current => ({ ...current, [field.key]: event.target.value }))} />}<small className="tiny">{field.help}</small></label>)}</div>
          <div className="toolbar" style={{ marginTop: 18 }}><button className="btn" disabled={saving || !dirty} onClick={() => { setSettings({ ...baseline }); setError(''); setMessage(''); }}>Revert changes</button><button className="btn primary" disabled={saving} onClick={() => void save()}>{saving ? 'Saving…' : 'Save runtime settings'}</button></div>
        </>}
      </section>
    </div>}
  </PageFrame>;
}
