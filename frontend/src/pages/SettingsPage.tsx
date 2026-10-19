import { useEffect, useState } from 'react';
import { RefreshCw, Sun, Moon, Monitor } from 'lucide-react';
import { api, record } from '../lib/react/api';
import { PageFrame } from '../components/PageFrame';
import { accessibility, type MotionPreference } from '../state/accessibility';
import { theme, type ThemeMode } from '../state/theme';
import { accountSummaryFacts, backupListFacts, backupPreviewFacts, modelPolicyFacts } from './securitySettingsSummary';
import './settings-page.css';

type Row = Record<string, unknown>;
function SummaryFacts({ facts, empty = 'No configured values are available.' }: { facts: Array<{ label: string; value: string }>; empty?: string }) {
  return facts.length ? <dl className="facts">{facts.map(fact => <div key={fact.label}><dt>{fact.label}</dt><dd>{fact.value}</dd></div>)}</dl> : <p className="empty">{empty}</p>;
}
type Field = { label: string; help: string; key: string; kind?: string; options?: string[]; writable?: boolean };
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
    { label: 'Speed model', help: 'Selected by the runtime; it is not writable through the current settings service.', key: 'speed_model', writable: false },
    { label: 'Low-resource code model', help: 'Selected by the runtime; it is not writable through the current settings service.', key: 'low_resource_code_model', writable: false },
    { label: 'Extraction model', help: 'Selected by the runtime; it is not writable through the current settings service.', key: 'extraction_model', writable: false },
    { label: 'High-context research model', help: 'Selected by the runtime; it is not writable through the current settings service.', key: 'high_context_research_model', writable: false },
  ],
};
const runtimeCore = new Set(['ollama_url', 'ollama_model', 'llamacpp_url', 'llamacpp_model']);
const modelCore = new Set(['chat_model', 'code_model', 'research_model', 'voice_model']);
export function settingsFields(section: 'runtime' | 'models') {
  const all = groups[section];
  const core = section === 'runtime' ? runtimeCore : modelCore;
  return { primary: all.filter(field => core.has(field.key)), advanced: all.filter(field => !core.has(field.key)) };
}

function SettingField({ field, settings, update }: { field: Field; settings: Row; update: (key: string, value: string) => void }) {
  const current = String(settings[field.key] ?? '');
  return <label key={field.key} className="stack">
    <strong>{field.label}</strong>
    {field.writable === false ? <output className="settings-readonly">{current || 'Not reported'}</output>
      : field.options ? <select className="field" value={current} onChange={event => update(field.key, event.target.value)}><option value="">Not configured</option>{field.options.map(option => <option key={option} value={option}>{option}</option>)}</select>
        : <input className="field" type={field.kind ?? 'text'} min={field.kind === 'number' ? 0 : undefined} step={field.kind === 'number' ? 'any' : undefined} placeholder="Not configured" value={current} onChange={event => update(field.key, event.target.value)} />}
    <small className="tiny">{field.help}</small>
  </label>;
}

export function SettingsPage() {
  const [settings, setSettings] = useState<Row>({});
  const [baseline, setBaseline] = useState<Row>({});
  const [policy, setPolicy] = useState<Row>({});
  const [security, setSecurity] = useState<Row>({});
  const [user, setUser] = useState<Row>({});
  const [backups, setBackups] = useState<Row[]>([]);
  const [backupPath, setBackupPath] = useState('');
  const [backupPreview, setBackupPreview] = useState<Row>({});
  const [section, setSection] = useState('appearance');
  const [themeMode, setThemeMode] = useState<ThemeMode>(() => theme.read());
  const [motionPreference, setMotionPreference] = useState<MotionPreference>(() => accessibility.read());
  const [loading, setLoading] = useState(true);
  const [readIssues, setReadIssues] = useState<string[]>([]);
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
    const labels = ['Runtime configuration', 'Model routing', 'Privacy status', 'Account details', 'Backup list'];
    const issues = settled.flatMap((item, index) => item.status === 'rejected' ? [labels[index]] : []);
    setReadIssues(issues);
    if (issues.length) setError(`Could not read ${issues.join(', ').toLowerCase()}. Available data is still shown.`);
    setLoading(false);
  }
  useEffect(() => { void refresh(); }, []);
  useEffect(() => {
    const syncTheme = () => setThemeMode(theme.read());
    const syncMotion = () => setMotionPreference(accessibility.read());
    window.addEventListener(theme.eventName, syncTheme);
    window.addEventListener(accessibility.eventName, syncMotion);
    return () => {
      window.removeEventListener(theme.eventName, syncTheme);
      window.removeEventListener(accessibility.eventName, syncMotion);
    };
  }, []);

  async function save() {
    setSaving(true); setError(''); setMessage('');
    const payload: Row = {};
    for (const field of [...groups.runtime, ...groups.models]) {
      if (field.writable !== false && Object.prototype.hasOwnProperty.call(settings, field.key)) payload[field.key] = settings[field.key];
    }
    try {
      const response = record(await api.post<unknown>('/api/settings/runtime', { settings: payload }));
      const confirmed = record(response.settings);
      if (!Object.keys(confirmed).length) throw new Error('No saved settings were returned.');
      setSettings({ ...confirmed }); setBaseline({ ...confirmed }); setMessage('Settings saved.');
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
  const dirty = [...groups.runtime, ...groups.models].some(field => field.writable !== false && String(settings[field.key] ?? '') !== String(baseline[field.key] ?? ''));

  const activeFields = section === 'runtime' || section === 'models' ? settingsFields(section) : null;
  return <PageFrame title="Settings" actions={<button className="btn settings-refresh" aria-label="Refresh settings" title={dirty ? 'Save or revert changes before refreshing' : 'Refresh settings'} onClick={() => void refresh()} disabled={loading || saving || dirty}><RefreshCw size={15}/></button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{message && <p className="notice success" role="status">{message}</p>}
    {loading ? <p className="empty">Reading runtime configuration…</p> : <div className="workspace settings-workspace">
      <nav className="panel panel-body stack settings-sections" aria-label="Settings sections">{[['appearance', 'Appearance & accessibility'], ['runtime', 'Runtime'], ['models', 'Model defaults'], ['privacy', 'Privacy & routing'], ['account', 'Account'], ['backups', 'Backups']].map(([id, label]) => <button key={id} className="btn" onClick={() => { setSection(id); setMessage(''); }} aria-current={section === id ? 'page' : undefined}>{label}</button>)}</nav>
      <section className="panel panel-body">
        {section === 'appearance' ? <><h2>Appearance & accessibility</h2><p className="settings-local-note">These preferences are saved in this browser and do not change server configuration.</p><div className="settings-preference-groups"><section className="settings-preference-group"><div><h3>Appearance</h3><p>Choose how OpenAgent looks on this device.</p></div><div className="settings-theme-options" role="group" aria-label="Theme">{([{value:'light' as const,label:'Light',description:'Bright workspace',Icon:Sun},{value:'dark' as const,label:'Dark',description:'Low-light workspace',Icon:Moon},{value:'system' as const,label:'System',description:'Follow device setting',Icon:Monitor}]).map(({value,label,description,Icon})=><button type="button" data-theme={value} className={`settings-theme-option${themeMode===value?' selected':''}`} aria-pressed={themeMode===value} key={value} onClick={()=>{theme.set(value);setThemeMode(value)}}><span className="settings-theme-icon"><Icon size={17}/></span><span><strong>{label}</strong><small>{description}</small></span></button>)}</div></section><section className="settings-preference-group"><div><h3>Accessibility</h3><p>Limit interface motion where possible.</p></div><label><span>Reduced motion</span><select className="field" aria-label="Reduced motion" value={motionPreference} onChange={event => { const value = event.currentTarget.value as MotionPreference; accessibility.set(value); setMotionPreference(value); }}><option value="system">Use system setting</option><option value="reduce">Reduce motion</option><option value="allow">Allow motion</option></select><small className="tiny">{motionPreference === 'system' ? 'Follows your operating system preference.' : motionPreference === 'reduce' ? 'Motion effects and CSS animations are reduced.' : 'Motion effects are allowed, regardless of the system preference.'}</small></label></section></div></> : section === 'privacy' ? <><h2>Privacy and routing</h2><SummaryFacts facts={modelPolicyFacts(policy, security)} empty="Routing and privacy details are unavailable."/></> : section === 'account' ? <><h2>Account</h2>{Object.keys(user).length ? <SummaryFacts facts={accountSummaryFacts(user)} empty="Account details aren't reported by the current session."/> : <p className="empty">{readIssues.includes('Account details') ? 'Account details could not be loaded.' : 'Account details are unavailable in this session.'}</p>}</> : section === 'backups' ? <><h2>Backups</h2><div className="toolbar"><button className="btn primary" disabled={saving} onClick={() => void createBackup()}>{saving ? 'Working…' : 'Create backup'}</button></div>{backups.length === 0 && <p className="empty">{readIssues.includes('Backup list') ? 'The backup list could not be loaded.' : 'No backups are registered in this local store yet.'}</p>}<div className="list">{backups.map((backup, index) => { const path = String(backup.path ?? backup.archive ?? ''); const facts = backupListFacts(backup); return <button key={String(backup.name ?? path ?? index)} onClick={() => setBackupPath(path)}><strong>{String(backup.name ?? 'Backup')}</strong><small>{facts.map(item => `${item.label}: ${item.value}`).join(' · ')} · Select</small></button>; })}</div><label>Backup archive path<input className="field" value={backupPath} onChange={event => { setBackupPath(event.target.value); setBackupPreview({}); }} /></label><div className="toolbar"><button className="btn" disabled={saving || !backupPath.trim()} onClick={() => void previewRestore()}>Preview restore</button><button className="btn danger" disabled={saving || !backupPath.trim() || backupPreview.status !== 'ready'} onClick={() => void restoreBackup()}>Restore database</button></div>{Object.keys(backupPreview).length > 0 && <details open><summary>Restore preview/result</summary><SummaryFacts facts={backupPreviewFacts(backupPreview)}/></details>}<p className="tiny">Restore preview is read-only. Applying restore replaces current data; the service attempts a safety copy.</p></> : <>
          <h2>{section === 'models' ? 'Role model defaults' : 'Runtime'}</h2>
          {Object.keys(baseline).length === 0 ? <p className="empty">{readIssues.includes('Runtime configuration') ? 'Runtime settings could not be loaded.' : 'The runtime returned no configurable settings.'}</p> : <>
            <div className="two-col">{activeFields?.primary.map(field => <SettingField key={field.key} field={field} settings={settings} update={(key, value) => setSettings(current => ({ ...current, [key]: value }))}/>)}</div>
            {!!activeFields?.advanced.length && <details className="settings-advanced"><summary>Advanced settings <span>{activeFields.advanced.length}</span></summary><div className="two-col">{activeFields.advanced.map(field => <SettingField key={field.key} field={field} settings={settings} update={(key, value) => setSettings(current => ({ ...current, [key]: value }))}/>)}</div></details>}
          </>}
        </>}
        {dirty && !loading && <div className="settings-savebar"><span>Unsaved changes</span><div><button className="btn" disabled={saving} onClick={() => { setSettings({ ...baseline }); setError(''); setMessage(''); }}>Revert</button><button className="btn primary" disabled={saving} onClick={() => void save()}>{saving ? 'Saving…' : 'Save settings'}</button></div></div>}
      </section>
    </div>}
  </PageFrame>;
}
