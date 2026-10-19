import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, CircleAlert, Copy, Eye, EyeOff, Play, RefreshCw, X } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import { api, record, records } from '../lib/react/api';
import type { ResourceAction, ResourceRead } from '../lib/react/contracts';
import { resourcePageConfigs } from './ResourcePages';
import '../features/specialists/specialist-workspace.css';
import './platform-workspaces.css';

const IDS = ['developer', 'engine', 'finetune', 'science', 'scraper'] as const;
export type PlatformWorkspaceId = typeof IDS[number];
type Snapshot = { data?: unknown; error?: string; loading: boolean };
const currentProject = () => new URL(location.href).searchParams.get('project') || (() => { try { return localStorage.getItem('oa-project') || 'default'; } catch { return 'default'; } })();
const scoped = (path: string, project: string) => path.replaceAll('{project}', encodeURIComponent(project));
const visibleValue = (item: unknown) => typeof item === 'string' || typeof item === 'number' ? String(item) : '';
const text = (item: Record<string, unknown>, key: string) => typeof item[key] === 'string' ? item[key] as string : '';
const title = (key: string) => key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
const readRecords = (source: unknown, item?: ResourceRead) => {
  if (!item) return [];
  const rows = records(source, item.key);
  if (rows.length || Array.isArray(source) || item.label !== 'Hardware inventory') return rows;
  return records(record(record(source).profile), 'devices');
};
const readableValue = (value: unknown) => {
  if (typeof value === 'boolean') return '';
  if (typeof value !== 'string') return String(value);
  const normalized = value.trim().toLowerCase();
  if (['ok', 'ready', 'healthy', 'active', 'available', 'success', 'succeeded'].includes(normalized)) return 'Ready';
  if (['false', 'true', 'enabled', 'unknown', 'registered'].includes(normalized)) return '';
  if (['inactive', 'disabled', 'unavailable', 'failed', 'error'].includes(normalized)) return 'Unavailable';
  return value;
};

function Facts({ value }: { value: unknown }) {
  const hiddenKeys = ['id', 'project', 'project_id', 'raw', 'metadata', 'config', 'settings', 'schema', 'details', 'content', 'secret', 'token'];
  const facts = Object.entries(record(value)).filter(([key]) => !hiddenKeys.includes(key.toLowerCase())).map(([key, raw]) => [key, visibleValue(raw) ? readableValue(raw) : ''] as const).filter(([, value]) => value !== '').slice(0, 18);
  return facts.length ? <dl className="platform-facts">{facts.map(([key, v]) => <div key={key}><dt>{title(key)}</dt><dd>{v}</dd></div>)}</dl> : null;
}

function readinessLabel(value: unknown) {
  if (typeof value !== 'string') return '';
  const normalized = value.toLowerCase();
  if (['true', 'false', 'unknown', 'registered'].includes(normalized)) return '';
  if (['ok', 'ready', 'healthy', 'available', 'active', 'success', 'succeeded'].includes(normalized)) return 'Ready';
  if (['failed', 'error', 'unavailable', 'disabled', 'blocked'].includes(normalized)) return 'Needs attention';
  return value;
}

function WorkflowAction({ action, onOpen, quiet = false }: { action?: ResourceAction; onOpen: (action: ResourceAction) => void; quiet?: boolean }) {
  if (!action) return null;
  return <button type="button" className={`platform-workflow-action${quiet ? ' quiet' : ''}`} onClick={() => onOpen(action)}><span>{action.label}</span>{!quiet && <span aria-hidden="true">›</span>}</button>;
}

function WorkflowSurface({
  id, reads, data, actions, onAction, selection, onSelect,
}: {
  id: PlatformWorkspaceId; reads: ResourceRead[]; data: Snapshot[]; actions: ResourceAction[]; onAction: (action: ResourceAction) => void;
  selection: { label: string; index: number } | null; onSelect: (selection: { label: string; index: number } | null) => void;
}) {
  const indexFor = (label: string) => reads.findIndex(item => item.label === label);
  const raw = (label: string) => { const index = indexFor(label); return index < 0 ? undefined : data[index]?.data; };
  const rows = (label: string) => { const index = indexFor(label); return index < 0 ? [] : readRecords(data[index]?.data, reads[index]); };
  const loading = (label: string) => { const index = indexFor(label); return index >= 0 && Boolean(data[index]?.loading); };
  const readError = (label: string) => { const index = indexFor(label); return index < 0 ? '' : data[index]?.error ?? ''; };
  const countLabel = (label: string) => loading(label) ? '…' : readError(label) ? '—' : rows(label).length;
  const action = (label: string) => actions.find(item => item.label === label);
  const statusRecord = record(raw(id === 'engine' ? 'Adaptive runtime' : id === 'finetune' ? 'Fine-tuning capability' : id === 'science' ? 'Scientific capability' : id === 'scraper' ? 'Scraper capability' : 'Developer platform'));
  const hardwareProfile = record(record(raw('Hardware inventory')).profile);
  const status = readinessLabel(statusRecord.status ?? statusRecord.health ?? statusRecord.readiness ?? statusRecord.state);
  const statusClass = status === 'Ready' ? 'is-ready' : status ? 'is-attention' : '';
  const list = (label: string, nameOf: (row: Record<string, unknown>, index: number) => string, stateOf?: (row: Record<string, unknown>) => string) => {
    if (loading(label)) return <p className="platform-workflow-muted">Loading {label.toLowerCase()}…</p>;
    if (readError(label)) return <p className="platform-workflow-error" role="alert">Couldn’t load {label.toLowerCase()}. Use refresh to retry.</p>;
    const items = rows(label);
    return items.length ? <div className="platform-workflow-collection"><ul className="platform-workflow-list">{items.slice(0, 6).map((row, index) => <li key={String(row.id ?? `${label}-${index}`)}><button type="button" className={selection?.label === label && selection.index === index ? 'selected' : ''} aria-pressed={selection?.label === label && selection.index === index} onClick={() => onSelect(selection?.label === label && selection.index === index ? null : { label, index })}><span>{nameOf(row, index)}</span>{stateOf && <small>{stateOf(row)}</small>}</button></li>)}</ul>{items.length > 6 && <p className="platform-workflow-muted">Showing 6 of {items.length}. Open technical data for the full collection.</p>}{selection?.label === label && items[selection.index] && <aside className="platform-workflow-inspector" aria-label={`${label} selected details`}><div><strong>{nameOf(items[selection.index], selection.index)}</strong><button type="button" aria-label="Close selected details" onClick={() => onSelect(null)}>×</button></div><Facts value={items[selection.index]}/></aside>}</div> : <p className="platform-workflow-muted">No {label.toLowerCase()} yet.</p>;
  };

  if (id === 'engine') return <section className="platform-workflow" aria-label="Engine readiness">
    <div className="platform-workflow-heading"><div><span>Runtime readiness</span><h2>Local inference capacity</h2></div>{status && <span className={`platform-readiness ${statusClass}`}>{status}</span>}</div>
    <p className="platform-workflow-lede">Review available hardware and measured performance before assigning model roles.</p>
    <div className="platform-workflow-stats"><div><small>Devices</small><strong>{countLabel('Hardware inventory')}</strong></div><div><small>Placement plans</small><strong>{countLabel('Placement plans')}</strong></div><div><small>Benchmarks</small><strong>{countLabel('Benchmark evidence')}</strong></div></div>
    <div className="platform-workflow-columns"><section><h3>Detected hardware</h3>{list('Hardware inventory', row => String(row.name ?? row.label ?? row.device_name ?? row.device_id ?? 'Device'), row => readinessLabel(String(row.status ?? row.availability ?? '')))}{(hardwareProfile.cpu_threads !== undefined || hardwareProfile.available_ram_gb !== undefined) && <dl className="platform-engine-capacity">{hardwareProfile.cpu_threads !== undefined && <div><dt>Host processor</dt><dd>{String(hardwareProfile.cpu_threads)} CPU threads</dd></div>}{hardwareProfile.available_ram_gb !== undefined && <div><dt>Available memory</dt><dd>{String(hardwareProfile.available_ram_gb)} GB</dd></div>}</dl>}</section><section><h3>Recent benchmark evidence</h3>{list('Benchmark evidence', row => String(row.device_name ?? row.device_id ?? row.model ?? row.id ?? 'Benchmark'), row => String(row.created_at ?? row.status ?? ''))}</section></div>
    <div className="platform-workflow-actions"><WorkflowAction action={action('Benchmark a device')} onOpen={onAction}/><WorkflowAction action={action('Inspect a local model file')} onOpen={onAction} quiet/><WorkflowAction action={action('Compile a local model')} onOpen={onAction} quiet/></div>
  </section>;

  if (id === 'finetune') return <section className="platform-workflow" aria-label="Fine-tuning pipeline">
    <div className="platform-workflow-heading"><div><span>Adaptation pipeline</span><h2>Dataset to adapter</h2></div>{status && <span className={`platform-readiness ${statusClass}`}>{status}</span>}</div>
    <p className="platform-workflow-lede">Build a dataset, create a training job, then review and activate its adapter.</p>
    <div className="platform-pipeline">{[
      { label: 'Datasets', sub: 'Prepare examples', count: rows('Datasets').length, action: 'Create dataset record', name: (row: Record<string, unknown>) => String(row.name ?? row.id ?? 'Dataset') },
      { label: 'Training jobs', sub: 'Track runs', count: rows('Training jobs').length, action: 'Create a training job', name: (row: Record<string, unknown>) => String(row.name ?? row.base_model ?? row.id ?? 'Training job') },
      { label: 'Adapters', sub: 'Review outputs', count: rows('Adapters').length, action: 'Register adapter', name: (row: Record<string, unknown>) => String(row.name ?? row.base_model ?? row.id ?? 'Adapter') },
    ].map(stage => <article className="platform-pipeline-stage" key={stage.label}><div className="platform-pipeline-stage-head"><div><h3>{stage.label}</h3><span>{stage.sub}</span></div><strong>{countLabel(stage.label)}</strong></div>{list(stage.label, stage.name, row => readinessLabel(String(row.status ?? row.state ?? '')))}<WorkflowAction action={action(stage.action)} onOpen={onAction}/>{stage.label === 'Datasets' && <WorkflowAction action={action('Add training example')} onOpen={onAction} quiet/>}{stage.label === 'Adapters' && <WorkflowAction action={action('Activate a registered adapter')} onOpen={onAction} quiet/>}</article>)}</div>
    <details className="platform-workflow-secondary"><summary>Recipes and training examples</summary><div>{list('Curated recipes', row => String(row.name ?? row.title ?? row.id ?? 'Recipe'))}</div></details>
  </section>;

  if (id === 'science') return <section className="platform-workflow" aria-label="Scientific analysis">
    <div className="platform-workflow-heading"><div><span>Analysis workspace</span><h2>Choose data and run an analysis</h2></div>{status && <span className={`platform-readiness ${statusClass}`}>{status}</span>}</div>
    <p className="platform-workflow-lede">Select a catalog dataset for a bounded analysis, or start a research run from a goal.</p>
    <div className="platform-workflow-columns science-columns"><section><h3>Catalog datasets</h3>{list('Bundled catalog', row => String(row.name ?? row.title ?? row.label ?? row.id ?? 'Dataset'), row => String(row.license ?? row.status ?? ''))}<div className="platform-workflow-actions"><WorkflowAction action={action('Analyze a catalog dataset')} onOpen={onAction}/><WorkflowAction action={action('Plan a bounded science run')} onOpen={onAction} quiet/></div></section><section><h3>Available tools</h3>{list('Discovered tools', row => String(row.name ?? row.label ?? row.id ?? 'Tool'), row => readinessLabel(String(row.status ?? row.availability ?? '')))}<div className="platform-workflow-actions"><WorkflowAction action={action('Add a scientific note')} onOpen={onAction} quiet/></div></section></div>
    <details className="platform-workflow-secondary"><summary>Project memory and datasets</summary><div>{list('Scientific memory', row => String(row.name ?? row.title ?? row.id ?? 'Note'))}<div className="platform-workflow-sublist"><h3>Materials datasets</h3>{list('Materials datasets', row => String(row.name ?? row.title ?? row.id ?? 'Dataset'))}</div></div></details>
  </section>;

  if (id === 'scraper') return <section className="platform-workflow" aria-label="Scraper jobs">
    <div className="platform-workflow-heading"><div><span>Collection jobs</span><h2>Scheduled scraping</h2></div>{status && <span className={`platform-readiness ${statusClass}`}>{status}</span>}</div>
    <p className="platform-workflow-lede">Review scheduled jobs and start a one-time fetch when needed. External access remains subject to server policy.</p>
    <section className="platform-workflow-jobs"><div className="platform-workflow-subhead"><h3>Scheduled jobs</h3><WorkflowAction action={action('Schedule a scrape')} onOpen={onAction}/></div>{list('Scheduled jobs', row => String(row.label ?? row.name ?? row.url ?? row.id ?? 'Scheduled job'), row => String(row.next_run_at ?? row.status ?? row.state ?? ''))}</section>
    <div className="platform-workflow-actions"><WorkflowAction action={action('Run due scheduled scrapes')} onOpen={onAction} quiet/><WorkflowAction action={action('Scrape this URL now')} onOpen={onAction} quiet/></div>
  </section>;

  return <section className="platform-workflow" aria-label="Developer platform overview">
    <div className="platform-workflow-heading"><div><span>Developer platform</span><h2>Integrations and credentials</h2></div>{status && <span className={`platform-readiness ${statusClass}`}>{status}</span>}</div>
    <p className="platform-workflow-lede">Manage scoped API access and review the tools connected to this project.</p>
    <div className="platform-workflow-stats"><div><small>Tools</small><strong>{countLabel('Registered tools')}</strong></div><div><small>Plugins</small><strong>{countLabel('Plugins')}</strong></div><div><small>MCP servers</small><strong>{countLabel('MCP servers')}</strong></div><div><small>Webhooks</small><strong>{countLabel('Webhooks')}</strong></div></div>
    <div className="platform-workflow-columns"><section><h3>Connected tools</h3>{list('Registered tools', row => String(row.name ?? row.label ?? row.id ?? 'Tool'), row => readinessLabel(String(row.status ?? '')))}</section><section><h3>Integration services</h3>{list('MCP servers', row => String(row.name ?? row.label ?? row.id ?? 'Server'), row => readinessLabel(String(row.status ?? '')))}</section></div>
    <div className="platform-workflow-actions"><WorkflowAction action={action('Create scoped API key')} onOpen={onAction}/><WorkflowAction action={action('Register a webhook')} onOpen={onAction} quiet/><WorkflowAction action={action('Generate SDK artifact')} onOpen={onAction} quiet/></div>
  </section>;
}

export function PlatformWorkspaces({ id }: { id: PlatformWorkspaceId }) {
  const config = resourcePageConfigs[id];
  const project = currentProject();
  const reads = useMemo(() => config.reads.map(read => ({ ...read, path: scoped(read.path, project) })), [config.reads, project]);
  const [data, setData] = useState<Snapshot[]>(() => reads.map(() => ({ loading: true })));
  const [active, setActive] = useState(0);
  const [selected, setSelected] = useState(0);
  const [inspecting, setInspecting] = useState(false);
  const [workflowSelection, setWorkflowSelection] = useState<{ label: string; index: number } | null>(null);
  const [action, setAction] = useState<ResourceAction | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [result, setResult] = useState<unknown>();
  const [oneTimeKey, setOneTimeKey] = useState('');
  const [keyRevealed, setKeyRevealed] = useState(false);
  const [keyCopyState, setKeyCopyState] = useState('');
  const keyExpiry = useRef<number | undefined>(undefined);

  const clearOneTimeKey = useCallback(() => {
    if (keyExpiry.current !== undefined) window.clearTimeout(keyExpiry.current);
    keyExpiry.current = undefined;
    setOneTimeKey(''); setKeyRevealed(false); setKeyCopyState('');
  }, []);
  const copyOneTimeKey = async () => {
    try {
      if (!navigator.clipboard?.writeText) throw new Error('Clipboard access is unavailable.');
      await navigator.clipboard.writeText(oneTimeKey);
      setKeyCopyState('Copied to clipboard.');
    } catch { setKeyCopyState('Clipboard access is unavailable; reveal the key and select it to copy.'); }
  };
  useEffect(() => () => { if (keyExpiry.current !== undefined) window.clearTimeout(keyExpiry.current); }, []);

  const load = useCallback(async (signal?: AbortSignal) => {
    setData(reads.map(() => ({ loading: true })));
    const next = await Promise.all(reads.map(async read => {
      try { return { data: await api.get<unknown>(read.path, signal), loading: false } satisfies Snapshot; }
      catch (cause) { return { error: cause instanceof Error ? cause.message : 'Service request failed.', loading: false } satisfies Snapshot; }
    }));
    if (!signal?.aborted) setData(next);
  }, [reads]);
  useEffect(() => { const c = new AbortController(); void load(c.signal); return () => c.abort(); }, [load]);

  const read = reads[active];
  const snap = data[active] ?? { loading: true };
  const rows = readRecords(snap.data, read);
  const isList = Array.isArray(snap.data) || (!!read?.key && Array.isArray(record(snap.data)[read.key])) || read?.label === 'Hardware inventory';
  const detail = isList ? rows[selected] : snap.data;
  const actions = config.actions ?? [];

  const invoke = async (event: React.FormEvent) => {
    event.preventDefault(); if (!action) return;
    if (action.confirmText && !confirm) { setConfirm(true); return; }
    setBusy(true); setError(''); setNotice('');
    try {
      const fieldValues = Object.fromEntries(action.fields.map(field => [field.name, values[field.name] ?? '']));
      const path = scoped(action.path(fieldValues), project);
      const payload = action.payload(fieldValues);
      const response = action.method === 'GET' ? await api.get<unknown>(path) : action.method === 'PUT' ? await api.put<unknown>(path, payload) : action.method === 'PATCH' ? await api.patch<unknown>(path, payload) : action.method === 'DELETE' ? await api.delete<unknown>(path) : await api.post<unknown>(path, payload);
      const generatedKey = action.label === 'Create scoped API key' ? text(record(record(response).api_key), 'key') : '';
      if (generatedKey) {
        clearOneTimeKey(); setOneTimeKey(generatedKey); setKeyRevealed(true); setKeyCopyState('');
        keyExpiry.current = window.setTimeout(clearOneTimeKey, 60_000);
        setResult(undefined); setNotice('API key created. Copy it now; it will be cleared from this page after one minute.');
      } else { setResult(response); setNotice(`${action.label}: service acknowledged the request.`); }
      setAction(null); setValues({}); setConfirm(false); await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The request did not complete.'); setConfirm(false); }
    finally { setBusy(false); }
  };

  const optionsFor = (label: string) => {
    const sourceIndex = reads.findIndex(item => item.label === label);
    return sourceIndex < 0 ? [] : readRecords(data[sourceIndex]?.data, reads[sourceIndex]).map(row => {
      const idValue = row.id ?? row.device_id ?? row.adapter_id ?? row.dataset_id ?? row.application_id;
      if (typeof idValue !== 'string' && typeof idValue !== 'number') return null;
      const display = row.name ?? row.title ?? row.label ?? row.status ?? idValue;
      return { id: String(idValue), label: String(display) };
    }).filter((v): v is { id: string; label: string } => !!v);
  };
  const openAction = (item: ResourceAction) => { setAction(item); setValues({}); setConfirm(false); setError(''); };

  return <section className="specialist-page platform-page" aria-label={config.title} aria-busy={data.some(item => item.loading)}>
    <header className="specialist-header"><div><h1>{config.title}</h1></div><div className="specialist-header-actions"><button type="button" className="specialist-icon-button" aria-label="Refresh data" disabled={data.some(item => item.loading)} onClick={() => void load()}><RefreshCw size={15}/></button></div></header>
    {error && !action && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={() => setError('')}>×</button></div>}
    {notice && <div className="specialist-feedback success" role="status"><Check size={15}/>{notice}<button type="button" onClick={() => setNotice('')}>×</button></div>}
    {oneTimeKey && <section className="platform-one-time-key" aria-label="One-time API key"><div><strong>Copy this API key now</strong><p>It is held in page memory only. Clear it when finished; it also clears when you leave this view or after one minute.</p></div><output aria-label={keyRevealed ? 'Generated API key' : 'Generated API key concealed'}>{keyRevealed ? oneTimeKey : '•'.repeat(Math.min(oneTimeKey.length, 32))}</output><div className="platform-one-time-key-actions"><button type="button" onClick={() => void copyOneTimeKey()}><Copy size={14}/>{keyCopyState || 'Copy key'}</button><button type="button" onClick={() => setKeyRevealed(value => !value)} aria-label={keyRevealed ? 'Conceal key' : 'Reveal key'}>{keyRevealed ? <EyeOff size={14}/> : <Eye size={14}/>} {keyRevealed ? 'Conceal' : 'Reveal'}</button><button type="button" onClick={clearOneTimeKey} aria-label="Dismiss and clear generated key"><X size={14}/>Clear</button></div>{keyCopyState.startsWith('Clipboard') && <p role="status">{keyCopyState}</p>}</section>}
    <div className="platform-frame">
      <WorkflowSurface id={id} reads={reads} data={data} actions={actions} onAction={openAction} selection={workflowSelection} onSelect={setWorkflowSelection}/>
      <details className="platform-technical">
        <summary>Technical data and operations</summary>
        <div className="platform-technical-layout">
          <nav className="platform-nav" aria-label={`${config.title} technical data`}><span className="platform-nav-label">DATA SOURCES</span>{reads.map((item, index) => <button key={item.label} type="button" className={inspecting && index === active ? 'selected' : ''} aria-current={inspecting && index === active ? 'page' : undefined} onClick={() => { setActive(index); setSelected(0); setResult(undefined); setInspecting(true); }}><span>{item.label}</span><small>{data[index]?.loading ? '…' : data[index]?.error ? '!' : Array.isArray(data[index]?.data) || item.key || item.label === 'Hardware inventory' ? readRecords(data[index]?.data, item).length : '•'}</small></button>)}</nav>
          <div className="platform-technical-content">{actions.length > 0 && <div className="platform-operation-list"><span className="platform-nav-label">OPERATIONS</span>{actions.map(item => <button key={item.label} type="button" className="platform-operation" onClick={() => openAction(item)}><Play size={12}/><span>{item.label}</span></button>)}</div>}
          {inspecting && <main className="platform-main" aria-live="polite"><div className="platform-main-heading"><div><span>{read?.label}</span><h2>{detail ? String(record(detail).name ?? record(detail).title ?? record(detail).label ?? read?.label ?? config.title) : read?.label ?? config.title}</h2></div></div>
        {snap.loading ? <div className="specialist-loading" role="status" aria-label={`Loading ${read?.label}`}><i/><i/><i/></div> : snap.error ? <div className="specialist-state-panel error"><CircleAlert/><strong>Couldn’t load {read?.label.toLowerCase()}</strong><p>{snap.error}</p><button type="button" onClick={() => void load()}>Try again</button></div> : isList && !rows.length ? <div className="platform-empty"><strong>No {read?.label.toLowerCase()} yet</strong></div> : isList ? <div className="platform-record-grid">{rows.map((row, index) => { const name = row.name ?? row.title ?? row.label ?? row.filename ?? row.id ?? `Item ${index + 1}`; const status = row.status ?? row.state ?? row.available; const statusText = readableValue(status); return <button key={String(row.id ?? `${name}-${index}`)} type="button" className={`platform-record ${selected === index ? 'active' : ''}`} aria-pressed={selected === index} onClick={() => setSelected(index)}><span><strong>{String(name)}</strong>{statusText && <small>{statusText}</small>}</span><span className="platform-chevron">›</span></button>; })}<div className="platform-selected"><h3>{String(record(detail).name ?? record(detail).title ?? record(detail).label ?? 'Selected record')}</h3><Facts value={detail}/></div></div> : snap.data === null || snap.data === undefined ? <div className="platform-empty"><strong>No data available</strong></div> : <><Facts value={snap.data}/>{result !== undefined && <div className="platform-result"><strong>Latest operation result</strong><Facts value={result}/></div>}</>}
          </main>}</div>
        </div>
      </details>
    </div>
    <Dialog open={!!action} onOpenChange={open => { if (!open && !busy) { setAction(null); setConfirm(false); setError(''); } }}><DialogContent className="specialist-dialog platform-dialog"><DialogTitle>{confirm ? 'Confirm operation' : action?.label}</DialogTitle><DialogDescription>{confirm ? action?.confirmText : `This request will be sent to the local ${config.title} service for project ${project}.`}</DialogDescription><form onSubmit={event => void invoke(event)}>{!confirm && action?.fields.map(field => { const opts = field.optionsFrom ? optionsFor(field.optionsFrom) : (field.options ?? []).map(value => ({ id: value, label: value })); return <label className="specialist-field" key={field.name}><span>{field.label}{field.required && <b>Required</b>}</span>{field.type === 'textarea' ? <textarea required={field.required} rows={3} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}/> : field.type === 'select' || field.optionsFrom ? <select required={field.required} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}><option value="">Choose…</option>{opts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</select> : <input type={field.type ?? 'text'} required={field.required} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}/>}</label>; })}{confirm && <p className="platform-confirm"><CircleAlert size={16}/>{action?.confirmText}</p>}{error && <p className="specialist-feedback error" role="alert">{error}</p>}<div className="specialist-dialog-actions">{confirm && <button className="specialist-secondary" type="button" onClick={() => setConfirm(false)}>Back</button>}<button type="button" className="specialist-secondary" onClick={() => setAction(null)} disabled={busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={busy}>{busy ? 'Working…' : confirm ? 'Confirm and continue' : action?.confirmText ? 'Review operation' : 'Continue'}</button></div></form></DialogContent></Dialog>
  </section>;
}
