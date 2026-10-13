import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, CircleAlert, Play, RefreshCw } from 'lucide-react';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import { api, record, records } from '../lib/react/api';
import type { ResourceAction, ResourceRead } from '../lib/react/contracts';
import { resourcePageConfigs } from './ResourcePages';
import '../features/specialists/specialist-workspace.css';
import './platform-workspaces.css';

const IDS = ['developer', 'engine', 'finetune', 'phd', 'science', 'scraper'] as const;
export type PlatformWorkspaceId = typeof IDS[number];
type Snapshot = { data?: unknown; error?: string; loading: boolean };
const currentProject = () => new URL(location.href).searchParams.get('project') || (() => { try { return localStorage.getItem('oa-project') || 'default'; } catch { return 'default'; } })();
const scoped = (path: string, project: string) => path.replaceAll('{project}', encodeURIComponent(project));
const visibleValue = (item: unknown) => typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean' ? String(item) : '';
const title = (key: string) => key.replaceAll('_', ' ').replace(/\b\w/g, c => c.toUpperCase());

function Facts({ value }: { value: unknown }) {
  const facts = Object.entries(record(value)).filter(([key, v]) => !['id', 'project', 'project_id', 'raw', 'metadata', 'config', 'settings', 'schema', 'details', 'content', 'secret', 'token'].includes(key.toLowerCase()) && visibleValue(v) !== '').slice(0, 18);
  return facts.length ? <dl className="platform-facts">{facts.map(([key, v]) => <div key={key}><dt>{title(key)}</dt><dd>{visibleValue(v)}</dd></div>)}</dl> : <p className="platform-empty-detail">The service returned no displayable summary fields.</p>;
}

export function PlatformWorkspaces({ id }: { id: PlatformWorkspaceId }) {
  const config = resourcePageConfigs[id];
  const project = currentProject();
  const reads = useMemo(() => config.reads.map(read => ({ ...read, path: scoped(read.path, project) })), [config.reads, project]);
  const [data, setData] = useState<Snapshot[]>(() => reads.map(() => ({ loading: true })));
  const [active, setActive] = useState(0);
  const [selected, setSelected] = useState(0);
  const [action, setAction] = useState<ResourceAction | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [confirm, setConfirm] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [result, setResult] = useState<unknown>();

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
  const rows = records(snap.data, read?.key);
  const isList = Array.isArray(snap.data) || (!!read?.key && Array.isArray(record(snap.data)[read.key]));
  const detail = isList ? rows[selected] : snap.data;
  const actions = config.actions ?? [];
  const fields = detail ? Object.entries(record(detail)).filter(([key, v]) => !['id', 'project', 'project_id', 'raw', 'metadata', 'config', 'settings', 'schema', 'details', 'content', 'secret', 'token'].includes(key.toLowerCase()) && visibleValue(v) !== '').slice(0, 18) : [];

  const invoke = async (event: React.FormEvent) => {
    event.preventDefault(); if (!action) return;
    if (action.confirmText && !confirm) { setConfirm(true); return; }
    setBusy(true); setError(''); setNotice('');
    try {
      const path = scoped(action.path(values), project);
      const payload = action.payload(values);
      const response = action.method === 'GET' ? await api.get<unknown>(path) : action.method === 'PUT' ? await api.put<unknown>(path, payload) : action.method === 'PATCH' ? await api.patch<unknown>(path, payload) : action.method === 'DELETE' ? await api.delete<unknown>(path) : await api.post<unknown>(path, payload);
      setResult(response); setNotice(`${action.label} completed.`); setAction(null); setValues({}); setConfirm(false); await load();
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'The request did not complete.'); setConfirm(false); }
    finally { setBusy(false); }
  };

  const optionsFor = (label: string) => {
    const sourceIndex = reads.findIndex(item => item.label === label);
    return sourceIndex < 0 ? [] : records(data[sourceIndex]?.data, reads[sourceIndex].key).map(row => {
      const idValue = row.id ?? row.device_id ?? row.adapter_id ?? row.dataset_id ?? row.application_id;
      if (typeof idValue !== 'string' && typeof idValue !== 'number') return null;
      const display = row.name ?? row.title ?? row.label ?? row.status ?? idValue;
      return { id: String(idValue), label: String(display) };
    }).filter((v): v is { id: string; label: string } => !!v);
  };

  return <section className="specialist-page platform-page" aria-label={config.title} aria-busy={data.some(item => item.loading)}>
    <header className="specialist-header"><div><span className="specialist-eyebrow">{config.eyebrow}</span><h1>{config.title}</h1></div><div className="specialist-header-actions"><span className="specialist-project">{project}</span><button type="button" className="specialist-icon-button" aria-label="Refresh data" disabled={data.some(item => item.loading)} onClick={() => void load()}><RefreshCw size={15}/></button></div></header>
    {error && !action && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={() => setError('')}>×</button></div>}
    {notice && <div className="specialist-feedback success" role="status"><Check size={15}/>{notice}<button type="button" onClick={() => setNotice('')}>×</button></div>}
    <div className="platform-frame">
      <nav className="platform-nav" aria-label="Platform inventories"><span className="platform-nav-label">INVENTORY</span>{reads.map((item, index) => <button key={item.label} type="button" className={index === active ? 'selected' : ''} aria-current={index === active ? 'page' : undefined} onClick={() => { setActive(index); setSelected(0); setResult(undefined); }}><span>{item.label}</span><small>{data[index]?.loading ? '…' : data[index]?.error ? '!' : Array.isArray(data[index]?.data) || item.key ? records(data[index]?.data, item.key).length : '•'}</small></button>)}{actions.length > 0 && <><span className="platform-nav-label platform-ops-label">OPERATIONS</span>{actions.map(item => <button key={item.label} type="button" className="platform-operation" onClick={() => { setAction(item); setValues({}); setConfirm(false); setError(''); }}><Play size={12}/><span>{item.label}</span></button>)}</>}</nav>
      <main className="platform-main" aria-live="polite"><div className="platform-main-heading"><div><span>{read?.label}</span><h2>{detail ? String(record(detail).name ?? record(detail).title ?? record(detail).label ?? read?.label ?? config.title) : read?.label ?? config.title}</h2></div><span className="platform-scope">{project}</span></div>
        {snap.loading ? <div className="specialist-loading" role="status" aria-label={`Loading ${read?.label}`}><i/><i/><i/></div> : snap.error ? <div className="specialist-state-panel error"><CircleAlert/><strong>Couldn’t load this inventory</strong><p>{snap.error}</p><button type="button" onClick={() => void load()}>Try again</button></div> : isList && !rows.length ? <div className="platform-empty"><strong>No {read?.label.toLowerCase()} yet</strong><span>The service returned an empty collection for this project.</span></div> : isList ? <div className="platform-record-grid">{rows.map((row, index) => { const name = row.name ?? row.title ?? row.label ?? row.filename ?? row.id ?? `Item ${index + 1}`; const status = row.status ?? row.state ?? row.available; return <button key={String(row.id ?? `${name}-${index}`)} type="button" className={`platform-record ${selected === index ? 'active' : ''}`} aria-pressed={selected === index} onClick={() => setSelected(index)}><span><strong>{String(name)}</strong><small>{typeof status === 'boolean' ? status ? 'Available' : 'Unavailable' : String(status ?? 'Registered')}</small></span><span className="platform-chevron">›</span></button>; })}<div className="platform-selected"><h3>{String(record(detail).name ?? record(detail).title ?? record(detail).label ?? 'Selected record')}</h3><Facts value={detail}/></div></div> : snap.data === null || snap.data === undefined ? <div className="platform-empty"><strong>No data returned</strong><span>This endpoint returned no content.</span></div> : <><Facts value={snap.data}/>{result !== undefined && <div className="platform-result"><strong>Latest operation result</strong><Facts value={result}/></div>}{fields.length === 0 && <p className="platform-empty-detail">Service is reachable. No summary fields are currently available.</p>}</>}
      </main>
    </div>
    <Dialog open={!!action} onOpenChange={open => { if (!open && !busy) { setAction(null); setConfirm(false); setError(''); } }}><DialogContent className="specialist-dialog platform-dialog"><DialogTitle>{confirm ? 'Confirm operation' : action?.label}</DialogTitle><DialogDescription>{confirm ? action?.confirmText : `This request will be sent to the local ${config.title} service for project ${project}.`}</DialogDescription><form onSubmit={event => void invoke(event)}>{!confirm && action?.fields.map(field => { const opts = field.optionsFrom ? optionsFor(field.optionsFrom) : (field.options ?? []).map(value => ({ id: value, label: value })); return <label className="specialist-field" key={field.name}><span>{field.label}{field.required && <b>Required</b>}</span>{field.type === 'textarea' ? <textarea required={field.required} rows={3} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}/> : field.type === 'select' || field.optionsFrom ? <select required={field.required} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}><option value="">Choose…</option>{opts.map(o => <option key={o.id} value={o.id}>{o.label}</option>)}</select> : <input type={field.type ?? 'text'} required={field.required} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={e => setValues(v => ({ ...v, [field.name]: e.target.value }))}/>}</label>; })}{confirm && <p className="platform-confirm"><CircleAlert size={16}/>{action?.confirmText}</p>}{error && <p className="specialist-feedback error" role="alert">{error}</p>}<div className="specialist-dialog-actions">{confirm && <button className="specialist-secondary" type="button" onClick={() => setConfirm(false)}>Back</button>}<button type="button" className="specialist-secondary" onClick={() => setAction(null)} disabled={busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={busy}>{busy ? 'Working…' : confirm ? 'Confirm and continue' : action?.confirmText ? 'Review operation' : 'Continue'}</button></div></form></DialogContent></Dialog>
  </section>;
}
