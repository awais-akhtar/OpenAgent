import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import type { ResourceAction, ResourceRead } from '../lib/react/contracts';
import { resourcePageConfigs } from './ResourcePages';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import './personal-work-surfaces.css';

export type PersonalSurfaceId = 'assistant' | 'calendar' | 'communications' | 'twin';
type SurfaceConfig = { title: string; reads: ResourceRead[]; actions: ResourceAction[] };
const configs = resourcePageConfigs as Record<string, SurfaceConfig>;
const labels: Record<PersonalSurfaceId, string> = { assistant: 'Assistant', calendar: 'Calendar', communications: 'Communications', twin: 'Personal context' };
export function projectScope(href: string, storedProject?: string | null) { return new URL(href).searchParams.get('project') || storedProject || 'default'; }
export function scopeReadPath(path: string, project: string) { return path.replaceAll('{project}', encodeURIComponent(project)); }
export function actionBody(action: ResourceAction, values: Record<string, string>) { return action.payload(values); }
function currentProject() { try { return projectScope(location.href, localStorage.getItem('oa-project')); } catch { return projectScope(location.href); } }
function send(action: ResourceAction, values: Record<string, string>) {
  const path = action.path(values), body = actionBody(action, values);
  if (action.method === 'GET') return api.get(path);
  if (action.method === 'PUT') return api.put(path, body);
  if (action.method === 'PATCH') return api.patch(path, body);
  if (action.method === 'DELETE') return api.delete(path);
  return api.post(path, body);
}
function titleOf(row: Record<string, unknown>) { const v = row.title ?? row.name ?? row.subject ?? row.label ?? row.id ?? 'Untitled'; return String(v); }
function factsOf(value: unknown) {
  const obj = record(value), ignored = new Set(['id','project','project_id','created_at','updated_at','metadata','payload','content','body','description','notes']);
  return Object.entries(obj).filter(([key, v]) => !ignored.has(key) && ['string','number','boolean'].includes(typeof v) && String(v).length < 180).slice(0, 6);
}
export function PersonalWorkSurface({ id }: { id: PersonalSurfaceId }) {
  const config = configs[id], project = currentProject(), reads = config.reads;
  const scoped = useMemo(() => reads.map(read => ({ ...read, path: scopeReadPath(read.path, project) })), [reads, project]);
  const [data, setData] = useState<Array<{ value?: unknown; error?: string; loading: boolean }>>(() => reads.map(() => ({ loading: true })));
  const [selected, setSelected] = useState(0), [rowIndex, setRowIndex] = useState(0);
  const [action, setAction] = useState<ResourceAction | null>(null), [confirm, setConfirm] = useState(false), [values, setValues] = useState<Record<string,string>>({});
  const [busy, setBusy] = useState(false), [feedback, setFeedback] = useState(''), [error, setError] = useState('');
  const load = useCallback(async (signal?: AbortSignal) => {
    setData(scoped.map(() => ({ loading: true })));
    const snapshots = await Promise.all(scoped.map(async read => { try { return { value: await api.get(read.path, signal), loading: false }; } catch (e) { return signal?.aborted ? { loading: false } : { error: e instanceof Error ? e.message : 'Could not load data.', loading: false }; } }));
    if (!signal?.aborted) setData(snapshots);
  }, [scoped]);
  useEffect(() => { const controller = new AbortController(); void load(controller.signal); return () => controller.abort(); }, [load]);
  const active = data[selected] ?? { loading: true }, read = scoped[selected];
  const rows = active.value === undefined ? [] : records(active.value, read?.key);
  const collection = Array.isArray(active.value) || (!!read?.key && Array.isArray(record(active.value)[read.key]));
  const item = rows[rowIndex] ?? rows[0], detail = item ?? active.value;
  const actions = config.actions;
  const primaryAction = id === 'assistant' ? 'Add a task' : id === 'calendar' ? 'Create a calendar event' : id === 'communications' ? 'Save a local draft' : 'Add a personal goal';
  const openAction = (next: ResourceAction) => { setAction(next); setValues({}); setError(''); setConfirm(false); };
  async function submit(e: React.FormEvent) { e.preventDefault(); if (!action) return; if (action.confirmText && !confirm) { setConfirm(true); return; }
    setBusy(true); setError(''); try { await send(action, values); setFeedback(`${action.label} completed.`); setAction(null); setConfirm(false); await load(); } catch (err) { setError(err instanceof Error ? err.message : 'Request failed.'); setConfirm(false); } finally { setBusy(false); }
  }
  const source = (label: string) => reads.findIndex(item => item.label === label);
  const lookup = (label: string) => { const index = source(label); return index < 0 ? [] : records(data[index]?.value, reads[index]?.key).map(r => { const idValue = r.id ?? r.task_id ?? r.calendar_id ?? r.provider_id ?? r.conflict_id ?? r.approval_id; return idValue == null ? null : { value: String(idValue), label: titleOf(r) }; }).filter((r): r is {value:string;label:string} => !!r); };
  return <section className="personal-surface" aria-label={labels[id]} aria-busy={active.loading}>
    <header className="pw-head"><div><span className="pw-kicker">{labels[id]}</span><h1>{labels[id]}</h1></div><div className="pw-head-actions"><span>{project}</span><button type="button" onClick={() => void load()} disabled={data.some(x => x.loading)} aria-label="Refresh">↻</button><button type="button" className="pw-primary" onClick={() => { const found = actions.find(a => a.label === primaryAction); if (found) openAction(found); }}>{primaryAction}</button></div></header>
    {feedback && <div className="pw-feedback" role="status">{feedback}<button onClick={() => setFeedback('')} aria-label="Dismiss">×</button></div>}
    <div className="pw-layout"><nav className="pw-nav" aria-label={`${labels[id]} workspace`}>
      <div className="pw-nav-caption">WORKSPACE</div>{reads.map((r,i) => <button type="button" key={r.label} aria-current={selected===i?'page':undefined} className={selected===i?'selected':''} onClick={() => {setSelected(i);setRowIndex(0);}}>{r.label}<small>{data[i]?.loading?'…':data[i]?.error?'!':records(data[i]?.value,r.key).length || '·'}</small></button>)}
      <details className="pw-more"><summary>More actions <span>{actions.length-1}</span></summary>{actions.filter(a=>a.label!==primaryAction).map(a=><button key={a.label} type="button" onClick={()=>openAction(a)}>{a.label}</button>)}</details>
    </nav><main className="pw-content" aria-live="polite"><div className="pw-title-row"><div><small>{read?.label}</small><h2>{item ? titleOf(item) : read?.label}</h2></div><span>{project}</span></div>
      {active.loading ? <div className="pw-empty" role="status">Loading {read?.label.toLowerCase()}…</div>
      : active.error ? <div className="pw-empty" role="alert"><strong>Couldn’t load this view</strong><p>{active.error}</p><button onClick={()=>void load()}>Try again</button></div>
      : collection && !rows.length ? <div className="pw-empty"><strong>Nothing here yet</strong><p>The service returned no {read?.label.toLowerCase()} for this project.</p></div>
      : collection ? <div className="pw-records"><div className="pw-list" aria-label={read?.label}>{rows.map((row,i)=><button type="button" key={String(row.id??`${titleOf(row)}-${i}`)} className={rowIndex===i?'selected':''} onClick={()=>setRowIndex(i)}><strong>{titleOf(row)}</strong><small>{String(row.status??row.state??row.date??'')}</small></button>)}</div><div className="pw-inspector">{factsOf(detail).length ? <dl>{factsOf(detail).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{String(v)}</dd></div>)}</dl> : <p>No additional details.</p>}{typeof record(detail).description==='string'&&<p className="pw-description">{String(record(detail).description)}</p>}</div></div>
      : active.value == null ? <div className="pw-empty"><strong>No data returned</strong></div>
      : <div className="pw-inspector">{factsOf(detail).length?<dl>{factsOf(detail).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{String(v)}</dd></div>)}</dl>:<p>No summary available.</p>}</div>}
    </main></div>
    <Dialog open={!!action} onOpenChange={open=>{if(!open&&!busy){setAction(null);setConfirm(false);}}}>{action&&<DialogContent className="pw-dialog"><DialogTitle>{confirm?'Review and confirm':action.label}</DialogTitle><DialogDescription>{confirm?action.confirmText:`Sent to the local ${labels[id]} service. External or approval-gated actions remain separate.`}</DialogDescription><form onSubmit={e=>void submit(e)}>{!confirm&&action.fields.map(field=>{const options=field.optionsFrom?lookup(field.optionsFrom):(field.options??[]).map(value=>({value,label:value}));return <label key={field.name}>{field.label}{field.type==='textarea'?<textarea required={field.required} value={values[field.name]??''} placeholder={field.hint} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}/>:field.type==='select'||field.optionsFrom?<select required={field.required} value={values[field.name]??''} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}><option value="">Choose…</option>{options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</select>:<input required={field.required} type={field.type??'text'} value={values[field.name]??''} placeholder={field.hint} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}/>}</label>})}{confirm&&<div className="pw-confirm">{action.confirmText}</div>}{error&&<p role="alert" className="pw-error">{error}</p>}<footer>{confirm&&<button type="button" onClick={()=>setConfirm(false)}>Back</button>}<button type="button" onClick={()=>setAction(null)}>Cancel</button><button type="submit" className="pw-primary" disabled={busy}>{busy?'Working…':confirm?'Confirm':'Continue'}</button></footer></form></DialogContent>}</Dialog>
  </section>;
}
