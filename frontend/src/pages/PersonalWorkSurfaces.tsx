import { useCallback, useEffect, useMemo, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import type { ResourceAction, ResourceRead } from '../lib/react/contracts';
import { resourcePageConfigs } from './ResourcePages';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import './personal-work-surfaces.css';

export type PersonalSurfaceId = 'assistant' | 'calendar' | 'communications' | 'twin';
type SurfaceConfig = { title: string; reads: ResourceRead[]; actions: ResourceAction[] };
const configs = resourcePageConfigs as Record<string, SurfaceConfig>;
const labels: Record<PersonalSurfaceId, string> = { assistant: 'Personal assistant', calendar: 'Calendar', communications: 'Communications', twin: 'Personal context' };
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
export function messageBodyOf(row: Record<string, unknown>) {
  for (const key of ['body', 'content', 'text', 'message']) {
    if (typeof row[key] === 'string' && row[key].trim()) return row[key] as string;
  }
  return typeof row.preview === 'string' ? row.preview : typeof row.snippet === 'string' ? row.snippet : '';
}
export function factsOf(value: unknown) {
  const obj = record(value), ignored = new Set(['id','project','project_id','created_at','updated_at','metadata','payload','content','body','description','notes','success','ok','status','healthy','available','enabled']);
  return Object.entries(obj).filter(([key, v]) => {
    const normalizedKey = key.toLowerCase().replace(/[^a-z0-9]/g, '');
    const sensitive = normalizedKey === 'raw' || ['secret','token','password','credential','apikey','authorization'].some(part => normalizedKey.includes(part));
    return !ignored.has(key.toLowerCase()) && !sensitive && typeof v !== 'boolean' && ['string','number'].includes(typeof v) && String(v).length < 180 && !['success','ok','true','false'].includes(String(v).toLowerCase());
  }).slice(0, 6);
}
const dateFields = ['starts_at','start_at','start','date','datetime','scheduled_at','due_at','created_at','timestamp','time'];
function dateOf(row: Record<string, unknown>) {
  for (const key of dateFields) { const value = row[key]; if (typeof value === 'string' || typeof value === 'number') { const date = new Date(value); if (!Number.isNaN(date.getTime())) return date; } }
  return null;
}
export function sortAgenda<T extends Record<string, unknown>>(items: T[]) {
  return [...items].sort((a,b) => (dateOf(a)?.getTime() ?? Number.MAX_SAFE_INTEGER) - (dateOf(b)?.getTime() ?? Number.MAX_SAFE_INTEGER));
}
function formatDay(date: Date) { return new Intl.DateTimeFormat(undefined, { weekday:'long', month:'long', day:'numeric', year:'numeric' }).format(date); }
function formatTime(date: Date | null) { return date ? new Intl.DateTimeFormat(undefined, { hour:'numeric', minute:'2-digit' }).format(date) : 'Time not set'; }
function readName(id: PersonalSurfaceId, name: string) {
  if (id === 'calendar') return ({ Events:'Agenda', 'Calendar service':'Overview', 'External providers':'Providers', 'Unresolved conflicts':'Conflicts', 'Due reminders':'Reminders' } as Record<string,string>)[name] ?? name;
  if (id === 'assistant') return ({ 'Daily brief':'Today', Tasks:'Tasks', 'Productivity coaching':'Coaching', 'CRM records':'People' } as Record<string,string>)[name] ?? name;
  if (id === 'communications') return ({ 'Connector status':'Connectors', 'Pending send approvals':'Approvals', 'Scheduled outbox':'Scheduled' } as Record<string,string>)[name] ?? name;
  return ({ 'Personal context status':'Overview', 'Observed patterns':'Patterns', Suggestions:'Suggestions' } as Record<string,string>)[name] ?? name;
}
export function defaultSectionIndex(id: PersonalSurfaceId, reads: ResourceRead[]) {
  const defaultLabel = id === 'calendar' ? 'Events' : id === 'communications' ? 'Inbox' : undefined;
  return defaultLabel ? Math.max(0, reads.findIndex(read => read.label === defaultLabel)) : 0;
}
export function calendarWeekDays(value: Date) {
  const monday = new Date(value.getFullYear(), value.getMonth(), value.getDate());
  monday.setDate(monday.getDate() - ((monday.getDay() + 6) % 7));
  monday.setHours(0, 0, 0, 0);
  return Array.from({ length: 7 }, (_, index) => {
    const day = new Date(monday);
    day.setDate(monday.getDate() + index);
    return day;
  });
}
export function calendarWeekLabel(value: Date) {
  const days = calendarWeekDays(value), first = days[0], last = days[6];
  const formatter = new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
  const range = typeof formatter.formatRange === 'function' ? formatter.formatRange(first, last) : `${formatter.format(first)} – ${formatter.format(last)}`;
  return `Week of ${range}`;
}
export function calendarEventsPath(path: string, project: string, anchor: Date) {
  const days = calendarWeekDays(anchor), from = new Date(days[0]), to = new Date(days[6]);
  to.setHours(23, 59, 59, 999);
  const url = new URL(scopeReadPath(path, project), location.origin);
  url.searchParams.set('from', from.toISOString());
  url.searchParams.set('to', to.toISOString());
  return `${url.pathname}${url.search}`;
}
function localDayKey(date: Date) { return `${date.getFullYear()}-${date.getMonth()}-${date.getDate()}`; }
const calendarStartHour = 8;
const calendarEndHour = 19;
const calendarPixelsPerHour = 66;
export function calendarEventStyle(event: Record<string, unknown>) {
  const start = dateOf(event);
  if (!start || event.all_day === true) return undefined;
  const rawEnd = typeof event.ends_at === 'string' ? new Date(event.ends_at) : start;
  const startMinute = Math.max(calendarStartHour * 60, Math.min(calendarEndHour * 60, start.getHours() * 60 + start.getMinutes()));
  const endMinute = Math.max(startMinute + 20, Math.min(calendarEndHour * 60, Number.isNaN(rawEnd.getTime()) ? startMinute + 30 : rawEnd.getHours() * 60 + rawEnd.getMinutes()));
  return { top: `${((startMinute - calendarStartHour * 60) / 60) * calendarPixelsPerHour}px`, height: `${Math.max(24, ((endMinute - startMinute) / 60) * calendarPixelsPerHour - 2)}px` };
}
function CalendarAgenda({ rows, onSelect, selected, anchor, onPrevious, onToday, onNext, reads, read, setSelected, setRowIndex }: {
  rows: Record<string,unknown>[]; onSelect: (row:Record<string,unknown>)=>void; selected?: Record<string,unknown>;
  anchor: Date; onPrevious: () => void; onToday: () => void; onNext: () => void;
  reads: ResourceRead[]; read?: ResourceRead; setSelected: (index: number) => void; setRowIndex: (index: number) => void;
}) {
  const days = calendarWeekDays(anchor), byDay = new Map(days.map(day => [localDayKey(day), [] as Array<{ row: Record<string, unknown>; index: number; continues: boolean }>]));
  rows.forEach((event, index) => {
    const date = dateOf(event);
    if (!date) return;
    let day = date, continues = false;
    const eventEnd = typeof event.ends_at === 'string' ? new Date(event.ends_at) : date;
    if (!Number.isNaN(eventEnd.getTime()) && date < days[0] && eventEnd >= days[0]) { day = days[0]; continues = true; }
    const key = localDayKey(day);
    byDay.get(key)?.push({ row: event, index, continues });
  });
  const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const isToday = days.some(day => localDayKey(day) === localDayKey(new Date()));
  const hours = Array.from({ length: calendarEndHour - calendarStartHour }, (_, index) => calendarStartHour + index);
  return <section className="pw-calendar" aria-label="Calendar week">
    <div className="pw-calendar-toolbar">
      <strong>{calendarWeekLabel(anchor)}</strong>
      <div className="pw-calendar-view-switch" aria-label="Calendar sections">{['Events', 'Calendars', 'External providers', 'Unresolved conflicts', 'Due reminders'].map(label => { const target = reads.findIndex(item => item.label === label); return target < 0 ? null : <button type="button" key={label} className={read?.label === label ? 'selected' : ''} aria-current={read?.label === label ? 'page' : undefined} onClick={() => { setSelected(target); setRowIndex(0); }}>{readName('calendar', label)}</button>; })}</div>
      <div className="pw-calendar-navigation" aria-label="Calendar date navigation">
        <button type="button" aria-label="Previous week" title="Previous week" onClick={onPrevious}>‹</button>
        <button type="button" onClick={onToday} disabled={isToday}>Today</button>
        <button type="button" aria-label="Next week" title="Next week" onClick={onNext}>›</button>
      </div>
      <small>Times in {timezone || 'your local time zone'}</small>
    </div>
    <div className="pw-calendar-week" aria-label={calendarWeekLabel(anchor)}>
      <div className="pw-calendar-corner" aria-hidden="true">{timezone || 'Local time'}</div>
      {days.map(day => { const key = localDayKey(day), today = key === localDayKey(new Date()); return <div className={`pw-calendar-day-heading${today ? ' today' : ''}`} key={key}><span>{new Intl.DateTimeFormat(undefined, { weekday: 'short' }).format(day)}</span><time dateTime={`${day.getFullYear()}-${String(day.getMonth() + 1).padStart(2, '0')}-${String(day.getDate()).padStart(2, '0')}`}>{day.getDate()}</time></div>; })}
      <div className="pw-calendar-time-axis" aria-label="Hours">{hours.map(hour => <span key={hour}>{new Intl.DateTimeFormat(undefined, { hour: 'numeric', hour12: false }).format(new Date(2020, 0, 1, hour))}</span>)}</div>
      {days.map(day => {
        const key = localDayKey(day), events = byDay.get(key) ?? [];
        return <section className="pw-calendar-day" key={key} aria-label={formatDay(day)}><div className="pw-calendar-events" aria-label={`${formatDay(day)} events`}><div className="pw-calendar-hour-lines" aria-hidden="true">{hours.map(hour => <span key={hour} />)}</div>{events.map(({ row: event, index, continues }) => {
          const start = dateOf(event), end = typeof event.ends_at === 'string' ? new Date(event.ends_at) : null;
          const validEnd = end && !Number.isNaN(end.getTime()) ? end : null, allDay = event.all_day === true;
          return <button type="button" aria-pressed={selected === event} key={String(event.id ?? event.event_id ?? `${titleOf(event)}-${index}`)} className={`pw-agenda-event${selected === event ? ' selected' : ''}${allDay ? ' all-day' : ''}`} style={calendarEventStyle(event)} onClick={() => onSelect(event)}>
            <time>{allDay ? 'All day' : continues ? `Continues${validEnd ? ` · until ${formatTime(validEnd)}` : ''}` : `${formatTime(start)}${validEnd ? ` – ${formatTime(validEnd)}` : ''}`}</time>
            <span><strong>{titleOf(event)}</strong>{typeof event.location === 'string' && event.location && <small>{event.location}</small>}</span>
          </button>;
        })}</div></section>;
      })}
    </div>
    {rows.length === 0 && <p className="pw-calendar-empty">No events this week.</p>}
  </section>;
}
export function PersonalWorkSurface({ id }: { id: PersonalSurfaceId }) {
  const config = configs[id], project = currentProject(), reads = config.reads;
  const [calendarDate, setCalendarDate] = useState(() => new Date());
  const scoped = useMemo(() => reads.map(read => ({ ...read, path: id === 'calendar' && read.label === 'Events' ? calendarEventsPath(read.path, project, calendarDate) : scopeReadPath(read.path, project) })), [reads, project, id, calendarDate]);
  const [data, setData] = useState<Array<{ value?: unknown; error?: string; loading: boolean }>>(() => reads.map(() => ({ loading: true })));
  const [selected, setSelected] = useState(defaultSectionIndex(id, reads)), [rowIndex, setRowIndex] = useState(0);
  const [action, setAction] = useState<ResourceAction | null>(null), [confirm, setConfirm] = useState(false), [values, setValues] = useState<Record<string,string>>({});
  const [busy, setBusy] = useState(false), [feedback, setFeedback] = useState(''), [error, setError] = useState('');
  const [actionResult, setActionResult] = useState<{ label: string; value: unknown } | null>(null);
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
  const agendaRows = id === 'calendar' && read?.label === 'Events' ? sortAgenda(rows) : rows;
  const actions = config.actions;
  const primaryAction = id === 'assistant' ? 'Add a task' : id === 'calendar' ? 'Create a calendar event' : id === 'communications' ? 'Save a local draft' : 'Add a personal goal';
  const shiftCalendarWeek = (amount: number) => { setRowIndex(0); setCalendarDate(current => { const next = new Date(current); next.setDate(next.getDate() + amount * 7); return next; }); };
  const goToToday = () => { setCalendarDate(new Date()); setRowIndex(0); };
  const openAction = (next: ResourceAction) => { setAction(next); setValues({}); setError(''); setConfirm(false); };
  async function submit(e: React.FormEvent) { e.preventDefault(); if (!action) return; if (action.confirmText && !confirm) { setConfirm(true); return; }
    setBusy(true); setError(''); try {
      const result = await send(action, values);
      if (action.method === 'GET') { setActionResult({ label: action.label, value: result }); setFeedback(`${action.label} loaded.`); }
      else { setActionResult(null); setFeedback(`${action.label} completed.`); }
      setAction(null); setConfirm(false); if (action.method !== 'GET') await load();
    } catch (err) { setError(err instanceof Error ? err.message : 'Request failed.'); setConfirm(false); } finally { setBusy(false); }
  }
  const source = (label: string) => reads.findIndex(item => item.label === label);
  const lookup = (label: string) => { const index = source(label); return index < 0 ? [] : records(data[index]?.value, reads[index]?.key).map(r => { const idValue = r.id ?? r.task_id ?? r.calendar_id ?? r.provider_id ?? r.conflict_id ?? r.approval_id; return idValue == null ? null : { value: String(idValue), label: titleOf(r) }; }).filter((r): r is {value:string;label:string} => !!r); };
  return <section className={`personal-surface pw-${id}`} aria-label={labels[id]} aria-busy={active.loading}>
    <header className="pw-head"><h1>{labels[id]}</h1><div className="pw-head-actions"><button type="button" onClick={() => void load()} disabled={data.some(x => x.loading)} aria-label="Refresh">↻</button><button type="button" className="pw-primary" onClick={() => { const found = actions.find(a => a.label === primaryAction); if (found) openAction(found); }}>{id === 'calendar' ? 'Schedule event' : primaryAction}</button></div></header>
    {feedback && <div className="pw-feedback" role="status">{feedback}<button onClick={() => setFeedback('')} aria-label="Dismiss">×</button></div>}
    {id === 'calendar' && actionResult?.label === 'Find available time' && <section className="pw-availability" aria-label="Available times">
      <div className="pw-availability-head"><h2>Available times</h2><button type="button" onClick={() => setActionResult(null)} aria-label="Dismiss available times">×</button></div>
      {records(actionResult.value, 'slots').length ? <ul>{records(actionResult.value, 'slots').map((slot, index) => {
        const start = slot.start ?? slot.starts_at ?? slot.from;
        const end = slot.end ?? slot.ends_at ?? slot.to;
        const format = (value: unknown) => { const date = typeof value === 'string' || typeof value === 'number' ? new Date(value) : null; return date && !Number.isNaN(date.getTime()) ? new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' }).format(date) : typeof value === 'string' ? value : 'Time not provided'; };
        return <li key={String(slot.id ?? `${start ?? 'slot'}-${index}`)}><strong>{format(start)}</strong><span>{format(end)}</span></li>;
      })}</ul> : <p>No available times in the selected range.</p>}
    </section>}
    <div className={`pw-layout${id === 'calendar' ? ' calendar-layout' : ''}`}><nav className="pw-nav" aria-label={`${labels[id]} sections`}>
      {reads.map((r,i) => <button type="button" key={r.label} aria-current={selected===i?'page':undefined} className={selected===i?'selected':''} onClick={() => {setSelected(i);setRowIndex(0);}}>{readName(id,r.label)}<small>{data[i]?.loading?'…':data[i]?.error?'!':records(data[i]?.value,r.key).length || ''}</small></button>)}
      <details className="pw-more"><summary>Other actions</summary>{actions.filter(a=>a.label!==primaryAction).map(a=><button key={a.label} type="button" onClick={()=>openAction(a)}>{a.label}</button>)}</details>
    </nav><main className="pw-content" aria-live="polite">{!(id === 'calendar' && read?.label === 'Events') && <div className="pw-title-row"><h2>{readName(id,read?.label ?? labels[id])}</h2></div>}
      {active.loading ? <div className="pw-empty" role="status">Loading {read?.label.toLowerCase()}…</div>
      : active.error ? <div className="pw-empty" role="alert"><strong>Couldn’t load this view</strong><p>{active.error}</p><button onClick={()=>void load()}>Try again</button></div>
      : collection && id === 'calendar' && read?.label === 'Events' ? <><CalendarAgenda rows={agendaRows} onSelect={row => setRowIndex(rows.indexOf(row))} selected={item} anchor={calendarDate} onPrevious={() => shiftCalendarWeek(-1)} onToday={goToToday} onNext={() => shiftCalendarWeek(1)} reads={reads} read={read} setSelected={setSelected} setRowIndex={setRowIndex} />
        <aside className="pw-calendar-rail" aria-label="Calendar details">
          {item && <section className="pw-calendar-detail"><h2>Selected event</h2><strong>{titleOf(item)}</strong><p>{dateOf(item) ? new Intl.DateTimeFormat(undefined, { dateStyle: 'full', timeStyle: item.all_day === true ? undefined : 'short' }).format(dateOf(item)!) : 'Date not provided by the calendar service.'}</p>{typeof item.location === 'string' && item.location && <small>{item.location}</small>}{typeof item.description === 'string' && item.description && <p>{item.description}</p>}</section>}
          <section className="pw-calendar-upcoming"><h2>Upcoming events</h2>{agendaRows.filter(event => { const date = dateOf(event); return date && date >= new Date(); }).slice(0, 5).map((event, index) => <button key={String(event.id ?? event.event_id ?? index)} type="button" className={item === event ? 'selected' : ''} onClick={() => setRowIndex(rows.indexOf(event))}><strong>{titleOf(event)}</strong><time>{dateOf(event) ? new Intl.DateTimeFormat(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' }).format(dateOf(event)!) : ''}</time></button>)}{!agendaRows.some(event => { const date = dateOf(event); return date && date >= new Date(); }) && <p>No upcoming events were returned for this week.</p>}</section>
          <section className="pw-calendar-assist"><h2>Scheduling</h2><p>Check the connected calendar for available time before creating an event.</p><button type="button" onClick={() => { const found = actions.find(a => a.label === 'Find available time'); if (found) openAction(found); }}>Find available time</button></section>
        </aside></>
      : collection && !rows.length ? <div className="pw-empty"><strong>No {readName(id,read?.label ?? 'items').toLowerCase()}</strong></div>
      : collection && id === 'assistant' && read?.label === 'Tasks' ? <div className="pw-task-list" aria-label="Tasks">{rows.map((row,i)=><button type="button" key={String(row.id??row.task_id??`${titleOf(row)}-${i}`)} className={rowIndex===i?'selected':''} onClick={()=>setRowIndex(i)}><span className="pw-task-check" aria-hidden="true">{/complete|done/i.test(String(row.status??row.state??''))?'✓':''}</span><span><strong>{titleOf(row)}</strong><small>{[row.due_at,row.priority,row.status].filter(Boolean).join(' · ')}</small></span></button>)}</div>
      : collection && id === 'communications' && read?.label === 'Inbox' ? <div className="pw-inbox-view"><div className="pw-inbox" aria-label="Inbox">{rows.map((row,i)=><button type="button" key={String(row.id??row.message_id??`${titleOf(row)}-${i}`)} className={rowIndex===i?'selected':''} aria-pressed={rowIndex===i} onClick={()=>setRowIndex(i)}><span className="pw-inbox-top">{(row.sender??row.from??row.name) != null && <strong>{String(row.sender??row.from??row.name)}</strong>}<time>{String(row.received_at??row.created_at??row.date??'')}</time></span><b>{String(row.subject??titleOf(row))}</b><small>{String(row.preview??row.snippet??'')}</small></button>)}</div>{item&&<article className="pw-message-detail" aria-label="Selected message"><header>{(item.sender??item.from??item.name)!=null&&<p>{String(item.sender??item.from??item.name)}</p>}<h3>{String(item.subject??titleOf(item))}</h3>{(item.received_at??item.created_at??item.date)!=null&&<time>{String(item.received_at??item.created_at??item.date)}</time>}</header><div>{messageBodyOf(item)||'No message content.'}</div></article>}</div>
      : collection && id === 'twin' && read?.label === 'Suggestions' ? <div className="pw-suggestions" aria-label="Suggestions">{rows.map((row,i)=><button type="button" key={String(row.id??`${titleOf(row)}-${i}`)} className={rowIndex===i?'selected':''} onClick={()=>setRowIndex(i)}><strong>{titleOf(row)}</strong><small>{String(row.created_at??row.updated_at??row.status??'')}</small>{typeof row.description==='string'&&<span>{row.description}</span>}</button>)}</div>
      : collection ? <div className="pw-records"><div className="pw-list" aria-label={read?.label}>{rows.map((row,i)=><button type="button" key={String(row.id??`${titleOf(row)}-${i}`)} className={rowIndex===i?'selected':''} onClick={()=>setRowIndex(i)}><strong>{titleOf(row)}</strong><small>{String(row.status??row.state??row.date??'')}</small></button>)}</div><div className="pw-inspector">{factsOf(detail).length ? <dl>{factsOf(detail).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{String(v)}</dd></div>)}</dl> : <p>No additional details.</p>}{typeof record(detail).description==='string'&&<p className="pw-description">{String(record(detail).description)}</p>}</div></div>
      : active.value == null ? <div className="pw-empty"><strong>No details available</strong></div>
      : <div className="pw-inspector">{factsOf(detail).length?<dl>{factsOf(detail).map(([k,v])=><div key={k}><dt>{k.replaceAll('_',' ')}</dt><dd>{String(v)}</dd></div>)}</dl>:<p>No summary available.</p>}</div>}
    </main></div>
    <Dialog open={!!action} onOpenChange={open=>{if(!open&&!busy){setAction(null);setConfirm(false);}}}>{action&&<DialogContent className="pw-dialog"><DialogTitle>{confirm?'Review and confirm':action.label}</DialogTitle><DialogDescription>{confirm ? action.confirmText : id==='communications' && action.label.includes('Send') ? 'Outbound delivery remains subject to server approval.' : null}</DialogDescription><form onSubmit={e=>void submit(e)}>{!confirm&&action.fields.map(field=>{const options=field.optionsFrom?lookup(field.optionsFrom):(field.options??[]).map(value=>({value,label:value}));return <label key={field.name}>{field.label}{field.type==='textarea'?<textarea required={field.required} value={values[field.name]??''} placeholder={field.hint} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}/>:field.type==='select'||field.optionsFrom?<select required={field.required} value={values[field.name]??''} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}><option value="">Choose…</option>{options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</select>:<input required={field.required} type={field.type??'text'} value={values[field.name]??''} placeholder={field.hint} onChange={e=>setValues(v=>({...v,[field.name]:e.target.value}))}/>}</label>})}{error&&<p role="alert" className="pw-error">{error}</p>}<footer>{confirm&&<button type="button" onClick={()=>setConfirm(false)}>Back</button>}<button type="button" onClick={()=>setAction(null)}>Cancel</button><button type="submit" className="pw-primary" disabled={busy}>{busy?'Working…':confirm?'Confirm':'Continue'}</button></footer></form></DialogContent>}</Dialog>
  </section>;
}
