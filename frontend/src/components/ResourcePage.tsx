import { useCallback, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, CircleAlert, Plus, RefreshCw } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import type { ResourceAction, ResourceRead } from '../lib/react/contracts';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from './ui/dialog';
import { SpecialistInspector } from '../features/specialists/SpecialistInspector';
import '../features/specialists/specialist-workspace.css';
import './resource-workspace.css';

type Snapshot = { data?: unknown; error?: string; loading: boolean };
type Props = { title: string; eyebrow: string; description: string; reads: ResourceRead[]; actions?: ResourceAction[]; primaryRead?: string; primaryAction?: string };

function activeProject() {
  const queryProject = new URL(location.href).searchParams.get('project');
  try { return queryProject || localStorage.getItem('oa-project') || 'default'; } catch { return queryProject || 'default'; }
}

function actionRequest(action: ResourceAction, path: string, values: Record<string, string>) {
  const method = action.method ?? 'POST';
  if (method === 'GET') return api.get<unknown>(path);
  if (method === 'PUT') return api.put<unknown>(path, action.payload(values));
  if (method === 'PATCH') return api.patch<unknown>(path, action.payload(values));
  if (method === 'DELETE') return api.delete<unknown>(path);
  return api.post<unknown>(path, action.payload(values));
}

export function ResourcePage({ title, eyebrow, reads, actions = [], primaryRead, primaryAction }: Props) {
  const project = activeProject();
  const scopedReads = useMemo(() => reads.map(read => ({
    ...read,
    path: read.path.replaceAll('{project}', encodeURIComponent(project)),
  })), [reads, project]);
  const [snapshots, setSnapshots] = useState<Snapshot[]>(() => reads.map(() => ({ loading: true })));
  const [selectedRead, setSelectedRead] = useState(() => Math.max(0, reads.findIndex(item => item.label === primaryRead)));
  const [selectedRecord, setSelectedRecord] = useState(0);
  const [dialogAction, setDialogAction] = useState<ResourceAction | null>(null);
  const [confirming, setConfirming] = useState(false);
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const load = useCallback(async (signal?: AbortSignal) => {
    setSnapshots(scopedReads.map(() => ({ loading: true })));
    const results = await Promise.all(scopedReads.map(async read => {
      try { return { data: await api.get<unknown>(read.path, signal), loading: false } satisfies Snapshot; }
      catch (cause) {
        if (signal?.aborted) return { loading: false } satisfies Snapshot;
        return { error: cause instanceof Error ? cause.message : 'Could not load this information.', loading: false } satisfies Snapshot;
      }
    }));
    if (!signal?.aborted) setSnapshots(results);
  }, [scopedReads]);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const snapshot = snapshots[selectedRead] ?? { loading: true };
  const read = scopedReads[selectedRead];
  const rows = snapshot.data === undefined ? [] : records(snapshot.data, read?.key);
  const isCollection = Array.isArray(snapshot.data) || (read?.key ? Array.isArray(record(snapshot.data)[read.key]) : false);
  const selectedRow = rows[selectedRecord] ?? rows[0];
  const detail = selectedRow ?? snapshot.data;
  const displayData = selectedRow ? { [read?.key ?? 'items']: [selectedRow] } : snapshot.data;
  const lookupOptions = useCallback((label: string) => {
    const sourceIndex = scopedReads.findIndex(item => item.label === label);
    if (sourceIndex < 0) return [] as Array<{ value: string; label: string }>;
    return records(snapshots[sourceIndex]?.data, scopedReads[sourceIndex]?.key)
      .map(row => {
        const value = row.id ?? row.agent_id ?? row.workflow_id ?? row.application_id ?? row.calendar_id ?? row.provider_id ?? row.conflict_id ?? row.task_id ?? row.dataset_id ?? row.device_id ?? row.adapter_id ?? row.approval_id;
        if (typeof value !== 'string' && typeof value !== 'number') return null;
        const labelValue = row.name ?? row.title ?? row.label ?? row.subject ?? row.filename ?? row.event ?? row.email ?? value;
        const status = row.status ?? row.state;
        return { value: String(value), label: `${String(labelValue)}${typeof status === 'string' ? ` · ${status}` : ''}` };
      }).filter((option): option is { value: string; label: string } => option !== null);
  }, [scopedReads, snapshots]);

  function openAction(action: ResourceAction) {
    setDialogAction(action);
    setConfirming(false);
    setValues({});
    setError('');
  }

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!dialogAction) return;
    if (dialogAction.confirmText && !confirming) { setConfirming(true); return; }
    setBusy(true); setError(''); setNotice('');
    try {
      const path = dialogAction.path(values);
      await actionRequest(dialogAction, path, values);
      await load();
      setNotice(`${dialogAction.label}: request acknowledged by the service.`);
      setDialogAction(null);
      setConfirming(false);
      setValues({});
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'The request did not complete.');
      setConfirming(false);
    } finally { setBusy(false); }
  }

  const featuredAction = actions.find(action => action.label === primaryAction);
  return <section className="specialist-page resource-page" aria-label={title} aria-busy={snapshot.loading}>
    <header className="specialist-header">
      <div><span className="specialist-eyebrow">{eyebrow}</span><h1>{title}</h1></div>
      <div className="specialist-header-actions">
        <span className="specialist-project" title="Active project">{project}</span>
        {featuredAction && <button type="button" className="specialist-primary resource-primary-action" onClick={() => openAction(featuredAction)}><Plus size={14}/>{featuredAction.label}</button>}
        <button type="button" className="specialist-icon-button" aria-label="Refresh data" onClick={() => void load()} disabled={snapshots.some(item => item.loading)}><RefreshCw size={15}/></button>
      </div>
    </header>

    {error && !dialogAction && <div className="specialist-feedback error" role="alert"><CircleAlert size={16}/>{error}</div>}
    {notice && <div className="specialist-feedback success" role="status">{notice}<button type="button" aria-label="Dismiss notification" onClick={() => setNotice('')}>×</button></div>}

    <div className="specialist-workspace">
      <nav className="specialist-list" aria-label={`${title} data and actions`}>
        <div className="specialist-list-heading"><span>Workspace</span><ChevronDown size={14}/></div>
        <div className="specialist-source-list" role="tablist" aria-label="Data sources">
          {scopedReads.map((item, index) => {
            const current = snapshots[index] ?? { loading: true };
            const itemRows = current.data === undefined ? [] : records(current.data, item.key);
            const collection = Array.isArray(current.data) || (item.key ? Array.isArray(record(current.data)[item.key]) : false);
            return <button type="button" role="tab" tabIndex={selectedRead === index ? 0 : -1} aria-selected={selectedRead === index} className={`specialist-source ${selectedRead === index ? 'active' : ''}`} key={`${item.label}:${item.path}`} onClick={() => { setSelectedRead(index); setSelectedRecord(0); }} onKeyDown={event => {
              if (!['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End'].includes(event.key) || scopedReads.length < 2) return;
              event.preventDefault();
              const next = event.key === 'Home' ? 0 : event.key === 'End' ? scopedReads.length - 1 : (index + (event.key === 'ArrowRight' || event.key === 'ArrowDown' ? 1 : -1) + scopedReads.length) % scopedReads.length;
              setSelectedRead(next); setSelectedRecord(0);
              event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
            }}>
              <span className="specialist-source-name">{item.label}</span>
              <span className={`specialist-source-state ${current.error ? 'error' : ''}`}>{current.loading ? '…' : current.error ? '!' : collection ? itemRows.length : '•'}</span>
            </button>;
          })}
        </div>
        {!!actions.length && <details className="specialist-action-menu">
          <summary className="specialist-action-summary"><span>Actions</span><span className="specialist-action-count">{actions.length}</span><ChevronDown size={14}/></summary>
          <div className="specialist-action-list" aria-label="Available actions">{actions.map(action => <button className="specialist-action-link" key={action.label} type="button" onClick={() => openAction(action)}><Plus size={14}/><span>{action.label}</span></button>)}</div>
        </details>}
      </nav>

      <main className="specialist-detail" aria-live="polite">
        <div className="specialist-detail-head">
          <div><span className="specialist-context">{read?.label ?? title}</span><h2>{selectedRow ? String(selectedRow.name ?? selectedRow.title ?? selectedRow.label ?? 'Details') : read?.label ?? title}</h2></div>
          {read && <span className="specialist-scope">{project}</span>}
        </div>
        {snapshot.loading ? <div className="specialist-loading" role="status" aria-label={`Loading ${read?.label ?? title}`}><i/><i/><i/></div>
          : snapshot.error ? <div className="specialist-state-panel error"><CircleAlert size={20}/><strong>Couldn’t load this view</strong><p>{snapshot.error}</p><button type="button" onClick={() => void load()}>Try again</button></div>
          : isCollection && !rows.length ? <div className="specialist-state-panel"><strong>No {read?.label.toLowerCase() ?? 'items'} yet</strong></div>
          : isCollection ? <div className="specialist-record-layout">
            <div className="specialist-record-list" aria-label={read?.label}>
              {rows.map((row, index) => {
                const label = row.name ?? row.title ?? row.label ?? row.subject ?? row.filename ?? row.id ?? 'Untitled item';
                const status = row.status ?? row.state;
                return <button type="button" key={String(row.id ?? `${label}-${index}`)} className={`specialist-record ${selectedRecord === index ? 'selected' : ''}`} aria-current={selectedRecord === index ? 'true' : undefined} onClick={() => setSelectedRecord(index)}>
                  <span className="specialist-record-copy"><strong>{String(label)}</strong>{typeof status === 'string' && <small>{status}</small>}</span><ChevronRight size={15}/>
                </button>;
              })}
            </div>
            <SpecialistInspector value={detail} emptyLabel="No additional details are available." />
          </div>
          : snapshot.data === null || snapshot.data === undefined ? <div className="specialist-state-panel"><strong>No details available</strong></div>
          : <SpecialistInspector value={displayData} emptyLabel="No details available." />}
      </main>
    </div>

    <Dialog open={!!dialogAction} onOpenChange={open => { if (!open && !busy) { setDialogAction(null); setConfirming(false); setError(''); } }}>
      {dialogAction && <DialogContent className="specialist-dialog" aria-describedby="specialist-dialog-description">
        <div className="specialist-dialog-head"><div><DialogTitle>{confirming ? 'Confirm action' : dialogAction.label}</DialogTitle><DialogDescription id="specialist-dialog-description">{confirming ? dialogAction.confirmText : `This action will be sent to the local ${title} service.`}</DialogDescription></div></div>
        <form onSubmit={event => void submit(event)}>
          {!confirming && dialogAction.fields.map(field => { const options: Array<{ value: string; label: string }> = field.optionsFrom ? lookupOptions(field.optionsFrom) : (field.options ?? []).map(option => ({ value: option, label: option })); return <label className="specialist-field" key={field.name}>
            <span>{field.label}{field.required && <b>Required</b>}</span>
            {field.type === 'textarea' ? <textarea name={field.name} rows={4} required={field.required} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={event => setValues(old => ({ ...old, [field.name]: event.currentTarget.value }))}/>
              : field.type === 'select' || field.optionsFrom ? <select name={field.name} required={field.required} value={values[field.name] ?? ''} onChange={event => setValues(old => ({ ...old, [field.name]: event.currentTarget.value }))}><option value="">{field.optionsFrom && !options.length ? `No ${field.optionsFrom.toLowerCase()} available` : 'Choose…'}</option>{options.map(option => <option value={option.value} key={option.value}>{option.label}</option>)}</select>
              : <input name={field.name} type={field.type ?? 'text'} required={field.required} placeholder={field.hint ?? ''} value={values[field.name] ?? ''} onChange={event => setValues(old => ({ ...old, [field.name]: event.currentTarget.value }))}/>}
          </label>;})}
          {confirming && <div className="specialist-confirmation"><CircleAlert size={17}/><span>{dialogAction.confirmText}</span></div>}
          {error && <p className="specialist-feedback error" role="alert">{error}</p>}
          <div className="specialist-dialog-actions">
            {confirming && <button type="button" className="specialist-secondary" onClick={() => setConfirming(false)} disabled={busy}>Back</button>}
            <button type="button" className="specialist-secondary" onClick={() => setDialogAction(null)} disabled={busy}>Cancel</button>
            <button type="submit" className="specialist-primary" disabled={busy}>{busy ? 'Working…' : confirming ? 'Confirm and continue' : dialogAction.confirmText ? 'Review action' : 'Continue'}</button>
          </div>
        </form>
      </DialogContent>}
    </Dialog>
  </section>;
}
