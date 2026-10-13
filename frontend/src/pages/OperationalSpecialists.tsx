import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronRight, CircleAlert, Clock3, Play, Plus, RefreshCw, Search, ShieldCheck, Workflow } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import '../features/specialists/specialist-workspace.css';
import './operational-specialists.css';

type AgentTab = 'agents' | 'marketplace' | 'approvals' | 'permissions';
type ActionKind = 'run' | 'delegate' | 'request' | 'grant' | 'resolve' | 'create' | 'rename' | 'live' | 'dry';
type Item = Record<string, unknown>;
const project = () => { try { return new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default'; } catch { return 'default'; } };
const label = (item: Item) => String(item.name ?? item.title ?? item.label ?? item.subject ?? item.id ?? 'Untitled');
const status = (item: Item) => String(item.status ?? item.state ?? item.enabled ?? '');
const text = (item: Item, key: string) => typeof item[key] === 'string' || typeof item[key] === 'number' ? String(item[key]) : '';
function useProjectData(paths: Array<[string, string, string?]>) {
  const scope = project();
  const [data, setData] = useState<Record<string, unknown>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refreshKey, setRefreshKey] = useState(0);
  const reload = useCallback(() => setRefreshKey(v => v + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true); setError('');
    Promise.all(paths.map(async ([key, path]) => [key, await api.get<unknown>(`${path}${path.includes('?') ? '&' : '?'}project=${encodeURIComponent(scope)}`, controller.signal)] as const))
      .then(rows => { if (!controller.signal.aborted) setData(Object.fromEntries(rows)); })
      .catch(cause => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Could not load this workspace.'); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [scope, refreshKey]);
  return { data, loading, error, reload };
}
function useMutation(reload: () => void) {
  const [busy, setBusy] = useState(false); const busyRef = useRef(false); const [feedback, setFeedback] = useState(''); const [error, setError] = useState('');
  const run = async (call: () => Promise<unknown>, done: string) => {
    if (busyRef.current) return false;
    busyRef.current = true;
    setBusy(true); setFeedback(''); setError('');
    try { await call(); setFeedback(done); reload(); return true; }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'The request did not complete.'); return false; }
    finally { busyRef.current = false; setBusy(false); }
  };
  return { busy, feedback, error, setError, setFeedback, run };
}
function listFor(value: unknown, key: string) { return records(value, key); }
function selectedDetails(item: Item | undefined, fallback: string) {
  if (!item) return <div className="oa-ops-empty">{fallback}</div>;
  const ignored = new Set(['id', 'steps', 'metadata', 'payload', 'config', 'trigger']);
  const fields = Object.entries(item).filter(([key, value]) => !ignored.has(key.toLowerCase()) && (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') && String(value).trim());
  return <dl className="oa-ops-facts">{fields.slice(0, 12).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{String(value)}</dd></div>)}</dl>;
}

export function AgentsWorkspace() {
  const { data, loading, error, reload } = useProjectData([
    ['agents', '/api/agents'], ['marketplace', '/api/agents/marketplace'], ['approvals', '/api/agents/approvals?status=pending'], ['permissions', '/api/agents/permissions'],
  ]);
  const mutation = useMutation(reload);
  const [tab, setTab] = useState<AgentTab>('agents'); const [selected, setSelected] = useState(0); const [query, setQuery] = useState('');
  const [action, setAction] = useState<ActionKind | null>(null); const [target, setTarget] = useState<Item | undefined>(); const [values, setValues] = useState<Record<string, string>>({}); const [confirm, setConfirm] = useState(false);
  const sources: Record<AgentTab, { label: string; key: string }> = { agents: { label: 'Assistants', key: 'agents' }, marketplace: { label: 'Available', key: 'agents' }, approvals: { label: 'Approvals', key: 'approvals' }, permissions: { label: 'Project access', key: 'permissions' } };
  const endpoint = tab === 'agents' ? data.agents : tab === 'marketplace' ? data.marketplace : data[tab];
  const rows = listFor(endpoint, sources[tab].key).filter(item => `${label(item)} ${status(item)}`.toLowerCase().includes(query.toLowerCase()));
  const current = rows[selected] ?? rows[0];
  const open = (kind: ActionKind, item?: Item) => { setAction(kind); setTarget(item); setValues({}); setConfirm(false); mutation.setError(''); };
  useEffect(() => { setSelected(0); }, [tab, query]);
  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); if (!action) return;
    if (['run', 'delegate', 'grant', 'live', 'resolve'].includes(action) && !confirm) { setConfirm(true); return; }
    const p = project(); const v = values;
    const ok = await mutation.run(async () => {
      if (action === 'run') return api.post('/api/agents/run-checked', { agent: text(target ?? {}, 'id') || text(target ?? {}, 'name') || 'executive', goal: v.goal, project: p, require_approval: true, verify: true });
      if (action === 'delegate') return api.post('/api/agents/delegate', { from_agent: v.from_agent || 'executive', to_agent: v.to_agent, goal: v.goal, project: p });
      if (action === 'request') return api.post('/api/agents/approvals/request', { requested_by: 'user', action: v.action, reason: v.reason, project: p, payload: {} });
      if (action === 'grant') return api.post('/api/agents/permissions/grant', { agent: text(target ?? {}, 'id') || v.agent, scope: v.scope, project: p, granted_by: 'user' });
      if (action === 'resolve') return api.post('/api/agents/approvals/resolve', { approval_id: text(target ?? {}, 'id') || v.approval_id, approved: v.approved === 'true' && confirm });
      return undefined;
    }, `${action === 'resolve' ? 'Approval decision' : action[0].toUpperCase() + action.slice(1)} sent to the server.`);
    if (ok) { setAction(null); setConfirm(false); }
  };
  const approvalAction = (item: Item, approved: boolean) => { setTarget(item); setValues({ approved: String(approved) }); setAction('resolve'); setConfirm(false); mutation.setError(''); };
  const title = action === 'run' ? 'Run assistant' : action === 'delegate' ? 'Delegate work' : action === 'request' ? 'Request approval' : action === 'grant' ? 'Grant project access' : action === 'resolve' ? 'Resolve approval' : 'Agent action';
  return <section className="specialist-page oa-ops-page" aria-label="Agent workspace" aria-busy={loading}>
    <header className="specialist-header"><div><span className="specialist-eyebrow">Assistants</span><h1>Agents</h1></div><div className="specialist-header-actions"><span className="specialist-project" title="Active project">{project()}</span><button className="specialist-icon-button" type="button" onClick={reload} aria-label="Refresh agents" disabled={loading}><RefreshCw size={15}/></button></div></header>
    {error && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={reload}>Retry</button></div>}
    {mutation.feedback && <div className="specialist-feedback success" role="status">{mutation.feedback}</div>}
    {mutation.error && !action && <div className="specialist-feedback error" role="alert">{mutation.error}</div>}
    <div className="oa-ops-layout">
      <aside className="oa-ops-rail" aria-label="Agent workspace sections">
        <nav className="oa-ops-tabs">{(Object.keys(sources) as AgentTab[]).map(key => <button type="button" key={key} className={tab === key ? 'active' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => setTab(key)}>{sources[key].label}<small>{listFor(key === 'agents' ? data.agents : key === 'marketplace' ? data.marketplace : data[key], sources[key].key).length}</small></button>)}</nav>
        <div className="oa-ops-list-head"><strong>{sources[tab].label}</strong><span>{rows.length}</span></div>
        <label className="oa-ops-search"><Search size={14}/><input aria-label={`Search ${sources[tab].label}`} value={query} onChange={event => setQuery(event.target.value)} placeholder="Filter"/></label>
        <div className="oa-ops-items" role="listbox" aria-label={sources[tab].label}>
          {loading ? <div className="oa-ops-muted">Loading…</div> : rows.map((item, index) => <button role="option" aria-selected={selected === index} type="button" className={selected === index ? 'selected' : ''} key={text(item, 'id') || `${label(item)}-${index}`} onClick={() => setSelected(index)}><span><b>{label(item)}</b><small>{status(item) || text(item, 'description') || sources[tab].label}</small></span><ChevronRight size={14}/></button>)}
          {!loading && !rows.length && <div className="oa-ops-muted">{query ? 'No matches.' : `No ${sources[tab].label.toLowerCase()} yet.`}</div>}
        </div>
        {tab === 'agents' && <button className="oa-ops-rail-action" onClick={() => open('delegate')} type="button"><Plus size={14}/>Delegate work</button>}
      </aside>
      <main className="oa-ops-content">
        <div className="oa-ops-toolbar"><div><span className="specialist-context">{sources[tab].label}</span><h2>{current ? label(current) : tab === 'approvals' ? 'Approval queue' : 'Agent workspace'}</h2></div>
          <div className="oa-ops-toolbar-actions">{tab === 'agents' && current && <><button type="button" onClick={() => open('run', current)}><Play size={14}/>Run</button><button type="button" onClick={() => open('grant', current)}><ShieldCheck size={14}/>Access</button></>}{tab === 'approvals' && <button type="button" onClick={() => open('request')}><Plus size={14}/>Request</button>}</div>
        </div>
        {loading ? <div className="specialist-loading" role="status" aria-label="Loading agents"><i/><i/><i/></div> : !current ? <div className="oa-ops-empty">{error ? 'Agent data is unavailable.' : tab === 'approvals' ? 'No approvals waiting.' : `No ${sources[tab].label.toLowerCase()} are available for this project.`}</div> : <>
          <div className="oa-ops-status-line"><span className="oa-ops-status-dot"/>{status(current) || 'Registered'}<span>·</span>{project()}</div>
          {tab === 'approvals' && <div className="oa-ops-approval-actions"><button type="button" onClick={() => approvalAction(current, true)}><Check size={14}/>Approve</button><button type="button" onClick={() => approvalAction(current, false)}>Deny</button></div>}
          {selectedDetails(current, 'No additional details were returned.')}
          {tab === 'agents' && <div className="oa-ops-inline-actions"><button type="button" onClick={() => open('run', current)}>Run checked</button><button type="button" onClick={() => open('delegate')}>Delegate</button></div>}
        </>}
      </main>
    </div>
    <Dialog open={!!action} onOpenChange={openState => { if (!openState && !mutation.busy) { setAction(null); setConfirm(false); } }}><DialogContent className="specialist-dialog oa-ops-dialog"><DialogTitle>{title}</DialogTitle><DialogDescription>{confirm ? 'Review this action before sending it. The server remains authoritative for permissions.' : `This request is scoped to ${project()}.`}</DialogDescription>
      <form onSubmit={event => void submit(event)}>
        {!confirm && (action === 'run' || action === 'delegate') && <label className="specialist-field"><span>{action === 'run' ? 'Goal' : 'Task'}</span><textarea required rows={3} value={values.goal ?? ''} onChange={event => setValues({ ...values, goal: event.target.value })}/></label>}
        {!confirm && action === 'delegate' && <label className="specialist-field"><span>Receiving assistant</span><select required value={values.to_agent ?? ''} onChange={event => setValues({ ...values, to_agent: event.target.value })}><option value="">Choose assistant</option>{listFor(data.agents, 'agents').map((agent, index) => <option key={text(agent, 'id') || index} value={text(agent, 'id') || text(agent, 'name')}>{label(agent)}</option>)}</select></label>}
        {!confirm && action === 'request' && <><label className="specialist-field"><span>Requested action</span><input required value={values.action ?? ''} onChange={event => setValues({ ...values, action: event.target.value })}/></label><label className="specialist-field"><span>Reason and scope</span><textarea required rows={3} value={values.reason ?? ''} onChange={event => setValues({ ...values, reason: event.target.value })}/></label></>}
        {!confirm && action === 'grant' && <label className="specialist-field"><span>Permission scope</span><input required placeholder="e.g. files:read" value={values.scope ?? ''} onChange={event => setValues({ ...values, scope: event.target.value })}/></label>}
        {confirm && <div className="specialist-confirmation"><CircleAlert size={17}/><span>{action === 'run' ? 'Run through the checked agent path. The service can still require approval.' : action === 'grant' ? `Grant the selected assistant access for ${project()}?` : action === 'delegate' ? 'Delegate the task to the chosen assistant?' : action === 'resolve' ? `${values.approved === 'true' ? 'Approve' : 'Deny'} this pending request? The decision will be recorded by the server.` : 'Submit this action for server authorization?'}</span></div>}
        {mutation.error && <p className="specialist-feedback error" role="alert">{mutation.error}</p>}
        <div className="specialist-dialog-actions">{confirm && <button type="button" className="specialist-secondary" onClick={() => setConfirm(false)} disabled={mutation.busy}>Back</button>}<button type="button" className="specialist-secondary" onClick={() => setAction(null)} disabled={mutation.busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={mutation.busy}>{mutation.busy ? 'Working…' : confirm ? 'Confirm' : ['run','delegate','grant'].includes(action ?? '') ? 'Review' : 'Submit'}</button></div>
      </form>
    </DialogContent></Dialog>
  </section>;
}

type WorkflowAction = { label: string; category?: string; requires_network?: boolean };
type WorkflowStep = { action: string; values: Record<string, string> };
const stepFields: Record<string, { key: string; label: string }[]> = {
  add_event: [{ key: 'title', label: 'Event title' }, { key: 'content', label: 'Content' }], run_agent: [{ key: 'agent', label: 'Assistant ID' }, { key: 'goal', label: 'Goal' }], search_context: [{ key: 'query', label: 'Search query' }], chat: [{ key: 'message', label: 'Message' }], file_read: [{ key: 'path', label: 'Project file path' }], file_write: [{ key: 'path', label: 'Project file path' }, { key: 'content', label: 'File content' }], file_list: [{ key: 'path', label: 'Project folder path' }], report_generate: [{ key: 'title', label: 'Report title' }, { key: 'content', label: 'Report content' }], audit: [{ key: 'event', label: 'Audit event' }], browser_fetch: [{ key: 'url', label: 'URL' }], api_request: [{ key: 'url', label: 'API URL' }], alert: [{ key: 'message', label: 'Alert message' }], monitor_check: [{ key: 'target', label: 'Monitor target' }], database_query: [{ key: 'path', label: 'Database path' }, { key: 'query', label: 'Read-only SQL query' }], spreadsheet_read: [{ key: 'path', label: 'CSV file path' }], spreadsheet_write: [{ key: 'path', label: 'CSV file path' }, { key: 'content', label: 'CSV content' }], condition: [{ key: 'field', label: 'Context field' }, { key: 'value', label: 'Expected value' }],
};
function workflowLabel(item: Item) { return label(item); }
export function AutomationWorkspace() {
  const { data, loading, error, reload } = useProjectData([['workflows', '/api/workflows'], ['builder', '/api/workflows/builder'], ['runs', '/api/workflow-runs?limit=60']]);
  const mutation = useMutation(reload);
  const [tab, setTab] = useState<'workflows' | 'runs'>('workflows'); const [selected, setSelected] = useState(0); const [query, setQuery] = useState(''); const [action, setAction] = useState<ActionKind | null>(null); const [target, setTarget] = useState<Item>(); const [confirm, setConfirm] = useState(false); const [values, setValues] = useState<Record<string, string>>({}); const [steps, setSteps] = useState<WorkflowStep[]>([{ action: 'run_agent', values: {} }]); const [triggerType, setTriggerType] = useState('manual');
  const allWorkflows = records(data.workflows, 'workflows'); const workflows = allWorkflows.filter(item => workflowLabel(item).toLowerCase().includes(query.toLowerCase())); const runs = records(data.runs, 'runs'); const rows = tab === 'workflows' ? workflows : runs; const current = rows[selected] ?? rows[0];
  const builder = record(data.builder); const builderActions = record(builder.actions) as Record<string, WorkflowAction>;
  const triggerEntries = Object.entries(record(builder.triggers)) as Array<[string, unknown]>;
  useEffect(() => { setSelected(0); }, [tab, query]);
  const open = (kind: ActionKind, item?: Item) => { setTarget(item); setAction(kind); setConfirm(false); setValues({}); if (kind === 'create') { const firstAction = Object.keys(builderActions)[0] ?? 'run_agent'; setSteps([{ action: firstAction, values: {} }]); setTriggerType(triggerEntries.some(([key]) => key === 'manual') ? 'manual' : triggerEntries[0]?.[0] ?? 'manual'); } mutation.setError(''); };
  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); if (!action) return;
    if (['live', 'rename'].includes(action) && !confirm) { setConfirm(true); return; }
    const p = project(); const v = values; const selectedWorkflow = target ?? current;
    let ok = false;
    if (action === 'create') {
      const configuredSteps = steps.map(step => {
        const values = Object.fromEntries(Object.entries(step.values).map(([key, value]) => [key, value.trim()]));
        return { action: step.action, ...values };
      });
      const trigger = triggerType === 'scheduled'
        ? { interval_seconds: Math.max(60, Number(v.interval_seconds || 3600)) }
        : triggerType === 'event' ? { event_name: v.event_name?.trim() } : {};
      ok = await mutation.run(() => api.post('/api/workflows', { name: v.name?.trim(), description: v.description?.trim() || '', project: p, trigger_type: triggerType, trigger, steps: configuredSteps }), 'Workflow created and registered with the selected trigger.');
    } else if (action === 'rename' && selectedWorkflow) {
      ok = await mutation.run(() => api.put(`/api/workflows/${encodeURIComponent(text(selectedWorkflow, 'id'))}`, { name: v.name?.trim() || text(selectedWorkflow, 'name'), description: v.description ?? text(selectedWorkflow, 'description'), project: p }), 'Workflow updated.');
    } else if (action === 'live' && selectedWorkflow) {
      ok = await mutation.run(() => api.post('/api/workflows/run', { workflow_id: text(selectedWorkflow, 'id'), dry_run: false, allow_network: false, project: p }), 'Live run accepted. Refresh run history to see server status.');
    } else if (action === 'dry' && selectedWorkflow) {
      ok = await mutation.run(() => api.post('/api/workflows/run', { workflow_id: text(selectedWorkflow, 'id'), dry_run: true, allow_network: false, project: p }), 'Dry run accepted. Refresh run history to see server status.');
    }
    if (ok) { setAction(null); setConfirm(false); }
  };
  const builderInfo = record(data.builder);
  return <section className="specialist-page oa-ops-page" aria-label="Automation workspace" aria-busy={loading}>
    <header className="specialist-header"><div><span className="specialist-eyebrow">Scheduled</span><h1>Workflows</h1></div><div className="specialist-header-actions"><span className="specialist-project">{project()}</span><button className="specialist-icon-button" type="button" aria-label="Refresh workflows" onClick={reload} disabled={loading}><RefreshCw size={15}/></button></div></header>
    {error && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={reload}>Retry</button></div>}{mutation.feedback && <div className="specialist-feedback success" role="status">{mutation.feedback}</div>}{mutation.error && !action && <div className="specialist-feedback error" role="alert">{mutation.error}</div>}
    <div className="oa-ops-layout">
      <aside className="oa-ops-rail"><nav className="oa-ops-tabs" aria-label="Automation views"><button type="button" className={tab === 'workflows' ? 'active' : ''} onClick={() => setTab('workflows')}>Workflows<small>{workflows.length}</small></button><button type="button" className={tab === 'runs' ? 'active' : ''} onClick={() => setTab('runs')}>Run history<small>{runs.length}</small></button></nav>
        <div className="oa-ops-list-head"><strong>{tab === 'workflows' ? 'Workflow library' : 'Recent runs'}</strong><span>{rows.length}</span></div>
        <label className="oa-ops-search"><Search size={14}/><input aria-label={`Search ${tab}`} placeholder="Filter" value={query} onChange={event => setQuery(event.target.value)}/></label>
        <div className="oa-ops-items" role="listbox" aria-label={tab === 'workflows' ? 'Workflows' : 'Workflow runs'}>{loading ? <div className="oa-ops-muted">Loading…</div> : rows.map((item, index) => <button type="button" role="option" aria-selected={selected === index} className={selected === index ? 'selected' : ''} key={text(item,'id') || `${workflowLabel(item)}-${index}`} onClick={() => setSelected(index)}><span><b>{workflowLabel(item)}</b><small>{status(item) || text(item,'created_at') || 'Manual trigger'}</small></span><ChevronRight size={14}/></button>)}{!loading && !rows.length && <div className="oa-ops-muted">{query ? 'No matches.' : tab === 'runs' ? 'No workflow runs recorded.' : 'No workflows in this project.'}</div>}</div>
        {tab === 'workflows' && <button type="button" className="oa-ops-rail-action" onClick={() => open('create')}><Plus size={14}/>New workflow</button>}
      </aside>
      <main className="oa-ops-content"><div className="oa-ops-toolbar"><div><span className="specialist-context">{tab === 'workflows' ? 'Workflow' : 'Run'}</span><h2>{current ? workflowLabel(current) : tab === 'runs' ? 'Run history' : 'Workflow library'}</h2></div><div className="oa-ops-toolbar-actions">{tab === 'workflows' && current && <><button type="button" onClick={() => open('dry', current)}><Play size={14}/>Dry run</button><button type="button" onClick={() => open('rename', current)}>Edit</button><button type="button" onClick={() => open('live', current)}><Play size={14}/>Live run</button></>}</div></div>
        {loading ? <div className="specialist-loading" role="status" aria-label="Loading workflows"><i/><i/><i/></div> : !current ? <div className="oa-ops-empty">{error ? 'Workflow data is unavailable.' : tab === 'runs' ? 'Runs will appear here after a workflow starts.' : 'Create a workflow to get started.'}</div> : <><div className="oa-ops-status-line"><Workflow size={14}/>{status(current) || 'Manual trigger'}<span>·</span>{project()}</div>{selectedDetails(current, 'No workflow details were returned.')}
          {tab === 'workflows' && <div className="oa-ops-inline-actions"><strong>Steps</strong>{Array.isArray(current.steps) ? (current.steps as unknown[]).map((step, index) => { const item = record(step); const actionName = text(item, 'action'); return <div className="oa-ops-step" key={`${actionName}-${index}`}><span>{index + 1}</span><b>{builderActions[actionName]?.label ?? actionName}</b></div>; }) : <span className="oa-ops-muted">No steps configured.</span>}</div>}
          {tab === 'runs' && <div className="oa-ops-inline-actions"><Clock3 size={15}/><span>Run details are read from the workflow service and refresh with the history list.</span></div>}
        </>}
        <details className="oa-ops-capability"><summary>Builder support</summary><span>{Object.keys(builderActions).length ? `${Object.keys(builderActions).length} server-registered actions · network is disabled by default` : 'Builder metadata unavailable.'}</span>{Object.keys(builderInfo.triggers ?? {}).length > 0 && <small>Trigger types: {Object.keys(record(builderInfo.triggers)).join(', ')}</small>}</details>
      </main>
    </div>
    <Dialog open={!!action} onOpenChange={openState => { if (!openState && !mutation.busy) { setAction(null); setConfirm(false); } }}><DialogContent className="specialist-dialog oa-ops-dialog"><DialogTitle>{action === 'create' ? 'New workflow' : action === 'rename' ? 'Edit workflow' : action === 'dry' ? 'Dry run' : 'Run workflow'}</DialogTitle><DialogDescription>{confirm ? 'This workflow can execute its configured steps. Network access remains disabled for this run.' : `Changes apply to ${project()}.`}</DialogDescription>
      <form onSubmit={event => void submit(event)}>
        {!confirm && action === 'create' && <><label className="specialist-field"><span>Name</span><input required value={values.name ?? ''} onChange={event => setValues({ ...values, name: event.target.value })}/></label><label className="specialist-field"><span>Purpose</span><textarea rows={2} value={values.description ?? ''} onChange={event => setValues({ ...values, description: event.target.value })}/></label><label className="specialist-field"><span>Trigger</span><select aria-label="Workflow trigger" value={triggerType} onChange={event => setTriggerType(event.target.value)}>{(triggerEntries.length ? triggerEntries : [['manual', { label: 'Manual' }]] as Array<[string, unknown]>).map(([key, item]) => <option key={key} value={key}>{text(record(item), 'label') || key}</option>)}</select></label>{triggerType === 'scheduled' && <label className="specialist-field"><span>Repeat every (seconds)</span><input aria-label="Repeat every seconds" type="number" min={60} required value={values.interval_seconds ?? '3600'} onChange={event => setValues({ ...values, interval_seconds: event.target.value })}/></label>}{triggerType === 'event' && <label className="specialist-field"><span>Event name</span><input required value={values.event_name ?? ''} onChange={event => setValues({ ...values, event_name: event.target.value })}/></label>}<div className="oa-workflow-steps"><div className="oa-workflow-steps-head"><strong>Steps</strong><button type="button" onClick={() => setSteps(current => [...current, { action: Object.keys(builderActions)[0] ?? 'run_agent', values: {} }])}><Plus size={13}/>Add step</button></div>{steps.map((step, index) => <fieldset className="oa-workflow-step-editor" key={index}><legend>Step {index + 1}</legend><div className="oa-workflow-step-controls"><label className="specialist-field"><span>Action</span><select aria-label={`Step ${index + 1} action`} value={step.action} onChange={event => setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { action: event.target.value, values: {} } : item))}>{Object.entries(builderActions).map(([key, item]) => <option key={key} value={key}>{item.label || key}{item.requires_network ? ' · network' : ''}</option>)}</select></label><div className="oa-workflow-step-order"><button type="button" aria-label={`Move step ${index + 1} up`} disabled={index === 0} onClick={() => setSteps(current => { const next = [...current]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; return next; })}>↑</button><button type="button" aria-label={`Move step ${index + 1} down`} disabled={index === steps.length - 1} onClick={() => setSteps(current => { const next = [...current]; [next[index + 1], next[index]] = [next[index], next[index + 1]]; return next; })}>↓</button><button type="button" aria-label={`Remove step ${index + 1}`} disabled={steps.length === 1} onClick={() => setSteps(current => current.filter((_, itemIndex) => itemIndex !== index))}>×</button></div></div>{(stepFields[step.action] ?? []).map(field => <label className="specialist-field" key={field.key}><span>{field.label}</span><input required={field.key !== 'agent'} aria-label={`Step ${index + 1} ${field.label}`} value={step.values[field.key] ?? ''} onChange={event => setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, values: { ...item.values, [field.key]: event.target.value } } : item))}/></label>)}</fieldset>)}</div></>}
        {!confirm && action === 'rename' && <><label className="specialist-field"><span>Name</span><input required defaultValue={text(target ?? current ?? {}, 'name')} onChange={event => setValues({ ...values, name: event.target.value })}/></label><label className="specialist-field"><span>Purpose</span><textarea rows={2} defaultValue={text(target ?? current ?? {}, 'description')} onChange={event => setValues({ ...values, description: event.target.value })}/></label></>}
        {confirm && <div className="specialist-confirmation"><CircleAlert size={17}/><span>{action === 'dry' ? 'Evaluate this workflow in dry-run mode. No network access is allowed.' : 'Start a live workflow run? Side effects in its steps will be governed by server policy.'}</span></div>}
        {mutation.error && <p className="specialist-feedback error" role="alert">{mutation.error}</p>}<div className="specialist-dialog-actions">{confirm && <button className="specialist-secondary" type="button" onClick={() => setConfirm(false)} disabled={mutation.busy}>Back</button>}<button className="specialist-secondary" type="button" onClick={() => setAction(null)} disabled={mutation.busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={mutation.busy}>{mutation.busy ? 'Working…' : confirm ? 'Confirm run' : action === 'live' ? 'Review live run' : action === 'dry' ? 'Start dry run' : action === 'create' ? 'Create workflow' : 'Save changes'}</button></div>
      </form>
    </DialogContent></Dialog>
  </section>;
}
