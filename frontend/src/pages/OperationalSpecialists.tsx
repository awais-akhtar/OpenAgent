import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronRight, CircleAlert, Clock3, Play, Plus, RefreshCw, Search, ShieldCheck, Workflow } from 'lucide-react';
import { api, record, records } from '../lib/react/api';
import { Dialog, DialogContent, DialogDescription, DialogTitle } from '../components/ui/dialog';
import '../features/specialists/specialist-workspace.css';
import './operational-specialists.css';

type AgentTab = 'agents' | 'marketplace' | 'approvals' | 'permissions';
export const agentSections = { primary: ['agents', 'approvals'] as const, secondary: ['marketplace', 'permissions'] as const };
type ActionKind = 'run' | 'delegate' | 'request' | 'grant' | 'resolve' | 'create' | 'rename' | 'live' | 'dry' | 'toggle' | 'delete';
type Item = Record<string, unknown>;
const project = () => { try { return new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default'; } catch { return 'default'; } };
const label = (item: Item) => String(item.name ?? item.title ?? item.label ?? item.subject ?? item.id ?? 'Untitled');
const status = (item: Item) => { const value = item.status ?? item.state ?? item.enabled ?? ''; return typeof value === 'boolean' ? value ? 'Enabled' : 'Disabled' : String(value); };
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
  const run = async (call: () => Promise<unknown>, done: string | ((result: unknown) => string | { message: string; error?: boolean })) => {
    if (busyRef.current) return false;
    busyRef.current = true;
    setBusy(true); setFeedback(''); setError('');
    try { const result = await call(); const outcome = typeof done === 'function' ? done(result) : done; if (typeof outcome === 'string') setFeedback(outcome); else { setFeedback(outcome.error ? '' : outcome.message); if (outcome.error) setError(outcome.message); } reload(); return true; }
    catch (cause) { setError(cause instanceof Error ? cause.message : 'The request did not complete.'); return false; }
    finally { busyRef.current = false; setBusy(false); }
  };
  return { busy, feedback, error, setError, setFeedback, run };
}
function listFor(value: unknown, key: string) { return records(value, key); }
function selectedDetails(item: Item | undefined, fallback: string) {
  if (!item) return <div className="oa-ops-empty">{fallback}</div>;
  const ignored = new Set(['id', 'project', 'project_id', 'steps', 'metadata', 'payload', 'config', 'trigger', 'enabled']);
  const fields = Object.entries(item).filter(([key, value]) => !ignored.has(key.toLowerCase()) && (typeof value === 'string' || typeof value === 'number') && String(value).trim() && !['true','false','ok','success'].includes(String(value).toLowerCase()));
  return <dl className="oa-ops-facts">{fields.slice(0, 12).map(([key, value]) => <div key={key}><dt>{key.replaceAll('_', ' ')}</dt><dd>{String(value)}</dd></div>)}</dl>;
}
const itemId = (item: Item) => text(item, 'id') || text(item, 'approval_id');
export function approvalContext(item: Item, activeProject: string) {
  return { title: text(item, 'action') || label(item), requester: text(item, 'requested_by') || text(item, 'agent') || 'User', reason: text(item, 'reason') || text(item, 'scope'), project: text(item, 'project') || activeProject };
}
export function permissionContext(item: Item, activeProject: string) {
  return { agent: text(item, 'agent_name') || text(item, 'agent') || text(item, 'agent_id') || 'Assistant', scope: text(item, 'scope') || text(item, 'permission') || 'Scope unavailable', project: text(item, 'project') || activeProject };
}

export function AgentsWorkspace() {
  const { data, loading, error, reload } = useProjectData([
    ['agents', '/api/agents'], ['marketplace', '/api/agents/marketplace'], ['approvals', '/api/agents/approvals?status=pending'], ['permissions', '/api/agents/permissions'],
  ]);
  const mutation = useMutation(reload);
  const [tab, setTab] = useState<AgentTab>('agents'); const [selected, setSelected] = useState(0); const [query, setQuery] = useState('');
  const [action, setAction] = useState<ActionKind | null>(null); const [target, setTarget] = useState<Item | undefined>(); const [values, setValues] = useState<Record<string, string>>({}); const [confirm, setConfirm] = useState(false);
  const sources: Record<AgentTab, { label: string; key: string }> = { agents: { label: 'Agents', key: 'agents' }, marketplace: { label: 'Available', key: 'agents' }, approvals: { label: 'Approvals', key: 'approvals' }, permissions: { label: 'Project access', key: 'permissions' } };
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
      if (action === 'resolve') return api.post('/api/agents/approvals/resolve', { approval_id: itemId(target ?? {}) || v.approval_id, approved: v.approved === 'true' && confirm });
      return undefined;
    }, action === 'resolve' ? 'Approval decision recorded.' : action === 'run' ? 'Agent run started.' : action === 'grant' ? 'Project access updated.' : action === 'delegate' ? 'Task delegated.' : action === 'request' ? 'Approval request submitted.' : 'Agent settings updated.');
    if (ok) { setAction(null); setConfirm(false); }
  };
  const approvalAction = (item: Item, approved: boolean) => { setTarget(item); setValues({ approved: String(approved) }); setAction('resolve'); setConfirm(false); mutation.setError(''); };
  const approvalRows = listFor(data.approvals, 'approvals');
  const agentRows = listFor(data.agents, 'agents');
  const title = action === 'run' ? 'Run assistant' : action === 'delegate' ? 'Delegate work' : action === 'request' ? 'Request approval' : action === 'grant' ? 'Grant project access' : action === 'resolve' ? 'Resolve approval' : 'Agent action';
  return <section className="specialist-page oa-ops-page" aria-label="Agent workspace" aria-busy={loading}>
    <header className="specialist-header"><h1>Agents</h1><div className="specialist-header-actions"><button className="specialist-icon-button" type="button" onClick={reload} aria-label="Refresh agents" disabled={loading}><RefreshCw size={15}/></button></div></header>
    {error && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={reload}>Retry</button></div>}
    {mutation.feedback && <div className="specialist-feedback success" role="status">{mutation.feedback}</div>}
    {mutation.error && !action && <div className="specialist-feedback error" role="alert">{mutation.error}</div>}
    <div className="oa-ops-layout">
      <aside className="oa-ops-rail" aria-label="Agent workspace sections">
        <nav className="oa-ops-tabs" aria-label="Assistant work">{agentSections.primary.map(key => <button type="button" key={key} className={tab === key ? 'active' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => setTab(key)}>{sources[key].label}<small>{listFor(data[key], sources[key].key).length}</small></button>)}</nav>
        <details className="oa-agent-secondary"><summary>Manage</summary><div>{agentSections.secondary.map(key => <button type="button" key={key} className={tab === key ? 'active' : ''} aria-current={tab === key ? 'page' : undefined} onClick={() => setTab(key)}>{sources[key].label}<small>{listFor(key === 'marketplace' ? data.marketplace : data.permissions, sources[key].key).length}</small></button>)}</div></details>
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
        {loading ? <div className="specialist-loading" role="status" aria-label="Loading assistants"><i/><i/><i/></div> : !current ? <div className="oa-ops-empty">{error ? 'Assistant data is unavailable.' : tab === 'approvals' ? 'No approvals waiting.' : `No ${sources[tab].label.toLowerCase()} available.`}</div> : <>
          <div className="oa-ops-status-line"><span className="oa-ops-status-dot"/>{status(current) || 'Registered'}<span>·</span>{project()}</div>
          {tab === 'approvals' && <div className="oa-ops-approval-actions"><button type="button" onClick={() => approvalAction(current, true)}><Check size={14}/>Approve</button><button type="button" onClick={() => approvalAction(current, false)}>Deny</button></div>}
          {tab === 'permissions' && <div className="oa-agent-scope"><ShieldCheck size={16}/><span>Access for <strong>{permissionContext(current, project()).agent}</strong><code>{permissionContext(current, project()).scope}</code><small>Project: {permissionContext(current, project()).project}</small></span></div>}
          {tab !== 'permissions' && selectedDetails(current, 'No additional details were returned.')}
          {tab === 'agents' && <div className="oa-ops-inline-actions"><button type="button" onClick={() => open('run', current)}>Run checked</button><button type="button" onClick={() => open('delegate')}>Delegate</button></div>}
        </>}
        {tab === 'agents' && !loading && <section className="oa-agent-queue" aria-label="Pending approval queue"><header><div><h3>Pending approvals</h3><small>{approvalRows.length} awaiting review · {project()}</small></div><button type="button" onClick={() => open('request')}><Plus size={14}/>Request approval</button></header>{approvalRows.length ? <div className="oa-agent-queue-list">{approvalRows.map((approval,index) => { const context=approvalContext(approval,project()); return <article key={itemId(approval)||`${context.title}-${index}`}><div><strong>{context.title}</strong><small>Requested by {context.requester} · {context.project}</small>{context.reason&&<p>{context.reason}</p>}</div><div className="oa-ops-approval-actions"><button type="button" onClick={()=>approvalAction(approval,true)}><Check size={14}/>Approve</button><button type="button" onClick={()=>approvalAction(approval,false)}>Deny</button></div></article>; })}</div> : <p>No approvals waiting.</p>}</section>}
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
type StepField = { key: string; label: string; kind?: 'textarea' | 'select'; options?: string[]; optional?: boolean };
const stepFields: Record<string, StepField[]> = {
  add_event: [{ key: 'title', label: 'Event title' }, { key: 'content', label: 'Content', optional: true }], run_agent: [{ key: 'agent', label: 'Assistant ID', optional: true }, { key: 'goal', label: 'Goal' }], search_context: [{ key: 'query', label: 'Search query' }], chat: [{ key: 'message', label: 'Message' }], file_read: [{ key: 'path', label: 'Project file path' }], file_write: [{ key: 'path', label: 'Project file path' }, { key: 'content', label: 'File content' }], file_list: [{ key: 'path', label: 'Project folder path' }], report_generate: [{ key: 'path', label: 'Report path', optional: true }, { key: 'title', label: 'Report title', optional: true }], audit: [{ key: 'audit_action', label: 'Audit event' }, { key: 'resource', label: 'Resource', optional: true }], browser_fetch: [{ key: 'url', label: 'URL' }], api_request: [{ key: 'url', label: 'API URL' }, { key: 'method', label: 'Method', kind: 'select', options: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE'], optional: true }, { key: 'body', label: 'Request body', kind: 'textarea', optional: true }], alert: [{ key: 'message', label: 'Alert message' }], monitor_check: [{ key: 'path', label: 'Context path', optional: true }, { key: 'op', label: 'Condition', kind: 'select', options: ['truthy', 'exists', 'equals', 'not_equals', 'contains'], optional: true }, { key: 'value', label: 'Expected value', optional: true }, { key: 'message', label: 'Message', optional: true }], database_query: [{ key: 'path', label: 'Database path', optional: true }, { key: 'query', label: 'Read-only SQL query' }], spreadsheet_read: [{ key: 'path', label: 'CSV file path' }], spreadsheet_write: [{ key: 'path', label: 'CSV file path' }, { key: 'csv', label: 'CSV data (first row is headers)', kind: 'textarea' }], condition: [{ key: 'path', label: 'Context path' }, { key: 'op', label: 'Condition', kind: 'select', options: ['truthy', 'exists', 'equals', 'not_equals', 'contains'], optional: true }, { key: 'value', label: 'Expected value', optional: true }],
};
export function validateWorkflowDraft(name: string, triggerType: string, values: Record<string, string>, steps: WorkflowStep[], actions: Record<string, WorkflowAction>, triggers: Record<string, unknown>) {
  if (!name.trim()) return 'Enter a workflow name.';
  if (!Object.hasOwn(triggers, triggerType)) return 'Choose a supported trigger.';
  if (triggerType === 'scheduled' && (!Number.isInteger(Number(values.interval_seconds)) || Number(values.interval_seconds) < 60)) return 'Set a schedule interval of at least 60 seconds.';
  if (triggerType === 'event' && !values.event_name?.trim()) return 'Enter the event name that starts this workflow.';
  if (!steps.length) return 'Add at least one workflow step.';
  for (const [index, step] of steps.entries()) {
    if (!Object.hasOwn(actions, step.action)) return `Choose a supported action for step ${index + 1}.`;
    if (!Object.hasOwn(stepFields, step.action)) return `The editor does not yet support inputs for step ${index + 1}.`;
    const required = (stepFields[step.action] ?? []).filter(field => !field.optional);
    const missing = required.find(field => !step.values[field.key]?.trim());
    if (missing) return `Enter ${missing.label.toLowerCase()} for step ${index + 1}.`;
    const operator = step.values.op || 'truthy';
    if (['condition', 'monitor_check'].includes(step.action) && ['equals', 'not_equals', 'contains'].includes(operator) && !step.values.value?.trim()) return `Enter an expected value for step ${index + 1}.`;
    if (step.action === 'spreadsheet_write' && parseCsvRows(step.values.csv ?? '').length === 0) return `Add a CSV header and at least one data row for step ${index + 1}.`;
  }
  return '';
}
function parseCsvRows(source: string): Record<string, string>[] {
  const rows: string[][] = [];
  let row: string[] = []; let cell = ''; let quoted = false;
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (char === '"' && quoted && source[index + 1] === '"') { cell += '"'; index += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === ',' && !quoted) { row.push(cell); cell = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) { if (char === '\r' && source[index + 1] === '\n') index += 1; row.push(cell); rows.push(row); row = []; cell = ''; }
    else cell += char;
  }
  if (cell || row.length) { row.push(cell); rows.push(row); }
  const [headerRow, ...dataRows] = rows;
  const headers = (headerRow ?? []).map(header => header.trim()).filter(Boolean);
  if (!headers.length) return [];
  return dataRows.filter(values => values.some(value => value.trim())).map(values => Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ''])));
}
export function workflowStepPayload(step: WorkflowStep) {
  const values = Object.fromEntries(Object.entries(step.values).map(([key, value]) => [key, value.trim()]));
  if (step.action === 'condition' || step.action === 'monitor_check') {
    const { path, op, value, ...rest } = values;
    return { action: step.action, ...rest, condition: { path: path || '', op: op || 'truthy', ...(value ? { value } : {}) } };
  }
  if (step.action === 'spreadsheet_write') {
    const { csv, ...rest } = values;
    return { action: step.action, ...rest, rows: parseCsvRows(csv ?? '') };
  }
  if (step.action === 'api_request' && values.body) {
    try { values.body = JSON.stringify(JSON.parse(values.body)); } catch { /* Keep a plain-text request body as entered. */ }
  }
  return { action: step.action, ...values };
}
function workflowRunMessage(response: unknown, dryRun: boolean) {
  const run = record(record(response).run);
  const runStatus = text(run, 'status').toLowerCase();
  if (runStatus === 'failed') return { error: true, message: `${dryRun ? 'Dry run' : 'Workflow run'} failed. Review the recorded run for the failing step.` };
  if (runStatus === 'completed') return { message: `${dryRun ? 'Dry run' : 'Workflow'} completed. The run is in history.` };
  return { message: `Run recorded with status ${runStatus || 'unavailable'}.` };
}
type StepUpdate = (value: WorkflowStep[] | ((current: WorkflowStep[]) => WorkflowStep[])) => void;
function WorkflowStepInputs({ step, index, setSteps, setDraftError }: { step: WorkflowStep; index: number; setSteps: StepUpdate; setDraftError: (value: string) => void }) {
  return <>{(stepFields[step.action] ?? []).map(field => <label className="specialist-field" key={field.key}><span>{field.label}</span>{field.kind === 'select' ? <select aria-label={`Step ${index + 1} ${field.label}`} required={!field.optional} value={step.values[field.key] ?? field.options?.[0] ?? ''} onChange={event => { setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, values: { ...item.values, [field.key]: event.target.value } } : item)); setDraftError(''); }}>{(field.options ?? []).map(option => <option key={option} value={option}>{option.replaceAll('_', ' ')}</option>)}</select> : field.kind === 'textarea' ? <textarea rows={3} aria-label={`Step ${index + 1} ${field.label}`} required={!field.optional} value={step.values[field.key] ?? ''} onChange={event => { setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, values: { ...item.values, [field.key]: event.target.value } } : item)); setDraftError(''); }}/> : <input required={!field.optional} aria-label={`Step ${index + 1} ${field.label}`} value={step.values[field.key] ?? ''} onChange={event => { setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { ...item, values: { ...item.values, [field.key]: event.target.value } } : item)); setDraftError(''); }}/>}</label>)}</>;
}
type DraftBuilderProps = {
  values: Record<string, string>;
  setValues: (value: Record<string, string> | ((current: Record<string, string>) => Record<string, string>)) => void;
  steps: WorkflowStep[];
  setSteps: StepUpdate;
  triggerType: string;
  setTriggerType: (value: string) => void;
  triggerEntries: Array<[string, unknown]>;
  builderActions: Record<string, WorkflowAction>;
  draftError: string;
  setDraftError: (value: string) => void;
  mutationError: string;
  busy: boolean;
  onCancel: () => void;
  onSubmit: (event: React.FormEvent<HTMLFormElement>) => void;
};
function WorkflowDraftBuilder(props: DraftBuilderProps) {
  const { values, setValues, steps, setSteps, triggerType, setTriggerType, triggerEntries, builderActions, draftError, setDraftError, mutationError, busy, onCancel, onSubmit } = props;
  return <section className="oa-workflow-builder" aria-label="Workflow builder"><div className="oa-ops-toolbar"><div><h2>New workflow</h2><p>Set a trigger and order the actions to run.</p></div></div><form className="oa-workflow-form" onSubmit={onSubmit}>
    <div className="oa-workflow-setup"><label className="specialist-field"><span>Name</span><input required autoFocus value={values.name ?? ''} onChange={event => { setValues({ ...values, name: event.target.value }); setDraftError(''); }}/></label><label className="specialist-field"><span>Purpose</span><textarea rows={2} value={values.description ?? ''} onChange={event => setValues({ ...values, description: event.target.value })}/></label><label className="specialist-field"><span>Trigger</span><select aria-label="Workflow trigger" value={triggerType} onChange={event => { setTriggerType(event.target.value); setDraftError(''); }}>{triggerEntries.map(([key, item]) => <option key={key} value={key}>{text(record(item), 'label') || key}</option>)}</select></label>
      {triggerType === 'scheduled' && <label className="specialist-field"><span>Repeat every (seconds)</span><input aria-label="Repeat every seconds" type="number" min={60} required value={values.interval_seconds ?? '3600'} onChange={event => { setValues({ ...values, interval_seconds: event.target.value }); setDraftError(''); }}/></label>}{triggerType === 'event' && <label className="specialist-field"><span>Event name</span><input required value={values.event_name ?? ''} onChange={event => { setValues({ ...values, event_name: event.target.value }); setDraftError(''); }}/></label>}</div>
    <div className="oa-workflow-steps"><div className="oa-workflow-steps-head"><strong>Steps</strong><button type="button" onClick={() => { const action = Object.hasOwn(builderActions, 'run_agent') ? 'run_agent' : Object.keys(builderActions)[0] ?? 'run_agent'; setSteps(current => [...current, { action, values: {} }]); setDraftError(''); }}><Plus size={13}/>Add step</button></div>{steps.map((step, index) => <fieldset className="oa-workflow-step-editor" key={index}><legend>Step {index + 1}</legend><div className="oa-workflow-step-controls"><label className="specialist-field"><span>Action</span><select aria-label={`Step ${index + 1} action`} value={step.action} onChange={event => { setSteps(current => current.map((item, itemIndex) => itemIndex === index ? { action: event.target.value, values: {} } : item)); setDraftError(''); }}>{Object.entries(builderActions).map(([key, item]) => <option key={key} value={key}>{item.label || key}{item.requires_network ? ' · network' : ''}</option>)}</select></label><div className="oa-workflow-step-order"><button type="button" aria-label={`Move step ${index + 1} up`} disabled={index === 0} onClick={() => setSteps(current => { const next = [...current]; [next[index - 1], next[index]] = [next[index], next[index - 1]]; return next; })}>↑</button><button type="button" aria-label={`Move step ${index + 1} down`} disabled={index === steps.length - 1} onClick={() => setSteps(current => { const next = [...current]; [next[index + 1], next[index]] = [next[index], next[index + 1]]; return next; })}>↓</button><button type="button" aria-label={`Remove step ${index + 1}`} disabled={steps.length === 1} onClick={() => setSteps(current => current.filter((_, itemIndex) => itemIndex !== index))}>×</button></div></div><WorkflowStepInputs step={step} index={index} setSteps={setSteps} setDraftError={setDraftError}/></fieldset>)}</div>
    {draftError && <p className="specialist-feedback error" role="alert">{draftError}</p>}{mutationError && <p className="specialist-feedback error" role="alert">{mutationError}</p>}<div className="specialist-dialog-actions"><button className="specialist-secondary" type="button" onClick={onCancel} disabled={busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={busy || !Object.keys(builderActions).length}>{busy ? 'Working…' : 'Create workflow'}</button></div>
  </form></section>;
}
function workflowLabel(item: Item) { return label(item); }
function workflowTime(value: unknown) {
  if (typeof value !== 'string' || !value.trim()) return 'Time unavailable';
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? 'Time unavailable' : parsed.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}
function workflowTrigger(item: Item) {
  const trigger = record(item.trigger); const config = record(trigger.config);
  const type = text(trigger, 'trigger_type') || text(item, 'trigger_type') || 'manual';
  const name = type === 'scheduled' ? 'Scheduled' : type === 'event' ? 'Event' : 'Manual';
  const interval = Number(config.interval_seconds);
  const schedule = type === 'scheduled' && Number.isFinite(interval) && interval > 0
    ? `Every ${interval % 3600 === 0 ? `${interval / 3600} hr` : interval % 60 === 0 ? `${interval / 60} min` : `${interval} sec`}`
    : type === 'event' ? text(config, 'event_name') || 'Event trigger' : 'Run on demand';
  const enabled = typeof item.enabled === 'boolean' ? item.enabled : typeof trigger.enabled === 'boolean' ? trigger.enabled : undefined;
  return { name, schedule, enabled, nextRun: text(trigger, 'next_run_at') };
}
function runStepResult(step: Item, actions: Record<string, WorkflowAction>) {
  const actionName = text(step, 'action');
  const error = text(step, 'error');
  const useful = ['path', 'url', 'row_count', 'bytes', 'message', 'passed'].find(key => typeof step[key] === 'string' || typeof step[key] === 'number' || typeof step[key] === 'boolean');
  const detail = error || (useful ? `${useful.replaceAll('_', ' ')}: ${String(step[useful])}` : '');
  return { action: actions[actionName]?.label || actionName || 'Workflow step', state: text(step, 'status') || 'Status unavailable', detail };
}
export function AutomationWorkspace() {
  const { data, loading, error, reload } = useProjectData([['workflows', '/api/workflows'], ['builder', '/api/workflows/builder'], ['runs', '/api/workflow-runs?limit=60']]);
  const mutation = useMutation(reload);
  const [tab, setTab] = useState<'workflows' | 'runs'>('workflows'); const [selected, setSelected] = useState(0); const [selectedId, setSelectedId] = useState(''); const [query, setQuery] = useState(''); const [action, setAction] = useState<ActionKind | null>(null); const [target, setTarget] = useState<Item>(); const [confirm, setConfirm] = useState(false); const [values, setValues] = useState<Record<string, string>>({}); const [steps, setSteps] = useState<WorkflowStep[]>([{ action: 'run_agent', values: {} }]); const [triggerType, setTriggerType] = useState('manual'); const [draftError, setDraftError] = useState('');
  const allWorkflows = records(data.workflows, 'workflows'); const workflows = allWorkflows.filter(item => workflowLabel(item).toLowerCase().includes(query.toLowerCase())); const runs = records(data.runs, 'runs'); const workflowNames = new Map(allWorkflows.map(item => [text(item, 'id'), workflowLabel(item)])); const visibleRuns = runs.filter(item => `${workflowNames.get(text(item, 'workflow_id')) ?? 'Workflow'} ${status(item)}`.toLowerCase().includes(query.toLowerCase())); const rows = tab === 'workflows' ? workflows : visibleRuns; const current = rows.find(item => text(item, 'id') === selectedId) ?? rows[selected] ?? rows[0];
  const rowLabel = (item: Item) => tab === 'workflows' ? workflowLabel(item) : workflowNames.get(text(item, 'workflow_id')) ?? 'Workflow run';
  const rowSummary = (item: Item) => tab === 'workflows' ? `${status(item) || 'Status unavailable'} · ${workflowTrigger(item).schedule}` : `${status(item) || 'Status unavailable'} · ${workflowTime(item.created_at)}`;
  const builder = record(data.builder); const builderActions = record(builder.actions) as Record<string, WorkflowAction>;
  const triggerEntries = Object.entries(record(builder.triggers)) as Array<[string, unknown]>;
  const triggerMap = record(builder.triggers);
  useEffect(() => { setSelected(0); }, [tab, query]);
  const defaultAction = Object.hasOwn(builderActions, 'run_agent') ? 'run_agent' : Object.keys(builderActions)[0] ?? 'run_agent';
  const open = (kind: ActionKind, item?: Item) => { setTarget(item); setAction(kind); setConfirm(false); setValues({}); if (kind === 'create') { setSteps([{ action: defaultAction, values: {} }]); setTriggerType(triggerEntries.some(([key]) => key === 'manual') ? 'manual' : triggerEntries[0]?.[0] ?? 'manual'); setDraftError(''); } mutation.setError(''); };
  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); if (!action) return;
    if (action === 'create') { const validation = validateWorkflowDraft(values.name ?? '', triggerType, values, steps, builderActions, triggerMap); if (validation) { setDraftError(validation); return; } setDraftError(''); }
    if (['live', 'rename', 'toggle', 'delete'].includes(action) && !confirm) { setConfirm(true); return; }
    const p = project(); const v = values; const selectedWorkflow = target ?? current;
    let ok = false;
    if (action === 'create') {
      const configuredSteps = steps.map(workflowStepPayload);
      const trigger = triggerType === 'scheduled'
        ? { interval_seconds: Math.max(60, Number(v.interval_seconds || 3600)) }
        : triggerType === 'event' ? { event_name: v.event_name?.trim() } : {};
      ok = await mutation.run(() => api.post('/api/workflows', { name: v.name?.trim(), description: v.description?.trim() || '', project: p, trigger_type: triggerType, trigger, steps: configuredSteps }), 'Workflow created and registered with the selected trigger.');
    } else if (action === 'rename' && selectedWorkflow) {
      ok = await mutation.run(() => api.put(`/api/workflows/${encodeURIComponent(text(selectedWorkflow, 'id'))}`, { name: v.name?.trim() || text(selectedWorkflow, 'name'), description: v.description ?? text(selectedWorkflow, 'description'), project: p }), 'Workflow updated.');
    } else if (action === 'dry' && selectedWorkflow) {
      ok = await mutation.run(() => api.post('/api/workflows/run', { workflow_id: text(selectedWorkflow, 'id'), dry_run: true, allow_network: false, project: p }), response => workflowRunMessage(response, true));
    } else if (action === 'live' && selectedWorkflow) {
      ok = await mutation.run(() => api.post('/api/workflows/run', { workflow_id: text(selectedWorkflow, 'id'), dry_run: false, allow_network: false, project: p }), response => workflowRunMessage(response, false));
    } else if (action === 'toggle' && selectedWorkflow) {
      const enabled = !Boolean(selectedWorkflow.enabled);
      ok = await mutation.run(() => api.put(`/api/workflows/${encodeURIComponent(text(selectedWorkflow, 'id'))}`, { enabled, project: p }), `Workflow ${enabled ? 'enabled' : 'disabled'}.`);
    } else if (action === 'delete' && selectedWorkflow) {
      ok = await mutation.run(() => api.delete(`/api/workflows/${encodeURIComponent(text(selectedWorkflow, 'id'))}`), 'Workflow and its stored run history deleted.');
    }
    if (ok) { setAction(null); setConfirm(false); }
  };
  const builderInfo = record(data.builder);
  return <section className="specialist-page oa-ops-page" aria-label="Automation workspace" aria-busy={loading}>
    <header className="specialist-header"><h1>Automations</h1><div className="specialist-header-actions"><button className="specialist-icon-button" type="button" aria-label="Refresh automations" onClick={reload} disabled={loading}><RefreshCw size={15}/></button><button className="oa-automation-new" type="button" aria-label="New workflow" onClick={() => open('create')} disabled={loading || !Object.keys(builderActions).length}><Plus size={14}/>New</button></div></header>
    {error && <div className="specialist-feedback error" role="alert"><CircleAlert size={15}/>{error}<button type="button" onClick={reload}>Retry</button></div>}{mutation.feedback && <div className="specialist-feedback success" role="status">{mutation.feedback}</div>}{mutation.error && !action && <div className="specialist-feedback error" role="alert">{mutation.error}</div>}
    <div className="oa-ops-layout">
      <aside className="oa-ops-rail"><nav className="oa-ops-tabs" aria-label="Automation views"><button type="button" className={tab === 'workflows' ? 'active' : ''} onClick={() => setTab('workflows')}>Workflows<small>{workflows.length}</small></button><button type="button" className={tab === 'runs' ? 'active' : ''} onClick={() => setTab('runs')}>Run history<small>{runs.length}</small></button></nav>
        <label className="oa-ops-search"><Search size={14}/><input aria-label={`Search ${tab}`} placeholder="Filter" value={query} onChange={event => setQuery(event.target.value)}/></label>
        <div className="oa-ops-items" role="listbox" aria-label={tab === 'workflows' ? 'Workflows' : 'Workflow runs'}>{loading ? <div className="oa-ops-muted">Loading…</div> : rows.map((item, index) => <button type="button" role="option" aria-selected={current === item} className={current === item ? 'selected' : ''} key={text(item,'id') || `${rowLabel(item)}-${index}`} onClick={() => { setSelected(index); setSelectedId(text(item, 'id')); }}><span><b>{rowLabel(item)}</b><small>{rowSummary(item)}</small></span><ChevronRight size={14}/></button>)}{!loading && !rows.length && <div className="oa-ops-muted">{query ? 'No matches.' : error ? 'Unable to load this list. Retry the request above.' : tab === 'runs' ? 'No workflow runs recorded.' : 'No workflows in this project.'}</div>}</div>
      </aside>
      <main className="oa-ops-content">{action === 'create' ? <WorkflowDraftBuilder values={values} setValues={setValues} steps={steps} setSteps={setSteps} triggerType={triggerType} setTriggerType={setTriggerType} triggerEntries={triggerEntries} builderActions={builderActions} draftError={draftError} setDraftError={setDraftError} mutationError={mutation.error} busy={mutation.busy} onCancel={() => setAction(null)} onSubmit={event => void submit(event)}/> : <>
        <div className="oa-ops-toolbar"><div><h2>{current ? rowLabel(current) : tab === 'runs' ? 'Run history' : 'Workflows'}</h2></div><div className="oa-ops-toolbar-actions">{tab === 'workflows' && current && <><button type="button" onClick={() => open('dry', current)}><Play size={14}/>Preview run</button><button type="button" onClick={() => open('rename', current)}>Edit</button><button type="button" onClick={() => open('toggle', current)}>{workflowTrigger(current).enabled === false ? 'Enable' : 'Disable'}</button><button type="button" onClick={() => open('live', current)}><Play size={14}/>Live run</button><button type="button" onClick={() => open('delete', current)} aria-label="Delete workflow">Delete</button></>}</div></div>
        {loading ? <div className="specialist-loading" role="status" aria-label="Loading workflows"><i/><i/><i/></div> : !current ? <div className="oa-ops-empty">{error ? 'Workflow data is unavailable.' : tab === 'runs' ? 'Runs will appear here after a workflow starts.' : 'Create a workflow to get started.'}</div> : tab === 'workflows' ? <>
          {(() => { const trigger = workflowTrigger(current); return <div className="oa-ops-status-line"><Workflow size={14}/>{trigger.enabled === false ? 'Disabled' : trigger.enabled === true ? 'Enabled' : 'Status unavailable'}<span>·</span>{trigger.name}: {trigger.schedule}{trigger.nextRun ? <><span>·</span>Next {workflowTime(trigger.nextRun)}</> : null}</div>; })()}
          {text(current, 'description') && <p className="oa-automation-description">{text(current, 'description')}</p>}
          <div className="oa-ops-inline-actions oa-automation-steps"><strong>Steps</strong>{Array.isArray(current.steps) && current.steps.length ? (current.steps as unknown[]).map((step, index) => { const item = record(step); const actionName = text(item, 'action'); return <div className="oa-ops-step" key={`${actionName}-${index}`}><span>{index + 1}</span><b>{builderActions[actionName]?.label || actionName || 'Workflow step'}</b></div>; }) : <span className="oa-ops-muted">No steps configured.</span>}</div>
          <section className="oa-automation-recent"><header><h3>Recent runs</h3><button type="button" onClick={() => setTab('runs')}>View history <ChevronRight size={14}/></button></header>{runs.filter(run => text(run, 'workflow_id') === text(current, 'id')).slice(0, 3).map(run => <button className="oa-automation-run-row" type="button" key={text(run, 'id')} onClick={() => { setSelectedId(text(run, 'id')); setTab('runs'); }}><span>{status(run) || 'Status unavailable'}</span><time>{workflowTime(run.created_at)}</time></button>)}{!runs.some(run => text(run, 'workflow_id') === text(current, 'id')) && <p>No runs recorded for this workflow.</p>}</section>
        </> : <>
          <div className="oa-ops-status-line"><Clock3 size={14}/>{status(current) || 'Status unavailable'}<span>·</span>{workflowTime(current.created_at)}{text(current, 'completed_at') ? <><span>·</span>Finished {workflowTime(current.completed_at)}</> : null}</div>
          <div className="oa-automation-run-summary">{record(current.result).dry_run === true && <span>Dry run</span>}{text(current, 'workflow_id') && <span>{workflowNames.get(text(current, 'workflow_id')) ?? 'Workflow'}</span>}</div>
          <section className="oa-automation-run-steps"><h3>Steps</h3>{records(record(current.result), 'steps').map((step, index) => { const result = runStepResult(step, builderActions); return <article className={`oa-automation-run-step is-${result.state.toLowerCase()}`} key={`${text(step, 'action')}-${index}`}><span className="oa-automation-run-index">{index + 1}</span><div><strong>{result.action}</strong><small>{result.state}</small>{result.detail && <p>{result.detail}</p>}</div></article>; })}{!records(record(current.result), 'steps').length && <p>Step results were not included in this run record.</p>}</section>
        </>}
        <details className="oa-ops-capability"><summary>Builder support</summary><span>{Object.keys(builderActions).length ? `${Object.keys(builderActions).length} available actions · network access is off by default` : 'Builder metadata unavailable.'}</span>{Object.keys(builderInfo.triggers ?? {}).length > 0 && <small>Triggers: {Object.values(record(builderInfo.triggers)).map(item => text(record(item), 'label')).filter(Boolean).join(', ')}</small>}</details>
      </>}</main>
    </div>
    <Dialog open={!!action && action !== 'create'} onOpenChange={openState => { if (!openState && !mutation.busy) { setAction(null); setConfirm(false); } }}><DialogContent className="specialist-dialog oa-ops-dialog"><DialogTitle>{action === 'rename' ? 'Edit workflow' : action === 'dry' ? 'Preview run' : action === 'toggle' ? current?.enabled ? 'Disable workflow?' : 'Enable workflow?' : action === 'delete' ? 'Delete workflow?' : 'Run workflow'}</DialogTitle><DialogDescription>{confirm ? action === 'live' ? 'This executes workflow steps now. Network access remains disabled for this run.' : action === 'delete' ? 'This also deletes the stored run history for this workflow.' : action === 'toggle' ? 'This changes the server enabled state for the selected workflow.' : 'Review this change before saving.' : action === 'dry' ? 'The server will record a dry run without executing workflow actions.' : `Changes apply to ${project()}.`}</DialogDescription>
      <form onSubmit={event => void submit(event)}>
        {!confirm && action === 'rename' && <><label className="specialist-field"><span>Name</span><input required defaultValue={text(target ?? current ?? {}, 'name')} onChange={event => setValues({ ...values, name: event.target.value })}/></label><label className="specialist-field"><span>Purpose</span><textarea rows={2} defaultValue={text(target ?? current ?? {}, 'description')} onChange={event => setValues({ ...values, description: event.target.value })}/></label></>}
        {confirm && <div className="specialist-confirmation"><CircleAlert size={17}/><span>{action === 'live' ? 'Start a live run? Each step executes now; network actions remain blocked by default.' : action === 'delete' ? 'Delete this workflow and its run records?' : action === 'toggle' ? `Set this workflow to ${current?.enabled ? 'disabled' : 'enabled'}?` : 'Save these workflow changes?'}</span></div>}
        {mutation.error && <p className="specialist-feedback error" role="alert">{mutation.error}</p>}<div className="specialist-dialog-actions">{confirm && <button className="specialist-secondary" type="button" onClick={() => setConfirm(false)} disabled={mutation.busy}>Back</button>}<button className="specialist-secondary" type="button" onClick={() => setAction(null)} disabled={mutation.busy}>Cancel</button><button className="specialist-primary" type="submit" disabled={mutation.busy}>{mutation.busy ? 'Working…' : confirm ? action === 'delete' ? 'Delete workflow' : action === 'toggle' ? (current?.enabled ? 'Disable workflow' : 'Enable workflow') : 'Confirm run' : action === 'live' ? 'Review live run' : action === 'dry' ? 'Start preview' : 'Save changes'}</button></div>
      </form>
    </DialogContent></Dialog>
  </section>;
}
