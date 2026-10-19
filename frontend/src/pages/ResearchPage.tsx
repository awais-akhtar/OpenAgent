import { useEffect, useRef, useState } from 'react';
import { api, ApiError, record, records } from '../lib/react/api';
import { getEvents } from '../lib/react/stream';
import { MessageContent } from '../components/chat/MessageContent';
import { PageFrame } from '../components/PageFrame';
import './research-knowledge-voice.css';

type Row = Record<string, unknown>;
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const val = (r: Row, k: string, fallback = '—') => r[k] == null || r[k] === '' ? fallback : String(r[k]);

export function ResearchPage() {
  const [sessions, setSessions] = useState<Row[]>([]);
  const [selected, setSelected] = useState<Row | null>(null);
  const [checkpoints, setCheckpoints] = useState<Row[]>([]);
  const [sourceGraph, setSourceGraph] = useState<Row>({});
  const [report, setReport] = useState<Row>({});
  const [inspectorTab, setInspectorTab] = useState<'Sources' | 'Activity'>('Sources');
  const [query, setQuery] = useState('');
  const [external, setExternal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [following, setFollowing] = useState(false);
  const [followError, setFollowError] = useState('');
  const [includeArchived, setIncludeArchived] = useState(false);
  const [sessionsState, setSessionsState] = useState<'loading' | 'ready' | 'error'>('loading');
  const [selectedId, setSelectedId] = useState('');
  const [cursor, setCursor] = useState(0);
  const [streamRevision, setStreamRevision] = useState(0);
  const openRevision = useRef(0);
  const listRequest = useRef<AbortController | null>(null);
  const path = (id: string, suffix = '') => `/api/research/sessions/${encodeURIComponent(id)}${suffix}`;
  const requestError = (cause: unknown, subject: string) => {
    if (cause instanceof ApiError) {
      if (cause.status === 401) return `Sign in to access ${subject}.`;
      if (cause.status === 403) return `You do not have permission to access ${subject}.`;
      if (cause.status === 404) return `${subject} could not be found. Refresh the session list and try again.`;
      if (cause.status >= 500 || cause.status === 408) return `${subject} is unavailable. Check the local service and retry.`;
    }
    return cause instanceof Error ? cause.message : `${subject} could not be loaded.`;
  };

  async function open(id: string) {
    const revision = ++openRevision.current;
    try {
      const [detailData, checkpointData, reportData, graphData] = await Promise.all([
        api.get<unknown>(path(id)),
        api.get<unknown>(path(id, '/checkpoints')),
        api.get<unknown>(path(id, '/report')),
        api.get<unknown>(path(id, '/source-graph')),
      ]);
      if (revision !== openRevision.current) return;
      const session = record(record(detailData).session);
      setSelected(session);
      setSelectedId(id);
      const loadedCheckpoints = records(checkpointData, 'checkpoints');
      setCheckpoints(loadedCheckpoints);
      setCursor(Math.max(0, ...loadedCheckpoints.map(item => Number(item.id ?? 0))));
      setReport(record(reportData));
      setSourceGraph(record(graphData));
      setInspectorTab('Sources');
      setStreamRevision(value => value + 1);
      setQuery(val(session, 'query', ''));
      setError('');
    } catch (cause) {
      if (revision === openRevision.current) setError(requestError(cause, 'This research session'));
    }
  }

  async function load(selection: string | null | undefined = undefined) {
    listRequest.current?.abort();
    const controller = new AbortController();
    listRequest.current = controller;
    setSessionsState('loading'); setBusy(true); setError('');
    try {
      const p = encodeURIComponent(project());
      const sessionData = await api.get<unknown>(`/api/research/sessions?project=${p}&limit=50&include_archived=${includeArchived}`, controller.signal);
      if (controller.signal.aborted) return;
      const rows = records(sessionData, 'sessions');
      setSessions(rows);
      setSessionsState('ready');
      const selectedId = selection === undefined ? selected?.id ?? rows[0]?.id : selection;
      if (selectedId && rows.some(item => item.id === selectedId)) await open(String(selectedId));
      else if (selectedId) { ++openRevision.current; setSelected(null); setSelectedId(''); setCheckpoints([]); setReport({}); setSourceGraph({}); }
    } catch (cause) {
      if (!controller.signal.aborted) { setSessionsState('error'); setError(requestError(cause, 'Research history')); }
    } finally { if (!controller.signal.aborted) setBusy(false); }
  }

  useEffect(() => { void load(); return () => listRequest.current?.abort(); }, [includeArchived]);

  // The Python endpoint streams durable events from SQLite. Reconnect from the last
  // checkpoint cursor after each bounded stream, so navigation/reload is recoverable.
  useEffect(() => {
    if (!selectedId || !selected || !['queued', 'running'].includes(statusLabel(selected))) return;
    const controller = new AbortController();
    let cursorNow = cursor;
    let mounted = true;
    setFollowing(true); setFollowError('');
    const follow = async () => {
      while (!controller.signal.aborted && mounted) {
        try {
          let terminal = false;
          for await (const frame of getEvents(`${path(selectedId, '/stream')}?after_id=${cursorNow}&timeout=15`, controller.signal)) {
            let event: Row;
            try { event = record(JSON.parse(frame.data)); } catch { continue; }
            if (event.type === 'checkpoint') {
              const item = record(event.checkpoint);
              const next = Number(item.id ?? event.cursor ?? 0);
              if (mounted) setCheckpoints(current => current.some(existing => existing.id === item.id) ? current : [...current, item]);
              if (Number.isFinite(next)) { cursorNow = Math.max(cursorNow, next); setCursor(cursorNow); }
            } else if (event.type === 'session' || event.type === 'complete') {
              const fresh = record(event.session);
              if (mounted && fresh.id) setSelected(fresh);
              if (event.type === 'complete') terminal = true;
            }
          }
          if (terminal) { if (mounted) { await open(selectedId); await load(); } break; }
        } catch (cause) {
          if (controller.signal.aborted || !mounted) break;
          setFollowError(cause instanceof Error ? cause.message : 'Research progress disconnected.');
          break;
        }
      }
    };
    void follow().finally(() => { if (mounted) setFollowing(false); });
    return () => { mounted = false; controller.abort(); };
  // streamRevision intentionally restarts the stream after opening/reloading a session.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId, streamRevision]);

  async function start() {
    if (!query.trim()) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const response = record(await api.post<unknown>('/api/research/sessions', {
        query: query.trim(), project: project(), max_iterations: 3, provider: 'auto', allow_external: external,
      }));
      const session = record(response.session);
      if (!session.id) throw new Error('Research could not start. No session record was returned.');
      setSessions(current => [session, ...current.filter(item => item.id !== session.id)]);
      await open(String(session.id));
      setNotice('Research started.');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Research could not start.');
    } finally { setBusy(false); }
  }

  async function act(action: 'cancel' | 'retry' | 'branch' | 'archive' | 'restore' | 'delete') {
    if (!selected) return;
    if (['delete', 'archive'].includes(action) && !confirm(action === 'delete' ? 'Delete this research session permanently? This cannot be undone.' : 'Archive this research session? It will be hidden from the recent list.')) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const endpointAction = action === 'restore' ? 'archive' : action;
      const payload = action === 'branch' ? { query: String(selected.query ?? ''), label: 'Follow-up branch' } : action === 'restore' ? { archived: false } : {};
      const result = record(await api.post<unknown>(path(String(selected.id), `/${endpointAction}`), payload));
      let nextSessionId: string | undefined;
      if (action === 'branch' || action === 'retry') {
        const next = record(result.session);
        if (next.id) nextSessionId = String(next.id);
      }
      const actionCopy = { cancel: 'cancelled', retry: 'restarted', branch: 'branched', archive: 'archived', restore: 'restored', delete: 'deleted' } as const;
      setNotice(`Research session ${actionCopy[action]}.`);
      if (action === 'delete') { ++openRevision.current; setSelected(null); setSelectedId(''); setCheckpoints([]); setReport({}); setSourceGraph({}); }
      await load(action === 'delete' ? null : nextSessionId);
    } catch (cause) {
      setError(requestError(cause, `Research session (${action})`));
    } finally { setBusy(false); }
  }

  async function saveReportToChat() {
    if (!selected) return;
    const reportText = String(report.report ?? record(report.result).report ?? selected.report ?? '').trim();
    if (!reportText) { setError('This session has no completed report to save.'); return; }
    setBusy(true); setError('');
    try {
      const result = record(await api.post<unknown>('/api/chat/research-record', {
        query: String(selected.query ?? ''), report: reportText, project: project(),
        sources, research_session_id: String(selected.id),
      }));
      const conversationId = String(result.conversation_id ?? record(result.conversation).id ?? '');
      if (!conversationId) throw new Error('Chat service did not return the saved conversation ID.');
      const url = new URL(location.href); url.searchParams.set('view', 'chat'); url.searchParams.set('project', project()); url.searchParams.set('conversation_id', conversationId);
      history.pushState({ view: 'chat', project: project() }, '', url); window.dispatchEvent(new PopStateEvent('popstate'));
    } catch (cause) { setError(cause instanceof Error ? cause.message : 'Research report could not be saved to conversation history.'); }
    finally { setBusy(false); }
  }

  const sources = Array.isArray(report.sources) ? report.sources : Array.isArray(selected?.sources) ? selected.sources : [];
  const statusLabel = (session: Row) => val(session, 'status').toLowerCase();
  const reportText = val(report, 'report', val(record(report.result), 'report', val(selected ?? {}, 'report', ['queued', 'running'].includes(statusLabel(selected ?? {})) ? 'Research is in progress. Waiting for the service to save a report.' : 'No report returned.')));

  return <PageFrame title="Research">
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <div className={`workspace research-workspace ${selected ? 'has-session' : sessions.length ? 'new-session' : 'no-history'}`}>
      <section className="panel research-main"><div className="panel-head"><div className="research-heading-copy"><h2>{selected ? val(selected, 'title', val(selected, 'query', 'Research report')) : 'New research'}</h2>{selected ? <span className={`research-status research-status-${statusLabel(selected)}`}>{statusLabel(selected).replaceAll('_', ' ')}</span> : null}</div><div className="research-heading-actions"><button className="btn" onClick={() => void load()} disabled={busy}>Refresh</button>{sessions.length > 0 && <label className="research-session-select"><span className="sr-only">Recent research</span><select aria-label="Recent research" value={selected?.id == null ? '' : String(selected.id)} onChange={event => void open(event.target.value)}><option value="" disabled>Recent research</option>{sessions.map(session => <option key={String(session.id)} value={String(session.id)}>{val(session, 'query', 'Research session')}</option>)}</select></label>}{selected && String(report.report ?? record(report.result).report ?? selected.report ?? '').trim() && <button className="btn" onClick={() => void saveReportToChat()} disabled={busy}>Save to Chat</button>}<button className="btn" aria-pressed={includeArchived} onClick={() => { ++openRevision.current; setSelected(null); setSelectedId(''); setCheckpoints([]); setReport({}); setSourceGraph({}); setIncludeArchived(value => !value); }}>{includeArchived ? 'Hide archived' : 'Archived'}</button></div></div><div className="panel-body stack">
        {!selected ? <>{sessionsState === 'loading' && <p className="research-history-state" role="status">Loading recent research…</p>}<label className="research-prompt-label">Research question<textarea className="field research-prompt" rows={5} value={query} onChange={event => setQuery(event.target.value)} placeholder="What do you need to find out?" /></label><div className="research-source-choice"><label className="research-consent"><input type="checkbox" checked={external} onChange={event => setExternal(event.target.checked)} /> Include external sources</label><small>When enabled, research may use sources outside this workspace.</small></div><button className="btn primary" onClick={() => void start()} disabled={busy || !query.trim()}>Start research</button></> : <>
          <p className="research-question">{val(selected, 'query')}</p>{Boolean(selected.error) && <p className="notice error">{String(selected.error)}</p>}
          {following && <span className="research-live-label" aria-live="polite">Live updates</span>}
          {followError && <button className="btn" onClick={() => { setFollowError(''); setStreamRevision(value => value + 1); }}>Reconnect to progress</button>}
          {followError && <p className="notice error" role="alert">{followError}</p>}
          <article className="content research-report" aria-live="polite"><MessageContent content={reportText} /></article>
          <details className="research-advanced research-actions"><summary>More actions</summary><div className="research-advanced-content"><p>Archiving hides this session from the recent list. Deleting removes it permanently.</p><div className="toolbar">{(['retry', 'branch', ...(selected.archived ? ['restore' as const] : ['archive' as const]), 'delete', 'cancel'] as const).map(action => { const labels = { retry: 'Retry research', branch: 'Start follow-up', restore: 'Restore', archive: 'Archive', delete: 'Delete', cancel: 'Cancel research' }; return <button className={`btn ${['delete', 'cancel'].includes(action) ? 'danger' : ''}`} disabled={busy || (action === 'cancel' && !['queued', 'running'].includes(statusLabel(selected))) || (action === 'retry' && !['failed', 'cancelled'].includes(statusLabel(selected)))} key={action} onClick={() => void act(action)}>{labels[action]}</button>; })}</div></div></details>
        </>}
      </div></section>
      {selected && <aside className="panel research-sources"><div className="research-inspector-tabs" role="tablist" aria-label="Research details">{(['Sources', 'Activity'] as const).map(name => <button type="button" role="tab" aria-selected={inspectorTab === name} key={name} onClick={() => setInspectorTab(name)}>{name}{name === 'Sources' && sources.length > 0 ? ` · ${sources.length}` : name === 'Activity' && checkpoints.length > 0 ? ` · ${checkpoints.length}` : ''}</button>)}</div><div className="panel-body stack">
        {inspectorTab === 'Sources' ? <>{sources.length ? sources.map((source, index) => { const item = record(source); const safeUrl = typeof item.url === 'string' && /^https?:\/\//i.test(item.url) ? item.url : ''; return <article className="research-source" key={String(item.id ?? item.url ?? index)}><strong>{typeof source === 'string' ? source : val(item, 'title', val(item, 'name', val(item, 'url', 'Source')))}</strong>{safeUrl && <a href={safeUrl} target="_blank" rel="noopener noreferrer">Open source ↗</a>}{(item.type ?? item.quality) != null && <small>{val(item, 'type', val(item, 'quality'))}</small>}{item.snippet != null && <p>{String(item.snippet)}</p>}</article>; }) : <p className="empty">No sources returned yet.</p>}<details className="research-advanced"><summary>Provenance details</summary><ul>{records(sourceGraph, 'edges').map((edge, index) => <li key={String(edge.to ?? index)}>{val(edge, 'relation')} → {val(edge, 'to')}</li>)}</ul></details></> : <div className="research-activity-list">{checkpoints.map((checkpoint, index) => <article key={String(checkpoint.id ?? index)}><span aria-hidden="true"/><div><strong>{val(checkpoint, 'stage', 'Progress update')}</strong><small>{val(checkpoint, 'progress', val(checkpoint, 'status'))}</small>{val(checkpoint, 'message', '') && <p>{val(checkpoint, 'message')}</p>}</div></article>)}{!checkpoints.length && <p className="empty">No saved activity has been returned yet.</p>}</div>}
      </div></aside>}
    </div>
  </PageFrame>;
}
