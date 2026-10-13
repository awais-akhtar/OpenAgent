import { useEffect, useState } from 'react';
import { api, record, records } from '../lib/react/api';
import { getEvents } from '../lib/react/stream';
import { PageFrame } from '../components/PageFrame';

type Row = Record<string, unknown>;
const project = () => new URL(location.href).searchParams.get('project') || localStorage.getItem('oa-project') || 'default';
const val = (r: Row, k: string, fallback = '—') => r[k] == null || r[k] === '' ? fallback : String(r[k]);

export function ResearchPage() {
  const [sessions, setSessions] = useState<Row[]>([]);
  const [selected, setSelected] = useState<Row | null>(null);
  const [checkpoints, setCheckpoints] = useState<Row[]>([]);
  const [sourceGraph, setSourceGraph] = useState<Row>({});
  const [report, setReport] = useState<Row>({});
  const [status, setStatus] = useState<Row>({});
  const [query, setQuery] = useState('');
  const [external, setExternal] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [following, setFollowing] = useState(false);
  const [followError, setFollowError] = useState('');
  const [includeArchived, setIncludeArchived] = useState(false);
  const [selectedId, setSelectedId] = useState('');
  const [cursor, setCursor] = useState(0);
  const [streamRevision, setStreamRevision] = useState(0);
  const path = (id: string, suffix = '') => `/api/research/sessions/${encodeURIComponent(id)}${suffix}`;

  async function open(id: string) {
    try {
      const [detailData, checkpointData, reportData, graphData] = await Promise.all([
        api.get<unknown>(path(id)),
        api.get<unknown>(path(id, '/checkpoints')),
        api.get<unknown>(path(id, '/report')),
        api.get<unknown>(path(id, '/source-graph')),
      ]);
      const session = record(record(detailData).session);
      setSelected(session);
      setSelectedId(id);
      const loadedCheckpoints = records(checkpointData, 'checkpoints');
      setCheckpoints(loadedCheckpoints);
      setCursor(Math.max(0, ...loadedCheckpoints.map(item => Number(item.id ?? 0))));
      setReport(record(reportData));
      setSourceGraph(record(graphData));
      setStreamRevision(value => value + 1);
      setQuery(val(session, 'query', ''));
      setError('');
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Session details or checkpoints could not be opened.');
    }
  }

  async function load(selection: string | null | undefined = undefined) {
    setBusy(true);
    setError('');
    try {
      const p = encodeURIComponent(project());
      const [statusData, sessionData] = await Promise.all([
        api.get<unknown>(`/api/research/status?project=${p}`),
        api.get<unknown>(`/api/research/sessions?project=${p}&limit=50&include_archived=${includeArchived}`),
      ]);
      setStatus(record(statusData));
      const rows = records(sessionData, 'sessions');
      setSessions(rows);
      const selectedId = selection === undefined ? selected?.id : selection;
      if (selectedId && rows.some(item => item.id === selectedId)) await open(String(selectedId));
      else if (selectedId) { setSelected(null); setCheckpoints([]); setReport({}); }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Research service unavailable.');
    } finally { setBusy(false); }
  }

  useEffect(() => { void load(); }, [includeArchived]);

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
      if (!session.id) throw new Error('Research service did not return a persisted session.');
      setSessions(current => [session, ...current.filter(item => item.id !== session.id)]);
      await open(String(session.id));
      setNotice(`Research session ${String(session.id)} was queued. Progress is read from its saved checkpoints.`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : 'Research could not start.');
    } finally { setBusy(false); }
  }

  async function act(action: 'cancel' | 'retry' | 'branch' | 'archive' | 'restore' | 'delete') {
    if (!selected) return;
    if (['delete', 'archive'].includes(action) && !confirm(`${action === 'delete' ? 'Delete' : 'Archive'} this research session?`)) return;
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
      setNotice(action === 'delete' ? 'Research session deleted.' : `Session ${action} confirmed by the service.`);
      if (action === 'delete') { setSelected(null); setSelectedId(''); setCheckpoints([]); setReport({}); setSourceGraph({}); }
      await load(action === 'delete' ? null : nextSessionId);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : `Could not ${action} session.`);
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

  return <PageFrame eyebrow="Evidence workspace" title="Research" description="Run and revisit auditable investigations with server-owned checkpoints, sources and reports." actions={<button className="btn" onClick={() => void load()} disabled={busy}>Refresh</button>}>
    {error && <p className="notice error" role="alert">{error}</p>}{notice && <p className="notice success" role="status">{notice}</p>}
    <div className="workspace">
      <section className="panel"><div className="panel-head"><h2>Sessions</h2><button className="btn" aria-pressed={includeArchived} onClick={() => { setSelected(null); setSelectedId(''); setIncludeArchived(value => !value); }}>{includeArchived ? 'Hide archived' : 'Include archived'}</button></div><div className="panel-body list">
        {sessions.map(session => <button key={String(session.id)} aria-current={selected?.id === session.id} onClick={() => void open(String(session.id))}><strong>{val(session, 'query', 'Research session')}</strong><small>{val(session, 'status')} · {val(session, 'created_at')}</small></button>)}
        {!sessions.length && <p className="empty">{busy ? 'Loading sessions…' : 'No research sessions for this project.'}</p>}
      </div></section>
      <section className="panel"><div className="panel-head"><h2>{selected ? 'Session report' : 'Start an investigation'}</h2>{selected && <span className="tag">{val(selected, 'status')}</span>}</div><div className="panel-body stack">
        {!selected ? <><label>Research question<textarea className="field" rows={4} value={query} onChange={event => setQuery(event.target.value)} /></label><label className="row"><input type="checkbox" checked={external} onChange={event => setExternal(event.target.checked)} /> Allow external research sources for this run</label><small className="tiny">External access is off unless explicitly enabled. Backend privacy policy still applies.</small><button className="btn primary" onClick={() => void start()} disabled={busy || !query.trim()}>Start research</button></> : <>
          <p>{val(selected, 'query')}</p>{Boolean(selected.error) && <p className="notice error">{String(selected.error)}</p>}
          <div className="row"><span className="tag">{following ? 'Receiving saved progress' : statusLabel(selected)}</span>{followError && <button className="btn" onClick={() => { setFollowError(''); setStreamRevision(value => value + 1); }}>Reconnect to progress</button>}</div>
          {followError && <p className="notice error" role="alert">{followError}</p>}
          <article className="content">{val(report, 'report', val(record(report.result), 'report', val(selected, 'report', ['queued', 'running'].includes(statusLabel(selected)) ? 'Research is in progress. Waiting for the service to save a report.' : 'No report returned.')))}</article>
          {String(report.report ?? record(report.result).report ?? selected.report ?? '').trim() && <button className="btn" onClick={() => void saveReportToChat()} disabled={busy}>Save report to Chat history</button>}
          <h3>Checkpoints</h3>{checkpoints.map((checkpoint, index) => <article className="record-row" key={String(checkpoint.id ?? index)}><strong>{val(checkpoint, 'stage', 'Checkpoint')}</strong><span>{val(checkpoint, 'progress', val(checkpoint, 'status'))}</span><p>{val(checkpoint, 'message')}</p></article>)}
          <div className="toolbar">{(['retry', 'branch', ...(selected.archived ? ['restore' as const] : ['archive' as const]), 'delete', 'cancel'] as const).map(action => <button className={`btn ${['delete', 'cancel'].includes(action) ? 'danger' : ''}`} disabled={busy || (action === 'cancel' && !['queued', 'running'].includes(statusLabel(selected))) || (action === 'retry' && !['failed', 'cancelled'].includes(statusLabel(selected)))} key={action} onClick={() => void act(action)}>{action}</button>)}</div>
        </>}
      </div></section>
      <aside className="panel"><div className="panel-head"><h2>Sources & provenance</h2></div><div className="panel-body stack"><div>Service status · {val(status, 'status', val(status, 'available', 'Not reported'))}</div>
        {selected ? <>{sources.map((source, index) => { const item = record(source); const safeUrl = typeof item.url === 'string' && /^https?:\/\//i.test(item.url) ? item.url : ''; return <article className="panel panel-body" key={String(item.id ?? item.url ?? index)}><strong>{typeof source === 'string' ? source : val(item, 'title', val(item, 'name', val(item, 'url', 'Source')))}</strong>{safeUrl && <a href={safeUrl} target="_blank" rel="noopener noreferrer">Open source</a>}<small>{val(item, 'type', val(item, 'quality', 'Source record supplied by backend'))}</small>{item.snippet != null && <p>{String(item.snippet)}</p>}</article>; })}<details><summary>Source graph · {records(sourceGraph, 'nodes').length} nodes</summary><ul>{records(sourceGraph, 'edges').map((edge, index) => <li key={String(edge.to ?? index)}>{val(edge, 'relation')} → {val(edge, 'to')}</li>)}</ul></details></> : <p className="empty">Select a session to inspect source provenance.</p>}
        <small className="tiny">Source presence does not mean independent verification.</small>
      </div></aside>
    </div>
  </PageFrame>;
}
