import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, type JsonRecord } from '../api/client';

// Phase 6: a Work Graph view of the persisted CodeQ graph, run tasks and evidence.
// It reads only backend records. Demo mode shows a notice instead of sample events.

export type GraphState = 'neutral' | 'active' | 'verified' | 'attention' | 'failed';

export const STATE_LABEL: Record<GraphState, string> = {
  neutral: 'Queued / unstarted',
  active: 'Active',
  verified: 'Verified',
  attention: 'Blocked / approval',
  failed: 'Failed / stale',
};

export const STATE_COLOR: Record<GraphState, string> = {
  neutral: '#8a8f98',
  active: '#2f6fdb',
  verified: '#1f9d55',
  attention: '#d98e04',
  failed: '#d1343c',
};

/** Map a backend status string to the panel's state colour. Colour shows state, not agent role. */
export function taskState(status: unknown): GraphState {
  const value = String(status ?? '');
  if (value === 'completed' || value === 'verified' || value === 'passed') return 'verified';
  if (value === 'running' || value === 'active') return 'active';
  if (value === 'blocked' || value === 'approval' || value === 'waiting_approval' || value === 'deferred') return 'attention';
  if (value === 'failed' || value === 'stale' || value === 'needs_fix' || value === 'cancelled') return 'failed';
  return 'neutral';
}

/** Map an executed check's status to a state. A stale check is never shown as verified. */
export function checkState(status: unknown): GraphState {
  const value = String(status ?? '');
  if (value === 'passed') return 'verified';
  if (value === 'stale') return 'failed';
  if (value === 'running') return 'active';
  if (value === 'failed' || value === 'blocked') return 'failed';
  return 'neutral';
}

export const LEGEND: GraphState[] = ['neutral', 'active', 'verified', 'attention', 'failed'];

export type GraphView = 'Project' | 'Code' | 'Tasks' | 'Evidence';
export const VIEWS: GraphView[] = ['Project', 'Code', 'Tasks', 'Evidence'];

/** Code nodes whose path matches the filter, in stable order, bounded for display. */
export function filterCodeNodes(nodes: JsonRecord[], query: string, limit = 200): { rows: JsonRecord[]; total: number } {
  const needle = query.trim().toLowerCase();
  const matches = nodes
    .filter(node => node.kind === 'file' && (!needle || [node.path, node.label, node.language].some(value => String(value ?? '').toLowerCase().includes(needle))))
    .sort((a, b) => String(a.path ?? '').localeCompare(String(b.path ?? '')));
  return { rows: matches.slice(0, limit), total: matches.length };
}

/** Tasks whose recorded code paths include this path. */
export function tasksForPath(tasks: JsonRecord[], path: string): JsonRecord[] {
  return tasks.filter(task => {
    const paths = [
      ...(Array.isArray(task.code_paths) ? task.code_paths : []),
      ...(Array.isArray(task.declared_files) ? task.declared_files : []),
    ];
    return paths.map(String).includes(path);
  });
}

/** Executed checks linked to any of these tasks, so evidence is shown only when the task is related. */
export function evidenceForTasks(checks: JsonRecord[], taskIds: string[]): JsonRecord[] {
  const wanted = new Set(taskIds);
  return checks.filter(check => Array.isArray(check.task_ids) && (check.task_ids as unknown[]).some(id => wanted.has(String(id))));
}

/** Acceptance-criterion ids carried by the linked tasks. */
export function criteriaForTasks(tasks: JsonRecord[]): string[] {
  const ids = new Set<string>();
  for (const task of tasks) {
    if (Array.isArray(task.criterion_ids)) for (const id of task.criterion_ids as unknown[]) ids.add(String(id));
  }
  return [...ids].sort();
}

/** Freshness of a code node: does the node's recorded hash still match the current file? */
export function freshnessOf(node: JsonRecord, currentSha256: string | undefined): 'fresh' | 'stale' | 'unknown' {
  const recorded = String(node.content_sha256 ?? '');
  if (!recorded || !currentSha256) return 'unknown';
  return recorded === currentSha256 ? 'fresh' : 'stale';
}

/** Counts of graph node kinds and edge relations for the Project view. */
export function graphSummary(graph: JsonRecord): { kinds: Record<string, number>; relations: Record<string, number> } {
  const kinds: Record<string, number> = {};
  const relations: Record<string, number> = {};
  for (const node of (graph.nodes as JsonRecord[] | undefined) ?? []) {
    const kind = String(node.kind ?? 'other');
    kinds[kind] = (kinds[kind] ?? 0) + 1;
  }
  for (const edge of (graph.edges as JsonRecord[] | undefined) ?? []) {
    const relation = String(edge.relation ?? 'other');
    relations[relation] = (relations[relation] ?? 0) + 1;
  }
  return { kinds, relations };
}

interface Props {
  projectId: string;
  runId: string;
  live: boolean;
  events?: JsonRecord[];
  streamError?: string;
}

export function CodeQWorkGraphPanel({ projectId, runId, live, events = [], streamError = '' }: Props) {
  const queryClient = useQueryClient();
  const [view, setView] = useState<GraphView>('Project');
  const [query, setQuery] = useState('');
  const [selectedId, setSelectedId] = useState('');
  const [selectedRunId, setSelectedRunId] = useState('');
  const [limit, setLimit] = useState(200);
  const [collapsed, setCollapsed] = useState(false);
  const filterRef = useRef<HTMLInputElement | null>(null);
  const selectedOptionRef = useRef<HTMLButtonElement | null>(null);
  const lastEventKey = useRef('');

  const graphQuery = useQuery({
    queryKey: ['codeq', 'work-graph', projectId],
    queryFn: ({ signal }) => apiGet<JsonRecord>(`/api/codebase/projects/${encodeURIComponent(projectId)}/graph?limit=2000`, signal),
    enabled: live && !!projectId,
    retry: false,
  });
  // Restore task/evidence context after a reload from the backend's
  // owner-scoped run history, constrained to the currently selected project.
  const runsQuery = useQuery({
    queryKey: ['codeq', 'work-graph-runs', projectId],
    queryFn: ({ signal }) => apiGet<JsonRecord>('/api/codebase/runs?project=default&limit=200', signal),
    enabled: live && !!projectId,
    retry: false,
  });
  const projectRuns = useMemo(() => ((runsQuery.data?.runs as JsonRecord[] | undefined) ?? [])
    .filter(item => String(item.codebase_project_id ?? '') === projectId), [projectId, runsQuery.data]);
  const displayRunId = selectedRunId || runId || String(projectRuns[0]?.id ?? '');
  useEffect(() => {
    setSelectedRunId('');
    setSelectedId('');
  }, [projectId, runId]);
  const runQuery = useQuery({
    queryKey: ['codeq', 'work-graph-run', displayRunId],
    queryFn: ({ signal }) => apiGet<JsonRecord>(`/api/codebase/runs/${encodeURIComponent(displayRunId)}`, signal),
    enabled: live && !!displayRunId,
    retry: false,
  });
  const eventHistoryQuery = useQuery({
    queryKey: ['codeq', 'work-graph-events', displayRunId],
    queryFn: ({ signal }) => apiGet<JsonRecord>(`/api/codebase/runs/${encodeURIComponent(displayRunId)}/events?after=0`, signal),
    enabled: live && !!displayRunId,
    retry: false,
  });
  // Goals outlive runs. Load the owner/project-scoped durable goal list so
  // the Project view remains useful after reload or before a run is selected.
  const goalQuery = useQuery({
    queryKey: ['codeq', 'work-graph-goals', projectId],
    queryFn: ({ signal }) => apiGet<JsonRecord>(`/api/codebase/goals?codebase_project_id=${encodeURIComponent(projectId)}&limit=100`, signal),
    enabled: live && !!projectId,
    retry: false,
  });

  const graph = (graphQuery.data ?? {}) as JsonRecord;
  const run = ((runQuery.data ?? {}) as JsonRecord).run as JsonRecord | undefined;
  const tasks = useMemo(() => ((run?.tasks as JsonRecord[] | undefined) ?? []), [run]);
  // Completion-audit checks and checks run inside tasks are both executed evidence.
  const checks = useMemo(() => [
    ...((run?.validation as JsonRecord[] | undefined) ?? []),
    ...((run?.task_checks as JsonRecord[] | undefined) ?? []),
  ], [run]);
  const nodes = useMemo(() => ((graph.nodes as JsonRecord[] | undefined) ?? []), [graph]);
  const code = useMemo(() => filterCodeNodes(nodes, query, limit), [nodes, query, limit]);
  const selected = nodes.find(node => String(node.id) === selectedId);
  const selectedPath = String(selected?.path ?? '');
  const linkedTasks = selected && selectedPath ? tasksForPath(tasks, selectedPath) : [];
  const linkedChecks = evidenceForTasks(checks, linkedTasks.map(task => String(task.id)));
  const persistedGoals = ((goalQuery.data?.goals as JsonRecord[] | undefined) ?? []);
  const runGoal = (run?.goal as JsonRecord | undefined) ?? undefined;
  const goal = runGoal ?? (persistedGoals.find(item => String(item.id ?? '') === String(run?.goal_id ?? '')) ?? persistedGoals[0]);
  const goalCriteria = (goal?.criteria as JsonRecord[] | undefined) ?? [];
  const milestones = (goal?.milestones as JsonRecord[] | undefined) ?? [];
  const graphEdges = useMemo(() => ((graph.edges as JsonRecord[] | undefined) ?? []), [graph]);
  const selectedEdges = selected ? graphEdges.filter(edge => String(edge.source) === String(selected.id) || String(edge.target) === String(selected.id)) : [];

  // Read the current file when a file node is opened, so provenance is compared with live source.
  const sourceQuery = useQuery({
    queryKey: ['codeq', 'work-graph-source', projectId, selectedPath],
    queryFn: ({ signal }) => apiGet<JsonRecord>(`/api/codebase/projects/${encodeURIComponent(projectId)}/file?path=${encodeURIComponent(selectedPath)}`, signal),
    enabled: live && !!selected && !!selectedPath,
    retry: false,
  });
  const currentFile = (sourceQuery.data?.file ?? undefined) as JsonRecord | undefined;
  const currentSha = typeof currentFile?.content_sha256 === 'string'
    ? currentFile.content_sha256
    : typeof currentFile?.sha256 === 'string' ? currentFile.sha256 : undefined;
  const currentSourceLines = String(currentFile?.content ?? '').split(/\r?\n/);
  const selectedLine = Math.max(0, Number(selected?.line ?? 0) || 0);
  const excerptStart = selectedLine ? Math.max(0, selectedLine - 4) : 0;
  const excerptEnd = selectedLine ? Math.min(currentSourceLines.length, selectedLine + 10) : Math.min(currentSourceLines.length, 120);
  const currentExcerpt = currentSourceLines.slice(excerptStart, excerptEnd).map((line, index) => `${excerptStart + index + 1}: ${line}`).join('\n').slice(0, 4000);

  // WorkspaceScreens owns the authenticated SSE connection. Consume those
  // genuine events here to refresh persisted graph/run records at the event
  // boundary; the panel never creates simulated events or another stream.
  const liveEvents = displayRunId === runId ? events : [];
  const persistedEvents = ((eventHistoryQuery.data?.events as JsonRecord[] | undefined) ?? []);
  const displayedEvents = useMemo(() => {
    const unique = new Map<string, JsonRecord>();
    for (const event of [...persistedEvents, ...liveEvents]) {
      const key = String(event.sequence ?? event.id ?? `${event.type ?? 'event'}:${event.created_at ?? ''}`);
      unique.set(key, event);
    }
    return [...unique.values()].sort((a, b) => Number(a.sequence ?? 0) - Number(b.sequence ?? 0));
  }, [liveEvents, persistedEvents]);
  const latestEvent = displayedEvents.at(-1);
  const latestEventKey = latestEvent ? `${displayRunId}:${String(latestEvent.sequence ?? latestEvent.id ?? latestEvent.type ?? '')}` : '';
  useEffect(() => { lastEventKey.current = ''; }, [displayRunId]);
  useEffect(() => {
    if (!live || !displayRunId || !latestEvent || !latestEventKey || lastEventKey.current === latestEventKey) return;
    lastEventKey.current = latestEventKey;
    if (String(latestEvent.type ?? '') === 'stream.heartbeat') return;
    void queryClient.invalidateQueries({ queryKey: ['codeq', 'work-graph-run', displayRunId] });
    void queryClient.invalidateQueries({ queryKey: ['codeq', 'work-graph-events', displayRunId] });
    const eventType = String(latestEvent.type ?? '');
    if (eventType.startsWith('run.')) {
      // The history selector is a separate persisted query. Refresh it on
      // lifecycle events so a run started while this panel is open appears
      // immediately and its terminal state does not remain stale.
      void queryClient.invalidateQueries({ queryKey: ['codeq', 'work-graph-runs', projectId] });
    }
    if (eventType === 'context.graph.retrieved' || eventType.startsWith('run.')) {
      void queryClient.invalidateQueries({ queryKey: ['codeq', 'work-graph', projectId] });
    }
  }, [latestEvent, latestEventKey, live, projectId, queryClient, displayRunId]);

  const onTabKey = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = VIEWS.indexOf(view);
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      const offset = event.key === 'ArrowRight' ? 1 : -1;
      const next = VIEWS[(index + offset + VIEWS.length) % VIEWS.length];
      const tablist = event.currentTarget;
      setView(next);
      window.requestAnimationFrame(() => tablist.querySelector<HTMLButtonElement>(`[data-view="${next}"]`)?.focus());
    }
  };

  const closeDetails = () => {
    setSelectedId('');
    window.requestAnimationFrame(() => selectedOptionRef.current?.focus());
  };

  const onDetailsKey = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Escape') { event.preventDefault(); closeDetails(); }
  };

  const onFileListKey = (event: KeyboardEvent<HTMLUListElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      setSelectedId('');
      filterRef.current?.focus();
      return;
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return;
    const options = [...event.currentTarget.querySelectorAll<HTMLButtonElement>('[role="option"]')];
    if (!options.length) return;
    event.preventDefault();
    const current = options.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? options.length - 1 :
      (current + (event.key === 'ArrowDown' ? 1 : -1) + options.length) % options.length;
    options[next]?.focus();
  };

  if (!live) {
    return <div className="codeq-work-graph" role="region" aria-label="Work Graph"><p className="note-panel">The Work Graph shows live CodeQ records. Demo mode shows no graph data.</p></div>;
  }
  if (!projectId) {
    return <div className="codeq-work-graph" role="region" aria-label="Work Graph"><p className="note-panel">Select a CodeQ project to see its work graph.</p></div>;
  }

  const graphError = graphQuery.error;
  const runError = runQuery.error;
  const updated = graphQuery.dataUpdatedAt ? new Date(graphQuery.dataUpdatedAt).toLocaleTimeString() : '';
  const runUpdated = runQuery.dataUpdatedAt ? new Date(runQuery.dataUpdatedAt).toLocaleTimeString() : '';

  return (
    <div className="codeq-work-graph" role="region" aria-label="Work Graph">
      <div className="codeq-work-graph-head" role="tablist" aria-label="Work Graph views" onKeyDown={onTabKey}>
        {VIEWS.map(name => (
          <button key={name} data-view={name} type="button" role="tab" aria-selected={view === name} tabIndex={view === name ? 0 : -1} className={`tab ${view === name ? 'active' : ''}`} onClick={() => setView(name)}>{name}</button>
        ))}
        <span className="caption" data-testid="freshness">
          {graphError || runError || runsQuery.error || eventHistoryQuery.error ? 'Connection problem — showing last known data' : streamError && displayRunId === runId ? 'Event stream reconnecting — showing persisted state' : `${displayRunId === runId && displayRunId ? `SSE ${liveEvents.length} events` : 'Persisted run history'} · graph ${updated || 'loading'} · run ${runUpdated || 'loading'}`}
        </span>
      </div>

      {projectRuns.length > 0 && <label className="caption codeq-work-graph-run-select">Run history <select aria-label="Work Graph run history" value={displayRunId} onChange={event => setSelectedRunId(event.target.value)}>
        {projectRuns.map(item => <option key={String(item.id)} value={String(item.id)}>{String(item.user_prompt ?? item.summary ?? item.id)} · {String(item.status ?? '')}</option>)}
      </select></label>}

      <ul className="codeq-work-graph-legend" aria-label="Status legend">
        {LEGEND.map(state => (
          <li key={state}><span aria-hidden="true" style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: STATE_COLOR[state], marginRight: 6 }} />{STATE_LABEL[state]}</li>
        ))}
      </ul>

      {graphError && <p className="note-panel" role="alert">The graph could not load from the backend: {String((graphError as Error).message ?? graphError)}</p>}
      {runError && <p className="note-panel" role="alert">The run could not load from the backend: {String((runError as Error).message ?? runError)}</p>}
      {runsQuery.error && <p className="note-panel" role="alert">Run history could not load from the backend: {String((runsQuery.error as Error).message ?? runsQuery.error)}</p>}
      {eventHistoryQuery.error && <p className="note-panel" role="alert">Persisted run events could not load from the backend: {String((eventHistoryQuery.error as Error).message ?? eventHistoryQuery.error)}</p>}
      {!displayRunId && !runsQuery.isLoading && <p className="caption" role="status">No persisted CodeQ runs are available for this project yet.</p>}
      {!graphError && graphQuery.isSuccess && graph.revision == null && <p className="note-panel" role="status">No graph has been recorded for this project yet. Run a CodeQ task to build one.</p>}

      {view === 'Project' && (() => {
        const summary = graphSummary(graph);
        return (
          <section aria-label="Project view">
            <p>{String(graph.node_count ?? nodes.length)} nodes, {String(graph.edge_count ?? 0)} relations{graph.truncated ? ' (display truncated)' : ''}.</p>
            {goal ? (
              <div aria-label="Project goal">
                <h4>{String(goal.goal ?? 'Project goal')}</h4>
                <p>Status: {String(goal.status ?? 'unknown')} · revision {String(goal.revision ?? '—')}</p>
                {milestones.length > 0 && <ul aria-label="Goal milestones">{milestones.map((milestone, index) => <li key={String(milestone.id ?? index)}>{String(milestone.title ?? 'Milestone')} — {String(milestone.status ?? 'open')}</li>)}</ul>}
                {goalCriteria.length > 0 && <ul aria-label="Acceptance criteria">{goalCriteria.map((criterion, index) => <li key={String(criterion.id ?? index)}><span aria-hidden="true" style={{ color: STATE_COLOR[taskState(criterion.status)] }}>●</span> {String(criterion.criterion ?? criterion.title ?? criterion.id)} — {String(criterion.status ?? 'open')}</li>)}</ul>}
              </div>
            ) : goalQuery.isLoading ? <p className="caption">Loading saved project goals…</p> : goalQuery.error ? <p className="caption" role="status">Saved project goals are unavailable.</p> : <p className="caption">No saved goal is linked to this project yet.</p>}
            <h4>Node kinds</h4>
            <ul>{Object.entries(summary.kinds).map(([kind, count]) => <li key={kind}>{kind}: {count}</li>)}</ul>
            <h4>Relations</h4>
            <ul>{Object.entries(summary.relations).map(([relation, count]) => <li key={relation}>{relation}: {count}</li>)}</ul>
          </section>
        );
      })()}

      {view === 'Code' && (
        <section aria-label="Code view">
          <div className="codeq-work-graph-tools">
            <label>Filter files <input ref={filterRef} value={query} onChange={event => setQuery(event.target.value)} placeholder="path, filename, or language…" /></label>
            <button type="button" aria-expanded={!collapsed} onClick={() => setCollapsed(value => !value)}>{collapsed ? 'Expand list' : 'Collapse list'}</button>
          </div>
          {!collapsed && (
            <ul role="listbox" aria-label="Code files" className="codeq-work-graph-rows" onKeyDown={onFileListKey}>
              {code.rows.map(node => {
                const linked = tasksForPath(tasks, String(node.path ?? ''));
                const state = linked.length ? taskState(linked[linked.length - 1].status) : 'neutral';
                return (
                  <li key={String(node.id)}>
                    <button type="button" role="option" aria-selected={selectedId === String(node.id)} onClick={event => { selectedOptionRef.current = event.currentTarget; setSelectedId(String(node.id)); }}>
                      <span aria-hidden="true" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: STATE_COLOR[state], marginRight: 6 }} />
                      {String(node.path)}
                    </button>
                  </li>
                );
              })}
            </ul>
          )}
          <p className="caption">Showing {code.rows.length} of {code.total} files.</p>
          {code.total > code.rows.length && <button type="button" onClick={() => setLimit(value => value + 200)}>Show more</button>}
        </section>
      )}

      {view === 'Tasks' && (
        <section aria-label="Tasks view">
          {tasks.length === 0 && <p className="caption">No tasks yet for this run.</p>}
          <ul>{tasks.map(task => (
            <li key={String(task.id)}><span aria-hidden="true" style={{ color: STATE_COLOR[taskState(task.status)] }}>●</span> {String(task.title)} — {String(task.status)}</li>
          ))}</ul>
        </section>
      )}

      {view === 'Evidence' && (
        <section aria-label="Evidence view">
          {checks.length === 0 && <p className="caption">No executed checks recorded for this run.</p>}
          <ul>{checks.map((check, index) => (
            <li key={String(check.id ?? index)}><span aria-hidden="true" style={{ color: STATE_COLOR[checkState(check.status)] }}>●</span> {String(check.kind)} — {String(check.status)} (exit {String(check.exit_code ?? '—')})</li>
          ))}</ul>
        </section>
      )}

      {displayRunId && displayedEvents.length > 0 && (
        <details className="codeq-work-graph-events">
          <summary>Run activity ({displayedEvents.length} persisted/received events)</summary>
          <ol aria-label="Recent CodeQ events">{displayedEvents.slice(-8).reverse().map((event, index) => <li key={String(event.sequence ?? event.id ?? `${event.type}-${index}`)}><strong>{String(event.type ?? 'event')}</strong>{event.task_id ? ` · task ${String(event.task_id)}` : ''}{event.created_at ? ` · ${new Date(String(event.created_at)).toLocaleTimeString()}` : ''}</li>)}</ol>
        </details>
      )}

      {selected && (
        <aside aria-label="Node details" className="codeq-work-graph-detail" tabIndex={-1} onKeyDown={onDetailsKey}>
          <h4>{String(selected.label ?? selected.path ?? selected.id)}</h4>
          <dl>
            <dt>Kind</dt><dd>{String(selected.kind)}</dd>
            <dt>Path</dt><dd>{String(selected.path ?? '—')}</dd>
            <dt>Provenance</dt><dd>{String(selected.provenance ?? '—')}</dd>
            <dt>Recorded hash</dt><dd>{String(selected.content_sha256 ?? '—')}</dd>
            <dt>Source range</dt><dd>{selected.line ? `line ${String(selected.line)}` : 'not recorded'}</dd>
            <dt>Freshness</dt><dd>{selected.content_sha256 ? (sourceQuery.isLoading ? 'checking…' : sourceQuery.error ? 'unavailable (could not read the current file)' : freshnessOf(selected, currentSha)) : 'unknown (no source hash recorded)'}</dd>
            <dt>Linked tasks</dt><dd>{linkedTasks.map(task => String(task.title)).join('; ') || 'none'}</dd>
            <dt>Linked criteria</dt><dd>{criteriaForTasks(linkedTasks).join(', ') || 'none'}</dd>
            <dt>Relevant evidence</dt><dd>{linkedChecks.map(check => `${String(check.kind)} ${String(check.status)}`).join('; ') || 'none'}</dd>
            <dt>Graph relations</dt><dd>{selectedEdges.map(edge => `${String(edge.relation)} (${String(edge.provenance ?? 'unknown')})`).join('; ') || 'none'}</dd>
            <dt>Current source</dt><dd>{currentFile ? <pre aria-label="Current source">{currentExcerpt || '(empty file)'}</pre> : sourceQuery.isLoading ? 'Loading current source…' : selectedPath ? 'Current source unavailable' : 'not associated with a source file'}</dd>
          </dl>
          <button type="button" onClick={closeDetails}>Close details</button>
        </aside>
      )}
    </div>
  );
}
