import { afterEach, describe, expect, it, vi, beforeEach } from 'vitest';
import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import {
  CodeQWorkGraphPanel, LEGEND, STATE_COLOR, checkState, criteriaForTasks, evidenceForTasks, filterCodeNodes,
  freshnessOf, graphSummary, taskState, tasksForPath,
} from '../src/screens/CodeQWorkGraphPanel';

const apiGet = vi.fn();
vi.mock('../src/api/client', () => ({ apiGet: (...args: unknown[]) => apiGet(...args) }));

function renderPanel(props: { live: boolean; projectId?: string; runId?: string; events?: JsonRecord[]; streamError?: string }) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const rendered = render(
    <QueryClientProvider client={client}>
      <CodeQWorkGraphPanel projectId={props.projectId ?? 'proj-1'} runId={props.runId ?? 'run-1'} live={props.live} events={props.events} streamError={props.streamError} />
    </QueryClientProvider>,
  );
  return { ...rendered, client };
}

const graphFixture = {
  revision: 'r1', node_count: 2, edge_count: 1,
  nodes: [
    { id: 'n1', kind: 'file', path: 'src/a.py', label: 'a.py', content_sha256: 'h-a', provenance: 'scanner' },
    { id: 'n2', kind: 'file', path: 'src/b.py', label: 'b.py', content_sha256: 'h-b' },
  ],
  edges: [{ relation: 'imports-local' }],
};

const runFixture = {
  run: {
    goal: { goal: 'Ship the account feature', status: 'active', revision: 2, milestones: [{ id: 'm1', title: 'Core feature', status: 'active' }], criteria: [{ id: 'c-1', criterion: 'Account records save', status: 'verified' }] },
    tasks: [{ id: 't1', title: 'Edit a', status: 'completed', code_paths: ['src/a.py'], criterion_ids: ['c-1'] }],
    validation: [{ id: 'e1', kind: 'test', status: 'passed', exit_code: 0, task_ids: ['t1'] }, { id: 'e2', kind: 'lint', status: 'stale', exit_code: 0, task_ids: ['t9'] }],
  },
};

function mockBackend(overrides: { graph?: unknown; run?: unknown; file?: unknown; goals?: unknown; events?: unknown; runs?: unknown } = {}) {
  apiGet.mockImplementation(async (path: string) => {
    if (path.includes('/graph')) return overrides.graph ?? graphFixture;
    if (path.includes('/api/codebase/goals?')) return overrides.goals ?? { goals: [] };
    if (path.includes('/events?after=')) return overrides.events ?? { events: [] };
    if (path.includes('/api/codebase/runs?')) return overrides.runs ?? { runs: [{ id: 'run-history-1', codebase_project_id: 'proj-1', user_prompt: 'Saved run', status: 'completed' }] };
    if (path.includes('/file?path=')) return overrides.file ?? { file: { content: 'print(1)\n', sha256: 'h-a' } };
    return overrides.run ?? runFixture;
  });
}

describe('Work Graph status mapping', () => {
  it('maps backend statuses to the five legend states', () => {
    expect(taskState('completed')).toBe('verified');
    expect(taskState('running')).toBe('active');
    expect(taskState('deferred')).toBe('attention');
    expect(taskState('needs_fix')).toBe('failed');
    expect(taskState('pending')).toBe('neutral');
    expect(taskState(undefined)).toBe('neutral');
  });

  it('never shows a stale check as verified', () => {
    expect(checkState('passed')).toBe('verified');
    expect(checkState('stale')).toBe('failed');
  });

  it('gives every legend state a distinct colour', () => {
    expect(new Set(LEGEND.map(state => STATE_COLOR[state])).size).toBe(LEGEND.length);
  });
});

describe('Work Graph data helpers', () => {
  const nodes = [
    { id: 'n2', kind: 'file', path: 'src/b.py' },
    { id: 'n1', kind: 'file', path: 'src/a.py' },
    { id: 'n3', kind: 'symbol', path: 'src/a.py' },
  ];

  it('filters only file nodes, sorts them, and bounds the rows', () => {
    const result = filterCodeNodes(nodes, '', 1);
    expect(result.total).toBe(2);
    expect(result.rows.map(node => node.path)).toEqual(['src/a.py']);
    expect(filterCodeNodes(nodes, 'B.PY').rows.map(node => node.path)).toEqual(['src/b.py']);
    expect(filterCodeNodes([{ id: 'ts', kind: 'file', path: 'src/profile.ts', label: 'profile.ts', language: 'typescript' }], 'TypeScript').total).toBe(1);
  });

  it('links paths, evidence and criteria only through the linked tasks', () => {
    const tasks = [{ id: 't1', title: 'Edit a', code_paths: ['src/a.py'], criterion_ids: ['c-2', 'c-1'] }, { id: 't2', title: 'Edit b', code_paths: ['src/b.py'] }];
    expect(tasksForPath(tasks, 'src/a.py').map(task => task.id)).toEqual(['t1']);
    expect(tasksForPath([{ id: 'declared', declared_files: ['src/new.py'] }], 'src/new.py').map(task => task.id)).toEqual(['declared']);
    expect(criteriaForTasks(tasks.slice(0, 1))).toEqual(['c-1', 'c-2']);
    const checks = [{ id: 'e1', task_ids: ['t1'] }, { id: 'e2', task_ids: ['t2'] }];
    expect(evidenceForTasks(checks, ['t1']).map(check => check.id)).toEqual(['e1']);
  });

  it('reports freshness by comparing recorded and current hashes', () => {
    expect(freshnessOf({ content_sha256: 'x' }, 'x')).toBe('fresh');
    expect(freshnessOf({ content_sha256: 'x' }, 'y')).toBe('stale');
    expect(freshnessOf({}, 'y')).toBe('unknown');
  });

  it('summarises node kinds and edge relations for the Project view', () => {
    const summary = graphSummary({ nodes, edges: [{ relation: 'imports-local' }, { relation: 'imports-local' }] });
    expect(summary.kinds).toEqual({ file: 2, symbol: 1 });
    expect(summary.relations).toEqual({ 'imports-local': 2 });
  });
});

describe('CodeQWorkGraphPanel', () => {
  beforeEach(() => {
    apiGet.mockReset();
  });

  afterEach(() => {
    cleanup();
  });

  it('shows a notice instead of sample data in demo mode', () => {
    renderPanel({ live: false });
    expect(screen.getByText(/Demo mode shows no graph data/)).toBeInTheDocument();
    expect(apiGet).not.toHaveBeenCalled();
  });

  it('scopes every request to the selected project and run', async () => {
    mockBackend();
    renderPanel({ live: true, projectId: 'proj-9', runId: 'run-7' });
    await waitFor(() => expect(apiGet).toHaveBeenCalledWith('/api/codebase/projects/proj-9/graph?limit=2000', expect.anything()));
    expect(apiGet).toHaveBeenCalledWith('/api/codebase/goals?codebase_project_id=proj-9&limit=100', expect.anything());
    expect(apiGet).toHaveBeenCalledWith('/api/codebase/runs/run-7', expect.anything());
  });

  it('shows a persisted project goal when no run is selected', async () => {
    mockBackend({
      run: { run: {} },
      goals: { goals: [{ id: 'goal-saved', goal: 'Maintain the billing workspace', status: 'active', revision: 4, milestones: [], criteria: [] }] },
    });
    renderPanel({ live: true, runId: '' });
    expect(await screen.findByLabelText('Project goal')).toHaveTextContent('Maintain the billing workspace');
    expect(screen.getByText('Status: active · revision 4')).toBeInTheDocument();
  });

  it('keeps an honest project-goal empty state when no durable goal exists', async () => {
    mockBackend({ run: { run: {} }, goals: { goals: [] } });
    renderPanel({ live: true, runId: '' });
    expect(await screen.findByText('No saved goal is linked to this project yet.')).toBeInTheDocument();
  });

  it('renders the legend and the project view from real backend records', async () => {
    mockBackend();
    renderPanel({ live: true });
    expect(screen.getByRole('region', { name: 'Work Graph' })).toBeInTheDocument();
    expect(screen.getByRole('list', { name: 'Status legend' })).toHaveTextContent('Blocked / approval');
    await waitFor(() => expect(screen.getByText(/2 nodes, 1 relations/)).toBeInTheDocument());
    expect(screen.getByTestId('freshness')).toHaveTextContent(/SSE 0 events/);
    fireEvent.click(screen.getByRole('tab', { name: 'Project' }));
    expect(await screen.findByLabelText('Project goal')).toHaveTextContent('Ship the account feature');
    expect(screen.getByRole('list', { name: 'Acceptance criteria' })).toHaveTextContent('Account records save');
  });

  it('shows an empty-state message when no graph has been recorded', async () => {
    mockBackend({ graph: { revision: null, nodes: [], edges: [] } });
    renderPanel({ live: true });
    await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent(/No graph has been recorded/));
  });

  it('shows a connection error and labels the data as last known', async () => {
    apiGet.mockRejectedValue(new Error('connection refused'));
    renderPanel({ live: true });
    await waitFor(() => expect(screen.getAllByRole('alert')[0]).toHaveTextContent(/connection refused/));
    expect(screen.getByTestId('freshness')).toHaveTextContent(/last known data/);
  });

  it('refreshes persisted graph and run records when the existing SSE event stream advances', async () => {
    mockBackend();
    const rendered = renderPanel({ live: true, events: [] });
    fireEvent.click(screen.getByRole('tab', { name: 'Tasks' }));
    expect(await screen.findByText(/Edit a — completed/)).toBeInTheDocument();
    const graphCallsBefore = apiGet.mock.calls.filter(call => String(call[0]).includes('/graph?limit=2000')).length;
    mockBackend({ run: { run: { tasks: [{ id: 't1', title: 'Edit a', status: 'needs_fix', code_paths: [] }], validation: [] } } });
    rendered.rerender(
      <QueryClientProvider client={rendered.client}>
        <CodeQWorkGraphPanel projectId="proj-1" runId="run-1" live events={[{ sequence: 2, type: 'context.graph.retrieved', task_id: 't1', created_at: '2026-11-12T10:00:00Z' }]} />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(screen.getByText(/Edit a — needs_fix/)).toBeInTheDocument());
    await waitFor(() => expect(apiGet.mock.calls.filter(call => String(call[0]).includes('/graph?limit=2000')).length).toBeGreaterThan(graphCallsBefore));
    expect(screen.getByText(/Run activity \(1 persisted\/received events\)/)).toBeInTheDocument();

    const runHistoryCallsBefore = apiGet.mock.calls.filter(call => String(call[0]).includes('/api/codebase/runs?project=default')).length;
    mockBackend({ runs: { runs: [{ id: 'run-1', codebase_project_id: 'proj-1', user_prompt: 'Saved run', status: 'failed' }] } });
    rendered.rerender(
      <QueryClientProvider client={rendered.client}>
        <CodeQWorkGraphPanel projectId="proj-1" runId="run-1" live events={[{ sequence: 3, type: 'run.failed', task_id: 't1', created_at: '2026-11-12T10:01:00Z' }]} />
      </QueryClientProvider>,
    );
    await waitFor(() => expect(apiGet.mock.calls.filter(call => String(call[0]).includes('/api/codebase/runs?project=default')).length).toBeGreaterThan(runHistoryCallsBefore));
    await waitFor(() => expect(screen.getByRole('combobox', { name: 'Work Graph run history' })).toHaveTextContent('failed'));
  });

  it('loads persisted run activity for a historical run after reopening the panel', async () => {
    mockBackend({ events: { events: [{ id: 'event-1', sequence: 1, type: 'task.completed', task_id: 't1', created_at: '2026-11-12T10:00:00Z' }] } });
    renderPanel({ live: true, runId: '' });
    expect(await screen.findByText(/Persisted run history/)).toBeInTheDocument();
    expect(await screen.findByText(/Run activity \(1 persisted\/received events\)/)).toBeInTheDocument();
    expect(screen.getByLabelText('Recent CodeQ events')).toHaveTextContent('task.completed');
  });

  it('shows failed task checks as failed evidence', async () => {
    mockBackend({ run: { run: {
      tasks: [{ id: 't-fail', title: 'Run tests', status: 'needs_fix', code_paths: ['src/a.py'] }],
      validation: [],
      task_checks: [{ id: 'check-fail', kind: 'pytest', status: 'failed', exit_code: 1, task_ids: ['t-fail'], output: '1 failed' }],
    } } });
    renderPanel({ live: true });
    fireEvent.click(screen.getByRole('tab', { name: 'Evidence' }));
    expect(await screen.findByText(/pytest — failed \(exit 1\)/)).toBeInTheDocument();
  });

  it('moves between views with arrow keys', async () => {
    mockBackend();
    renderPanel({ live: true });
    const project = screen.getByRole('tab', { name: 'Project' });
    project.focus();
    fireEvent.keyDown(project, { key: 'ArrowRight' });
    expect(screen.getByRole('tab', { name: 'Code' })).toHaveAttribute('aria-selected', 'true');
    await waitFor(() => expect(screen.getByRole('tab', { name: 'Code' })).toHaveFocus());
    fireEvent.keyDown(screen.getByRole('tab', { name: 'Code' }), { key: 'ArrowLeft' });
    expect(screen.getByRole('tab', { name: 'Project' })).toHaveAttribute('aria-selected', 'true');
  });

  it('opens node details with linked tasks, criteria, evidence and the current source', async () => {
    mockBackend();
    renderPanel({ live: true });
    fireEvent.click(screen.getByRole('tab', { name: 'Code' }));
    fireEvent.click(await screen.findByRole('option', { name: /src\/a\.py/ }));
    const details = await screen.findByLabelText('Node details');
    expect(details).toHaveTextContent('Edit a');
    expect(details).toHaveTextContent('c-1');
    expect(details).toHaveTextContent('test passed');
    expect(details).toHaveTextContent('Source range');
    expect(details).toHaveTextContent('Graph relations');
    expect(await screen.findByLabelText('Current source')).toHaveTextContent('print(1)');
    await waitFor(() => expect(screen.getByText('fresh')).toBeInTheDocument());
  });

  it('supports list arrow navigation, Escape focus recovery, and collapse', async () => {
    mockBackend();
    renderPanel({ live: true });
    fireEvent.click(screen.getByRole('tab', { name: 'Code' }));
    const first = await screen.findByRole('option', { name: /src\/a\.py/ });
    first.focus();
    fireEvent.keyDown(first, { key: 'ArrowDown' });
    expect(screen.getByRole('option', { name: /src\/b\.py/ })).toHaveFocus();
    fireEvent.click(first);
    const details = await screen.findByLabelText('Node details');
    fireEvent.keyDown(details, { key: 'Escape' });
    expect(screen.queryByLabelText('Node details')).not.toBeInTheDocument();
    await waitFor(() => expect(first).toHaveFocus());
    fireEvent.click(screen.getByRole('button', { name: 'Collapse list' }));
    expect(screen.queryByRole('listbox', { name: 'Code files' })).not.toBeInTheDocument();
  });
});
