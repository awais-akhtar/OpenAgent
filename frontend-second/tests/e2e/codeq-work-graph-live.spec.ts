import { expect, test } from '@playwright/test';

const projectId = 'project-graph-1';
const runId = 'run-graph-1';
const graph = {
  revision: 'revision-1', node_count: 1, edge_count: 0, truncated: false,
  nodes: [{ id: 'file:src/app.py', kind: 'file', path: 'src/app.py', label: 'app.py', language: 'python', content_sha256: 'hash-app', provenance: 'python-ast' }],
  edges: [],
};
const goal = {
  id: 'goal-1', codebase_project_id: projectId, goal: 'Complete the account feature', status: 'active', revision: 1,
  milestones: [{ id: 'm1', title: 'Core flow', status: 'active' }],
  criteria: [{ id: 'c1', criterion: 'Account data persists', status: 'open' }],
};

function run(status: string) {
  return {
    id: runId, codebase_project_id: projectId, goal_id: goal.id, user_prompt: 'Implement account persistence', status,
    tasks: [{ id: 'task-1', title: 'Implement account persistence', status: status === 'completed' ? 'completed' : 'running', code_paths: ['src/app.py'], criterion_ids: ['c1'] }],
    validation: [{ id: 'check-1', kind: 'pytest', status: status === 'completed' ? 'passed' : 'running', exit_code: status === 'completed' ? 0 : null, task_ids: ['task-1'] }],
    task_checks: [],
  };
}

test('Work Graph restores persisted goals, tasks and evidence from project-scoped backend records', async ({ page }) => {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path === '/api/codebase/run-admission') return route.fulfill({ json: { autonomous_run: 'eligible' } });
    if (path === '/api/models') return route.fulfill({ json: { providers: [] } });
    if (path === '/api/codebase/projects') return route.fulfill({ json: { projects: [{ id: projectId, name: 'Graph project', project: 'default' }] } });
    if (path === `/api/codebase/projects/${projectId}`) return route.fulfill({ json: { project: { id: projectId, name: 'Graph project', workspace: { workspace_path: 'C:/isolated/graph-project' } } } });
    if (path === `/api/codebase/projects/${projectId}/tree`) return route.fulfill({ json: { entries: [{ path: 'src/app.py', type: 'file' }] } });
    if (path === `/api/codebase/projects/${projectId}/graph`) return route.fulfill({ json: graph });
    if (path === '/api/codebase/goals') return route.fulfill({ json: { goals: [goal] } });
    if (path === '/api/codebase/runs') return route.fulfill({ json: { runs: [run('completed')] } });
    if (path === `/api/codebase/runs/${runId}`) return route.fulfill({ json: { run: { ...run('completed'), goal } } });
    if (path === `/api/codebase/runs/${runId}/events`) return route.fulfill({ json: { events: [{ sequence: 1, type: 'run.completed', created_at: '2026-11-12T11:59:00Z' }] } });
    if (path === `/api/codebase/projects/${projectId}/file`) return route.fulfill({ json: { file: { content: 'def save_account():\n    return True\n', sha256: 'hash-app' } } });
    return route.fulfill({ status: 404, json: { error: `Unmatched Work Graph request ${route.request().method()} ${path}` } });
  });

  await page.goto('/?view=codeq');
  await page.getByRole('combobox', { name: 'CodeQ project' }).selectOption(projectId);
  await page.getByRole('tab', { name: 'Work Graph' }).click();
  await expect(page.getByLabel('Project goal')).toContainText('Complete the account feature');
  await expect(page.getByRole('combobox', { name: 'Work Graph run history' })).toContainText('completed');
  await page.getByRole('tab', { name: 'Tasks' }).click();
  await expect(page.getByText('Implement account persistence — completed')).toBeVisible();
  await page.getByRole('tab', { name: 'Evidence' }).click();
  await expect(page.getByText('pytest — passed (exit 0)')).toBeVisible();
  await page.getByRole('region', { name: 'Work Graph' }).getByRole('tab', { name: 'Code' }).click();
  await page.getByRole('option', { name: 'src/app.py' }).click();
  await expect(page.getByLabel('Node details')).toContainText('fresh');
  await expect(page.getByLabel('Current source')).toContainText('def save_account');
});

test('Work Graph updates from the existing Live run SSE stream, without sample events', async ({ page }) => {
  const streamRequests: string[] = [];
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    if (!path.startsWith('/api/')) return route.continue();
    if (path === '/api/codebase/run-admission') return route.fulfill({ json: { autonomous_run: 'eligible' } });
    if (path === '/api/models') return route.fulfill({ json: { providers: [] } });
    if (path === '/api/codebase/projects') return route.fulfill({ json: { projects: [{ id: projectId, name: 'Graph project', project: 'default' }] } });
    if (path === `/api/codebase/projects/${projectId}`) return route.fulfill({ json: { project: { id: projectId, name: 'Graph project', workspace: { workspace_path: 'C:/isolated/graph-project' } } } });
    if (path === `/api/codebase/projects/${projectId}/tree`) return route.fulfill({ json: { entries: [] } });
    if (path === `/api/codebase/projects/${projectId}/graph`) return route.fulfill({ json: graph });
    if (path === '/api/codebase/goals') return route.fulfill({ json: { goals: [goal] } });
    if (path === '/api/codebase/runs' && route.request().method() === 'GET') return route.fulfill({ json: { runs: [] } });
    if (path === '/api/codebase/runs' && route.request().method() === 'POST') return route.fulfill({ status: 202, json: { run: { id: runId, status: 'queued' } } });
    if (path === `/api/codebase/runs/${runId}/events`) return route.fulfill({ json: { events: [] } });
    if (path === `/api/codebase/runs/${runId}/events/stream`) {
      streamRequests.push(url.search);
      const frames = [
        { type: 'context.graph.retrieved', sequence: 1, task_id: 'task-1', created_at: '2026-11-12T12:00:00Z' },
        { type: 'task.completed', sequence: 2, task_id: 'task-1', created_at: '2026-11-12T12:00:01Z' },
        { type: 'stream.closed', sequence: 3 },
      ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
      return route.fulfill({ status: 200, contentType: 'text/event-stream', body: frames });
    }
    if (path === `/api/codebase/runs/${runId}`) return route.fulfill({ json: { run: { ...run('completed'), goal } } });
    return route.fulfill({ status: 404, json: { error: `Unmatched Work Graph request ${route.request().method()} ${path}` } });
  });

  await page.goto('/?view=codeq');
  await page.getByRole('tab', { name: 'Work Graph' }).click();
  await expect(page.getByText('No persisted CodeQ runs are available for this project yet.')).toBeVisible();
  const composer = page.getByRole('textbox', { name: 'Ask CodeQ' });
  const prompt = 'Implement the account save function and run its tests.';
  await composer.fill(prompt);
  await page.getByRole('button', { name: 'Send to CodeQ' }).click();
  const thread = page.locator('.codeq-chat-thread');
  await expect(thread.locator('.chat-message-row.user')).toContainText(prompt);
  await expect(composer).toHaveValue('');
  await expect(thread.locator('.chat-message-row.assistant')).toContainText('Run completed.');
  await expect(page.getByText(/Run activity \(3 persisted\/received events\)/)).toBeVisible();
  await page.getByRole('tab', { name: 'Tasks' }).click();
  await expect(page.getByText('Implement account persistence — completed')).toBeVisible();
  expect(streamRequests).toContain('?after=0&timeout_seconds=120');
  await expect(page.getByText(/SSE 3 events/)).toBeVisible();
});
