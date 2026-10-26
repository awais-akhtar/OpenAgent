import { expect, test } from '@playwright/test';

/**
 * Covers incremental terminal output (polling) and cancellation on the CodeQ
 * Terminal tab in Live mode. All requests are intercepted — no real Python backend.
 */

const PROJECT_ID = 'proj-terminal-1';
const CWD = 'D:/openagent/data/workspaces/test-project';

function setupProjectRoutes(page: import('@playwright/test').Page) {
  return Promise.all([
    page.route('/api/codebase/projects?project=default', route =>
      route.fulfill({ json: { projects: [{ id: PROJECT_ID, name: 'Test Project', status: 'ready' }] } })
    ),
    page.route(`/api/codebase/projects/${PROJECT_ID}`, route =>
      route.fulfill({ json: { project: { id: PROJECT_ID, name: 'Test Project', status: 'ready', workspace: { workspace_path: CWD } } } })
    ),
    page.route(`/api/codebase/projects/${PROJECT_ID}/tree`, route =>
      route.fulfill({ json: { tree: [] } })
    ),
  ]);
}

test.describe('Terminal incremental output and cancellation', () => {
  test.use({ viewport: { width: 1440, height: 900 } });

  test('shows incremental stdout across polling cycles and exposes Stop while running', async ({ page }) => {
    let poll = 0;

    await setupProjectRoutes(page);
    await page.route('/api/terminal/runs?*', route =>
      route.fulfill({ json: { runs: [] } })
    );
    await page.route('/api/terminal/runs', async route => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 202,
          json: { run: { id: 'run-1', command: ['git', 'status', '--short'], cwd: CWD, status: 'running', exit_code: -1, events: [], started_at: '2026-10-25T13:00:00Z' } },
        });
      } else {
        await route.fulfill({ json: { runs: [] } });
      }
    });
    await page.route(`/api/terminal/runs/run-1`, route => {
      poll += 1;
      const events = poll >= 2
        ? [{ id: 'e1', stream: 'stdout', chunk: 'M  src/App.tsx\n' }, { id: 'e2', stream: 'stdout', chunk: 'M  src/api/client.ts\n' }]
        : [{ id: 'e1', stream: 'stdout', chunk: 'M  src/App.tsx\n' }];
      const status = poll >= 3 ? 'completed' : 'running';
      route.fulfill({ json: { run: { id: 'run-1', command: ['git', 'status', '--short'], cwd: CWD, status, exit_code: status === 'completed' ? 0 : -1, events, started_at: '2026-10-25T13:00:00Z' } } });
    });

    await page.goto('/?view=codeq');
    await page.getByRole('tab', { name: 'Terminal' }).click();
    // Live Terminal shows the command form (not the demo placeholder)
    const cmdInput = page.getByRole('textbox', { name: 'Terminal command' });
    await expect(cmdInput).toBeVisible({ timeout: 10_000 });
    await cmdInput.fill('git status --short');
    // Allowlisted command — no approval needed
    await expect(page.getByRole('status').filter({ hasText: /Allowlisted/ })).toBeVisible();
    await page.getByRole('button', { name: 'Run command' }).click();

    const log = page.getByRole('log', { name: 'Terminal run output' });
    await expect(log).toContainText('M  src/App.tsx', { timeout: 10_000 });
    // Stop button visible while running
    await expect(page.getByRole('button', { name: 'Stop' })).toBeVisible();

    // Wait for the second polling cycle to arrive
    await expect(log).toContainText('M  src/api/client.ts', { timeout: 10_000 });
    // Stop button disappears after completion
    await expect(page.getByRole('button', { name: 'Stop' })).toBeHidden({ timeout: 10_000 });
  });

  test('cancels a running terminal run via Stop and updates exit code to 130', async ({ page }) => {
    let cancelled = false;

    await setupProjectRoutes(page);
    await page.route('/api/terminal/runs?*', route =>
      route.fulfill({ json: { runs: [] } })
    );
    await page.route('/api/terminal/runs', async route => {
      if (route.request().method() === 'POST') {
        await route.fulfill({
          status: 202,
          json: { run: { id: 'run-2', command: ['npm', 'test'], cwd: CWD, status: 'running', exit_code: -1, events: [], started_at: '2026-10-25T13:00:00Z' } },
        });
      } else {
        await route.fulfill({ json: { runs: [] } });
      }
    });
    await page.route(`/api/terminal/runs/run-2`, async route => {
      if (route.request().method() === 'DELETE') {
        cancelled = true;
        await route.fulfill({ json: { run: { id: 'run-2', status: 'cancelled', exit_code: 130, events: [{ id: 'e1', stream: 'stdout', chunk: 'Test suite running...\n' }], started_at: '2026-10-25T13:00:00Z' }, cancel_requested: true } });
      } else {
        await route.fulfill({ json: { run: { id: 'run-2', command: ['npm', 'test'], cwd: CWD, status: cancelled ? 'cancelled' : 'running', exit_code: cancelled ? 130 : -1, events: [{ id: 'e1', stream: 'stdout', chunk: 'Test suite running...\n' }], started_at: '2026-10-25T13:00:00Z' } } });
      }
    });

    await page.goto('/?view=codeq');
    await page.getByRole('tab', { name: 'Terminal' }).click();
    const cmdInput = page.getByRole('textbox', { name: 'Terminal command' });
    await expect(cmdInput).toBeVisible({ timeout: 10_000 });
    await cmdInput.fill('npm test');
    // Non-default command needs explicit one-run approval
    await expect(page.getByRole('checkbox', { name: /Approve this non-default command/ })).toBeVisible();
    await page.getByRole('checkbox', { name: /Approve this non-default command/ }).check();
    await page.getByRole('button', { name: 'Run command' }).click();

    const log = page.getByRole('log', { name: 'Terminal run output' });
    await expect(log).toContainText('Test suite running', { timeout: 8_000 });
    const stopBtn = page.getByRole('button', { name: 'Stop' });
    await expect(stopBtn).toBeVisible();
    await stopBtn.click();

    // After cancellation Stop button disappears and exit code 130 is shown
    await expect(stopBtn).toBeHidden({ timeout: 8_000 });
    await expect(log.locator('..').getByText(/exit 130/)).toBeVisible({ timeout: 8_000 });
  });

  test('rejects shell-syntax commands before sending to the backend', async ({ page }) => {
    const terminalApiCalls: string[] = [];

    await setupProjectRoutes(page);
    await page.route('/api/terminal/runs?*', route =>
      route.fulfill({ json: { runs: [] } })
    );
    page.on('request', req => {
      if (new URL(req.url()).pathname.startsWith('/api/terminal')) terminalApiCalls.push(req.url());
    });

    await page.goto('/?view=codeq');
    await page.getByRole('tab', { name: 'Terminal' }).click();
    const cmdInput = page.getByRole('textbox', { name: 'Terminal command' });
    await expect(cmdInput).toBeVisible({ timeout: 10_000 });
    await cmdInput.fill('rm -rf /; echo done');
    await expect(page.getByRole('alert').filter({ hasText: /Shell operators/ })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Run command' })).toBeDisabled();
    // No POST to /api/terminal/runs should have been sent
    expect(terminalApiCalls.filter(u => !u.includes('?'))).toHaveLength(0);
  });
});
