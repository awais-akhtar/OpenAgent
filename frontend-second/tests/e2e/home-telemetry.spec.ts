import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Home technical telemetry auto-polls at ${viewport.width} ${theme}`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(themeName => { localStorage.setItem('oa-second-data-mode', 'live'); localStorage.setItem('oa-second-theme', themeName); }, theme);
      let cpu = 22;
      const requests: string[] = [];
      await page.route('**/api/**', async route => {
        const pathname = new URL(route.request().url()).pathname;
        if (!pathname.startsWith('/api/')) { await route.continue(); return; }
        requests.push(pathname);
        const data = pathname === '/api/system/metrics' ? { cpu: { count: 16, usage_percent: cpu }, memory: { used_percent: 51, used_gb: 16, total_gb: 32 }, disk: { used_percent: 58, free_gb: 160, total_gb: 390 }, platform: { system: 'Windows', release: '11' } }
          : pathname === '/api/health' ? { status: 'ok', checked_at: '2026-11-09T10:00:00Z' }
          : pathname === '/api/local-ai/setup/status' ? { ollama: { server_available: true, model_ready: true, selected_model: 'local-model' }, llamacpp: { available: false } }
          : pathname === '/api/agents/jobs' ? { jobs: [{ id: 'agent-1', goal: 'Local code review', status: 'running' }] }
          : pathname === '/api/local-ai/jobs' ? { jobs: [] }
          : { error: 'unexpected endpoint' };
        await route.fulfill({ status: pathname in { '/api/system/metrics': true, '/api/health': true, '/api/local-ai/setup/status': true, '/api/agents/jobs': true, '/api/local-ai/jobs': true } ? 200 : 404, contentType: 'application/json', body: JSON.stringify(data) });
      });
      await page.goto('/?view=home');
      await expect(page.getByRole('meter', { name: 'CPU utilization' })).toBeVisible();
      await expect(page.getByRole('meter', { name: 'CPU utilization' })).toHaveAttribute('aria-valuenow', '22');
      await expect(page.getByText('local-model')).toBeVisible();
      await expect(page.getByText(/Local code review/)).toBeVisible();
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
      cpu = 67;
      await expect(page.getByRole('meter', { name: 'CPU utilization' })).toHaveAttribute('aria-valuenow', '67', { timeout: 8000 });
      await expect(page.getByRole('button', { name: 'Pause' })).toHaveCount(0);
      await expect(page.getByRole('button', { name: 'Refresh live measurements' })).toHaveCount(0);
      await page.getByRole('button', { name: '5 min' }).click();
      await expect(page.getByRole('button', { name: '5 min' })).toHaveAttribute('aria-pressed', 'true');
      expect(requests).toContain('/api/system/metrics');
      expect(requests).not.toContain('/api/assistant/tasks');
      expect(requests).not.toContain('/api/codebase/projects');
      expect(requests).not.toContain('/api/events');
    });
  }
}

test('Home Demo telemetry is explicitly simulated and sends zero API requests', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  const requests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) requests.push(request.url()); });
  await page.goto('/?view=home');
  await expect(page.getByText('Simulated demo')).toBeVisible();
  await expect(page.getByRole('meter', { name: 'CPU utilization' })).toHaveAttribute('aria-valuenow', '24.6');
  await expect(page.getByRole('button', { name: 'Pause' })).toHaveCount(0);
  expect(requests).toEqual([]);
});
