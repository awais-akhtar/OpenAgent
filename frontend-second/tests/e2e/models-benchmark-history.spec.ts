import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Models reads persisted benchmark history in Live and stays offline in Demo (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      const apiRequests: string[] = [];
      const fixtureRequests: string[] = [];
      page.on('request', request => {
        const path = new URL(request.url()).pathname;
        if (path.startsWith('/api/')) apiRequests.push(path);
        if (path.startsWith('/demo/')) fixtureRequests.push(path);
      });
      const benchmark = { id: 'benchmark-e2e-1', model_id: 'local-1', model: 'Local test model', provider: 'local', prompt_tokens: 3, output_tokens: 9, duration_ms: 250, tokens_per_second: 36, created_at: '2026-10-24T10:00:00Z' };
      for (const [path, payload] of [
        ['/api/models', { providers: [] }],
        ['/api/local-ai/models', { models: [] }],
        ['/api/local-ai/hardware', { cpu_count: 8, ram_gb: 16 }],
        ['/api/model-policy', { routing_policy: 'local-first' }],
        ['/api/local-ai/setup/status', {}],
        ['/api/local-ai/marketplace', { models: [] }],
        ['/api/local-ai/adapters', { adapters: [] }],
        ['/api/local-ai/jobs', { jobs: [] }],
        ['/api/local-ai/benchmarks', { benchmarks: [benchmark] }],
      ] as const) {
        await page.route(`**${path}**`, route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(payload) }));
      }
      await page.goto('/?view=models');
      await page.getByText('Provider setup', { exact: true }).click();
      await page.getByRole('tab', { name: 'Setup activity' }).click();
      await expect(page.getByText('Local test model', { exact: true })).toBeVisible();
      await expect(page.getByText('250 ms', { exact: true })).toBeVisible();
      expect(apiRequests).toContain('/api/local-ai/benchmarks');
      expect(fixtureRequests).toEqual([]);

      const liveApiRequestCount = apiRequests.length;
      await page.getByRole('button', { name: 'Data mode: live. Click to switch to demo' }).click();
      await expect(page.getByRole('heading', { name: 'No measured benchmarks in Demo' })).toBeVisible();
      expect(apiRequests).toHaveLength(liveApiRequestCount);
    });
  }
}
