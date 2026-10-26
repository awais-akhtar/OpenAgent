import { expect, test } from '@playwright/test';

for (const viewport of [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1280x800', width: 1280, height: 800 },
]) {
  for (const theme of ['light', 'dark'] as const) {
    for (const view of ['agents', 'home'] as const) {
      test(`Live ${view} keeps agent history in its authorized workspace · ${viewport.name} · ${theme}`, async ({ page }) => {
        await page.setViewportSize({ width: viewport.width, height: viewport.height });
        await page.addInitScript(({ selectedTheme }) => {
          localStorage.setItem('oa-second-data-mode', 'live');
          localStorage.setItem('oa-second-theme', selectedTheme);
        }, { selectedTheme: theme });

        const historyRequests: string[] = [];
        page.on('request', request => {
          const url = new URL(request.url());
          if (url.pathname === '/api/agent-runs') historyRequests.push(request.url());
        });
        await page.route('**/*', async route => {
          const url = new URL(route.request().url());
          if (!url.pathname.startsWith('/api/')) { await route.continue(); return; }
          if (url.pathname === '/api/agents') {
            await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ agents: [] }) });
            return;
          }
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
        });

        await page.goto(`/?view=${view}`);
        if (view === 'agents') {
          await expect(page.getByRole('heading', { name: 'Agent workspace' })).toBeVisible();
          await expect(page.getByText('No agent runs are recorded for this project.')).toBeVisible();
          expect(historyRequests.length).toBeGreaterThanOrEqual(1);
          for (const requestUrl of historyRequests) {
            const request = new URL(requestUrl);
            expect(request.searchParams.get('project')).toBe('default');
            expect(request.searchParams.has('owner')).toBe(false);
          }
        } else {
          await expect(page.getByText(/Agent run history is shown in Agents when this account has the explicit agents:read scope for the active project/i)).toBeVisible();
        }
        await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
        if (view === 'home') expect(historyRequests).toEqual([]);
      });
    }
  }
}
