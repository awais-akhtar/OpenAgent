import { expect, test } from '@playwright/test';

for (const viewport of [
  { name: '1440x900', width: 1440, height: 900 },
  { name: '1280x800', width: 1280, height: 800 },
]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live automation overview reports no run instead of sample success · ${viewport.name} · ${theme}`, async ({ page }) => {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });

      const workflow = {
        id: 'wf-empty-history', name: 'Review local notes', description: 'Summarise workspace notes',
        project: 'default', enabled: true, trigger_type: 'manual', steps: [{ action: 'summarize' }],
      };
      await page.route('**/api/workflows/builder', route => route.fulfill({
        status: 200, contentType: 'application/json',
        body: JSON.stringify({ actions: { summarize: { label: 'Summarise' } } }),
      }));
      await page.route('**/api/workflows?project=default*', route => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ workflows: [workflow] }),
      }));
      await page.route('**/api/workflow-runs?project=default*', route => route.fulfill({
        status: 200, contentType: 'application/json', body: JSON.stringify({ runs: [] }),
      }));

      await page.goto('/?view=automations');
      await page.locator('.automation-record').getByText('Review local notes').click();
      const metrics = page.locator('.metric-strip');
      await expect(metrics).toContainText('Last run');
      await expect(metrics).toContainText('No runs reported');
      await expect(metrics).not.toContainText('Example · Completed');
      await expect(page.getByRole('button', { name: 'Preview run' })).toBeVisible();
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
    });
  }
}
