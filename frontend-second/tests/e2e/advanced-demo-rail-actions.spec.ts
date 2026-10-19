import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Advanced Demo rail export/reset succeed locally (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });
      await page.goto('/?view=voice');

      const downloadPromise = page.waitForEvent('download');
      await page.getByRole('button', { name: 'Export demo data' }).click();
      expect((await downloadPromise).suggestedFilename()).toBe('openagent-voice-demo.json');
      await expect(page.getByRole('status').filter({ hasText: 'Demo data export downloaded locally.' })).toBeVisible();

      await page.getByRole('button', { name: 'Reset this area' }).click();
      await expect(page.getByRole('status').filter({ hasText: 'Local demo data was reset to the approved examples.' })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });

    test(`Advanced Demo rail export/reset failures are visible and handled (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      const pageErrors: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', themeName);
        URL.createObjectURL = () => { throw new Error('Local download unavailable for this test.'); };
      }, { themeName: theme });
      await page.goto('/?view=voice');
      await page.getByRole('button', { name: 'Export demo data' }).click();
      await expect(page.getByRole('alert')).toContainText('Local download unavailable for this test.');

      await page.route('**/demo/reference-fixtures.json', route => route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Fixture reset unavailable for this test.' }),
      }));
      await page.getByRole('button', { name: 'Reset this area' }).click();
      await expect(page.getByRole('alert')).toContainText('The approved demo fixtures could not be loaded.');
      expect(apiRequests).toEqual([]);
      expect(pageErrors).toEqual([]);
    });
  }
}
