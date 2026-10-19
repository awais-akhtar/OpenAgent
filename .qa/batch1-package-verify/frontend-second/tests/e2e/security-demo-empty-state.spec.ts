import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Security Demo lists explain empty searches (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ mode, themeName }) => {
        localStorage.setItem('oa-second-data-mode', mode);
        localStorage.setItem('oa-second-theme', themeName);
      }, { mode: 'demo', themeName: theme });
      await page.goto('/?view=security');

      const assertEmptySearch = async (label: string) => {
        const search = page.getByRole('textbox', { name: label, exact: true });
        await search.fill('no security record matches this query');
        await expect(page.getByRole('status').filter({ hasText: 'No matching items' })).toBeVisible();
        await expect(page.getByText('Try a different search or create a sample record.', { exact: true })).toBeVisible();
        await search.fill('');
      };

      await assertEmptySearch('Search policies');
      await page.getByRole('tab', { name: 'Access' }).click();
      await assertEmptySearch('Search access');
      await page.getByRole('tab', { name: 'Sessions' }).click();
      await assertEmptySearch('Search sessions');
      await page.getByRole('tab', { name: 'Audit' }).click();
      await assertEmptySearch('Search audit');
      expect(apiRequests).toEqual([]);
    });
  }
}
