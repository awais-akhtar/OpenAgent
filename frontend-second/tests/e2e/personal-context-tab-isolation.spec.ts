import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Personal Context tab state is isolated by collection (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ themeName }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', themeName);
      }, { themeName: theme });
      await page.goto('/?view=twin');

      await page.getByRole('tab', { name: 'Sources' }).click();
      const sourceSearch = page.getByRole('textbox', { name: 'Search sources' });
      await sourceSearch.fill('Uploaded documents');
      await expect(page.getByText('Uploaded documents', { exact: true })).toBeVisible();
      await expect(page.getByText('Workspace conversations', { exact: true })).toHaveCount(0);

      await page.getByRole('tab', { name: 'Data controls' }).click();
      const historySearch = page.getByRole('textbox', { name: 'Search history' });
      await expect(historySearch).toHaveValue('');
      await expect(page.getByText('Updated response preference', { exact: true })).toBeVisible();
      await expect(page.getByText('Reviewed data source', { exact: true })).toBeVisible();

      await page.getByRole('tab', { name: 'Sources' }).click();
      await expect(page.getByRole('textbox', { name: 'Search sources' })).toHaveValue('');
      await expect(page.getByText('Workspace conversations', { exact: true })).toBeVisible();
      await expect(page.getByText('Uploaded documents', { exact: true })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}
