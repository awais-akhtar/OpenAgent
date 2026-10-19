import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Personal Assistant Tasks shows useful search and create states (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ mode, themeName }) => {
        localStorage.setItem('oa-second-data-mode', mode);
        localStorage.setItem('oa-second-theme', themeName);
      }, { mode: 'demo', themeName: theme });
      await page.goto('/?view=assistant');
      await page.getByRole('tab', { name: 'Tasks' }).click();

      const taskSearch = page.getByRole('textbox', { name: 'Search tasks' });
      await expect(page.locator('.assistant-demo-row').first()).toBeVisible();
      await taskSearch.fill('no task matches this exact query');
      await expect(page.getByText('No matching tasks', { exact: true })).toBeVisible();
      await expect(page.getByText('Try another search or clear the filter.', { exact: true })).toBeVisible();

      await taskSearch.fill('');
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await page.getByRole('textbox', { name: 'Task', exact: true }).fill('Review local empty-state fix');
      await page.getByRole('button', { name: 'Save locally' }).click();
      await expect(page.locator('.assistant-demo-list').getByText('Review local empty-state fix', { exact: true })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}
