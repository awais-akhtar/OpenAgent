import { expect, test } from '@playwright/test';

const viewports = [
  { width: 1440, height: 900 },
  { width: 1280, height: 800 },
];

for (const viewport of viewports) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Knowledge source editing persists locally in Demo (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ mode, selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', mode);
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { mode: 'demo', selectedTheme: theme });
      await page.goto('/?view=knowledge');
      await page.getByRole('button', { name: 'Open Product requirements' }).click();

      const details = page.getByRole('dialog', { name: 'Product requirements' });
      await expect(details).toBeVisible();
      await details.getByRole('button', { name: 'Edit source' }).click();
      const editor = page.getByRole('dialog', { name: 'Edit knowledge source' });
      await editor.getByRole('textbox', { name: 'Knowledge source title' }).fill('Updated local requirements');
      await editor.getByRole('textbox', { name: 'Knowledge source text' }).fill('Revised project decisions remain searchable in the local demo.');
      await editor.getByRole('button', { name: 'Save source' }).click();
      await expect(editor).toHaveCount(0);
      await expect(page.getByRole('heading', { name: 'Updated local requirements' })).toBeVisible();
      await expect(page.getByRole('status').filter({ hasText: 'Source updated in local demo data.' })).toBeVisible();

      await page.getByRole('dialog', { name: 'Updated local requirements' }).getByRole('button', { name: 'Close source details' }).click();
      await expect(page.getByRole('button', { name: 'Open Updated local requirements' })).toBeVisible();
      await page.reload();
      await expect(page.getByRole('button', { name: 'Open Updated local requirements' })).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}
