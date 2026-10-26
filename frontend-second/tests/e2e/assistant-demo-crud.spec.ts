import { expect, test } from '@playwright/test';

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Personal Assistant task and goal CRUD stays local (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      await page.goto('/?view=assistant');

      await page.getByRole('tab', { name: 'Tasks', exact: true }).click();
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await page.getByLabel('Task', { exact: true }).fill('Local assistant task to edit');
      await page.getByRole('button', { name: 'Save locally', exact: true }).click();
      const taskRow = page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant task to edit' });
      await expect(taskRow).toBeVisible();
      await taskRow.getByRole('button', { name: 'More options for Local assistant task to edit' }).click();
      await taskRow.getByRole('button', { name: 'Edit', exact: true }).click();
      await page.getByLabel('Edit task', { exact: true }).fill('Local assistant task revised');
      await page.getByRole('button', { name: 'Save changes', exact: true }).click();
      await expect(page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant task revised' })).toBeVisible();
      const revisedTask = page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant task revised' });
      await revisedTask.getByRole('button', { name: 'More options for Local assistant task revised' }).click();
      page.once('dialog', dialog => dialog.accept());
      await revisedTask.getByRole('button', { name: 'Delete', exact: true }).click();
      await expect(page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant task revised' })).toHaveCount(0);

      await page.getByRole('tab', { name: 'Goals', exact: true }).click();
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      const goalDialog = page.getByRole('dialog', { name: 'New goal' });
      await goalDialog.getByLabel('Name').fill('Local assistant goal');
      await goalDialog.getByRole('button', { name: 'Create item', exact: true }).click();
      const goalRow = page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant goal' });
      await expect(goalRow).toBeVisible();
      await goalRow.getByRole('button', { name: 'Edit Local assistant goal' }).click();
      const editGoal = page.getByRole('dialog', { name: 'Edit goal' });
      await editGoal.getByLabel('Name').fill('Local assistant goal revised');
      await editGoal.getByRole('button', { name: 'Save changes', exact: true }).click();
      const revisedGoal = page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant goal revised' });
      await expect(revisedGoal).toBeVisible();
      await revisedGoal.getByRole('button', { name: 'More actions for Local assistant goal revised' }).click();
      page.once('dialog', dialog => dialog.accept());
      await revisedGoal.getByRole('button', { name: 'Delete', exact: true }).click();
      await expect(page.locator('.assistant-demo-row').filter({ hasText: 'Local assistant goal revised' })).toHaveCount(0);
      expect(apiRequests).toEqual([]);
    });
  }
}
