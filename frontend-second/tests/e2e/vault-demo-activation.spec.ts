import { expect, test } from '@playwright/test';

async function selectTheme(page: import('@playwright/test').Page, theme: 'light' | 'dark') {
  await page.addInitScript(value => localStorage.setItem('oa-second-theme', value), theme);
  await page.addInitScript(() => {
    localStorage.setItem('oa-second-data-mode', 'demo');
    localStorage.removeItem('oa-second-isolated-demo-v1');
    localStorage.removeItem('oa-second-demo-vault-activity');
  });
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Vault Access activation: invalid/valid create, edit, deny, duplicate and delete (${viewport.width} ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await selectTheme(page, theme);
      await page.setViewportSize(viewport);
      await page.goto('/?view=vault');
      await page.getByRole('tab', { name: 'Access', exact: true }).click();
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);

      const rows = page.locator('.vault-access-records .demo-record');
      const add = page.getByRole('button', { name: 'Add', exact: true });
      await add.click();
      const create = page.getByRole('dialog', { name: 'New request' });
      const name = create.getByLabel('Name');
      await expect(name).toHaveAttribute('required', '');
      await expect(name).toHaveJSProperty('validity.valid', false);
      const createButton = create.getByRole('button', { name: 'Create item' });
      await expect(createButton).toBeEnabled();
      await createButton.click();
      await expect(create).toBeVisible();
      await expect(name).toHaveJSProperty('validity.valid', false);
      await name.fill('Desktop access review');
      await create.getByLabel('Description').fill('Read-only access to test metadata');
      await create.getByRole('button', { name: 'Create item' }).click();
      await expect(create).toBeHidden();
      await expect(rows.first()).toContainText('Desktop access review');

      const newRow = rows.filter({ hasText: 'Desktop access review' });
      await newRow.getByRole('button', { name: 'Edit Desktop access review' }).click();
      const edit = page.getByRole('dialog', { name: 'Edit request' });
      await edit.getByLabel('Description').fill('Updated read-only review');
      await edit.getByRole('button', { name: 'Save changes' }).click();
      await expect(newRow).toContainText('Updated read-only review');

      await rows.filter({ hasText: 'Agent needs GitHub read access' }).locator('.demo-record-main').click();
      const detail = page.getByRole('dialog', { name: 'Agent needs GitHub read access' });
      await expect(detail.locator('.vault-demo-detail-actions button')).toHaveText(['Approve', 'Deny', 'Edit', 'Delete']);
      await detail.getByRole('button', { name: 'Deny', exact: true }).click();
      await expect(rows.filter({ hasText: 'Agent needs GitHub read access' })).toContainText('Denied');

      await newRow.locator('button[aria-label="More actions for Desktop access review"]').click();
      const actions = page.getByRole('dialog', { name: 'Desktop access review' });
      await actions.getByRole('button', { name: 'Duplicate', exact: true }).click();
      await expect(rows.filter({ hasText: 'Desktop access review (copy)' })).toHaveCount(1);

      await newRow.locator('button[aria-label="More actions for Desktop access review"]').click();
      await page.getByRole('dialog', { name: 'Desktop access review' }).getByRole('button', { name: 'Delete', exact: true }).click();
      const confirm = page.getByRole('dialog', { name: 'Delete “Desktop access review”?' });
      await confirm.getByRole('button', { name: 'Cancel' }).click();
      await expect(rows.filter({ hasText: 'Desktop access review' }).filter({ hasNotText: '(copy)' })).toHaveCount(1);
      const detailAfterCancel = page.getByRole('dialog', { name: 'Desktop access review' });
      await detailAfterCancel.getByRole('button', { name: 'Delete', exact: true }).click();
      await page.getByRole('dialog', { name: 'Delete “Desktop access review”?' }).getByRole('button', { name: 'Delete record' }).click();
      await expect(rows.filter({ hasText: 'Desktop access review' }).filter({ hasNotText: '(copy)' })).toHaveCount(0);
      await expect(page.locator('.advanced-demo-activity')).toContainText('Agent needs GitHub read access · Denied');
      expect(apiRequests).toEqual([]);
    });

    test(`Vault Audit activation: detail, simulated run, create, search and keyboard recovery (${viewport.width} ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
      });
      await selectTheme(page, theme);
      await page.setViewportSize(viewport);
      await page.goto('/?view=vault');
      await page.getByRole('tab', { name: 'Audit', exact: true }).click();
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
      const rows = page.locator('.vault-access-records .demo-record');
      const target = rows.filter({ hasText: 'Token rotated' });

      const menuButton = target.getByRole('button', { name: 'More actions for Token rotated' });
      await menuButton.click();
      const actions = page.getByRole('dialog', { name: 'Token rotated' });
      await expect(actions.getByRole('button', { name: 'Open details' })).toBeVisible();
      await page.keyboard.press('Escape');
      await expect(actions).toBeHidden();
      await expect(menuButton).toBeFocused();

      await target.locator('.demo-record-main').click();
      const detail = page.getByRole('dialog', { name: 'Token rotated' });
      await expect(detail.getByRole('button', { name: 'Run example', exact: true })).toBeVisible();
      await expect(detail.locator('.vault-demo-detail-actions button')).toHaveText(['Run example', 'Edit', 'Delete']);
      await detail.getByRole('button', { name: 'Run example', exact: true }).click();
      await expect(detail).toBeHidden();
      const runningRow = rows.filter({ hasText: 'Token rotated' });
      await expect(runningRow).toContainText('Running');
      await expect(page.locator('.advanced-demo-activity')).toContainText('Started Token rotated');
      await expect(runningRow).toContainText('Completed', { timeout: 6500 });
      await expect(page.locator('.advanced-demo-activity')).toContainText('Completed Token rotated');

      const add = page.getByRole('button', { name: 'Add', exact: true });
      await add.click();
      const create = page.getByRole('dialog', { name: 'New audit' });
      await expect(create.getByLabel('Name')).toHaveAttribute('required', '');
      await create.getByLabel('Name').fill('Local audit note');
      await create.getByLabel('Description').fill('Synthetic review record');
      await create.getByRole('button', { name: 'Create item' }).click();
      await expect(create).toBeHidden();
      await expect(rows.first()).toContainText('Local audit note');
      await page.getByRole('textbox', { name: 'Search vault audit' }).fill('Local audit note');
      await expect(rows).toHaveCount(1);
      await page.getByRole('button', { name: 'All activity' }).click();
      const activity = page.getByRole('dialog', { name: 'Demo activity' });
      await expect(activity.getByText(/vault · (Completed|Running) · \d\d:\d\d/).first()).toBeVisible();
      await expect(activity.locator('.square-icon.green').first()).toBeVisible();
      expect(apiRequests).toEqual([]);
    });
  }
}
