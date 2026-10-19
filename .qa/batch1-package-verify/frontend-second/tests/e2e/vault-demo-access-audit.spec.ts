import { expect, test } from '@playwright/test';

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    localStorage.setItem('oa-second-data-mode', 'demo');
    localStorage.removeItem('oa-second-isolated-demo-v1');
    localStorage.removeItem('oa-second-demo-vault-activity');
  });
});

test('Vault Access demo restores the approved request workspace and supports local search, detail, review, edit, and create', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=vault');
  await page.getByRole('tab', { name: 'Access', exact: true }).click();

  await expect(page.getByRole('heading', { name: 'Security without friction' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Access · demo records' })).toBeVisible();
  const records = page.locator('.vault-access-records .demo-record');
  await expect(records).toHaveCount(2);
  await expect(records.nth(0)).toContainText('Agent needs GitHub read access');
  await expect(records.nth(0)).toContainText('Pending');
  await expect(records.nth(1)).toContainText('CodeQ needs staging key');
  await expect(page.getByRole('heading', { name: 'Recent activity' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Workspace overview' })).toBeVisible();

  await page.getByRole('textbox', { name: 'Search vault access' }).fill('CodeQ');
  await expect(records).toHaveCount(1);
  await expect(records.first()).toContainText('CodeQ needs staging key');
  await page.getByRole('textbox', { name: 'Search vault access' }).clear();
  await records.filter({ hasText: 'Agent needs GitHub read access' }).locator('.demo-record-main').click();
  const detail = page.getByRole('dialog', { name: 'Agent needs GitHub read access' });
  await expect(detail).toContainText('Read only');
  await detail.getByRole('button', { name: 'Approve', exact: true }).click();
  await expect(records.filter({ hasText: 'Agent needs GitHub read access' })).toContainText('Approved');
  await expect(page.locator('.advanced-demo-rail')).toContainText('Agent needs GitHub read access · Approved');
  await expect(page.locator('.advanced-demo-activity')).toContainText('Agent needs GitHub read access · Approved');

  await records.filter({ hasText: 'CodeQ needs staging key' }).getByRole('button', { name: 'Edit CodeQ needs staging key' }).click();
  const edit = page.getByRole('dialog', { name: 'Edit request' });
  await edit.getByLabel('Description').fill('Staging test scope reviewed');
  await edit.getByRole('button', { name: 'Save changes' }).click();
  await expect(records.filter({ hasText: 'CodeQ needs staging key' })).toContainText('Staging test scope reviewed');

  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const create = page.getByRole('dialog', { name: 'New request' });
  await expect(create.getByRole('button', { name: 'Create item' })).toBeEnabled();
  await expect(create.getByLabel('Name')).toHaveAttribute('required', '');
  await create.getByLabel('Name').fill('Calendar export request');
  await create.getByLabel('Description').fill('Read-only calendar metadata');
  await create.getByRole('button', { name: 'Create item' }).click();
  await expect(create).toBeHidden();
  await expect(records.first()).toContainText('Calendar export request');
  await expect(page.getByRole('status').filter({ hasText: 'Added Calendar export request' })).toBeVisible();
  await page.locator('.advanced-demo-rail').getByRole('button', { name: 'Reset this area' }).click();
  await expect(page.locator('.vault-access-records .demo-record')).toHaveCount(2);
  await expect(page.locator('.advanced-demo-rail')).toContainText('No activity yet.');
  expect(apiRequests).toEqual([]);
});

test('Vault Audit demo uses canonical fixture rows and keeps local record actions isolated', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto('/?view=vault');
  await page.getByRole('tab', { name: 'Audit', exact: true }).click();
  await expect(page.getByRole('heading', { name: 'Security without friction' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Audit · demo records' })).toBeVisible();
  const records = page.locator('.vault-access-records .demo-record');
  await expect(records).toHaveCount(2);
  await expect(records.nth(0)).toContainText('Token rotated');
  await records.nth(0).getByRole('button', { name: 'More actions for Token rotated' }).click();
  await page.getByRole('button', { name: 'Open details' }).click();
  await expect(page.getByRole('dialog', { name: 'Token rotated' })).toContainText('OpenAI development key');
  await page.keyboard.press('Escape');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const create = page.getByRole('dialog', { name: 'New audit' });
  await create.getByLabel('Name').fill('Local audit note');
  await create.getByLabel('Description').fill('Review requested');
  await create.getByRole('button', { name: 'Create item' }).click();
  await expect(records.first()).toContainText('Local audit note');
  await page.getByRole('textbox', { name: 'Search vault audit' }).fill('Access request');
  await expect(records).toHaveCount(1);
  await expect(records.first()).toContainText('Access request reviewed');
  expect(apiRequests).toEqual([]);
});
