import { expect, test } from '@playwright/test';

/**
 * Covers shell-level interaction behavior that the route/tab screenshot matrix
 * only discovers: keyboard opening/filtering/closing the command palette and
 * persistence of a local demo preference. These tests intentionally use the
 * desktop app and fail if a demo action reaches an API endpoint.
 */
test('command palette keyboard flow filters and navigates locally', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const apiRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });
  await page.addInitScript(() => {
    if (localStorage.getItem('e2e-shell-settings-initialized') !== 'yes') {
      localStorage.clear();
      localStorage.setItem('oa-second-data-mode', 'demo');
      localStorage.setItem('e2e-shell-settings-initialized', 'yes');
    }
  });
  await page.goto('/?view=home');

  await page.keyboard.press('Control+k');
  const palette = page.getByRole('dialog', { name: 'Search OpenAgent' });
  await expect(palette).toBeVisible();
  const search = palette.getByPlaceholder('Search workspaces');
  await expect(search).toBeFocused();
  await search.fill('vault');
  await expect(palette.getByRole('button', { name: 'Vault' })).toBeVisible();
  await expect(palette.getByRole('button')).toHaveCount(2); // Vault result and Close search.
  await palette.getByRole('button', { name: 'Vault' }).click();
  await expect(page).toHaveURL(/view=vault/);
  await expect(page.locator('#content h1').first()).toHaveText('Vault');
  await expect(palette).toHaveCount(0);
  expect(apiRequests).toEqual([]);
});

test('settings notification preferences change and persist as local demo state', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const apiRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });
  await page.addInitScript(() => {
    if (localStorage.getItem('e2e-shell-settings-initialized') !== 'yes') {
      localStorage.clear();
      localStorage.setItem('oa-second-data-mode', 'demo');
      localStorage.setItem('e2e-shell-settings-initialized', 'yes');
    }
  });
  await page.goto('/?view=settings');
  await page.getByRole('tab', { name: 'Notifications', exact: true }).click();

  const taskCompletion = page.getByRole('checkbox', { name: 'Task completion' });
  const approvalRequests = page.getByRole('checkbox', { name: 'Approval requests' });
  await expect(taskCompletion).toBeChecked();
  await expect(approvalRequests).toBeChecked();
  await taskCompletion.uncheck();
  await approvalRequests.uncheck();
  await expect(taskCompletion).not.toBeChecked();
  await expect(approvalRequests).not.toBeChecked();

  // Reload to prove the controls reflect stored state after a fresh React mount.
  await page.reload();
  await page.getByRole('tab', { name: 'Notifications', exact: true }).click();
  await expect(page.getByRole('checkbox', { name: 'Task completion' })).not.toBeChecked();
  await expect(page.getByRole('checkbox', { name: 'Approval requests' })).not.toBeChecked();
  const stored = await page.evaluate(() => JSON.parse(localStorage.getItem('oa-second-settings-demo-v1') ?? '{}'));
  expect(stored['Notifications|Task completion']).toBe(false);
  expect(stored['Notifications|Approval requests']).toBe(false);
  expect(apiRequests).toEqual([]);
});
