import { expect, test } from '@playwright/test';

test('Communications mobile demo matches the approved inbox controls in both themes', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));

  for (const theme of ['light', 'dark']) {
    await page.goto('/?view=communications');
    await page.evaluate(value => { localStorage.setItem('oa-second-theme', value); document.documentElement.dataset.theme = value; }, theme);
    const inbox = page.locator('.communications-demo-inbox');
    await expect(inbox.getByRole('heading', { name: 'Inbox · example mailbox' })).toBeVisible();
    await expect(inbox.locator('.communications-demo-copy').nth(0).locator('small')).toHaveText('Maya Chen · Please review the new acceptance items before Friday.');
    await expect(inbox.locator('.communications-demo-copy').nth(1).locator('small')).toHaveText('Daniel Park · The file permissions need one more pass.');
    await expect(inbox.locator('.communications-demo-status')).toHaveCount(2);
    await expect(inbox.locator('.communications-demo-status').first()).toBeVisible();
    await expect(inbox.locator('.communications-demo-row-action').first()).toBeVisible();

    const compose = await inbox.getByRole('button', { name: 'Compose message' }).boundingBox();
    const sync = await inbox.getByRole('button', { name: 'Sync demo inbox' }).boundingBox();
    const search = await inbox.locator('.communications-inbox-search').boundingBox();
    expect(compose?.width).toBeGreaterThan(156);
    expect(sync?.width).toBeGreaterThan(147);
    expect(search?.width).toBeGreaterThan(190);

    await inbox.getByRole('textbox', { name: 'Search inbox messages' }).fill('release');
    await expect(inbox.locator('.communications-demo-row')).toHaveCount(1);
    await inbox.getByRole('textbox', { name: 'Search inbox messages' }).clear();
  }
  expect(apiRequests).toEqual([]);
});
