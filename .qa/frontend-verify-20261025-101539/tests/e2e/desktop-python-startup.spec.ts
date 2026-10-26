import { expect, test } from '@playwright/test';

test('isolated Python desktop server renders the built React app without startup failures', async ({ page }) => {
  const pageErrors: string[] = [];
  const assetFailures: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('requestfailed', request => {
    if (['document', 'script', 'stylesheet', 'font'].includes(request.resourceType())) {
      assetFailures.push(`${request.resourceType()} ${request.url()}: ${request.failure()?.errorText ?? 'request failed'}`);
    }
  });
  page.on('response', response => {
    if (['document', 'script', 'stylesheet', 'font'].includes(response.request().resourceType()) && response.status() >= 400) {
      assetFailures.push(`${response.request().resourceType()} ${response.url()}: HTTP ${response.status()}`);
    }
  });

  const response = await page.goto('/', { waitUntil: 'networkidle' });
  expect(response?.status()).toBe(200);
  await expect(page.locator('#content .reference-screen-motion').first()).toBeVisible();
  await expect(page.locator('#openagent-startup-placeholder')).toHaveCount(0);
  expect(pageErrors).toEqual([]);
  expect(assetFailures).toEqual([]);
});
