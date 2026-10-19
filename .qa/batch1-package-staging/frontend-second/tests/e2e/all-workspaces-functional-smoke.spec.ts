import { expect, test } from '@playwright/test';

const workspaces = [
  'home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'models',
  'knowledge', 'voice', 'assistant', 'communications', 'security', 'developer', 'engine', 'science',
  'twin', 'vault', 'scraper', 'finetune', 'settings', 'tools',
];

test('every workspace tab is operable in isolated demo mode without browser or backend errors', async ({ page }) => {
  const pageErrors: string[] = [];
  const failedRequests: string[] = [];
  const apiRequests: string[] = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  page.on('requestfailed', request => failedRequests.push(`${request.method()} ${request.url()}`));
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));

  for (const workspace of workspaces) {
    await page.goto(`/?view=${workspace}`);
    await expect(page.locator('#content')).toBeVisible();
    await expect(page.locator('#content .reference-screen-motion, #content .codeq-root, #content .split-layout').first()).toBeVisible();

    const tabs = page.getByRole('tab');
    const count = await tabs.count();
    for (let index = 0; index < count; index += 1) {
      const tab = tabs.nth(index);
      const name = (await tab.innerText()).trim();
      await tab.click();
      await expect(tab, `${workspace} tab “${name}”`).toHaveAttribute('aria-selected', 'true');
      await expect(page.locator('#content')).toBeVisible();
    }
  }

  expect(pageErrors).toEqual([]);
  expect(failedRequests).toEqual([]);
  expect(apiRequests).toEqual([]);
});
