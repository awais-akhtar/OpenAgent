import { expect, test } from '@playwright/test';

const viewports = [{ width: 1440, height: 900 }, { width: 1280, height: 800 }];
const themes = ['light', 'dark'] as const;

for (const viewport of viewports) {
  for (const theme of themes) {
    test(`Demo workspace rails show actions actually recorded locally (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
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
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await page.getByLabel('Preference name').fill('Rail activity example');
      await page.getByLabel('Description').fill('Synthetic local action');
      await page.getByRole('button', { name: 'Save preference' }).click();
      const contextRail = page.locator('.advanced-demo-rail');
      await expect(contextRail.getByRole('heading', { name: 'Latest events' })).toBeVisible();
      await expect(contextRail).toContainText('Added personal context example');
      await expect(contextRail).toContainText('preferences');

      await page.goto('/?view=security');
      await page.getByRole('tab', { name: 'Access' }).click();
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      await page.getByLabel('Name').fill('Local access example');
      await page.getByRole('button', { name: 'Save locally' }).click();
      const securityRail = page.locator('.advanced-demo-rail');
      await expect(securityRail).toContainText('Added security policy');
      await expect(securityRail).toContainText('Saved locally · demo');

      await page.goto('/?view=voice');
      await page.getByRole('button', { name: 'Start recording' }).click();
      await page.getByRole('button', { name: 'Stop & save recording' }).click();
      const voiceRail = page.locator('.voice-rail');
      await expect(voiceRail).toContainText('Transcribed example recording');
      await expect(voiceRail).toContainText('Simulated locally · demo');
      expect(apiRequests).toEqual([]);
    });
  }
}
