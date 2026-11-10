import { expect, test } from '@playwright/test';

test('Home shows measured local telemetry without workspace shortcut lists', async ({ page }) => {
  await page.goto('/?view=home');
  await expect(page.getByRole('meter', { name: 'CPU utilization' })).toBeVisible();
  await expect(page.getByRole('meter', { name: 'CPU utilization' })).toBeVisible();
  await expect(page.getByRole('meter', { name: 'Memory utilization' })).toBeVisible();
  await expect(page.getByRole('meter', { name: 'Disk utilization' })).toBeVisible();
  await expect(page.getByText('Projects', { exact: true })).toHaveCount(0);
  await expect(page.getByText('Recent activity', { exact: true })).toHaveCount(0);
  await page.getByRole('button', { name: '5 min' }).click();
  await expect(page.getByRole('button', { name: '5 min' })).toHaveAttribute('aria-pressed', 'true');
});
test('All Tools desktop labels, group hierarchy and card affordances match the approved screen', async ({ page }, testInfo) => {
  await page.addInitScript(() => window.localStorage.setItem('oa-second-data-mode', 'demo'));
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(viewport);
    for (const theme of ['light', 'dark'] as const) {
      await page.goto('/?view=tools');
      await page.locator('#content .p-hub').waitFor();
      const currentTheme = await page.locator('html').getAttribute('data-theme');
      if (currentTheme !== theme) {
        await page.getByRole('button', { name: theme === 'dark' ? 'Switch to dark theme' : 'Switch to light theme' }).click();
      }
      await expect(page.locator('html')).toHaveAttribute('data-theme', theme);
      // 26 approved views plus the four added once their backends were
      // registered and scope-mapped: Personal Context (fact provenance),
      // Advanced AI, Voice conversations and Model serving. The figure stays
      // explicit so an unapproved addition still fails this check.
      await expect(page.getByText('30 existing feature views')).toBeVisible();
      await expect(page.getByRole('button', { name: /Personal assistant Tasks, goals and priorities/ })).toBeVisible();
      await expect(page.locator('.p-tool-link svg.lucide-arrow-right')).toHaveCount(30);
      const sectionWeight = await page.locator('.p-section-heading h2').first().evaluate(element => getComputedStyle(element).fontWeight);
      expect(Number(sectionWeight), 'category heading weight').toBeGreaterThanOrEqual(600);
      await page.mouse.move(0, 0);
      await page.waitForTimeout(220);
      const screenshotPath = testInfo.outputPath(`all-tools-${viewport.width}-${theme}.png`);
      await page.screenshot({ path: screenshotPath });
      await testInfo.attach(`all-tools-${viewport.width}-${theme}`, { path: screenshotPath, contentType: 'image/png' });
    }
  }
});
