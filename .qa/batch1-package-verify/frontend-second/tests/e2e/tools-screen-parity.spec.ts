import { expect, test } from '@playwright/test';

test('Home automation shortcut uses the approved bolt icon and navigates to Automations', async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=home');
  const automationShortcut = page.locator('.shortcut-card[data-nav="automations"]');
  await expect(automationShortcut).toBeVisible();
  await expect(automationShortcut.locator('svg.lucide-zap')).toHaveCount(1);
  await automationShortcut.click();
  await expect(page).toHaveURL(/view=automations/);
});

test('Home project picker and attachment affordance are complete in demo mode without backend calls', async ({ page }, testInfo) => {
  await page.addInitScript(() => window.localStorage.setItem('oa-second-data-mode', 'demo'));
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=home');
  await page.getByRole('button', { name: 'View projects' }).click();
  let picker = page.getByRole('dialog', { name: 'Choose workspace' });
  await expect(picker).toBeVisible();
  await testInfo.attach('home-project-picker-demo', { body: await page.screenshot(), contentType: 'image/png' });
  await page.keyboard.press('Escape');
  await expect(picker).toBeHidden();
  await page.getByRole('button', { name: 'View projects' }).click();
  picker = page.getByRole('dialog', { name: 'Choose workspace' });
  const researchProject = picker.getByRole('button', { name: /AI Research Assistant Python/ });
  await researchProject.click();
  await expect(picker).toBeHidden();
  await expect(page.getByRole('status').filter({ hasText: 'Live project scope is unchanged' })).toBeVisible();
  await page.getByRole('button', { name: 'Attach a file' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'File attachment is a design-only control.' })).toBeVisible();

  await page.getByRole('button', { name: 'View projects' }).click();
  picker = page.getByRole('dialog', { name: 'Choose workspace' });
  await picker.getByRole('button', { name: 'New project (preview)' }).click();
  const createDialog = page.getByRole('dialog', { name: 'Create a project' });
  await testInfo.attach('home-create-project-demo', { body: await page.screenshot(), contentType: 'image/png' });
  await createDialog.getByLabel('Project name').fill('Milestone A local sample');
  await createDialog.getByRole('button', { name: 'Create preview' }).click();
  picker = page.getByRole('dialog', { name: 'Choose workspace' });
  const newProject = picker.getByRole('button', { name: /Milestone A local sample Demo project/ });
  await expect(newProject).toBeVisible();
  await newProject.click();
  expect(apiRequests).toEqual([]);
});

test('Home project and attachment controls state their live-mode limitations', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=home');
  await page.getByRole('button', { name: 'View projects' }).click();
  const dialog = page.getByRole('dialog', { name: 'Choose workspace' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/does not provide a confirmed project-switch operation/i)).toBeVisible();
  await expect(dialog.getByRole('button', { name: /New project/ })).toHaveCount(0);
  await dialog.getByRole('button', { name: 'Close' }).click();
  await expect(page.getByRole('button', { name: 'Attach a file unavailable' })).toBeDisabled();
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
      await expect(page.getByText('22 existing feature views')).toBeVisible();
      await expect(page.getByRole('button', { name: /Personal assistant Tasks, goals and priorities/ })).toBeVisible();
      await expect(page.locator('.p-tool-link svg.lucide-arrow-right')).toHaveCount(22);
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
