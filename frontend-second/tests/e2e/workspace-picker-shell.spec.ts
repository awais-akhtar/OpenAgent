import { expect, test } from '@playwright/test';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

test('shell project picker is shared, persistent and Demo-local on desktop', async ({ page }) => {
  await page.addInitScript(() => window.localStorage.setItem('oa-second-data-mode', 'demo'));
  const apiRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });

  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(viewport);
    for (const theme of ['light', 'dark'] as const) {
      await page.goto('/?view=home');
      if (await page.locator('html').getAttribute('data-theme') !== theme) {
        await page.getByRole('button', { name: theme === 'dark' ? 'Switch to dark theme' : 'Switch to light theme' }).click();
      }
      const sidebarPicker = page.getByRole('button', { name: 'Choose project, current Personal Workspace' }).first();
      await expect(sidebarPicker).toBeEnabled();
      await sidebarPicker.click();
      let dialog = page.getByRole('dialog', { name: 'Choose workspace' });
      await expect(dialog).toBeVisible();
      const closePicker = dialog.getByRole('button', { name: 'Close project picker' });
      const createProject = dialog.getByRole('button', { name: 'New project (preview)' });
      await createProject.focus();
      await page.keyboard.press('Tab');
      await expect(closePicker).toBeFocused();
      await page.keyboard.press('Shift+Tab');
      await expect(createProject).toBeFocused();
      await page.keyboard.press('Escape');
      await expect(dialog).toBeHidden();
      await expect(sidebarPicker).toBeFocused();
      await sidebarPicker.click();
      dialog = page.getByRole('dialog', { name: 'Choose workspace' });
      await dialog.getByRole('button', { name: /AI Research Assistant Python/ }).click();
      await expect(page.locator('.breadcrumbs')).toContainText('AI Research Assistant');
      await expect(page.getByRole('button', { name: 'Choose project, current AI Research Assistant' })).toHaveCount(2);
      await expect(page.getByRole('button', { name: 'Choose project, current AI Research Assistant' }).first()).toBeVisible();
      const activeProject = viewport.width === 1440 && theme === 'light' ? 'Shell picker project' : 'AI Research Assistant';

      if (viewport.width === 1440 && theme === 'light') {
        await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'CodeQ', exact: true }).click();
        await page.getByRole('button', { name: 'Choose project, current AI Research Assistant' }).first().click();
        await page.getByRole('dialog', { name: 'Choose workspace' }).getByRole('button', { name: 'New project (preview)' }).click();
        await expect(page).toHaveURL(/view=home/);
        const createDialog = page.getByRole('dialog', { name: 'Create a project' });
        await createDialog.getByLabel('Project name').fill('Shell picker project');
        await createDialog.getByRole('button', { name: 'Create preview' }).click();
        await expect(page.locator('.breadcrumbs')).toContainText('Shell picker project');
      }

      await page.getByRole('navigation', { name: 'Main' }).getByRole('button', { name: 'CodeQ', exact: true }).click();
      await expect(page.locator('.breadcrumbs')).toContainText(activeProject);
      await page.reload();
      await expect(page.getByRole('button', { name: `Choose project, current ${activeProject}` })).toHaveCount(2);

      await page.getByRole('button', { name: `Choose project, current ${activeProject}` }).last().click();
      await expect(page.getByRole('dialog', { name: 'Choose workspace' })).toBeVisible();
      await page.getByRole('dialog', { name: 'Choose workspace' }).getByRole('button', { name: 'Personal Workspace' }).click();
      await expect(page.locator('.breadcrumbs')).toContainText('Personal Workspace');

    }
  }

  expect(apiRequests).toEqual([]);
});

test('Live shell project controls stay unavailable and do not claim a local Demo selection is active', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=home');
  await expect(page.getByRole('button', { name: 'Current workspace, switching unavailable' }).first()).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Current workspace, switching unavailable' }).last()).toBeDisabled();
  await expect(page.locator('.breadcrumbs')).toContainText('Personal Workspace');
});

test('shell project picker matches the approved chooser region on desktop', async ({ page }, testInfo) => {
  testInfo.setTimeout(60_000);
  await page.addInitScript(() => window.localStorage.setItem('oa-second-data-mode', 'demo'));
  const canonicalUrl = pathToFileURL(path.resolve('reference/frontend-final-approved.html')).href;
  const evidenceDir = path.resolve('evidence/project-picker-20261018');
  await fs.mkdir(evidenceDir, { recursive: true });
  const measurements: Array<Record<string, unknown>> = [];

  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(viewport);
    for (const theme of ['light', 'dark'] as const) {
      const state = `${viewport.width}x${viewport.height}-${theme}`;
      await test.step(`capture approved and React project chooser at ${state}`, async () => {
      const canonical = await page.context().newPage();
      await canonical.setViewportSize(viewport);
      await canonical.goto(canonicalUrl, { waitUntil: 'domcontentloaded', timeout: 10_000 });
      await canonical.evaluate(() => { location.hash = 'home'; });
      await canonical.locator('.home-grid').waitFor({ state: 'visible' });
      if (await canonical.locator('html').getAttribute('data-theme') !== theme) {
        await canonical.getByRole('button', { name: theme === 'dark' ? 'Switch to dark theme' : 'Switch to light theme' }).click();
      }
      await canonical.locator('.sidebar-project').click();
      const canonicalDialog = canonical.locator('.palette[role="dialog"]');
      await expect(canonicalDialog).toBeVisible();

      await page.goto('/?view=home', { waitUntil: 'domcontentloaded', timeout: 10_000 });
      if (await page.locator('html').getAttribute('data-theme') !== theme) {
        await page.getByRole('button', { name: theme === 'dark' ? 'Switch to dark theme' : 'Switch to light theme' }).click();
      }
      await page.getByRole('button', { name: 'Choose project, current Personal Workspace' }).first().click();
      const reactDialog = page.locator('.overlay-root .palette[role="dialog"]');
      await expect(reactDialog).toBeVisible();
      // Opening the React dialog starts its deterministic fixture query. Wait
      // for content, not merely the shell, before taking the parity capture.
      await expect(reactDialog.getByRole('button', { name: /E-commerce Platform/ })).toBeVisible({ timeout: 10_000 });
      await expect(reactDialog.getByRole('button', { name: /Data Analysis Tool/ })).toBeVisible({ timeout: 10_000 });

      await Promise.all([
        canonicalDialog.evaluate(element => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); }),
        reactDialog.evaluate(element => { if (document.activeElement instanceof HTMLElement) document.activeElement.blur(); }),
      ]);
      await Promise.all([
        canonical.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}' }),
        page.addStyleTag({ content: '*,*::before,*::after{animation:none!important;transition:none!important;caret-color:transparent!important}' }),
      ]);
      await Promise.all([canonical.waitForTimeout(220), page.waitForTimeout(220)]);
      const canonicalBounds = await canonicalDialog.boundingBox();
      const reactBounds = await reactDialog.boundingBox();
      if (!canonicalBounds || !reactBounds) throw new Error('The project chooser did not have measurable desktop geometry.');
      const canonicalPng = await canonical.screenshot({ clip: canonicalBounds, animations: 'disabled' });
      const reactPng = await page.screenshot({ clip: reactBounds, animations: 'disabled' });
      const canonicalName = `workspace-picker-${state}-approved.png`;
      const reactName = `workspace-picker-${state}-react.png`;
      await testInfo.attach(canonicalName, { body: canonicalPng, contentType: 'image/png' });
      await testInfo.attach(reactName, { body: reactPng, contentType: 'image/png' });
      await Promise.all([
        fs.writeFile(path.join(evidenceDir, canonicalName), canonicalPng),
        fs.writeFile(path.join(evidenceDir, reactName), reactPng),
      ]);
      const exactPercent = await page.evaluate(async ({ left, right }) => {
        const decodeTimeout = new Promise<never>((_, reject) => setTimeout(() => reject(new Error('Timed out decoding project chooser screenshots in the browser canvas.')), 10_000));
        const load = async (src: string) => {
          const image = new Image(); image.src = src; await image.decode(); return image;
        };
        const [a, b] = await Promise.race([Promise.all([load(left), load(right)]), decodeTimeout]);
        if (a.width !== b.width || a.height !== b.height) throw new Error('Chooser region captures have different dimensions.');
        const canvas = document.createElement('canvas'); canvas.width = a.width; canvas.height = a.height;
        const context = canvas.getContext('2d', { willReadFrequently: true });
        if (!context) throw new Error('Canvas is unavailable for chooser region scoring.');
        context.drawImage(a, 0, 0); const x = context.getImageData(0, 0, canvas.width, canvas.height).data;
        context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(b, 0, 0); const y = context.getImageData(0, 0, canvas.width, canvas.height).data;
        let equal = 0; for (let i = 0; i < x.length; i += 4) if (x[i] === y[i] && x[i + 1] === y[i + 1] && x[i + 2] === y[i + 2] && x[i + 3] === y[i + 3]) equal++;
        return Number((100 * equal / (canvas.width * canvas.height)).toFixed(2));
      }, { left: `data:image/png;base64,${canonicalPng.toString('base64')}`, right: `data:image/png;base64,${reactPng.toString('base64')}` });
      expect(exactPercent, `Chooser dialog-region strict RGBA score regressed at ${viewport.width}x${viewport.height}, ${theme}.`).toBeGreaterThanOrEqual(99);
      await testInfo.attach(`workspace-picker-${state}-strict-rgba.json`, { body: Buffer.from(JSON.stringify({ state, canonicalBounds, reactBounds, dialogRegionStrictRgbaPercent: exactPercent }, null, 2)), contentType: 'application/json' });
      expect(reactBounds, `Chooser geometry differs at ${viewport.width}x${viewport.height}, ${theme}.`).toEqual(canonicalBounds);
      measurements.push({ viewport, theme, canonicalBounds, reactBounds, dialogGeometryDelta: canonicalBounds && reactBounds ? { x: Number((reactBounds.x - canonicalBounds.x).toFixed(2)), y: Number((reactBounds.y - canonicalBounds.y).toFixed(2)), width: Number((reactBounds.width - canonicalBounds.width).toFixed(2)), height: Number((reactBounds.height - canonicalBounds.height).toFixed(2)) } : null, dialogRegionStrictRgbaPercent: exactPercent });
      await canonical.close();
      await page.keyboard.press('Escape');
      });
    }
  }

  const report = { canonical: 'reference/frontend-final-approved.html', canonicalSha256: 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1', deviceScaleFactor: 1, measurements };
  const json = JSON.stringify(report, null, 2);
  await fs.writeFile(path.join(evidenceDir, 'workspace-picker-region-measurements.json'), json);
  await testInfo.attach('workspace-picker-region-measurements.json', { body: Buffer.from(json), contentType: 'application/json' });
});
