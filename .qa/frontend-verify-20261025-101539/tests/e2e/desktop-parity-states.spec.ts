import { expect, test } from '@playwright/test';

test('Automations recent run actions open the approved demo detail and run-history tab locally', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=automations');
  await expect(page.locator('.automations-shell')).toBeVisible();
  await page.getByRole('button', { name: 'View output', exact: true }).first().click();
  const dialog = page.getByRole('dialog', { name: 'Run details' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText('Example run history from your offline workspace.')).toBeVisible();
  await expect(dialog.getByText('Today, 09:01')).toBeVisible();
  await expect(dialog.getByText('8 sources reviewed')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toHaveCount(0);
  await page.getByRole('button', { name: 'View all runs' }).click();
  await expect(page.getByRole('tab', { name: 'Runs' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText('Today, 09:01')).toBeVisible();
  await expect(page.getByText('Sunday, 09:00')).toBeVisible();
  await page.getByRole('button', { name: 'View error' }).click();
  await expect(page.getByRole('dialog', { name: 'Run details' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog', { name: 'Run details' })).toHaveCount(0);
  expect(apiRequests).toEqual([]);
});

test('CodeQ keeps its real editor and exposes local preview, terminal, diff, model and trace states', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=codeq');
  // Monaco is intentionally loaded as a split chunk; wait for the real editor
  // to mount rather than treating its editable source fallback as readiness.
  await expect(page.locator('.monaco-editor')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('Preview output', { exact: true })).toHaveClass(/tag orange/);
  await expect(page.locator('.codeq-model-select')).toHaveClass(/tag/);
  await expect(page.locator('.codeq-bottom .codeq-panel-head .tag')).toHaveText('Fixture output');
  await expect(page.locator('.codeq-desktop-trace')).toHaveCount(0);
  await page.getByRole('tab', { name: 'Preview' }).click();
  await expect(page.getByRole('heading', { name: 'Everyday Trainer' }).last()).toBeVisible();
  await page.getByRole('tab', { name: 'Terminal' }).click();
  await expect(page.getByText('Ready — connect a real terminal to run commands.')).toBeVisible();
  const dockPanels = await page.locator('.codeq-bottom .codeq-panel').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().width));
  expect(dockPanels).toHaveLength(2);
  expect(Math.abs(dockPanels[0] - dockPanels[1])).toBeLessThanOrEqual(2);
  await page.getByRole('tab', { name: 'Git' }).click();
  await page.getByRole('button', { name: 'Review proposed diff' }).first().click();
  await expect(page.getByText('openQuickView', { exact: false }).first()).toBeVisible();
  await page.getByRole('button', { name: 'GPT-4o' }).click();
  await page.getByRole('option', { name: 'Llama 3.1 8B' }).click();
  await expect(page.getByRole('button', { name: 'Llama 3.1 8B' })).toBeVisible();
  await page.getByRole('button', { name: 'Preview run', exact: true }).click();
  await expect(page.locator('.codeq-desktop-trace')).toBeVisible();
  const trace = page.locator('.codeq-desktop-trace');
  await expect(trace.locator('.trace-head-state')).toHaveText('Running');
  await expect(trace.locator('.trace-collapsed')).toBeVisible();
  await expect(page.locator('.monaco-editor')).toBeVisible();
  await trace.getByRole('button', { name: 'Expand' }).click();
  await expect(trace.getByText('PLAYING LOCAL EXAMPLE')).toBeVisible();
  const expandedDock = await page.locator('.codeq-bottom').boundingBox();
  expect(expandedDock?.y).toBeGreaterThan(600);
  expect(expandedDock?.height).toBeGreaterThan(100);
  await trace.getByRole('button', { name: 'Play example' }).click();
  await expect(trace.locator('.trace-head-state')).toHaveText('Running');
  await expect(trace.locator('.trace-collapsed')).toBeVisible();
  await trace.getByRole('button', { name: 'Expand' }).click();
  await expect(page.locator('.codeq-desktop-trace').getByText('PLAYING LOCAL EXAMPLE')).toBeVisible();
  await page.locator('.codeq-desktop-trace').getByRole('button', { name: 'Details' }).click();
  await expect(page.getByRole('dialog', { name: 'CodeQ execution trace' })).toBeVisible();
  await page.getByRole('dialog').getByRole('button', { name: 'Close', exact: true }).click();
  expect(apiRequests).toEqual([]);
});

test('Models fixture cards retain their approved icon and tone per model', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=models');
  const rows = page.locator('.models-shell .model-row');
  await expect(rows).toHaveCount(4);
  await expect(rows.nth(0).locator('.square-icon')).toHaveClass(/\bblue\b/);
  await expect(rows.nth(1).locator('.square-icon')).toHaveClass(/\borange\b/);
  await expect(rows.nth(2).locator('.square-icon')).toHaveClass(/\bviolet\b/);
  await expect(rows.nth(3).locator('.square-icon')).toHaveClass(/\bgreen\b/);
  await expect(rows.nth(0).locator('.square-icon svg')).toHaveClass(/lucide-bot/);
  await expect(rows.nth(1).locator('.square-icon svg')).toHaveClass(/lucide-brain/);
  await expect(rows.nth(2).locator('.square-icon svg')).toHaveClass(/lucide-sparkles/);
  await expect(rows.nth(3).locator('.square-icon svg')).toHaveClass(/lucide-cpu/);
  expect(apiRequests).toEqual([]);
});

test('Models Setup needed excludes configured provider records', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=models');
  await page.getByRole('tab', { name: 'Setup needed', exact: true }).click();
  const rows = page.locator('.models-shell .model-row');
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(0)).toContainText('Gemini Pro');
  await expect(rows.nth(1)).toContainText('Llama local');
  const setupNeededText = (await rows.allTextContents()).join(' ');
  expect(setupNeededText).not.toContain('GPT-4o');
  expect(setupNeededText).not.toContain('Claude Sonnet');
  expect(apiRequests).toEqual([]);
});

test('Voice transcript and settings tabs match the approved demo workflow and save locally', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=voice');
  await page.getByRole('tab', { name: 'Transcripts', exact: true }).click();
  await expect(page.getByText('Capture and understand ideas')).toBeVisible();
  await expect(page.getByText('We should make this workflow easy to understand from the first click.')).toBeVisible();
  await expect(page.getByText('The agent timeline needs to show real status, not just a spinner.')).toBeVisible();
  await page.getByRole('button', { name: 'Edit transcript' }).click();
  const editor = page.getByRole('dialog', { name: 'Edit demo transcript' });
  await expect(editor).toBeVisible();
  await editor.locator('textarea').fill('Saved deterministic transcript text.');
  await editor.getByRole('button', { name: 'Save transcript' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Saved' })).toBeVisible();
  await page.getByRole('tab', { name: 'Voice settings', exact: true }).click();
  await expect(page.getByText('Capture and understand ideas')).toBeVisible();
  await page.getByLabel('Transcription language').selectOption({ label: 'Urdu' });
  await page.getByRole('button', { name: 'Save preferences' }).click();
  await expect(page.getByRole('status').filter({ hasText: 'Preferences saved locally.' })).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Files links and notes retain the approved document workspace and demo Add is a local file fixture', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=files');
  await expect(page.locator('.files-shell')).toBeVisible();
  for (const tab of ['Links', 'Notes', 'Uploaded files']) {
    await page.getByRole('tab', { name: tab, exact: true }).click();
    await expect(page.locator('.files-record-list').getByRole('heading', { name: 'Product requirements specification' })).toBeVisible();
    await expect(page.locator('.files-record-list').getByText('Product review notes', { exact: true })).toBeVisible();
  }
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add a workspace file' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText('The selected file is represented in the offline demo. The contents are not sent anywhere.')).toBeVisible();
  const submit = dialog.getByRole('button', { name: 'Add to demo' });
  await expect(submit).toBeDisabled();
  await dialog.getByLabel('File').setInputFiles({ name: 'acceptance-notes.md', mimeType: 'text/markdown', buffer: Buffer.from('# Local demo file') });
  await dialog.getByLabel('Short description').fill('Created by the desktop parity test.');
  await expect(submit).toBeEnabled();
  await submit.click();
  await expect(page.locator('.files-record-list').getByRole('heading', { name: 'acceptance-notes.md' })).toBeVisible();
  await expect(page.getByRole('dialog')).toHaveCount(0);
  expect(apiRequests).toEqual([]);
});
