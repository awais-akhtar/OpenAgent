import { expect, test } from '@playwright/test';

test('Voice demo keeps recording and transcription states contextual without a generic trace', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('pageerror', error => console.error('Voice page error:', error.stack ?? error.message));
  page.on('console', message => { if (message.type() === 'error') console.error('Voice browser error:', message.text()); });
  page.on('requestfailed', request => console.error('Voice request failed:', request.url(), request.failure()?.errorText));
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=voice');
  // Advanced workspaces are lazy-loaded. Wait for the actual route content,
  // rather than racing the 5s default assertion timeout while Vite transforms it.
  const workspace = page.locator('.p-workspace--voice');
  await expect(workspace).toBeVisible({ timeout: 15_000 });
  await expect(workspace.getByRole('heading', { name: 'Capture your next idea.' })).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('.p-voice-hero')).toBeVisible();
  await expect(page.locator('.trace-mount')).toHaveCount(0);
  await page.getByRole('button', { name: 'Start recording' }).click();
  await expect(page.locator('.live-recording-indicator')).toHaveText('Simulated recording in progress');
  await expect(page.getByRole('button', { name: 'Stop & save recording' })).toBeVisible();
  await page.getByRole('button', { name: 'Stop & save recording' }).click();
  await expect(page.getByText('Saved a demo recording and transcript.', { exact: true })).toBeVisible();
  await expect(page.locator('.voice-demo-recordings')).toContainText(/New recording ·/);
  await page.getByRole('button', { name: 'More actions for Product planning sync' }).click();
  const actions = page.getByRole('dialog', { name: 'Product planning sync' });
  await actions.getByRole('button', { name: 'Run / test' }).click();
  await expect(page.getByRole('status', { name: 'Local workflow running' })).toBeVisible();
  await expect(page.getByRole('progressbar', { name: 'Simulated recording workflow progress' })).toBeVisible();
  await expect(page.locator('.trace-mount')).toHaveCount(0);
  await page.getByRole('button', { name: 'Cancel local run' }).click();
  await expect(page.getByRole('status', { name: 'Local workflow running' })).not.toBeVisible();
  await expect(page.getByRole('button', { name: 'Open Product planning sync details' }).locator('..').getByText('Cancelled')).toBeVisible();
  expect(apiRequests).toEqual([]);
});
