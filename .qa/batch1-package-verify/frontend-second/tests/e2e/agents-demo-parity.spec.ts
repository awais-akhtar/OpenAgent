import { expect, test } from '@playwright/test';

test('Agents preview activity updates the local execution trace and completes without backend calls', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=agents');

  await page.getByRole('button', { name: 'Preview activity' }).click();
  const trace = page.locator('.trace-mount.trace-agents');
  const stages = page.locator('.steps-line .step');
  await expect(page.locator('.live-inline')).toHaveCount(1);
  await expect(page.locator('.live-inline')).toContainText('Running · demo');
  await expect(stages.nth(0)).toContainText('In progress');
  await expect(stages.nth(1)).toContainText('Pending');
  await expect(stages.nth(0)).toHaveClass(/active/);
  await expect(trace).toContainText('Multi-agent activity started in the sample job engine');
  await expect(trace.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '0');
  await expect(trace.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25', { timeout: 2_000 });
  await trace.getByRole('button', { name: 'Pause' }).click();
  await expect(page.locator('.live-inline')).toContainText('Paused · demo');
  await expect(page.locator('.steps-line .step.active')).toContainText('Paused');
  await trace.getByRole('button', { name: 'Resume' }).click();
  await expect(trace).toContainText('Completed', { timeout: 6_000 });
  await expect(trace.locator('.trace-log-stream')).toContainText('Completed');
  await expect(trace.locator('.trace-log-stream')).not.toContainText('Multi-agent activity started in the sample job engine');
  await expect(page.locator('.live-inline')).toHaveCount(1);
  await expect(page.locator('.live-inline')).toContainText('Completed · demo');
  await expect(page.locator('.live-inline')).toContainText('Coordinator summary is ready to inspect');
  await expect(page.locator('.live-inline')).not.toContainText('working');
  await expect(page.locator('.live-inline')).toHaveAttribute('aria-label', 'Agents demo run completed');
  await expect(stages).toHaveCount(4);
  for (let i = 0; i < 4; i += 1) {
    await expect(stages.nth(i)).toContainText('Complete');
    await expect(stages.nth(i)).toHaveClass(/complete/);
  }
  await page.getByRole('button', { name: 'Preview activity' }).click();
  await expect(page.locator('.live-inline')).toContainText('Running · demo');
  await trace.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(page.locator('.live-inline')).toContainText('Cancelled · demo');
  await expect(page.locator('.live-inline')).not.toContainText('working');
  await expect(page.locator('.live-inline')).toHaveCount(1);
  const cancelledStage = page.locator('.steps-line .step.cancelled');
  await expect(cancelledStage).toHaveCount(1);
  await expect(cancelledStage).toContainText('Cancelled');
  await expect(page.locator('.steps-line .step').filter({ hasText: 'Not run' }).first()).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Agents demo mirrors the approved illustrative session and keeps interactions local on desktop', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=agents');

  await expect(page.getByRole('heading', { name: 'Product launch review' })).toBeVisible();
  await expect(page.locator('.agent-lane')).toHaveCount(3);
  await page.getByRole('button', { name: 'Agent directory' }).click();
  await expect(page.getByRole('heading', { name: 'AI agents' })).toBeVisible();
  await page.getByLabel('Search agents').fill('Researcher');
  await expect(page.getByRole('heading', { name: 'Researcher' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Coordinator' })).toHaveCount(0);
  await page.getByLabel('Search agents').clear();
  await page.getByRole('button', { name: 'New agent' }).click();
  await page.getByLabel('Agent name').fill('Planner');
  await page.getByLabel('Role').fill('Organises project milestones.');
  await page.getByRole('button', { name: 'Create agent' }).click();
  await expect(page.getByRole('heading', { name: 'Planner' })).toBeVisible();
  await page.getByRole('button', { name: 'View example session' }).click();
  await expect(page.getByRole('heading', { name: 'Product launch review' })).toBeVisible();

  await page.locator('.agent-context').getByRole('button', { name: 'Add', exact: true }).click();
  await page.getByLabel('Task name').fill('Prepare accessibility review');
  await page.getByRole('button', { name: 'Add task' }).click();
  await expect(page.getByText('Prepare accessibility review')).toBeVisible();

  await page.getByRole('textbox', { name: 'Message Coordinator' }).fill('Please review the launch checklist.');
  await page.getByRole('button', { name: 'Send message to Coordinator' }).click();
  await expect(page.getByText('Please review the launch checklist.')).toBeVisible();
  expect(apiRequests).toEqual([]);
});
