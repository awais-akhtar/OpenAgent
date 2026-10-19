import { expect, test } from '@playwright/test';

const executionRoutes = ['agents', 'chat', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'voice'];

async function setMode(page: import('@playwright/test').Page, mode: 'demo' | 'live') {
  await page.addInitScript(value => localStorage.setItem('oa-second-data-mode', value), mode);
}

test('routine workspace visits do not mount sample execution timelines', async ({ page }) => {
  await setMode(page, 'demo');
  for (const route of executionRoutes) {
    await page.goto(`/?view=${route}`);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0, { timeout: 5000 });
  }
  await page.goto('/?view=settings');
  await expect(page.locator('.live-top-button')).toHaveCount(0);
  await page.goto('/?view=voice');
  await expect(page.locator('.live-top-button')).toHaveCount(0);
});

test('CodeQ starts an explicitly labelled local trace only after Preview run and makes no API requests', async ({ page }) => {
  await setMode(page, 'demo');
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.goto('/?view=codeq');
  await expect(page.locator('#content .trace-mount')).toHaveCount(0);
  await page.getByRole('button', { name: 'Preview run' }).click();
  const trace = page.locator('#content .trace-mount.trace-codeq');
  await expect(trace).toBeVisible();
  await expect(trace.locator('.trace-head-state')).toHaveText('Running');
  await expect(trace.getByText('DEMO DATA', { exact: false })).toBeVisible();
  await expect(trace.locator('.trace-collapsed')).toBeVisible();
  await expect(page.locator('.codeq-root .monaco-editor, .codeq-root .code-lines').first()).toBeVisible();
  await trace.getByRole('button', { name: 'Expand' }).click();
  await expect(trace.locator('.trace-layout')).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Research starts a simulated inline trace only after its explicit preview action', async ({ page }) => {
  await setMode(page, 'demo');
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.goto('/?view=research');
  await expect(page.locator('#content .trace-mount')).toHaveCount(0);
  await page.getByRole('button', { name: /Preview running state · simulated/ }).click();
  const trace = page.locator('#content .trace-mount.trace-research');
  await expect(trace).toBeVisible();
  await expect(trace).toContainText('DEMO DATA');
  await expect(trace.locator('.trace-collapsed')).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Executive summary' })).toBeVisible();
  await trace.getByRole('button', { name: 'Expand' }).click();
  await expect(trace.getByRole('progressbar', { name: 'Simulated activity progress' })).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Agents show the inline simulated workflow only after Preview activity', async ({ page }) => {
  await setMode(page, 'demo');
  await page.goto('/?view=agents');
  await expect(page.locator('#content .trace-mount')).toHaveCount(0);
  await page.getByRole('button', { name: 'Preview activity' }).click();
  const trace = page.locator('#content .trace-mount.trace-agents');
  await expect(trace).toBeVisible();
  await expect(trace.getByText('PLAYING LOCAL EXAMPLE')).toBeVisible();
});

test('Live mode does not render synthetic trace content or the demo Activity center', async ({ page }) => {
  await page.route('**/api/**', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Offline Live-mode test fixture' }) }));
  await setMode(page, 'live');
  for (const route of ['agents', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'voice', 'chat']) {
    await page.goto(`/?view=${route}`);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0, { timeout: 5000 });
    await expect(page.locator('.live-top-button')).toHaveCount(0);
    await expect(page.getByText('LOCAL WORKFLOW SIMULATION')).toHaveCount(0);
    await expect(page.getByText('PLAYING LOCAL EXAMPLE')).toHaveCount(0);
  }
});
