import { expect, test, type Page } from '@playwright/test';

const apiRequests: string[] = [];
const watchedPages = new WeakSet<Page>();
const clockedPages = new WeakSet<Page>();

async function openDemo(page: Page, route: string) {
  if (!clockedPages.has(page)) {
    await page.clock.install({ time: Date.parse('2026-10-17T09:00:00.000Z') });
    clockedPages.add(page);
  }
  if (!watchedPages.has(page)) {
    page.on('request', request => {
      if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
    });
    watchedPages.add(page);
  }
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto(`/?view=${route}`);
}

async function assertDemoRunLifecycle(page: Page, route: string, start: () => Promise<void>) {
  await start();
  const trace = page.locator(`#content .trace-mount.trace-${route}`);
  await expect(trace).toBeVisible();
  await expect(trace.locator('.trace-head-state')).toHaveText('Running');
  await expect(trace).toContainText('DEMO DATA');
  await expect(trace).toContainText(/sample UI states only|local example|simulated/i);
  await trace.getByRole('button', { name: 'Stop', exact: true }).click();
  await expect(trace.locator('.trace-head-state')).toHaveText('Cancelled');
  await expect(trace).toContainText('no external operation occurred');
  await trace.getByRole('button', { name: 'Play example' }).click();
  await expect(trace.locator('.trace-head-state')).toHaveText('Running');
  await page.clock.runFor(10_000);
  await expect(trace.locator('.trace-head-state')).toHaveText('Completed', { timeout: 10_000 });
  await expect(trace).toContainText(/sample|example|illustrative/i);
  if (route === 'models') await expect(trace).toContainText('No download performed');
}

test.beforeEach(() => { apiRequests.length = 0; });

test('Models, Automations, Scientific AI, Fine-tuning and Web Scraper demo runs are explicit, local and stateful', async ({ page }) => {
  test.setTimeout(90_000);
  await page.setViewportSize({ width: 1440, height: 900 });

  await openDemo(page, 'models');
  await page.getByRole('tab', { name: 'Setup needed' }).click();
  const model = page.locator('.models-shell .model-row').first();
  await model.getByRole('button', { name: 'Set up', exact: true }).click();
  await page.getByRole('dialog').getByRole('button', { name: 'Preview setup' }).click();
  await assertDemoRunLifecycle(page, 'models', async () => {
    await expect(page.locator('#content .trace-mount.trace-models')).toBeVisible();
  });

  await openDemo(page, 'automations');
  await assertDemoRunLifecycle(page, 'automations', async () => {
    await page.getByRole('button', { name: 'Preview run', exact: true }).click();
  });

  await openDemo(page, 'science');
  await page.getByRole('tab', { name: 'Experiments' }).click();
  await assertDemoRunLifecycle(page, 'science', async () => {
    await page.getByRole('button', { name: 'Run demo experiment' }).click();
  });
  await expect(page.getByText(/no scientific service ran/i)).toBeVisible();

  await openDemo(page, 'finetune');
  await page.getByRole('tab', { name: 'Jobs' }).click();
  await assertDemoRunLifecycle(page, 'finetune', async () => {
    await page.getByRole('button', { name: 'Simulate training' }).click();
  });
  await expect(page.getByRole('status').filter({ hasText: /No model was trained/i })).toBeVisible();

  await openDemo(page, 'scraper');
  await page.getByLabel(/I confirm I have permission/i).check();
  await assertDemoRunLifecycle(page, 'scraper', async () => {
    await page.getByRole('button', { name: 'Start demo extraction' }).click();
  });
  await expect(page.getByText(/no network request was sent/i).first()).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Live research failure stays an honest backend error with no synthetic trace or success output', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url());
  });
  await page.route('**/api/research/sessions?project=default*', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) }));
  await page.route('**/api/research/sessions', route => route.request().method() === 'POST'
    ? route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Offline Live-mode test fixture' }) })
    : route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) }));
  await page.goto('/?view=research');
  await page.locator('.research-run-controls summary').click();
  await page.getByRole('textbox', { name: 'Research question' }).fill('A test query that cannot reach a backend');
  await page.getByRole('button', { name: 'Run research' }).click();
  await expect(page.getByRole('alert')).toContainText(/Offline Live-mode test fixture|503|unavailable/i);
  await expect(page.locator('#content .trace-mount')).toHaveCount(0);
  await expect(page.getByText(/Research execution completed/i)).toHaveCount(0);
  expect(apiRequests.some(url => url.includes('/api/research/sessions'))).toBe(true);
});
