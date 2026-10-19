import { chromium } from '@playwright/test';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';

const origin = process.env.OPENAGENT_PRODUCTION_PREVIEW_ORIGIN ?? 'http://127.0.0.1:5181';
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const apiCalls = [];
const errors = [];
const demoFixtureRequests = [];
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/demo/')) demoFixtureRequests.push(request.url()); });
  page.on('pageerror', error => errors.push(error.message));
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  // The production build intentionally locks data mode to Live. Keep API
  // responses offline here: this is a UI truthfulness check, not integration.
  await page.route('**/api/**', route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Offline production-preview fixture' }) }));
  await page.goto(`${origin}/?view=codeq`, { waitUntil: 'domcontentloaded' });
  await page.locator('.codeq-root').waitFor({ state: 'visible' });
  await page.getByText('Backend offline', { exact: true }).waitFor({ state: 'visible' });
  if (apiCalls.length === 0) throw new Error('Production Live did not attempt any API request after inheriting a stale Demo preference.');
  if (await page.evaluate(() => localStorage.getItem('oa-second-data-mode')) === 'demo') throw new Error('Production startup did not clear its stale Demo preference.');
  if (await page.locator('#content .trace-mount').count() !== 0) throw new Error('Production Live rendered a sample execution trace.');
  if (await page.getByText('PLAYING LOCAL EXAMPLE', { exact: true }).count() !== 0) throw new Error('Production Live exposed demo playback content.');
  await page.goto(`${origin}/?view=voice`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('button', { name: 'Integration guide' }).click();
  const guide = page.getByRole('dialog', { name: 'Demo and integration guide' });
  await guide.waitFor({ state: 'visible' });
  if (await guide.getByText('Only example data', { exact: true }).count()) throw new Error('Production Live Integration Guide presents Demo privacy copy.');
  if (await guide.getByRole('button', { name: 'Export sample data' }).count()) throw new Error('Production Live exposes the Demo fixture export button.');
  await guide.getByText('Live export is unavailable here. No backend export route is configured for this guide.', { exact: true }).waitFor({ state: 'visible' });
  await page.goto(`${origin}/?view=calendar`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: 'Upcoming events', exact: true }).waitFor({ state: 'visible' });
  if (await page.locator('.calendar-side').getByText('Examples', { exact: true }).count()) throw new Error('Production Live Calendar labels backend event state as Demo examples.');
  const evidenceDir = resolve('.qa/targeted-corrections-20261017');
  await mkdir(evidenceDir, { recursive: true });
  await page.screenshot({ path: resolve(evidenceDir, 'integration-guide-production-live-1440x900.png'), fullPage: false });
  if (demoFixtureRequests.length) throw new Error(`Production Live requested demo fixtures: ${JSON.stringify(demoFixtureRequests)}`);
  if (errors.length) throw new Error(`Browser errors: ${JSON.stringify(errors)}`);
  console.log(`Production preview regression: Live cleared stale preference; attempted ${apiCalls.length} API requests (503 offline stubs); demo fixture requests: ${demoFixtureRequests.length}; Integration Guide has no Demo privacy/export UI; page errors: 0. This is not a backend integration test.`);
} finally { await browser.close(); }
