/** Read-only smoke against the installed localhost UI. Never submits a form. */
import { chromium } from 'playwright';
import { existsSync } from 'node:fs';

const baseUrl = process.env.OPENAGENT_QA_URL ?? 'http://127.0.0.1:8765';
const views = [
  'home', 'chat', 'agents', 'research', 'files', 'automations', 'calendar',
  'models', 'settings', 'codeq', 'knowledge', 'voice', 'assistant',
  'communications', 'security', 'developer', 'engine', 'science', 'twin',
  'vault', 'scraper', 'finetune', 'tools',
];

const systemChrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const browser = await chromium.launch({ headless: true, ...(existsSync(systemChrome) ? { executablePath: systemChrome } : {}) });
const results = [];
try {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark']) {
    const context = await browser.newContext({ viewport });
    await context.addInitScript((selected) => {
      try { localStorage.setItem('oa-second-theme', selected); } catch { /* browser still renders */ }
    }, theme);
    for (const view of views) {
      const page = await context.newPage();
      const runtimeErrors = [];
      const assetFailures = [];
      page.on('pageerror', error => runtimeErrors.push(error.message));
      page.on('response', response => {
        if (response.status() >= 400 && /\.(?:js|css|woff2?)(?:\?|$)/.test(response.url())) {
          assetFailures.push(`${response.status()} ${new URL(response.url()).pathname}`);
        }
      });
      let status = 0;
      let contentLength = 0;
      let title = '';
      try {
        const response = await page.goto(`${baseUrl}/?view=${encodeURIComponent(view)}`, { waitUntil: 'domcontentloaded', timeout: 20_000 });
        status = response?.status() ?? 0;
        await page.locator('#content').waitFor({ state: 'visible', timeout: 15_000 });
        await page.waitForTimeout(400);
        contentLength = (await page.locator('#content').innerText()).trim().length;
        title = await page.title();
      } catch (error) {
        runtimeErrors.push(String(error));
      }
      results.push({ view, viewport: `${viewport.width}x${viewport.height}`, theme, status, title, contentLength, runtimeErrors, assetFailures });
      await page.close();
    }
    await context.close();
  }
  }
} finally {
  await browser.close();
}
const failed = results.filter(row => row.status !== 200 || row.contentLength < 20 || row.runtimeErrors.length || row.assetFailures.length);
console.log(JSON.stringify({ checked: results.length, failed: failed.length, failures: failed }, null, 2));
if (failed.length) process.exitCode = 1;
