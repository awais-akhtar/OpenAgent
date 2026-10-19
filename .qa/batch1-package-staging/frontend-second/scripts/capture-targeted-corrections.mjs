import { chromium } from '@playwright/test';
import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const origin = process.env.OPENAGENT_UI_ORIGIN ?? 'http://127.0.0.1:5180';
const output = resolve('.qa/targeted-corrections-20261017');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const captures = [];
try {
  for (const size of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({ viewport: size, deviceScaleFactor: 1 });
      const apiCalls = [];
      page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
      await page.addInitScript(themeValue => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', themeValue);
        localStorage.removeItem('oa-second-isolated-demo-v1');
      }, theme);
      const stem = `${size.width}x${size.height}-${theme}`;
      await page.goto(`${origin}/?view=finetune`);
      await page.getByRole('tab', { name: 'Datasets', exact: true }).waitFor();
      await page.locator('.demo-record').first().waitFor();
      await page.screenshot({ path: resolve(output, `finetune-datasets-${stem}.png`), fullPage: false });
      await page.getByRole('textbox', { name: 'Search datasets' }).fill('no matching targeted capture');
      await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
      await page.locator('.demo-record').first().waitFor();
      await page.screenshot({ path: resolve(output, `finetune-search-reset-jobs-${stem}.png`), fullPage: false });
      await page.getByRole('button', { name: 'Simulate training' }).click();
      await page.locator('.trace-mount.trace-finetune').waitFor();
      await page.screenshot({ path: resolve(output, `finetune-preview-explicit-start-${stem}.png`), fullPage: false });
      await page.getByRole('tab', { name: 'Results', exact: true }).click();
      await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
      await page.waitForTimeout(250);
      await page.screenshot({ path: resolve(output, `finetune-return-no-autoplay-${stem}.png`), fullPage: false });
      await page.getByRole('button', { name: 'Integration guide' }).click();
      await page.getByRole('dialog', { name: 'Demo and integration guide' }).waitFor();
      await page.screenshot({ path: resolve(output, `integration-guide-demo-${stem}.png`), fullPage: false });
      if (apiCalls.length) throw new Error(`Demo made API requests: ${JSON.stringify(apiCalls)}`);
      captures.push({ viewport: size, theme, route: 'finetune', captures: ['datasets', 'search-reset-jobs', 'explicit-start-preview', 'return-no-autoplay', 'integration-guide-demo'], apiCalls: apiCalls.length });
      await page.close();
    }
  }
  const images = (await readdir(output)).filter(name => name.endsWith('.png')).sort();
  await writeFile(resolve(output, 'manifest.json'), JSON.stringify({ createdAt: new Date().toISOString(), canonicalSha256: 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1', browser: await browser.version(), deviceScaleFactor: 1, captures, images }, null, 2));
  console.log(`Targeted desktop evidence captured: ${captures.length} viewport/theme combinations; ${captures.length * 5} screenshots; Demo /api/ requests: 0; ${output}`);
} finally { await browser.close(); }
