import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { chromium } from '@playwright/test';

const defaultRoutes = [
  'home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'models',
  'knowledge', 'voice', 'assistant', 'communications', 'security', 'developer', 'engine', 'science',
  'twin', 'vault', 'scraper', 'finetune', 'settings', 'tools',
];
const requestedRoutes = process.env.OPENAGENT_CAPTURE_ROUTES?.split(',').map(route => route.trim()).filter(Boolean);
const routes = requestedRoutes?.length ? requestedRoutes : defaultRoutes;
const allViewports = [
  { name: 'desktop-1440x900', width: 1440, height: 900 },
  { name: 'desktop-1280x800', width: 1280, height: 800 },
];
const requestedViewports = process.env.OPENAGENT_CAPTURE_VIEWPORTS?.split(',').map(viewport => viewport.trim()).filter(Boolean);
const viewports = requestedViewports?.length ? allViewports.filter(viewport => requestedViewports.includes(viewport.name)) : allViewports;
const themes = ['light', 'dark'];
const output = path.resolve(process.env.OPENAGENT_CAPTURE_OUTPUT || '.qa/react-parity-current');
const appOrigin = process.env.OPENAGENT_UI_ORIGIN?.replace(/\/$/, '') || 'http://127.0.0.1:5180';
const canonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync(path.resolve('visual-baseline.config.json'), 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
const sourcePath = path.resolve('reference/frontend-final-approved.html');
const observedHash = createHash('sha256').update(readFileSync(sourcePath)).digest('hex').toUpperCase();
if (observedHash !== canonicalHash) throw new Error(`Refusing React capture: local canonical HTML hash mismatch (${observedHash}).`);
const fixtureFiles = ['public/demo/reference-fixtures.json', 'public/demo/approved-shell-fixtures.json'];
const fixtureHashes = Object.fromEntries(fixtureFiles.map(file => [file, createHash('sha256').update(readFileSync(file)).digest('hex').toUpperCase()]));
const expectedFixtureHashes = {
  'public/demo/reference-fixtures.json': '855D1EEEEDC9DBABF1A0C60F301EBB2727B03053300F3092314835E242DE3903',
  'public/demo/approved-shell-fixtures.json': '500205CDC401EBBC95DE223EA81E68B34199DA1416A1B7B19B290CF3A2CB5FF4',
};
if (JSON.stringify(fixtureHashes) !== JSON.stringify(expectedFixtureHashes)) throw new Error('Refusing React capture: demo fixture hashes differ from the pinned baseline. Update visual-baseline.config.json after review.');
const settleMilliseconds = Math.max(180, Number(process.env.OPENAGENT_SCREENSHOT_SETTLE_MS || 700));
const codeqSettleMilliseconds = Math.max(settleMilliseconds, Number(process.env.OPENAGENT_CODEQ_SETTLE_MS || 5000));
mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const browserVersion = browser.version();
if (browserVersion !== expectedBrowserVersion) throw new Error(`Refusing unmatched capture: expected Chrome ${expectedBrowserVersion}, got ${browserVersion}.`);
const failures = [];
const records = [];
try {
  for (const theme of themes) for (const viewport of viewports) {
    const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.addInitScript(({ mode, selectedTheme }) => {
      localStorage.setItem('oa-second-data-mode', mode);
      localStorage.setItem('oa-second-theme', selectedTheme);
    }, { mode: 'demo', selectedTheme: theme });
    await page.addInitScript(() => { const NativeDate = Date; const fixed = NativeDate.parse('2026-10-14T12:00:00.000Z'); window.Date = class extends NativeDate { constructor(...args) { super(...(args.length ? args : [fixed])); } static now() { return fixed; } }; });
    for (const route of routes) {
      errors.length = 0;
      // Long-lived SSE and polling connections can prevent the load event from
      // settling. The screen itself is awaited below, so stop at DOM readiness.
      await page.goto(`${appOrigin}/?view=${route}`, { waitUntil: 'domcontentloaded', timeout: 15000 });
      await page.locator('#content').waitFor({ state: 'visible' });
      await page.evaluate(() => document.fonts.ready);
      await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
      await page.waitForTimeout(settleMilliseconds);
      // Capture the real backend response or its settled empty/error state,
      // not a transient loading skeleton. Do not wait for SSE/polling to stop.
      await page.waitForFunction(() => {
        const content = document.querySelector('#content');
        return !content || !/\b(loading|connecting|checking)\b/i.test(content.innerText);
      }, null, { timeout: 5000 }).catch(() => {});
      const screen = page.locator('#content .reference-screen-motion > *').first();
      if (route === 'chat') await page.locator('#content .split-layout').waitFor({ state: 'visible' });
      else if (route === 'codeq') {
        await page.locator('#content .codeq-root').waitFor({ state: 'visible' });
        await page.locator('#content .monaco-editor').waitFor({ state: 'visible', timeout: 15_000 }).catch(() => {});
      }
      else await page.locator('#content .reference-screen-motion h1').first().waitFor({ state: 'visible' });
      await page.waitForTimeout(route === 'codeq' ? codeqSettleMilliseconds : settleMilliseconds);
      const title = await page.locator('.breadcrumbs strong').textContent().catch(() => '');
      const overflow = await page.evaluate(() => {
        const root = document.documentElement;
        const content = document.querySelector('#content');
        return root.scrollWidth > root.clientWidth || Boolean(content && content.scrollWidth > content.clientWidth + 1);
      });
      const file = path.join(output, theme, viewport.name, `${route}.png`);
      mkdirSync(path.dirname(file), { recursive: true });
      await page.screenshot({ path: file, fullPage: false, animations: 'disabled' });
      const record = { route, theme, viewport: viewport.name, file, title: title?.trim(), overflow, errors: [...errors] };
      records.push(record);
      if (!(await screen.isVisible()) || !record.title || overflow || errors.length) failures.push(record);
    }
    await page.close();
  }
} finally {
  await browser.close();
}
writeFileSync(path.join(output, 'manifest.json'), JSON.stringify({ source: sourcePath, sourceSha256: observedHash, browser: `Chrome ${browserVersion}`, deviceScaleFactor: 1, fontSettle: 'document.fonts.ready', clock: '2026-10-14T12:00:00.000Z', capture: 'viewport-sized PNG at native pixels', fixtureHashes, fixtures: 'local deterministic demo mode', routes, viewports, themes, records }, null, 2));
console.log(`Captured ${records.length} screenshots; ${failures.length} route/layout/runtime issues.`);
if (failures.length) { console.error(JSON.stringify(failures, null, 2)); process.exitCode = 1; }
