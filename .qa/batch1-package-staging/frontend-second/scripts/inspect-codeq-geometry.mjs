import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const expected = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
const source = path.resolve('reference/frontend-final-approved.html');
const hash = createHash('sha256').update(readFileSync(source)).digest('hex').toUpperCase();
if (hash !== expected) throw new Error(`Canonical HTML hash mismatch: ${hash}`);
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== expectedBrowserVersion) throw new Error(`Expected Chrome ${expectedBrowserVersion}, got ${browser.version()}`);
const result = {};
const inspect = async (page) => page.evaluate(() => {
  const rect = (element) => { const box = element?.getBoundingClientRect(); return box ? { x: Math.round(box.x), y: Math.round(box.y), width: Math.round(box.width), height: Math.round(box.height) } : null; };
  const rows = [...document.querySelectorAll('.codeq-tree [role="treeitem"],.codeq-tree .tree-row')].map(element => ({ text: element.textContent?.trim(), ...rect(element) }));
  return { explorerHeading: rect(document.querySelector('.codeq-tree > .row')), projectHeading: rect(document.querySelector('.codeq-tree .codeq-project-heading') ?? document.querySelector('.codeq-tree > .row:nth-child(2)')), firstTreeRow: rows[0] ?? null, treeRows: rows, recentFilesHeading: rect([...document.querySelectorAll('h3')].find(element => element.textContent?.trim() === 'Recent files')),
    editor: rect(document.querySelector('.codeq-editor')), preview: rect(document.querySelector('.codeq-preview-head,.codeq-preview')), bottom: rect(document.querySelector('.codeq-bottom')) };
});
try {
  const reference = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  await reference.addInitScript(() => { const NativeDate = Date; const fixed = NativeDate.parse('2026-10-14T12:00:00.000Z'); window.Date = class extends NativeDate { constructor(...args) { super(...(args.length ? args : [fixed])); } static now() { return fixed; } }; localStorage.setItem('oa-demo-theme', JSON.stringify('light')); });
  await reference.goto(pathToFileURL(source).href, { waitUntil: 'domcontentloaded' });
  await reference.evaluate(() => { location.hash = 'codeq'; });
  await reference.waitForTimeout(350);
  result.reference = await inspect(reference);
  await reference.close();

  const target = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  await target.addInitScript(() => { const NativeDate = Date; const fixed = NativeDate.parse('2026-10-14T12:00:00.000Z'); window.Date = class extends NativeDate { constructor(...args) { super(...(args.length ? args : [fixed])); } static now() { return fixed; } }; localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.setItem('oa-second-theme', 'light'); });
  await target.goto(`${(process.env.OPENAGENT_UI_ORIGIN || 'http://127.0.0.1:5193').replace(/\/$/, '')}/?view=codeq`, { waitUntil: 'domcontentloaded' });
  await target.locator('.codeq-root').waitFor({ state: 'visible' });
  await target.waitForTimeout(1600);
  result.react = await inspect(target);
  await target.close();
} finally { await browser.close(); }
console.log(JSON.stringify({ sourceSha256: hash, browser: `Chrome ${browser.version()}`, viewport: '1440x900', result }, null, 2));
