import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

// QA-only fixture enrichment. The approved HTML is read and rendered as-is;
// only the ignored JSON fixture is written. No application source is changed.
if (process.env.OPENAGENT_ENABLE_VISUAL_FIXTURES !== '1' || process.env.NODE_ENV === 'production') {
  throw new Error('QA guard: set OPENAGENT_ENABLE_VISUAL_FIXTURES=1 in a non-production environment to enrich visual fixtures.');
}

const root = process.cwd();
if (path.basename(root) !== 'frontend-second') throw new Error('Run this QA helper from the frontend-second directory only.');
const fixturePath = path.resolve('.qa/visual-fixtures/reference-fixtures.json');
const referencePath = path.resolve('reference/frontend-final-approved.html');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf8'));
const source = readFileSync(referencePath);
const sourceSha256 = createHash('sha256').update(source).digest('hex').toUpperCase();
if (sourceSha256 !== 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1') throw new Error('Refusing fixture enrichment: canonical HTML SHA-256 mismatch.');
if (fixture.sourceSha256 !== sourceSha256) throw new Error('Fixture source hash does not match the approved reference HTML. Regenerate the visual fixture first.');
const files = fixture.data?.topLevel?.codeFiles;
if (!Array.isArray(files) || files.length !== 4) throw new Error('Expected the approved reference codeFiles metadata in the extracted fixture.');

console.warn('QA ONLY: extracting rendered reference code samples into ignored .qa fixture data. Never use these values in production.');
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await page.goto(pathToFileURL(referencePath).href, { waitUntil: 'domcontentloaded' });
  await page.locator('#app').waitFor({ state: 'visible' });
  await page.evaluate(() => { location.hash = '#/codeq'; });
  await page.locator('.codeq-root').waitFor({ state: 'visible' });
  const codeSamples = [];
  for (let index = 0; index < files.length; index += 1) {
    if (index > 0) await page.locator('.codeq-file-tab').nth(index).click();
    await page.waitForFunction(() => document.querySelectorAll('.codeq-editor .code-line .text').length > 0);
    codeSamples.push(await page.locator('.codeq-editor .code-line .text').allTextContents());
  }
  if (codeSamples.some(lines => !lines.length)) throw new Error('Reference editor did not render every code sample.');
  fixture.data.topLevel.codeFiles = files.map((file, index) => ({ ...file, content: codeSamples[index].join('\n') }));
  writeFileSync(fixturePath, `${JSON.stringify(fixture, null, 2)}\n`);
  console.log(`Added ${codeSamples.length} read-only code sample bodies to ${fixturePath}`);
} finally {
  await browser.close();
}
