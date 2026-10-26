import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

// Test-only extractor. It instruments an ignored temporary copy of the
// approved standalone HTML so screenshots can be compared against the same
// deterministic example records without shipping them to the application.
const root = process.cwd();
const source = path.join(root, 'reference', 'frontend-final-approved.html');
const outputDir = path.resolve(process.env.OPENAGENT_VISUAL_FIXTURE_DIR || '.qa/visual-fixtures');
const temporaryHtml = path.join(outputDir, '.reference-instrumented.html');
const fixtureJson = path.join(outputDir, 'reference-fixtures.json');
const original = readFileSync(source, 'utf8');
const sourceHash = createHash('sha256').update(original).digest('hex').toUpperCase();
if (sourceHash !== 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1') throw new Error('Refusing fixture extraction: canonical HTML SHA-256 mismatch.');
const seedAnchor = 'const demoSeed={';
if (!original.includes(seedAnchor)) throw new Error('Could not locate the reference demo seed without changing the source HTML.');
mkdirSync(outputDir, { recursive: true });
writeFileSync(temporaryHtml, original.replace(seedAnchor, 'const demoSeed=window.__OA_VISUAL_DEMO_SEED={'));

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  const page = await browser.newPage();
  await page.goto(`file:///${temporaryHtml.replaceAll('\\', '/')}`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => Boolean(document.querySelector('.app-shell')));
  const fixture = await page.evaluate(() => ({
    topLevel: window.OA_FIXTURES,
    advanced: window.__OA_VISUAL_DEMO_SEED,
  }));
  if (!fixture.topLevel || !fixture.advanced) throw new Error('The reference fixture stores were not available after page initialization.');
  const output = {
    schemaVersion: 1,
    purpose: 'Visual acceptance only. Never use these values in production data or production workflows.',
    sourceSha256: sourceHash,
    data: fixture,
  };
  writeFileSync(fixtureJson, `${JSON.stringify(output, null, 2)}\n`);
  console.log(`Extracted deterministic reference-only example data: ${fixtureJson}`);
  console.log(`Source SHA-256: ${output.sourceSha256}`);
} finally {
  await browser.close();
  rmSync(temporaryHtml, { force: true });
}
