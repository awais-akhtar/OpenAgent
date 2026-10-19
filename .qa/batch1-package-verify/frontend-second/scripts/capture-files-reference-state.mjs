import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

if (process.env.OPENAGENT_ALLOW_VISUAL_FIXTURES !== '1') {
  throw new Error('Refusing to load example data. Set OPENAGENT_ALLOW_VISUAL_FIXTURES=1 for an isolated visual-acceptance run only.');
}
const origin = process.env.OPENAGENT_VISUAL_ORIGIN || 'http://127.0.0.1:5180';
const fixturePath = path.resolve(process.env.OPENAGENT_VISUAL_FIXTURE_FILE || '.qa/visual-fixtures/reference-fixtures.json');
const outputPath = path.resolve(process.env.OPENAGENT_FILES_VISUAL_OUTPUT || '.qa/files-reference-state/light/desktop-1440x900/files.png');
const sourcePath = path.resolve('reference/frontend-final-approved.html');
const fixtureEnvelope = JSON.parse(readFileSync(fixturePath, 'utf8'));
const source = readFileSync(sourcePath);
const sourceHash = createHash('sha256').update(source).digest('hex').toUpperCase();
if (sourceHash !== 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1') throw new Error('Refusing Files fixture: canonical HTML SHA-256 mismatch.');
if (fixtureEnvelope.sourceSha256 !== sourceHash) throw new Error('Visual fixture source hash does not match the approved HTML reference. Re-extract the fixtures.');
if (!String(fixtureEnvelope.purpose).includes('Visual acceptance only')) throw new Error('Fixture file is not marked as visual-acceptance-only.');
const files = fixtureEnvelope.data?.topLevel?.files;
if (!Array.isArray(files) || files.length !== 6) throw new Error('The approved reference did not provide its expected six Files example records.');

const ago = (label) => {
  const text = String(label).toLowerCase();
  const amount = Number(text.match(/\d+/)?.[0] || 0);
  const days = text.includes('week') ? amount * 7 : text.includes('month') ? amount * 30 : amount;
  return new Date(Date.now() - days * 86_400_000).toISOString();
};
const sizeBytes = (label) => {
  const text = String(label).toUpperCase();
  const amount = Number(text.match(/[\d.]+/)?.[0] || 0);
  if (text.includes('GB')) return Math.round(amount * 1024 ** 3);
  if (text.includes('MB')) return Math.round(amount * 1024 ** 2);
  if (text.includes('KB')) return Math.round(amount * 1024);
  return 0;
};
const mapped = files.map((file, index) => ({
  doc_id: `visual-only-file-${index + 1}`,
  title: file.name,
  format: file.type,
  description: file.description,
  size_bytes: sizeBytes(file.size),
  ...(file.size === 'Website' ? { size_label: 'Website' } : {}),
  updated_at: ago(file.updated),
  project: 'default',
  version: 1,
  status: 'active',
  content: `${file.description} A connected application will render the actual document here and will respect file permissions and available preview formats.\n\n1. Overview\n${file.description}\n\n2. Requirements\n- Keep user tasks easy to discover.\n- Use a consistent interface across every workspace.\n- Make live execution status understandable and trustworthy.\n\n3. Implementation notes\nShow real file content from the existing backend; never substitute these examples for production data.`,
}));
const byId = new Map(mapped.map((file) => [file.doc_id, file]));
const interceptedReads = [];
const blockedWrites = [];
const response = (route, body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1, colorScheme: 'light' });
  await page.route('**/api/documents/library*', async (route) => {
    if (route.request().method() !== 'GET') {
      blockedWrites.push(`${route.request().method()} ${new URL(route.request().url()).pathname}`);
      return route.abort('blockedbyclient');
    }
    interceptedReads.push('GET /api/documents/library');
    return response(route, { items: mapped, count: mapped.length });
  });
  await page.route('**/api/document/**', async (route) => {
    const method = route.request().method();
    const url = new URL(route.request().url());
    const match = url.pathname.match(/\/api\/document\/([^/]+)(?:\/(versions))?$/);
    if (!match) return route.fallback();
    if (method !== 'GET') {
      blockedWrites.push(`${method} ${url.pathname}`);
      return route.abort('blockedbyclient');
    }
    const id = decodeURIComponent(match[1]);
    if (match[2]) {
      interceptedReads.push(`GET ${url.pathname}`);
      return response(route, { versions: [] });
    }
    const document = byId.get(id);
    if (!document) return response(route, { error: 'Unknown visual fixture document' }, 404);
    interceptedReads.push(`GET ${url.pathname}`);
    return response(route, { document });
  });
  await page.goto(`${origin}/?view=files`, { waitUntil: 'domcontentloaded' });
  await page.getByRole('heading', { name: 'Files & knowledge' }).waitFor({ state: 'visible', timeout: 15000 });
  await page.getByRole('button', { name: /Product requirements specification/ }).waitFor({ state: 'visible', timeout: 10000 });
  await page.locator('#file-preview-body .document-canvas').waitFor({ state: 'visible', timeout: 10000 });
  await page.waitForTimeout(250);
  await page.screenshot({ path: outputPath, animations: 'disabled' });
  if (blockedWrites.length) throw new Error(`Unexpected Files write was blocked: ${blockedWrites.join(', ')}`);
  const report = {
    purpose: 'Test-only visual fixture; never production data. Source records extracted from window.OA_FIXTURES.files.',
    sourceSha256: sourceHash,
    fixtureFile: path.relative(process.cwd(), fixturePath),
    interceptedReads,
    blockedWrites,
    viewport: '1440x900',
    theme: 'light',
    screenshot: path.relative(process.cwd(), outputPath),
    records: mapped.length,
  };
  writeFileSync(`${outputPath}.json`, `${JSON.stringify(report, null, 2)}\n`);
  console.log(`Captured Files reference-state screenshot: ${outputPath}`);
  console.log(`Read-only API fixtures used: ${interceptedReads.length}; blocked writes: ${blockedWrites.length}`);
} finally {
  await browser.close();
}
