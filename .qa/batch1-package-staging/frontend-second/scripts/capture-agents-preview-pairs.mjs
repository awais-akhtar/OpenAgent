import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const expectedCanonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const canonicalPath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'C:/Users/awais/Downloads/frontend final.html');
const canonicalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (canonicalHash !== expectedCanonicalHash) throw new Error(`Refusing Agents preview capture: canonical SHA-256 mismatch (${canonicalHash}).`);
const fixtureHashes = {
  'public/demo/reference-fixtures.json': createHash('sha256').update(readFileSync('public/demo/reference-fixtures.json')).digest('hex').toUpperCase(),
  'public/demo/approved-shell-fixtures.json': createHash('sha256').update(readFileSync('public/demo/approved-shell-fixtures.json')).digest('hex').toUpperCase(),
};
const expectedFixtureHashes = {
  'public/demo/reference-fixtures.json': '855D1EEEEDC9DBABF1A0C60F301EBB2727B03053300F3092314835E242DE3903',
  'public/demo/approved-shell-fixtures.json': '500205CDC401EBBC95DE223EA81E68B34199DA1416A1B7B19B290CF3A2CB5FF4',
};
if (JSON.stringify(fixtureHashes) !== JSON.stringify(expectedFixtureHashes)) throw new Error('Refusing Agents preview capture: demo fixture hashes differ from the pinned baseline.');

const viewports = [
  { name: 'desktop-1440x900', width: 1440, height: 900 },
  { name: 'desktop-1280x800', width: 1280, height: 800 },
];
const themes = ['light', 'dark'];
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedChrome = process.env.OPENAGENT_EXPECTED_CHROME || baseline.browser.version;
const output = path.resolve(process.env.OPENAGENT_AGENT_HUD_OUTPUT || '.qa/desktop-acceptance/milestone-a-resume/agents-run-final');
const appOrigin = process.env.OPENAGENT_UI_ORIGIN?.replace(/\/$/, '') || 'http://127.0.0.1:5180';
const fixedDate = Date.parse('2026-10-14T12:00:00.000Z');
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const browserVersion = browser.version();
if (browserVersion !== expectedChrome) { await browser.close(); throw new Error(`Expected Chrome ${expectedChrome}, got ${browserVersion}.`); }
const records = [];
const errors = [];
const apiCalls = [];

async function open(isCanonical, viewport, theme) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(`${isCanonical ? 'canonical' : 'React'} ${theme}/${viewport.name}: ${error.message}`));
  if (!isCanonical) page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  await page.clock.install({ time: fixedDate });
  if (isCanonical) {
    await page.addInitScript(({ selectedTheme }) => localStorage.setItem('oa-demo-theme', JSON.stringify(selectedTheme)), { selectedTheme: theme });
    await page.goto(pathToFileURL(canonicalPath).href, { waitUntil: 'domcontentloaded' });
    await page.evaluate(() => { location.hash = 'agents'; });
    await page.locator('.agent-top').waitFor({ state: 'visible' });
  } else {
    await page.addInitScript(({ selectedTheme }) => { localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.setItem('oa-second-theme', selectedTheme); }, { selectedTheme: theme });
    await page.goto(`${appOrigin}/?view=agents`, { waitUntil: 'domcontentloaded' });
    await page.locator('.agent-demo-workspace').waitFor({ state: 'visible' });
  }
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: '*,*::before,*::after{animation-play-state:paused!important;transition:none!important;caret-color:transparent!important}' });
  return page;
}

async function score(leftBytes, rightBytes, page) {
  return page.evaluate(async ({ left, right }) => {
    const load = source => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = source; });
    const [a, b] = await Promise.all([load(left), load(right)]);
    if (a.width !== b.width || a.height !== b.height) throw new Error('Agents preview captures differ in size.');
    const canvas = document.createElement('canvas'); canvas.width = a.width; canvas.height = a.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(a, 0, 0); const x = context.getImageData(0, 0, canvas.width, canvas.height).data;
    context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(b, 0, 0); const y = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let equalPixels = 0; for (let index = 0; index < x.length; index += 4) if (x[index] === y[index] && x[index + 1] === y[index + 1] && x[index + 2] === y[index + 2] && x[index + 3] === y[index + 3]) equalPixels++;
    const pixels = canvas.width * canvas.height;
    return { width: canvas.width, height: canvas.height, pixels, equalPixels, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)) };
  }, { left: `data:image/png;base64,${leftBytes.toString('base64')}`, right: `data:image/png;base64,${rightBytes.toString('base64')}` });
}

try {
  for (const theme of themes) for (const viewport of viewports) {
    const [approved, react] = await Promise.all([open(true, viewport, theme), open(false, viewport, theme)]);
    await approved.getByRole('button', { name: 'Preview activity', exact: true }).click();
    await react.getByRole('button', { name: 'Preview activity', exact: true }).click();
    const approvedInline = approved.locator('.live-inline');
    const reactInline = react.locator('.live-inline');
    const [approvedInlineCount, reactInlineCount] = await Promise.all([approvedInline.count(), reactInline.count()]);
    if (approvedInlineCount !== 2) throw new Error(`Expected the canonical Agents preview's two approved status rows, got ${approvedInlineCount}.`);
    if (reactInlineCount !== 1) throw new Error(`React Agents preview must keep a single contextual status banner, got ${reactInlineCount}.`);
    await Promise.all([approvedInline.first().waitFor({ state: 'visible' }), reactInline.first().waitFor({ state: 'visible' })]);
    await Promise.all([approved.clock.runFor(1100), react.clock.runFor(1100)]);
    await Promise.all([approved.locator('.trace-mount.trace-agents .trace-run-percent').getByText('25%', { exact: true }).waitFor({ state: 'visible' }), react.locator('.trace-mount.trace-agents .trace-run-percent').getByText('25%', { exact: true }).waitFor({ state: 'visible' })]);
    // Advance through the same reduced-motion layout transition before sampling
    // the settled state. The canonical action replaces its markup synchronously.
    await Promise.all([approved.clock.runFor(280), react.clock.runFor(280)]);
    for (const state of ['running', 'paused', 'completed']) {
      if (state === 'paused') {
        await approved.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Pause', exact: true }).click();
        await approved.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).waitFor({ state: 'visible' });
        await react.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Pause', exact: true }).click();
        await react.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).waitFor({ state: 'visible' });
        await Promise.all([
          approved.locator('.trace-mount.trace-agents .trace-head-state').getByText('Paused', { exact: true }).waitFor({ state: 'visible' }),
          react.locator('.trace-mount.trace-agents .trace-head-state').getByText('Paused', { exact: true }).waitFor({ state: 'visible' }),
        ]);
      }
      if (state === 'completed') {
        await approved.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).click();
        await react.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).click();
        await Promise.all([approved.clock.runFor(3000), react.clock.runFor(3000)]);
        await Promise.all([approved.locator('.trace-mount.trace-agents .trace-head-state').getByText('Completed', { exact: true }).waitFor({ state: 'visible' }), react.locator('.trace-mount.trace-agents .trace-head-state').getByText('Completed', { exact: true }).waitFor({ state: 'visible' })]);
        await Promise.all([approved.clock.runFor(280), react.clock.runFor(280)]);
      }
      // Both applications must be sampled at the same document origin. At 1280px
      // the React trace is taller than the viewport, and Playwright's button
      // activation can otherwise leave its page scrolled while the canonical
      // delegated handler leaves its page at the top.
      await Promise.all([approved.evaluate(() => { window.scrollTo(0, 0); const content = document.querySelector('#content'); if (content) content.scrollTop = 0; }), react.evaluate(() => { window.scrollTo(0, 0); const content = document.querySelector('#content'); if (content) content.scrollTop = 0; })]);
      await Promise.all([approved.evaluate(() => new Promise(requestAnimationFrame)), react.evaluate(() => new Promise(requestAnimationFrame))]);
      const name = `agents-preview-${state}.png`;
      const approvedFile = path.join(output, 'approved', theme, viewport.name, name);
      const reactFile = path.join(output, 'react', theme, viewport.name, name);
      mkdirSync(path.dirname(approvedFile), { recursive: true }); mkdirSync(path.dirname(reactFile), { recursive: true });
      const [approvedBytes, reactBytes] = await Promise.all([
        approved.screenshot({ path: approvedFile, fullPage: false, animations: 'disabled' }),
        react.screenshot({ path: reactFile, fullPage: false, animations: 'disabled' }),
      ]);
      records.push({ route: 'agents', state: `preview-${state}`, theme, viewport: viewport.name, approved: path.relative(process.cwd(), approvedFile), react: path.relative(process.cwd(), reactFile), ...await score(approvedBytes, reactBytes, react) });
    }
    await approved.close(); await react.close();
  }
} finally { await browser.close(); }

const finalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (finalHash !== expectedCanonicalHash) throw new Error('Canonical frontend changed during Agents preview capture.');
if (errors.length || apiCalls.length) throw new Error(`Capture issues: ${errors.length} browser errors, ${apiCalls.length} demo API requests.`);
const pixels = records.reduce((sum, item) => sum + item.pixels, 0);
const equalPixels = records.reduce((sum, item) => sum + item.equalPixels, 0);
const manifest = { canonicalSourceSha256: canonicalHash, browser: `Chrome ${browserVersion}`, deviceScaleFactor: 1, clock: '2026-10-14T12:00:00.000Z', themes, viewports, fixtureHashes, demoApiCalls: apiCalls.length, browserErrors: errors.length, methodology: 'Paired native-resolution desktop captures of the canonical Agents Preview activity action and its four-stage local run trace; both apps advance, pause, resume and complete the same deterministic fixture run before comparison. Strict RGBA equality.', intentionalBehaviourDifferences: [{ route: 'agents', state: 'preview-lifecycle', difference: 'The canonical document renders two inline workflow status rows; React intentionally renders one contextual status banner to avoid duplicate active-status announcements. This accepted behaviour difference is not counted as a component-matched UI pass.' }], overall: { captures: records.length, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)) }, captures: records };
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
console.log(`Captured ${records.length} Agents HUD pairs; exact pixel mean ${manifest.overall.exactPercent}%; browser errors: 0; demo API requests: 0.`);
