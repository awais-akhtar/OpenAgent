import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const canonicalPath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'reference/frontend-final-approved.html');
const canonicalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
const expectedCanonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
if (canonicalHash !== expectedCanonicalHash) throw new Error(`Refusing tab-state capture: canonical source mismatch (${canonicalHash}).`);
const fixtureHashes = {
  'public/demo/reference-fixtures.json': createHash('sha256').update(readFileSync('public/demo/reference-fixtures.json')).digest('hex').toUpperCase(),
  'public/demo/approved-shell-fixtures.json': createHash('sha256').update(readFileSync('public/demo/approved-shell-fixtures.json')).digest('hex').toUpperCase(),
};
const expectedFixtureHashes = {
  'public/demo/reference-fixtures.json': '855D1EEEEDC9DBABF1A0C60F301EBB2727B03053300F3092314835E242DE3903',
  'public/demo/approved-shell-fixtures.json': '500205CDC401EBBC95DE223EA81E68B34199DA1416A1B7B19B290CF3A2CB5FF4',
};
if (JSON.stringify(fixtureHashes) !== JSON.stringify(expectedFixtureHashes)) throw new Error('Refusing tab-state capture: fixture hashes differ from the approved visual baseline.');

const inventoryPath = path.resolve(process.env.OPENAGENT_INTERACTION_AUDIT_OUTPUT || '.qa/desktop-acceptance/continued-milestone-a/interaction-inventory.json');
const inventory = JSON.parse(readFileSync(inventoryPath, 'utf8'));
if (inventory.canonicalSourceSha256 !== expectedCanonicalHash) throw new Error('Interaction inventory source hash does not match the approved canonical HTML.');
const viewports = [{ name: 'desktop-1440x900', width: 1440, height: 900 }, { name: 'desktop-1280x800', width: 1280, height: 800 }];
const themes = ['light', 'dark'];
const output = path.resolve(process.env.OPENAGENT_DESKTOP_TABS_OUTPUT || '.qa/desktop-acceptance/continued-milestone-a/tab-states');
const origin = process.env.OPENAGENT_UI_ORIGIN?.replace(/\/$/, '') || 'http://127.0.0.1:5180';
const fixedDate = Date.parse('2026-10-14T12:00:00.000Z');
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== expectedBrowserVersion) throw new Error(`Expected Chrome ${expectedBrowserVersion}, got ${browser.version()}.`);
const captures = [];
const unmatched = [];
const errors = [];
const apiCalls = [];

async function openPage(isCanonical, viewport, theme, route) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(`${isCanonical ? 'canonical' : 'React'} ${route}: ${error.message}`));
  if (!isCanonical) page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  await page.clock.install({ time: fixedDate });
  if (isCanonical) {
    await page.addInitScript(({ selectedTheme }) => localStorage.setItem('oa-demo-theme', JSON.stringify(selectedTheme)), { selectedTheme: theme });
    await page.goto(pathToFileURL(canonicalPath).href, { waitUntil: 'domcontentloaded' });
    if (route !== 'home') await page.evaluate(value => { location.hash = value; }, route);
  } else {
    await page.addInitScript(({ selectedTheme }) => { localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.setItem('oa-second-theme', selectedTheme); }, { selectedTheme: theme });
    await page.goto(`${origin}/?view=${route}`, { waitUntil: 'domcontentloaded' });
  }
  await page.locator('#content').waitFor({ state: 'visible' });
  await page.evaluate(() => document.fonts.ready);
  if (route === 'codeq') await page.locator('.monaco-editor').waitFor({ state: 'visible', timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(route === 'codeq' ? 5000 : 300);
  await page.addStyleTag({ content: '*,*::before,*::after{animation-play-state:paused!important;transition:none!important;caret-color:transparent!important}' });
  return page;
}

const normalize = value => value.replace(/\s+/g, ' ').trim();
async function clickTab(page, label, occurrence, stateLabel) {
  const tabs = await page.locator('#content [role="tab"],#content .tab').all();
  const matches = [];
  for (const element of tabs) {
    const text = normalize(await element.innerText().catch(() => ''));
    if (text === label) matches.push(element);
  }
  if (!matches[occurrence]) return false;
  await matches[occurrence].click();
  await page.waitForFunction(({ label, occurrence: index }) => {
    const clean = value => value.replace(/\s+/g, ' ').trim();
    const candidates = [...document.querySelectorAll('#content [role="tab"],#content .tab')]
      .filter(element => clean(element.innerText || '') === label);
    const element = candidates[index];
    return element?.getAttribute('aria-selected') === 'true' || element?.classList.contains('active');
  }, { label, occurrence }, { timeout: 3000 }).catch(() => {});
  const selected = await matches[occurrence].getAttribute('aria-selected').catch(() => null);
  const active = await matches[occurrence].evaluate(node => node.classList.contains('active')).catch(() => false);
  if (selected !== 'true' && !active) throw new Error(`${stateLabel} tab did not activate.`);
  return true;
}

async function resetScroll(page) {
  await page.evaluate(() => {
    for (const element of document.querySelectorAll('*')) {
      if (element.scrollTop) element.scrollTop = 0;
      if (element.scrollLeft) element.scrollLeft = 0;
    }
    window.scrollTo(0, 0);
  });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
}

async function pixelScore(leftBytes, rightBytes, page) {
  return page.evaluate(async ({ left, right }) => {
    const load = source => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = source; });
    const [a, b] = await Promise.all([load(left), load(right)]);
    if (a.width !== b.width || a.height !== b.height) throw new Error('Paired tab screenshots differ in dimensions.');
    const canvas = document.createElement('canvas'); canvas.width = a.width; canvas.height = a.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(a, 0, 0); const x = context.getImageData(0, 0, canvas.width, canvas.height).data;
    context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(b, 0, 0); const y = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let equal = 0; for (let i = 0; i < x.length; i += 4) if (x[i] === y[i] && x[i + 1] === y[i + 1] && x[i + 2] === y[i + 2] && x[i + 3] === y[i + 3]) equal++;
    return { width: canvas.width, height: canvas.height, pixels: canvas.width * canvas.height, equalPixels: equal, exactPercent: Number((100 * equal / (canvas.width * canvas.height)).toFixed(2)) };
  }, { left: `data:image/png;base64,${leftBytes.toString('base64')}`, right: `data:image/png;base64,${rightBytes.toString('base64')}` });
}

const requestedRoutes = (process.env.OPENAGENT_DESKTOP_ROUTES || '').split(',').map(value => value.trim()).filter(Boolean);
const routesToCapture = requestedRoutes.length ? inventory.routes.filter(row => requestedRoutes.includes(row.route)) : inventory.routes;
for (const theme of themes) for (const viewport of viewports) for (const routeRecord of routesToCapture) {
  const route = routeRecord.route;
  const reference = await openPage(true, viewport, theme, route);
  const target = await openPage(false, viewport, theme, route);
  if (routeRecord.canonical.tabs.length === 0) {
    await Promise.all([resetScroll(reference), resetScroll(target)]);
    const safeLabel = 'initial';
    const name = `${route}-${safeLabel}.png`;
    const referencePath = path.join(output, 'approved', theme, viewport.name, name);
    const targetPath = path.join(output, 'react', theme, viewport.name, name);
    mkdirSync(path.dirname(referencePath), { recursive: true }); mkdirSync(path.dirname(targetPath), { recursive: true });
    const [referencePng, targetPng] = await Promise.all([
      reference.screenshot({ path: referencePath, fullPage: false, animations: 'disabled' }),
      target.screenshot({ path: targetPath, fullPage: false, animations: 'disabled' }),
    ]);
    captures.push({ route, state: 'initial:no-tabs', occurrence: 0, theme, viewport: viewport.name, approved: path.relative(process.cwd(), referencePath), react: path.relative(process.cwd(), targetPath), ...await pixelScore(referencePng, targetPng, target) });
  }
  const occurrences = new Map();
  for (const tab of routeRecord.canonical.tabs) {
    const label = normalize(tab.label);
    const occurrence = occurrences.get(label) ?? 0;
    occurrences.set(label, occurrence + 1);
    const stateLabel = `${route}/${label}/${occurrence}`;
    const sourceClicked = await clickTab(reference, label, occurrence, `canonical ${stateLabel}`);
    const targetClicked = await clickTab(target, label, occurrence, `React ${stateLabel}`);
    if (!sourceClicked || !targetClicked) {
      unmatched.push({ route, label, occurrence, canonicalFound: sourceClicked, reactFound: targetClicked, theme, viewport: viewport.name });
      continue;
    }
    await Promise.all([resetScroll(reference), resetScroll(target)]);
    const scrollPositions = await Promise.all([reference, target].map(page => page.evaluate(() => ({ windowX: window.scrollX, windowY: window.scrollY, elements: [...document.querySelectorAll('*')].filter(element => element.scrollTop || element.scrollLeft).map(element => [element.scrollTop, element.scrollLeft]) }))));
    if (JSON.stringify(scrollPositions[0]) !== JSON.stringify(scrollPositions[1])) throw new Error(`${stateLabel} scroll positions differ after normalization.`);
    const safeLabel = label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'tab';
    const name = `${route}-${String(tab.index + 1).padStart(2, '0')}-${safeLabel}-${occurrence}.png`;
    const referencePath = path.join(output, 'approved', theme, viewport.name, name);
    const targetPath = path.join(output, 'react', theme, viewport.name, name);
    mkdirSync(path.dirname(referencePath), { recursive: true }); mkdirSync(path.dirname(targetPath), { recursive: true });
    const [referencePng, targetPng] = await Promise.all([
      reference.screenshot({ path: referencePath, fullPage: false, animations: 'disabled' }),
      target.screenshot({ path: targetPath, fullPage: false, animations: 'disabled' }),
    ]);
    captures.push({ route, state: `tab:${label}`, occurrence, theme, viewport: viewport.name, approved: path.relative(process.cwd(), referencePath), react: path.relative(process.cwd(), targetPath), ...await pixelScore(referencePng, targetPng, target) });
  }
  await reference.close(); await target.close();
}

const finalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (finalHash !== expectedCanonicalHash) throw new Error('Canonical HTML changed during paired tab-state capture.');
if (errors.length || apiCalls.length || unmatched.length) throw new Error(`Tab-state capture issues: ${errors.length} browser errors, ${apiCalls.length} demo API calls, ${unmatched.length} unmatched tab clicks. See manifest.`);
const pixels = captures.reduce((sum, row) => sum + row.pixels, 0);
const equalPixels = captures.reduce((sum, row) => sum + row.equalPixels, 0);
const manifest = { canonicalSourceSha256: canonicalHash, browser: `Chrome ${browser.version()}`, deviceScaleFactor: 1, clock: '2026-10-14T12:00:00.000Z', themes, viewports, fixtureHashes, demoApiCalls: apiCalls.length, browserErrors: errors.length, unmatchedTabs: unmatched, methodology: 'Paired full-size screenshots after activating each visible canonical tab and its matching React tab. Window and nested scroll positions are reset to the top and verified equal before capture; reduced motion and deterministic clock; strict RGBA equality; desktop only.', overall: { captures: captures.length, exactPercent: pixels ? Number((100 * equalPixels / pixels).toFixed(2)) : null }, captures };
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'manifest.json'), `${JSON.stringify(manifest, null, 2)}\n`);
await browser.close();
console.log(`Captured ${captures.length} matched desktop tab-state pairs; strict mean ${manifest.overall.exactPercent}%; unmatched: ${unmatched.length}; API calls: ${apiCalls.length}.`);
