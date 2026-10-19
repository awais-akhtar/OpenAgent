import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const expectedHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
const canonicalPath = path.resolve('reference/frontend-final-approved.html');
const canonicalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (canonicalHash !== expectedHash) throw new Error(`Canonical hash mismatch: ${canonicalHash}`);
const output = path.resolve('.qa/living-ui-desktop-evidence-20261017-current');
const appOrigin = process.env.OPENAGENT_UI_ORIGIN || 'http://127.0.0.1:5180';
const viewports = [{ name: '1440x900', width: 1440, height: 900 }, { name: '1280x800', width: 1280, height: 800 }];
const themes = ['light', 'dark'];
const allRoutes = ['agents', 'chat', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'voice'];
const requestedRoutes = (process.env.OPENAGENT_LIVING_UI_ROUTES || '').split(',').map(route => route.trim()).filter(Boolean);
const routes = requestedRoutes.length ? allRoutes.filter(route => requestedRoutes.includes(route)) : allRoutes;
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== expectedBrowserVersion) throw new Error(`Expected Chrome ${expectedBrowserVersion}, got ${browser.version()}`);
const rows = [];
const errors = [];
const apiCalls = [];
const fixedTime = Date.parse('2026-10-14T12:00:00.000Z');
const readinessChecks = [];

const routeReady = {
  agents: async page => { await page.locator('.agent-top h1').getByText('Product launch review').waitFor({ state: 'visible' }); await page.waitForFunction(() => document.querySelectorAll('.agent-lane').length === 3); },
  chat: async page => { await page.locator('.chat-thread').waitFor({ state: 'visible' }); await page.locator('.chat-composer-outer textarea').waitFor({ state: 'visible' }); await page.waitForFunction(() => document.querySelectorAll('.chat-thread .chat-message').length > 0); },
  research: async page => { await page.locator('.research-shell').waitFor({ state: 'visible' }); await page.getByRole('heading', { name: 'Open-source AI agents' }).waitFor({ state: 'visible' }); await page.getByRole('heading', { name: 'Executive summary' }).waitFor({ state: 'visible' }); await page.waitForFunction(() => document.querySelectorAll('.research-side .source-row, .research-side .research-source-row').length > 0); },
  codeq: async page => { await page.locator('.codeq-root').waitFor({ state: 'visible' }); await page.locator('.codeq-statusbar').waitFor({ state: 'visible' }); await page.waitForFunction(() => Boolean(document.querySelector('.monaco-editor, .code-lines'))); },
  models: async page => { await page.getByRole('heading', { name: 'Models' }).waitFor({ state: 'visible' }); await page.waitForFunction(() => document.querySelectorAll('.models-shell .model-row').length > 0); },
  automations: async page => { await page.getByRole('heading', { name: 'Automations' }).waitFor({ state: 'visible' }); await page.waitForFunction(() => document.querySelectorAll('.automations-shell .automation-record').length > 0); },
  science: async page => { await page.locator('.p-workspace h1').waitFor({ state: 'visible' }); await page.locator('.demo-feature-hero h2').waitFor({ state: 'visible' }); },
  finetune: async page => { await page.locator('.p-workspace h1').waitFor({ state: 'visible' }); await page.locator('.demo-feature-hero h2').waitFor({ state: 'visible' }); },
  scraper: async page => { await page.locator('.p-workspace h1').waitFor({ state: 'visible' }); await page.locator('.demo-feature-hero h2').waitFor({ state: 'visible' }); },
  voice: async page => { await page.locator('.p-voice-hero').waitFor({ state: 'visible' }); await page.getByRole('button', { name: 'Start recording', exact: true }).waitFor({ state: 'visible' }); },
};

async function startContextualRun(page, route) {
  if (route === 'automations') await page.getByRole('button', { name: 'Preview run', exact: true }).click();
  if (route === 'science') await page.getByRole('button', { name: 'Run demo experiment', exact: true }).click();
  if (route === 'finetune') await page.getByRole('button', { name: 'Simulate training', exact: true }).click();
  if (route === 'scraper') { await page.getByLabel(/I confirm I have permission/i).check(); await page.getByRole('button', { name: 'Start demo extraction', exact: true }).click(); }
  const trace = page.locator(`.trace-mount.trace-${route}`);
  await trace.waitFor({ state: 'visible' });
  const state = (await trace.locator('.trace-head-state').textContent())?.trim();
  if (state !== 'Running') {
    const play = trace.getByRole('button', { name: /^(Play|Replay) example$/ });
    await play.waitFor({ state: 'visible' }); await play.click();
  }
  await page.waitForFunction(routeName => document.querySelector(`.trace-mount.trace-${routeName} .trace-head-state`)?.textContent?.trim() === 'Running', route);
  return trace;
}

async function captureWorkflowLifecycle(reference, react, route, viewport, theme) {
  if (route === 'science') await Promise.all([reference, react].map(page => page.getByRole('tab', { name: 'Experiments', exact: true }).click()));
  if (route === 'finetune') {
    await Promise.all([reference, react].map(page => page.getByRole('tab', { name: 'Jobs', exact: true }).click()));
    await Promise.all([reference, react].map(page => page.waitForFunction(() => { const button = [...document.querySelectorAll('button')].find(item => item.textContent?.includes('Simulate training')); return Boolean(button && !button.disabled); })));
  }
  if (route === 'scraper') await Promise.all([reference, react].map(page => page.getByRole('tab', { name: 'New extraction', exact: true }).click()));
  await Promise.all([reference.clock.pauseAt(fixedTime + 60_000), react.clock.pauseAt(fixedTime + 60_000)]);
  const [a, b] = await Promise.all([startContextualRun(reference, route), startContextualRun(react, route)]);
  await Promise.all([a.getByRole('button', { name: 'Pause', exact: true }).waitFor({ state: 'visible' }), b.getByRole('button', { name: 'Pause', exact: true }).waitFor({ state: 'visible' })]);
  await capturePair(reference, react, route, 'explicit-run-running', viewport, theme);
  await Promise.all([a.getByRole('button', { name: 'Pause', exact: true }).click(), b.getByRole('button', { name: 'Pause', exact: true }).click()]);
  await Promise.all([reference.clock.resume(), react.clock.resume()]);
  await capturePair(reference, react, route, 'explicit-run-paused', viewport, theme);
  await Promise.all([a.getByRole('button', { name: 'Resume', exact: true }).click(), b.getByRole('button', { name: 'Resume', exact: true }).click()]);
  await Promise.all([reference.clock.runFor(20_000), react.clock.runFor(20_000)]);
  await Promise.all([reference.locator(`.trace-mount.trace-${route} .trace-head-state`).filter({ hasText: 'Completed' }).waitFor({ state: 'visible' }), react.locator(`.trace-mount.trace-${route} .trace-head-state`).filter({ hasText: 'Completed' }).waitFor({ state: 'visible' })]);
  await capturePair(reference, react, route, 'explicit-run-completed', viewport, theme);
  for (const [page, trace] of [[reference, a], [react, b]]) {
    await trace.getByRole('button', { name: /^(Play|Replay) example$/ }).click();
    await trace.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'visible' });
  }
  await Promise.all([a.getByRole('button', { name: 'Stop', exact: true }).click(), b.getByRole('button', { name: 'Stop', exact: true }).click()]);
  await Promise.all([reference.locator(`.trace-mount.trace-${route} .trace-head-state`).filter({ hasText: 'Cancelled' }).waitFor({ state: 'visible' }), react.locator(`.trace-mount.trace-${route} .trace-head-state`).filter({ hasText: 'Cancelled' }).waitFor({ state: 'visible' })]);
  await capturePair(reference, react, route, 'explicit-run-cancelled', viewport, theme);
}

async function open(canonical, route, viewport, theme) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(`${canonical ? 'canonical' : 'React'} ${route}: ${error.message}`));
  if (!canonical) page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  await page.clock.install({ time: fixedTime });
  if (canonical) {
    await page.addInitScript(value => localStorage.setItem('oa-demo-theme', JSON.stringify(value)), theme);
    await page.goto(pathToFileURL(path.resolve(canonicalPath)).href, { waitUntil: 'domcontentloaded' });
    await page.evaluate(value => { location.hash = value; }, route);
  } else {
    await page.addInitScript(({ theme }) => { localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.setItem('oa-second-theme', theme); }, { theme });
    await page.goto(`${appOrigin}/?view=${route}`, { waitUntil: 'domcontentloaded' });
  }
  assert.ok(routeReady[route], `Missing content readiness contract for ${route}`);
  await routeReady[route](page, canonical);
  readinessChecks.push({ route, app: canonical ? 'canonical' : 'React', passed: true });
  await page.evaluate(() => document.fonts.ready);
  await page.addStyleTag({ content: '*,*::before,*::after{animation-play-state:paused!important;transition:none!important;caret-color:transparent!important}' });
  await page.evaluate(() => { window.scrollTo(0, 0); for (const node of document.querySelectorAll('*')) { node.scrollTop = 0; node.scrollLeft = 0; } });
  return page;
}

async function compare(a, b, page, rects) {
  return page.evaluate(async ({ a, b, rects }) => {
    const load = url => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url; });
    const [left, right] = await Promise.all([load(a), load(b)]);
    if (left.width !== right.width || left.height !== right.height) throw new Error('Pair dimensions differ');
    const canvas = document.createElement('canvas'); canvas.width = left.width; canvas.height = left.height;
    const ctx = canvas.getContext('2d', { willReadFrequently: true }); ctx.drawImage(left, 0, 0); const x = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(right, 0, 0); const y = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    let equalPixels = 0; for (let i = 0; i < x.length; i += 4) if (x[i] === y[i] && x[i + 1] === y[i + 1] && x[i + 2] === y[i + 2] && x[i + 3] === y[i + 3]) equalPixels++;
    const regions = {};
    for (const [name, pair] of Object.entries(rects)) {
      const [leftRect, rightRect] = pair;
      if (!leftRect || !rightRect) { regions[name] = { comparable: false, reason: 'region absent in one application', canonical: leftRect, react: rightRect }; continue; }
      const x0 = Math.max(0, Math.floor(Math.min(leftRect.x, rightRect.x)));
      const y0 = Math.max(0, Math.floor(Math.min(leftRect.y, rightRect.y)));
      const x1 = Math.min(canvas.width, Math.ceil(Math.max(leftRect.x + leftRect.width, rightRect.x + rightRect.width)));
      const y1 = Math.min(canvas.height, Math.ceil(Math.max(leftRect.y + leftRect.height, rightRect.y + rightRect.height)));
      const width = Math.max(1, x1 - x0); const height = Math.max(1, y1 - y0);
      let equal = 0; for (let row = y0; row < y1; row++) for (let column = x0; column < x1; column++) { const i = (row * canvas.width + column) * 4; if (x[i] === y[i] && x[i + 1] === y[i + 1] && x[i + 2] === y[i + 2] && x[i + 3] === y[i + 3]) equal++; }
      regions[name] = { comparable: true, pixels: width * height, strictRgbaPercent: Number((100 * equal / (width * height)).toFixed(2)), canonical: leftRect, react: rightRect, geometryDelta: { left: Number((rightRect.x - leftRect.x).toFixed(2)), top: Number((rightRect.y - leftRect.y).toFixed(2)), width: Number((rightRect.width - leftRect.width).toFixed(2)), height: Number((rightRect.height - leftRect.height).toFixed(2)) } };
    }
    const pixels = canvas.width * canvas.height; return { width: canvas.width, height: canvas.height, pixels, equalPixels, strictRgbaPercent: Number((100 * equalPixels / pixels).toFixed(2)), regions };
  }, { a: `data:image/png;base64,${a.toString('base64')}`, b: `data:image/png;base64,${b.toString('base64')}`, rects });
}

async function regionRects(page) {
  return page.evaluate(() => {
    const rect = (selector) => { const element = document.querySelector(selector); if (!element) return null; const box = element.getBoundingClientRect(); return { x: box.x, y: box.y, width: box.width, height: box.height }; };
    return { workspace: rect('#content .p-primary, #content .research-shell, #content .codeq-root, #content .p-voice-hero'), executionTrace: rect('#content .trace-mount') };
  });
}

async function capturePair(reference, react, route, state, viewport, theme, explicitClassification = null) {
  for (const page of [reference, react]) await page.evaluate(() => { window.scrollTo(0, 0); for (const node of document.querySelectorAll('*')) { node.scrollTop = 0; node.scrollLeft = 0; } });
  const name = `${route}-${state}.png`;
  const classification = explicitClassification || (state === 'idle-no-trace' ? 'intentionally-changed-idle' : 'equivalent-active-state');
  const approvedFile = path.join(output, classification, 'approved', theme, viewport.name, name);
  const reactFile = path.join(output, classification, 'react', theme, viewport.name, name);
  mkdirSync(path.dirname(approvedFile), { recursive: true }); mkdirSync(path.dirname(reactFile), { recursive: true });
  const [approved, target] = await Promise.all([reference.screenshot({ path: approvedFile, animations: 'disabled' }), react.screenshot({ path: reactFile, animations: 'disabled' })]);
  const [approvedRegions, reactRegions] = await Promise.all([regionRects(reference), regionRects(react)]);
  const rects = Object.fromEntries(Object.keys(approvedRegions).map(name => [name, [approvedRegions[name], reactRegions[name]]]));
  rows.push({ route, state, classification, theme, viewport: viewport.name, approved: path.relative(output, approvedFile), react: path.relative(output, reactFile), ...await compare(approved, target, react, rects) });
}

try {
  for (const theme of themes) for (const viewport of viewports) {
    for (const route of routes) {
      const [a, b] = await Promise.all([open(true, route, viewport, theme), open(false, route, viewport, theme)]);
      if (route === 'agents') {
        await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
        await Promise.all([a.getByRole('button', { name: 'Preview activity', exact: true }).click(), b.getByRole('button', { name: 'Preview activity', exact: true }).click()]);
        await Promise.all([a.locator('.trace-mount.trace-agents').waitFor(), b.locator('.trace-mount.trace-agents').waitFor()]);
        await capturePair(a, b, route, 'explicit-demo-run', viewport, theme);
        await Promise.all([a.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Pause', exact: true }).click(), b.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Pause', exact: true }).click()]);
        await capturePair(a, b, route, 'paused-demo-run', viewport, theme);
        await Promise.all([a.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).click(), b.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Resume', exact: true }).click()]);
        await Promise.all([a.clock.runFor(5000), b.clock.runFor(5000)]);
        await Promise.all([a.locator('.trace-mount.trace-agents .trace-head-state').filter({ hasText: 'Completed' }).waitFor({ state: 'visible' }), b.locator('.trace-mount.trace-agents .trace-head-state').filter({ hasText: 'Completed' }).waitFor({ state: 'visible' })]);
        await capturePair(a, b, route, 'completed-demo-run', viewport, theme);
        await Promise.all([a.getByRole('button', { name: 'Preview activity', exact: true }).click(), b.getByRole('button', { name: 'Preview activity', exact: true }).click()]);
        await Promise.all([a.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Stop', exact: true }).click(), b.locator('.trace-mount.trace-agents').getByRole('button', { name: 'Stop', exact: true }).click()]);
        await capturePair(a, b, route, 'cancelled-demo-run', viewport, theme);
      } else if (route === 'codeq') {
        await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
        // Navigation/readiness can advance Playwright's virtual clock a few ticks.
        // Freeze both apps at the same future instant, rather than asking Clock to
        // rewind to fixedTime (which fails and can leave unmatched run states).
        const freezeAt = fixedTime + 60_000;
        await Promise.all([a.clock.pauseAt(freezeAt), b.clock.pauseAt(freezeAt)]);
        await Promise.all([a.getByRole('button', { name: 'Preview run', exact: true }).click(), b.getByRole('button', { name: 'Preview run', exact: true }).click()]);
        await Promise.all([a.locator('.trace-mount.trace-codeq').waitFor(), b.locator('.trace-mount.trace-codeq').waitFor()]);
        await capturePair(a, b, route, 'explicit-demo-start', viewport, theme, 'intentional-behaviour-divergence');
        await Promise.all([a.locator('.trace-mount.trace-codeq').getByRole('button', { name: 'Play example' }).click(), b.locator('.trace-mount.trace-codeq .trace-head-state').filter({ hasText: 'Running' }).waitFor()]);
        await Promise.all([a.locator('.trace-mount.trace-codeq').getByRole('button', { name: 'Pause' }).click(), b.locator('.trace-mount.trace-codeq').getByRole('button', { name: 'Pause' }).click()]);
        const codeqExpand = b.locator('.trace-mount.trace-codeq').getByRole('button', { name: 'Expand' });
        if (await codeqExpand.isVisible()) await codeqExpand.click();
        await Promise.all([a.locator('.trace-mount.trace-codeq .trace-head-state').filter({ hasText: 'Paused' }).waitFor(), b.locator('.trace-mount.trace-codeq .trace-head-state').filter({ hasText: 'Paused' }).waitFor()]);
        await capturePair(a, b, route, 'paused-expanded-matched', viewport, theme);
      } else if (route === 'research') {
        await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
        const freezeAt = fixedTime + 60_000;
        await Promise.all([a.clock.pauseAt(freezeAt), b.clock.pauseAt(freezeAt)]);
        await Promise.all([a.getByRole('button', { name: 'Preview running state', exact: false }).click(), b.getByRole('button', { name: /Preview running state · simulated/ }).click()]);
        await Promise.all([a.locator('.trace-mount.trace-research').waitFor(), b.locator('.trace-mount.trace-research').waitFor()]);
        await capturePair(a, b, route, 'explicit-demo-start', viewport, theme, 'intentional-behaviour-divergence');
        await Promise.all([a.locator('.trace-mount.trace-research').getByRole('button', { name: 'Play example' }).click(), b.locator('.trace-mount.trace-research .trace-head-state').filter({ hasText: 'Running' }).waitFor()]);
        await Promise.all([a.locator('.trace-mount.trace-research').getByRole('button', { name: 'Pause' }).click(), b.locator('.trace-mount.trace-research').getByRole('button', { name: 'Pause' }).click()]);
        const researchExpand = b.locator('.trace-mount.trace-research').getByRole('button', { name: 'Expand' });
        if (await researchExpand.isVisible()) await researchExpand.click();
        await Promise.all([a.locator('.trace-mount.trace-research .trace-head-state').filter({ hasText: 'Paused' }).waitFor(), b.locator('.trace-mount.trace-research .trace-head-state').filter({ hasText: 'Paused' }).waitFor()]);
        await capturePair(a, b, route, 'paused-expanded-matched', viewport, theme);
      } else if (route === 'voice') {
        await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
        await Promise.all([a.getByRole('button', { name: 'Start recording', exact: true }).click(), b.getByRole('button', { name: 'Start recording', exact: true }).click()]);
        await Promise.all([a.locator('.live-recording-indicator').waitFor(), b.locator('.live-recording-indicator').waitFor()]);
        await Promise.all([a.clock.runFor(2000), b.clock.runFor(2000)]);
        await capturePair(a, b, route, 'recording-native-audio-state', viewport, theme);
        await Promise.all([a.getByRole('button', { name: 'Stop & save recording', exact: true }).click(), b.getByRole('button', { name: 'Stop & save recording', exact: true }).click()]);
        await capturePair(a, b, route, 'saved-native-audio-state', viewport, theme);
      } else if (['automations', 'science', 'finetune', 'scraper'].includes(route)) {
        await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
        await captureWorkflowLifecycle(a, b, route, viewport, theme);
      } else await capturePair(a, b, route, 'idle-no-trace', viewport, theme);
      await a.close(); await b.close();
    }
  }
} finally { await browser.close(); }

const currentHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (currentHash !== expectedHash) throw new Error(`Canonical changed during capture: ${currentHash}`);
if (errors.length || apiCalls.length) throw new Error(`Browser errors=${errors.length}; Demo API calls=${apiCalls.length}; ${JSON.stringify({ errors, apiCalls })}`);
assert.equal(readinessChecks.length, routes.length * themes.length * viewports.length * 2, `Expected a successful content readiness check for every route/app/theme/viewport; got ${readinessChecks.length}`);
const pixelCount = rows.reduce((sum, row) => sum + row.pixels, 0);
const equalCount = rows.reduce((sum, row) => sum + row.equalPixels, 0);
const byClass = classification => { const selected = rows.filter(row => row.classification === classification); const pixels = selected.reduce((sum, row) => sum + row.pixels, 0); const equal = selected.reduce((sum, row) => sum + row.equalPixels, 0); return { pairs: selected.length, weightedStrictRgbaPercent: pixels ? Number((100 * equal / pixels).toFixed(2)) : null }; };
const regionSummary = name => Object.fromEntries(['intentionally-changed-idle', 'intentional-behaviour-divergence', 'equivalent-active-state'].map(classification => { const selected = rows.filter(row => row.classification === classification && row.regions?.[name]?.comparable).map(row => row.regions[name]); const pixels = selected.reduce((sum, region) => sum + region.pixels, 0); const equal = selected.reduce((sum, region) => sum + region.strictRgbaPercent * region.pixels / 100, 0); return [classification, { captures: selected.length, weightedStrictRgbaPercent: pixels ? Number((100 * equal / pixels).toFixed(2)) : null }]; }));
const report = { canonicalSha256: currentHash, chrome: browser.version(), deviceScaleFactor: 1, fixedClock: '2026-10-14T12:00:00.000Z', themes, viewports, routes, demoApiCalls: apiCalls.length, browserErrors: errors.length, captures: rows.length, weightedStrictRgbaPercent: Number((100 * equalCount / pixelCount).toFixed(2)), intentionallyChangedIdle: byClass('intentionally-changed-idle'), intentionalBehaviourDivergences: byClass('intentional-behaviour-divergence'), equivalentActiveStates: byClass('equivalent-active-state'), regionSummary: { workspace: regionSummary('workspace'), executionTrace: regionSummary('executionTrace') }, readinessAssertions: readinessChecks.length, readinessChecks, methodology: 'Matched desktop route states captured from the pinned canonical and React demo using same Chrome version, DPR, theme, viewport, fixed clock, reduced motion, font readiness and scroll origin. Full-frame strict RGBA plus workspace and execution-trace region comparisons with geometry deltas. Synthetic traces are deliberately absent from idle states; initial explicit-start behavior for CodeQ/Research is reported as an intentional behavior divergence because the canonical HTML opens a preview while React enters its requested compact running state. Separate paused-expanded captures start both local sample runs, pause both, and align the expanded step presentation before comparison. Voice is captured in native idle, recording and saved states.', rows };
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'manifest.json'), `${JSON.stringify(report, null, 2)}\n`);
const md = [`# Desktop Living UI visual evidence`, '', `- Canonical SHA-256: \`${currentHash}\``, `- Chrome: ${report.chrome}; DPR 1; fixed clock ${report.fixedClock}`, `- Total: ${rows.length} paired states; ${report.weightedStrictRgbaPercent}% weighted strict RGBA equality`, `- Intentionally changed idle states: ${report.intentionallyChangedIdle.pairs} pairs; ${report.intentionallyChangedIdle.weightedStrictRgbaPercent}%`, `- Intentional behavior divergences (CodeQ/Research explicit start): ${report.intentionalBehaviourDivergences.pairs} pairs; score is descriptive, not acceptance`, `- Equivalent active states: ${report.equivalentActiveStates.pairs} pairs; ${report.equivalentActiveStates.weightedStrictRgbaPercent}%`, `- Route-content readiness assertions: ${readinessChecks.length}`, `- Demo API requests: ${apiCalls.length}; page errors: ${errors.length}`, `- States: ${[...new Set(rows.map(row => `${row.route}/${row.state}`))].join(', ')}`, '', 'Strict equality is a whole-frame metric and is not a claim of 100% design acceptance. The canonical reference can include example trace markup at idle while React intentionally suppresses it under the contextual Living UI requirement; these requested differences are classified separately. CodeQ/Research explicit-start preview-versus-running differences are marked intentional behavior divergence. Separate paused-expanded states align both local runs before visual scoring. Each route/app/theme/viewport capture waits for route-specific fixture content before screenshotting.', ''].join('\n');
writeFileSync(path.join(output, 'comparison-report.md'), md);
console.log(`Captured ${rows.length} paired desktop states; weighted strict RGBA ${report.weightedStrictRgbaPercent}%; Demo API requests ${apiCalls.length}; errors ${errors.length}.`);
