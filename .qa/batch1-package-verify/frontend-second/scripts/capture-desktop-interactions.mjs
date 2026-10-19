import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';

const expectedCanonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync('visual-baseline.config.json', 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
const expectedFixtureHashes = {
  'public/demo/reference-fixtures.json': '855D1EEEEDC9DBABF1A0C60F301EBB2727B03053300F3092314835E242DE3903',
  'public/demo/approved-shell-fixtures.json': '500205CDC401EBBC95DE223EA81E68B34199DA1416A1B7B19B290CF3A2CB5FF4',
};
const canonicalPath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'C:/Users/awais/Downloads/frontend final.html');
const canonicalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (canonicalHash !== expectedCanonicalHash) throw new Error(`Refusing desktop state capture: canonical SHA-256 mismatch (${canonicalHash}).`);
const fixtureHashes = Object.fromEntries(Object.entries(expectedFixtureHashes).map(([name]) => [name, createHash('sha256').update(readFileSync(name)).digest('hex').toUpperCase()]));
if (JSON.stringify(fixtureHashes) !== JSON.stringify(expectedFixtureHashes)) throw new Error('Refusing desktop state capture: demo fixture hashes changed.');

const viewports = [
  { name: 'desktop-1440x900', width: 1440, height: 900 },
  { name: 'desktop-1280x800', width: 1280, height: 800 },
];
const themes = ['light', 'dark'];
const output = path.resolve(process.env.OPENAGENT_DESKTOP_STATES_OUTPUT || '.qa/desktop-acceptance/interactions');
const appOrigin = process.env.OPENAGENT_UI_ORIGIN?.replace(/\/$/, '') || 'http://127.0.0.1:5180';
const fixedDate = Date.parse('2026-10-14T12:00:00.000Z');
const canonicalUrl = pathToFileURL(canonicalPath).href;
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== expectedBrowserVersion) throw new Error(`Refusing unmatched desktop state capture: expected Chrome ${expectedBrowserVersion}, got ${browser.version()}.`);
const captures = [];
const errors = [];
const apiCalls = [];

async function openCanonical(viewport, theme, route) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(`canonical ${route}: ${error.message}`));
  await page.clock.install({ time: fixedDate });
  await page.addInitScript(({ selectedTheme }) => {
    localStorage.setItem('oa-demo-theme', JSON.stringify(selectedTheme));
  }, { selectedTheme: theme });
  await page.goto(canonicalUrl, { waitUntil: 'domcontentloaded' });
  await page.evaluate(value => { location.hash = value; }, route);
  await page.evaluate(() => document.fonts.ready);
  const readySelector = route === 'voice' ? '.p-voice-hero' : route === 'codeq' ? '.codeq-root' : route === 'assistant' ? '#content' : '.trace-mount';
  await page.locator(readySelector).waitFor({ state: 'visible' });
  if (route === 'codeq') await page.locator('.codeq-center .code-lines, .codeq-center .monaco-editor').first().waitFor({ state: 'visible' });
  await page.waitForTimeout(700);
  await page.evaluate(() => window.scrollTo(0, 0));
  return page;
}

async function openReact(viewport, theme, route) {
  const page = await browser.newPage({ viewport: { width: viewport.width, height: viewport.height }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
  page.on('pageerror', error => errors.push(`React ${route}: ${error.message}`));
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
  await page.clock.install({ time: fixedDate });
  await page.addInitScript(({ selectedTheme }) => {
    localStorage.setItem('oa-second-data-mode', 'demo');
    localStorage.setItem('oa-second-theme', selectedTheme);
  }, { selectedTheme: theme });
  await page.goto(`${appOrigin}/?view=${route}`, { waitUntil: 'domcontentloaded' });
  const readySelector = route === 'voice' ? '.p-voice-hero' : route === 'codeq' ? '.codeq-root' : route === 'assistant' ? '#content' : '.trace-mount';
  await page.locator(readySelector).waitFor({ state: 'visible' });
  await page.evaluate(() => document.fonts.ready);
  if (route === 'codeq') await page.locator('.monaco-editor').waitFor({ state: 'visible', timeout: 15_000 });
  await page.waitForTimeout(route === 'codeq' ? 5000 : 700);
  await page.evaluate(() => window.scrollTo(0, 0));
  return page;
}

async function comparePngs(reference, target, targetPage) {
  return targetPage.evaluate(async ({ a, b }) => {
    const load = source => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = source; });
    const [leftImage, rightImage] = await Promise.all([load(a), load(b)]);
    if (leftImage.width !== rightImage.width || leftImage.height !== rightImage.height) throw new Error('Paired screenshots have different dimensions.');
    const canvas = document.createElement('canvas'); canvas.width = leftImage.width; canvas.height = leftImage.height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(leftImage, 0, 0); const left = context.getImageData(0, 0, canvas.width, canvas.height).data;
    context.clearRect(0, 0, canvas.width, canvas.height); context.drawImage(rightImage, 0, 0); const right = context.getImageData(0, 0, canvas.width, canvas.height).data;
    let equalPixels = 0; for (let i = 0; i < left.length; i += 4) if (left[i] === right[i] && left[i + 1] === right[i + 1] && left[i + 2] === right[i + 2] && left[i + 3] === right[i + 3]) equalPixels++;
    const pixels = canvas.width * canvas.height;
    return { width: canvas.width, height: canvas.height, pixels, equalPixels, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)) };
  }, { a: `data:image/png;base64,${reference.toString('base64')}`, b: `data:image/png;base64,${target.toString('base64')}` });
}

async function savePair(referencePage, targetPage, viewport, theme, route, state) {
  await resetScroll(referencePage); await resetScroll(targetPage);
  const referenceDir = path.join(output, 'approved', theme, viewport.name);
  const targetDir = path.join(output, 'react', theme, viewport.name);
  mkdirSync(referenceDir, { recursive: true }); mkdirSync(targetDir, { recursive: true });
  const name = `${route}-${state}.png`;
  const referencePath = path.join(referenceDir, name);
  const targetPath = path.join(targetDir, name);
  const [referencePng, targetPng] = await Promise.all([
    referencePage.screenshot({ path: referencePath, fullPage: false, animations: 'disabled' }),
    targetPage.screenshot({ path: targetPath, fullPage: false, animations: 'disabled' }),
  ]);
  captures.push({ route, state, theme, viewport: viewport.name, approved: path.relative(process.cwd(), referencePath), react: path.relative(process.cwd(), targetPath), ...await comparePngs(referencePng, targetPng, targetPage) });
}

async function freezeVisualState(...pages) {
  for (const page of pages) await page.addStyleTag({ content: `*,*::before,*::after{animation-play-state:paused!important;transition:none!important;caret-color:transparent!important}` });
}

async function freezeExecutionTimers(...pages) {
  await Promise.all(pages.map(async page => {
    const now = await page.evaluate(() => Date.now());
    await page.clock.pauseAt(new Date(now + 25));
  }));
}

async function resetScroll(page) {
  await page.evaluate(() => {
    for (const element of document.querySelectorAll('*')) {
      if (element.scrollTop !== 0) element.scrollTop = 0;
      if (element.scrollLeft !== 0) element.scrollLeft = 0;
    }
    window.scrollTo(0, 0);
  });
}

async function clickBoth(referencePage, targetPage, role, name) {
  await referencePage.getByRole(role, { name, exact: true }).click();
  await targetPage.getByRole(role, { name, exact: true }).click();
}

async function clickCodeQTab(referencePage, targetPage, name) {
  for (const page of [referencePage, targetPage]) {
    await page.locator('.codeq-toolbar .tab').filter({ hasText: new RegExp(`^\\s*${name}\\s*$`) }).click();
    const active = await page.locator('.codeq-toolbar .tab').evaluateAll(nodes => nodes.filter(node => node.classList.contains('active')).map(node => node.textContent?.trim()));
    if (active.length !== 1 || active[0] !== name) throw new Error(`CodeQ ${name} tab did not become active (${active.join(', ')}).`);
  }
}

async function assertButton(page, name, state) {
  const count = await page.getByRole('button', { name, exact: true }).count();
  if (count !== 1) {
    const buttons = await page.locator('.trace-mount button').allTextContents().catch(() => []);
    const progress = await page.locator('.trace-progress').getAttribute('aria-valuenow').catch(() => null);
    throw new Error(`Expected exactly one ${name} button in ${state}; found ${count}. Trace progress=${progress}; buttons=${JSON.stringify(buttons)}.`);
  }
}

for (const theme of themes) for (const viewport of viewports) {
  const sourceVoice = await openCanonical(viewport, theme, 'voice');
  const reactVoice = await openReact(viewport, theme, 'voice');
  await savePair(sourceVoice, reactVoice, viewport, theme, 'voice', 'idle');
  await clickBoth(sourceVoice, reactVoice, 'button', 'Start recording');
  await assertButton(sourceVoice, 'Stop & save recording', 'Voice recording');
  await assertButton(reactVoice, 'Stop & save recording', 'Voice recording');
  await sourceVoice.locator('.live-recording-indicator').waitFor({ state: 'visible' });
  await reactVoice.locator('.live-recording-indicator').waitFor({ state: 'visible' });
  await sourceVoice.locator('.trace-voice').waitFor({ state: 'visible' });
  await reactVoice.locator('.trace-voice').waitFor({ state: 'visible' });
  await Promise.all([sourceVoice.clock.runFor(2000), reactVoice.clock.runFor(2000)]);
  const sourceTimer = await sourceVoice.locator('.demo-rec-timer').textContent();
  const reactTimer = await reactVoice.locator('.demo-rec-timer').textContent();
  if (sourceTimer !== reactTimer) throw new Error(`Voice timers differ after deterministic advance: canonical=${sourceTimer}, React=${reactTimer}.`);
  await freezeVisualState(sourceVoice, reactVoice);
  await savePair(sourceVoice, reactVoice, viewport, theme, 'voice', 'recording');
  await clickBoth(sourceVoice, reactVoice, 'button', 'Stop & save recording');
  await assertButton(sourceVoice, 'Start recording', 'Voice saved');
  await assertButton(reactVoice, 'Start recording', 'Voice saved');
  await sourceVoice.getByText('Saved a demo recording and transcript.', { exact: true }).waitFor({ state: 'visible' });
  await reactVoice.getByText('Saved a demo recording and transcript.', { exact: true }).waitFor({ state: 'visible' });
  await freezeVisualState(sourceVoice, reactVoice);
  await savePair(sourceVoice, reactVoice, viewport, theme, 'voice', 'saved');
  await sourceVoice.close(); await reactVoice.close();

  const sourceCodeQ = await openCanonical(viewport, theme, 'codeq');
  const reactCodeQ = await openReact(viewport, theme, 'codeq');
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'initial');
  for (const tab of ['Preview', 'Terminal', 'Git']) {
    await clickCodeQTab(sourceCodeQ, reactCodeQ, tab);
    await sourceCodeQ.waitForTimeout(350); await reactCodeQ.waitForTimeout(350);
    await freezeVisualState(sourceCodeQ, reactCodeQ);
    await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', tab.toLowerCase());
  }
  const reviewButtons = [sourceCodeQ, reactCodeQ].map(page => page.getByRole('button', { name: 'Review proposed diff', exact: true }).last());
  await Promise.all(reviewButtons.map(button => button.click()));
  await Promise.all([sourceCodeQ, reactCodeQ].map(page => page.getByText('setQuickViewOpen(true)', { exact: false }).first().waitFor({ state: 'visible' })));
  await Promise.all([sourceCodeQ.waitForTimeout(300), reactCodeQ.waitForTimeout(300)]);
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'git-diff-expanded');
  await clickCodeQTab(sourceCodeQ, reactCodeQ, 'Preview');
  for (const page of [sourceCodeQ, reactCodeQ]) {
    const size = page.locator('.codeq-center .size-options button').filter({ hasText: /^10$/ }).first();
    await size.click();
    const selected = await size.getAttribute('aria-pressed') === 'true' || (await size.getAttribute('class') ?? '').split(/\s+/).includes('active');
    if (!selected) throw new Error('CodeQ example size selection did not update in the preview.');
  }
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'preview-size-selected');
  await clickBoth(sourceCodeQ, reactCodeQ, 'button', 'Expand');
  await assertButton(sourceCodeQ, 'Collapse', 'expanded CodeQ trace');
  await assertButton(reactCodeQ, 'Collapse', 'expanded CodeQ trace');
  await sourceCodeQ.waitForTimeout(250); await reactCodeQ.waitForTimeout(250);
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'trace-expanded');
  await Promise.all([
    sourceCodeQ.locator('.trace-mount .trace-step-head').nth(1).click(),
    reactCodeQ.locator('.trace-mount .trace-step-head').nth(1).click(),
  ]);
  await Promise.all([sourceCodeQ.waitForTimeout(350), reactCodeQ.waitForTimeout(350)]);
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'trace-step-expanded');
  await clickBoth(sourceCodeQ, reactCodeQ, 'button', 'Play example');
  await assertButton(sourceCodeQ, 'Pause', 'running CodeQ trace');
  await assertButton(reactCodeQ, 'Pause', 'running CodeQ trace');
  await freezeExecutionTimers(sourceCodeQ, reactCodeQ);
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'trace-running');
  await clickBoth(sourceCodeQ, reactCodeQ, 'button', 'Pause');
  await Promise.all([sourceCodeQ.clock.resume(), reactCodeQ.clock.resume()]);
  await assertButton(sourceCodeQ, 'Resume', 'paused CodeQ trace');
  await assertButton(reactCodeQ, 'Resume', 'paused CodeQ trace');
  await sourceCodeQ.waitForTimeout(150); await reactCodeQ.waitForTimeout(150);
  await freezeVisualState(sourceCodeQ, reactCodeQ);
  await savePair(sourceCodeQ, reactCodeQ, viewport, theme, 'codeq', 'trace-paused');
  await sourceCodeQ.close(); await reactCodeQ.close();
}

for (const theme of themes) for (const viewport of viewports) {
  const source = await openCanonical(viewport, theme, 'assistant');
  const target = await openReact(viewport, theme, 'assistant');
  await source.getByRole('tab', { name: 'Goals', exact: true }).click();
  await target.getByRole('tab', { name: 'Goals', exact: true }).click();
  await source.getByRole('button', { name: 'Add', exact: true }).click();
  await target.getByRole('button', { name: 'Add', exact: true }).click();
  await source.getByRole('dialog').waitFor({ state: 'visible' });
  await target.getByRole('dialog').waitFor({ state: 'visible' });
  await savePair(source, target, viewport, theme, 'assistant', 'goal-create-dialog');
  await source.close(); await target.close();
}

const workflowRoutes = ['chat', 'research', 'models', 'automations', 'science', 'finetune', 'scraper'];
if (process.env.OPENAGENT_SKIP_WORKFLOW_STATES !== '1') for (const theme of themes) for (const viewport of viewports) for (const route of workflowRoutes) {
  const source = await openCanonical(viewport, theme, route);
  const target = await openReact(viewport, theme, route);
  const sourceTrace = source.locator('.trace-mount'); const targetTrace = target.locator('.trace-mount');
  await sourceTrace.getByRole('button', { name: 'Play example', exact: true }).click();
  await targetTrace.getByRole('button', { name: 'Play example', exact: true }).click();
  await Promise.all([source.clock.runFor(250), target.clock.runFor(250)]);
  await assertButton(sourceTrace, 'Pause', `${route} running`);
  await assertButton(targetTrace, 'Pause', `${route} running`);
  await freezeExecutionTimers(source, target);
  await freezeVisualState(source, target);
  await savePair(source, target, viewport, theme, route, 'workflow-running');
  await sourceTrace.getByRole('button', { name: 'Pause', exact: true }).click();
  await targetTrace.getByRole('button', { name: 'Pause', exact: true }).click();
  await Promise.all([source.clock.resume(), target.clock.resume()]);
  await Promise.all([source.clock.runFor(100), target.clock.runFor(100)]);
  await assertButton(sourceTrace, 'Resume', `${route} paused`);
  await assertButton(targetTrace, 'Resume', `${route} paused`);
  await freezeVisualState(source, target);
  await savePair(source, target, viewport, theme, route, 'workflow-paused');
  await sourceTrace.getByRole('button', { name: 'Resume', exact: true }).click();
  await targetTrace.getByRole('button', { name: 'Resume', exact: true }).click();
  await Promise.all([source.clock.runFor(9000), target.clock.runFor(9000)]);
  await assertButton(sourceTrace, 'Replay example', `${route} completed`);
  await assertButton(targetTrace, 'Replay example', `${route} completed`);
  await freezeVisualState(source, target);
  await savePair(source, target, viewport, theme, route, 'workflow-completed');
  await source.close(); await target.close();
}

const finalHash = createHash('sha256').update(readFileSync(canonicalPath)).digest('hex').toUpperCase();
if (finalHash !== expectedCanonicalHash) throw new Error('Canonical HTML changed during desktop interaction capture.');
if (errors.length) throw new Error(`Browser errors during desktop state captures:\n${errors.join('\n')}`);
if (apiCalls.length) throw new Error(`Demo interactions attempted backend requests:\n${apiCalls.join('\n')}`);
const report = {
  canonicalSourceSha256: canonicalHash,
  browser: `Chrome ${browser.version()}`, deviceScaleFactor: 1, clock: '2026-10-14T12:00:00.000Z',
  themes, viewports, fixtureHashes, demoApiCalls: apiCalls.length, browserErrors: errors.length,
  methodology: 'Paired, native-size screenshots for equivalent deterministic Voice, CodeQ, Personal Assistant goal-creation dialog, and workflow execution states; strict RGBA equality; desktop-only.',
  overall: { captures: captures.length, exactPercent: Number((captures.reduce((sum, capture) => sum + capture.equalPixels, 0) / captures.reduce((sum, capture) => sum + capture.pixels, 0) * 100).toFixed(2)) },
  byRouteState: Object.fromEntries([...new Set(captures.map(item => `${item.route}/${item.state}`))].map(key => {
    const group = captures.filter(item => `${item.route}/${item.state}` === key);
    const pixels = group.reduce((sum, item) => sum + item.pixels, 0);
    const equalPixels = group.reduce((sum, item) => sum + item.equalPixels, 0);
    return [key, { captures: group.length, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)), minimumPercent: Math.min(...group.map(item => item.exactPercent)), maximumPercent: Math.max(...group.map(item => item.exactPercent)) }];
  })), captures,
};
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'manifest.json'), `${JSON.stringify(report, null, 2)}\n`);
await browser.close();
console.log(`Captured ${captures.length} matched interaction pairs; strict score ${report.overall.exactPercent}%; demo backend requests: 0; evidence: ${output}`);
