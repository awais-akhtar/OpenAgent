import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const canonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync(path.resolve('visual-baseline.config.json'), 'utf8'));
const expectedBrowser = `Chrome ${baseline.browser.version}`;
const referenceRoot = path.resolve(process.env.OPENAGENT_COMPARE_REFERENCE || '.qa/approved-html-current');
const targetRoot = path.resolve(process.env.OPENAGENT_COMPARE_TARGET || '.qa/react-parity-current');
const output = path.resolve(process.env.OPENAGENT_COMPARE_OUTPUT || '.qa/visual-match-current.json');
const readManifest = (root) => JSON.parse(readFileSync(path.join(root, 'manifest.json'), 'utf8'));
const referenceManifest = readManifest(referenceRoot);
const targetManifest = readManifest(targetRoot);
if (referenceManifest.sourceSha256 !== canonicalHash || targetManifest.sourceSha256 !== canonicalHash) {
  throw new Error('Refusing comparison: both capture manifests must identify the exact approved frontend SHA-256.');
}
if (referenceManifest.browser !== expectedBrowser || targetManifest.browser !== expectedBrowser) {
  throw new Error('Refusing comparison: browser versions do not match the pinned Chrome version.');
}
if (referenceManifest.deviceScaleFactor !== 1 || targetManifest.deviceScaleFactor !== 1) throw new Error('Refusing comparison: deviceScaleFactor must be 1.');
if (referenceManifest.clock !== '2026-10-14T12:00:00.000Z' || targetManifest.clock !== referenceManifest.clock) throw new Error('Refusing comparison: capture wall-clock fixtures do not match.');
const expectedFixtureHashes = {
  'public/demo/reference-fixtures.json': '855D1EEEEDC9DBABF1A0C60F301EBB2727B03053300F3092314835E242DE3903',
  'public/demo/approved-shell-fixtures.json': '500205CDC401EBBC95DE223EA81E68B34199DA1416A1B7B19B290CF3A2CB5FF4',
};
if (JSON.stringify(targetManifest.fixtureHashes) !== JSON.stringify(expectedFixtureHashes)) throw new Error('Refusing comparison: React capture fixture hashes are not the pinned demo datasets.');
const keyOf = (item) => `${item.route}|${item.theme}|${item.viewport}`;
const targetRecords = new Map(targetManifest.records.map(record => [keyOf(record), record]));
const pairs = referenceManifest.records.map(reference => ({ reference, target: targetRecords.get(keyOf(reference)) })).filter(pair => pair.target);
if (!pairs.length) throw new Error('No matching route/theme/viewport captures found.');
if (pairs.length !== referenceManifest.records.length || pairs.length !== targetManifest.records.length) throw new Error('Capture matrices are incomplete or contain unmatched states; refusing aggregate score.');

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
if (browser.version() !== baseline.browser.version) throw new Error(`Expected pinned Chrome ${baseline.browser.version}, got ${browser.version()}.`);
const results = [];
try {
  const page = await browser.newPage({ deviceScaleFactor: 1 });
  for (const { reference, target } of pairs) {
    const refPath = path.resolve(reference.file);
    const targetPath = path.resolve(target.file);
    const [refBytes, targetBytes] = [readFileSync(refPath), readFileSync(targetPath)];
    const score = await page.evaluate(async ({ aUrl, bUrl }) => {
      const load = (url) => new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = url; });
      const [a, b] = await Promise.all([load(aUrl), load(bUrl)]);
      if (a.width !== b.width || a.height !== b.height) throw new Error(`Capture dimensions differ: ${a.width}x${a.height} and ${b.width}x${b.height}.`);
      const canvas = document.createElement('canvas'); canvas.width = a.width; canvas.height = a.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      ctx.drawImage(a, 0, 0); const left = ctx.getImageData(0, 0, a.width, a.height).data;
      ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.drawImage(b, 0, 0); const right = ctx.getImageData(0, 0, b.width, b.height).data;
      let equalPixels = 0; let channelError = 0;
      for (let i = 0; i < left.length; i += 4) {
        if (left[i] === right[i] && left[i + 1] === right[i + 1] && left[i + 2] === right[i + 2] && left[i + 3] === right[i + 3]) equalPixels++;
        channelError += Math.abs(left[i] - right[i]) + Math.abs(left[i + 1] - right[i + 1]) + Math.abs(left[i + 2] - right[i + 2]);
      }
      const pixels = a.width * a.height;
      return { width: a.width, height: a.height, pixels, equalPixels, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)), meanAbsoluteChannelError: Number((channelError / (pixels * 3)).toFixed(3)) };
    }, { aUrl: `data:image/png;base64,${refBytes.toString('base64')}`, bUrl: `data:image/png;base64,${targetBytes.toString('base64')}` });
    results.push({ route: reference.route, theme: reference.theme, viewport: reference.viewport, reference: path.relative(process.cwd(), refPath), target: path.relative(process.cwd(), targetPath), ...score });
    console.log(`${reference.route} · ${reference.theme} · ${reference.viewport}: ${score.exactPercent}%`);
  }
} finally { await browser.close(); }

const byKey = (key) => {
  const groups = new Map();
  for (const item of results) { const label = item[key]; const group = groups.get(label) ?? []; group.push(item); groups.set(label, group); }
  return Object.fromEntries([...groups].map(([label, items]) => [label, aggregate(items)]));
};
function aggregate(items) {
  const pixels = items.reduce((sum, item) => sum + item.pixels, 0);
  const equalPixels = items.reduce((sum, item) => sum + item.equalPixels, 0);
  return { captures: items.length, pixels, equalPixels, exactPercent: Number((100 * equalPixels / pixels).toFixed(2)), minimumRoutePercent: Math.min(...items.map(item => item.exactPercent)), maximumRoutePercent: Math.max(...items.map(item => item.exactPercent)) };
}
const report = {
  canonicalSourceSha256: canonicalHash,
  browser: `Chrome ${browser.version()}`, deviceScaleFactor: 1,
  methodology: 'Full viewport-sized PNGs at native resolution; strict per-pixel RGBA equality. Captures matched by route, theme, and viewport; no scaling, cropping, or montage scoring.',
  captureCount: results.length,
  overall: aggregate(results),
  byTheme: byKey('theme'), byViewport: byKey('viewport'), byRoute: byKey('route'),
  captures: results,
};
mkdirSync(path.dirname(output), { recursive: true });
writeFileSync(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(`Overall: ${report.overall.exactPercent}% exact across ${report.captureCount} matched full-size screenshots; report: ${output}`);
