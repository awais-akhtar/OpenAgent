import { readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const routes = [
  'home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'models',
  'knowledge', 'voice', 'assistant', 'communications', 'security', 'developer', 'engine',
  'science', 'twin', 'vault', 'scraper', 'finetune', 'settings', 'tools',
];
const canonicalHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const targetMatrix = path.resolve(process.env.OPENAGENT_SHEET_TARGET_MATRIX || '.qa/react-parity-current');
const referenceMatrix = path.resolve(process.env.OPENAGENT_SHEET_REFERENCE_MATRIX || '.qa/approved-html-current');
const targetManifest = JSON.parse(readFileSync(path.join(targetMatrix, 'manifest.json'), 'utf8'));
const referenceManifest = JSON.parse(readFileSync(path.join(referenceMatrix, 'manifest.json'), 'utf8'));
if (targetManifest.sourceSha256 !== canonicalHash || referenceManifest.sourceSha256 !== canonicalHash) {
  throw new Error('Refusing comparison sheet: both matrices must be captured against the pinned approved frontend.');
}
const viewport = process.env.OPENAGENT_SHEET_VIEWPORT || 'desktop-1440x900';
const theme = process.env.OPENAGENT_SHEET_THEME || 'light';
const targetRoot = path.join(targetMatrix, theme, viewport);
const referenceRoot = path.join(referenceMatrix, theme, viewport);
const outputPath = path.resolve(process.env.OPENAGENT_SHEET_OUTPUT || '.qa/visual-match-current-comparison-sheet.png');
const captures = routes.map(route => ({
  route,
  target: `data:image/png;base64,${readFileSync(path.join(targetRoot, `${route}.png`)).toString('base64')}`,
  reference: `data:image/png;base64,${readFileSync(path.join(referenceRoot, `${route}.png`)).toString('base64')}`,
}));

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  // This downsampled route sheet is for triage only. The acceptance score is
  // computed from the full-size PNG pairs by measure-html-pixel-match.mjs.
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  await page.setContent('<html><body style="margin:0"><canvas id="sheet"></canvas></body></html>');
  const { width, height } = await page.evaluate(async items => {
    const canvas = document.querySelector('#sheet');
    const columnWidth = 720;
    const headerHeight = 26;
    const imageHeight = 450;
    const rowHeight = headerHeight + imageHeight + 12;
    canvas.width = columnWidth * 2;
    canvas.height = rowHeight * items.length;
    const context = canvas.getContext('2d');
    context.fillStyle = '#eef2f8';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const load = src => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = src;
    });
    for (let index = 0; index < items.length; index++) {
      const item = items[index];
      const y = index * rowHeight;
      context.fillStyle = '#142039';
      context.font = '600 14px Arial, sans-serif';
      context.fillText(`${item.route} · approved HTML reference`, 12, y + 18);
      context.fillText(`${item.route} · React frontend-second`, columnWidth + 12, y + 18);
      const [reference, target] = await Promise.all([load(item.reference), load(item.target)]);
      context.drawImage(reference, 0, y + headerHeight, columnWidth, imageHeight);
      context.drawImage(target, columnWidth, y + headerHeight, columnWidth, imageHeight);
    }
    return { width: canvas.width, height: canvas.height };
  }, captures);
  await page.screenshot({ path: outputPath, fullPage: true });
  writeFileSync(outputPath.replace(/\.png$/i, '.json'), JSON.stringify({ viewport: '1440x900', theme: 'light', routes, width, height, referenceRoot, targetRoot }, null, 2));
  console.log(`Created ${routes.length}-route comparison sheet (${width}×${height}): ${outputPath}`);
} finally {
  await browser.close();
}
