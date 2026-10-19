import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

// Keep the approved concept PNGs and current implementation captures at their
// original pixel dimensions. No screenshot is rescaled or cropped here.
const references = [
  { id: '01', route: 'home', file: '01_home.png', label: 'Home / Activity; shell' },
  { id: '02', route: 'chat', file: '02_chat.png', label: 'Chat / workspace context' },
  { id: '03', route: 'agents', file: '03_agents.png', label: 'Agents / live sessions' },
  { id: '04', route: 'research', file: '04_research.png', label: 'Research' },
  { id: '05', route: 'files', file: '05_files.png', label: 'Files / Knowledge' },
  { id: '06', route: 'automations', file: '06_automations.png', label: 'Automations' },
  { id: '07', route: 'calendar', file: '07_calendar.png', label: 'Calendar' },
  { id: '08', route: 'models', file: '08_models.png', label: 'Models' },
  { id: '09', route: 'settings', file: '09_settings.png', label: 'Settings' },
  { id: '10', route: 'codeq', file: '10_codeq.png', label: 'CodeQ' },
];

const referenceRoot = path.resolve(process.env.OPENAGENT_APPROVED_REFERENCE_ROOT || '../OpenAgent_HTML_Reference/references');
const targetRoot = path.resolve(process.env.OPENAGENT_APPROVED_TARGET_ROOT || '.qa/visual-final/light/desktop-1440x900');
const outputPath = path.resolve(process.env.OPENAGENT_APPROVED_SHEET_OUTPUT || '.qa/approved-reference-10-screen-sheet.png');
const outputDirectory = path.dirname(outputPath);
mkdirSync(outputDirectory, { recursive: true });

const entries = references.map(item => ({
  ...item,
  reference: `data:image/png;base64,${readFileSync(path.join(referenceRoot, item.file)).toString('base64')}`,
  target: `data:image/png;base64,${readFileSync(path.join(targetRoot, `${item.route}.png`)).toString('base64')}`,
}));

const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  const page = await browser.newPage({ viewport: { width: 2976, height: 1200 }, deviceScaleFactor: 1 });
  await page.setContent('<html><head><style>html,body{margin:0;background:#eef2f8;font:13px Arial,sans-serif}canvas{display:block}</style></head><body><canvas id="sheet"></canvas></body></html>');
  const layout = await page.evaluate(async items => {
    const canvas = document.querySelector('#sheet');
    const sourceWidth = 1448;
    const sourceHeight = 1086;
    const margin = 16;
    const gap = 24;
    const header = 43;
    const footer = 18;
    const rowHeight = header + sourceHeight + footer + margin;
    const rightColumn = margin + sourceWidth + gap;
    canvas.width = margin + sourceWidth + gap + sourceWidth + margin;
    canvas.height = margin + rowHeight * items.length;
    const context = canvas.getContext('2d');
    context.fillStyle = '#eef2f8';
    context.fillRect(0, 0, canvas.width, canvas.height);
    const load = source => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = reject;
      image.src = source;
    });
    const dimensions = [];
    for (let index = 0; index < items.length; index++) {
      const item = items[index];
      const y = margin + index * rowHeight;
      const [reference, target] = await Promise.all([load(item.reference), load(item.target)]);
      context.fillStyle = '#15233c';
      context.font = '600 16px Arial, sans-serif';
      context.fillText(`${item.id} · ${item.route} — ${item.label}`, margin, y + 17);
      context.fillStyle = '#53657f';
      context.font = '12px Arial, sans-serif';
      context.fillText(`APPROVED CONCEPT PNG · ${reference.naturalWidth} × ${reference.naturalHeight} · native pixels`, margin, y + 36);
      context.fillText(`frontend-second capture · ${target.naturalWidth} × ${target.naturalHeight} · native pixels`, rightColumn, y + 36);

      // Draw each image at its intrinsic size and align their top-left origins.
      context.drawImage(reference, margin, y + header);
      context.drawImage(target, rightColumn, y + header);

      // A visible frame exposes the actual capture bounds against the larger
      // concept image; it is deliberately not filled or stretched.
      context.strokeStyle = '#2580ff';
      context.lineWidth = 2;
      context.strokeRect(rightColumn + 1, y + header + 1, target.naturalWidth - 2, target.naturalHeight - 2);
      context.strokeStyle = '#8997ad';
      context.lineWidth = 1;
      context.strokeRect(margin + 0.5, y + header + 0.5, sourceWidth - 1, sourceHeight - 1);
      context.fillStyle = '#53657f';
      context.font = '11px Arial, sans-serif';
      context.fillText(`Capture is ${reference.naturalWidth - target.naturalWidth}px narrower and ${reference.naturalHeight - target.naturalHeight}px shorter; neither image was resized.`, rightColumn, y + header + target.naturalHeight + 14);
      dimensions.push({
        id: item.id,
        route: item.route,
        referenceFile: item.file,
        targetFile: `${item.route}.png`,
        reference: { width: reference.naturalWidth, height: reference.naturalHeight },
        target: { width: target.naturalWidth, height: target.naturalHeight },
        difference: { width: reference.naturalWidth - target.naturalWidth, height: reference.naturalHeight - target.naturalHeight },
        rendering: 'native dimensions, top-left aligned, no scaling or cropping',
      });
    }
    return { width: canvas.width, height: canvas.height, dimensions };
  }, entries);

  await page.screenshot({ path: outputPath, fullPage: true });
  const report = {
    title: 'Approved 10-image visual reference comparison',
    sourceType: 'Conceptual visual references, not running application screenshots; see OpenAgent_HTML_Reference/docs/original-blueprint/09_REFERENCE_IMAGE_MAP.md.',
    targetCapture: targetRoot,
    referenceRoot,
    comparisonMethod: 'Route-mapped side-by-side display at intrinsic PNG dimensions, aligned by top-left origin. No resizing, cropping, pixel-match percentage, or pixel-fidelity claim is made.',
    sheet: { path: outputPath, ...layout },
  };
  writeFileSync(outputPath.replace(/\.png$/i, '.json'), JSON.stringify(report, null, 2));
  console.log(`Created unscaled 10-screen reference sheet (${layout.width} × ${layout.height}): ${outputPath}`);
  for (const item of layout.dimensions) console.log(`${item.id} ${item.route}: ${item.reference.width}×${item.reference.height} ref vs ${item.target.width}×${item.target.height} target; target delta +${item.difference.width}/+${item.difference.height} px`);
} finally {
  await browser.close();
}
