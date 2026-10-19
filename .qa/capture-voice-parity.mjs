import { chromium } from '../frontend-second/node_modules/@playwright/test/index.mjs';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const out = path.resolve('.qa/voice-parity');
mkdirSync(out, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
try {
  for (const viewport of [{ name: 'desktop', width: 1440, height: 900 }, { name: 'mobile', width: 390, height: 844 }]) {
    const source = await browser.newPage({ viewport });
    await source.goto('file:///D:/openagent/frontend-second/reference/frontend-final-approved.html#/voice', { waitUntil: 'domcontentloaded' });
    await source.waitForTimeout(500);
    await source.screenshot({ path: path.join(out, `source-${viewport.name}.png`), fullPage: false, animations: 'disabled' });
    await source.close();
    const react = await browser.newPage({ viewport });
    const apiRequests = [];
    react.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
    await react.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
    await react.goto(`http://127.0.0.1:5180/?view=voice`);
    await react.locator('.p-voice-hero').waitFor({ state: 'visible' });
    await react.waitForTimeout(500);
    await react.screenshot({ path: path.join(out, `react-${viewport.name}.png`), fullPage: false, animations: 'disabled' });
    const geometry = await react.evaluate(() => ({ heroTop: document.querySelector('.p-voice-hero')?.getBoundingClientRect().top, traceTop: document.querySelector('.trace-mount')?.getBoundingClientRect().top }));
    await react.getByRole('button', { name: 'Start recording' }).click();
    await react.getByRole('button', { name: 'Stop & save recording' }).waitFor({ state: 'visible' });
    await react.waitForTimeout(1050);
    await react.getByRole('button', { name: 'Stop & save recording' }).click();
    await react.getByText(/New recording ·/).first().waitFor({ state: 'visible' });
    await react.screenshot({ path: path.join(out, `react-recorded-${viewport.name}.png`), fullPage: false, animations: 'disabled' });
    if (apiRequests.length) throw new Error(`Demo recording contacted the API: ${apiRequests.join(', ')}`);
    console.log(`${viewport.name} geometry ${JSON.stringify(geometry)}; recorded demo locally; API requests ${apiRequests.length}`);
    await react.close();
  }
} finally { await browser.close(); }
console.log(out);
