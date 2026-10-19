import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chromium } from '@playwright/test';
const defaultRoutes = ['home','chat','agents','research','files','automations','calendar','models','settings','codeq','knowledge','voice','assistant','communications','security','developer','engine','science','twin','vault','scraper','finetune','tools'];
const routes = process.env.APPROVED_CAPTURE_ROUTES?.split(',').map(route=>route.trim()).filter(Boolean) ?? defaultRoutes;
const defaultViewports = [{name:'desktop-1440x900',width:1440,height:900},{name:'desktop-1280x800',width:1280,height:800}];
const viewports = process.env.APPROVED_CAPTURE_VIEWPORTS?.split(',').map(name=>defaultViewports.find(viewport=>viewport.name===name)).filter(Boolean) ?? defaultViewports;
const output = path.resolve(process.env.OPENAGENT_APPROVED_CAPTURE_OUTPUT || '.qa/approved-html-current');
const sourcePath = path.resolve(process.env.OPENAGENT_APPROVED_HTML || 'C:/Users/awais/Downloads/frontend final.html');
const sourceHash = createHash('sha256').update(readFileSync(sourcePath)).digest('hex').toUpperCase();
const expectedHash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const baseline = JSON.parse(readFileSync(path.resolve('visual-baseline.config.json'), 'utf8'));
const expectedBrowserVersion = baseline.browser.version;
if (sourceHash !== expectedHash) throw new Error(`Refusing visual capture: canonical frontend SHA-256 mismatch (${sourceHash}).`);
if (path.resolve(output) === sourcePath) throw new Error('Refusing to write screenshots over the canonical HTML source.');
const fileUrl = pathToFileURL(sourcePath).href;
const settleMilliseconds = Math.max(180, Number(process.env.APPROVED_CAPTURE_SETTLE_MS || 800));
const browser = await chromium.launch({headless:true,executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe'});
if (browser.version() !== expectedBrowserVersion) throw new Error(`Expected pinned Chrome ${expectedBrowserVersion}, got ${browser.version()}.`);
const records=[];
const themes = process.env.APPROVED_CAPTURE_THEMES?.split(',').map(theme=>theme.trim()).filter(Boolean) ?? ['light','dark'];
try {
 for (const theme of themes) for (const viewport of viewports) {
  const page=await browser.newPage({viewport:{width:viewport.width,height:viewport.height},deviceScaleFactor:1,reducedMotion:'reduce'});
  // The source application restores its own theme from oa-demo-theme during
  // initialization, so set that exact preference before its scripts run.
  await page.addInitScript(value=>{localStorage.setItem('oa-demo-theme',JSON.stringify(value));},theme);
  await page.addInitScript(() => { const NativeDate = Date; const fixed = NativeDate.parse('2026-10-14T12:00:00.000Z'); window.Date = class extends NativeDate { constructor(...args) { super(...(args.length ? args : [fixed])); } static now() { return fixed; } }; });
  await page.goto(fileUrl,{waitUntil:'domcontentloaded'});
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(settleMilliseconds);
  for (const route of routes) {
   if(route!=='home') {
    // The approved source listens to hashchange and calls its own setRoute;
    // use that same internal navigation path for deterministic capture at mobile sizes.
    await page.evaluate(value=>{location.hash=value;},route);
    await page.waitForTimeout(settleMilliseconds);
   }
   const file=path.join(output,theme,viewport.name,`${route}.png`); mkdirSync(path.dirname(file),{recursive:true});
   await page.screenshot({path:file,fullPage:false,animations:'disabled'});
   records.push({route,theme,viewport:viewport.name,file});
  }
  await page.close();
 }
} finally { await browser.close(); }
const finalSourceHash = createHash('sha256').update(readFileSync(sourcePath)).digest('hex').toUpperCase();
if (finalSourceHash !== expectedHash) throw new Error('Canonical HTML changed while screenshot capture was running.');
writeFileSync(path.join(output,'manifest.json'),JSON.stringify({source:fileUrl,sourceSha256:sourceHash,browser:`Chrome ${browser.version()}`,deviceScaleFactor:1,fontSettle:'document.fonts.ready',clock:'2026-10-14T12:00:00.000Z',capture:'viewport-sized PNG at native pixels',routes,viewports,themes,records},null,2));
console.log(`Captured ${records.length} screenshots from the approved standalone HTML.`);
