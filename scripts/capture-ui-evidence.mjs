import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const views = [
  'dashboard', 'chat', 'codebase', 'voice', 'research', 'agents', 'knowledge', 'workspace',
  'automation', 'assistant', 'communications', 'models', 'security', 'developer', 'settings',
  'engine', 'science', 'calendar', 'twin', 'vault', 'scraper', 'finetune',
];
const viewports = [
  { name: 'desktop-1440x900', width: 1440, height: 900 },
  { name: 'desktop-1280x800', width: 1280, height: 800 },
  { name: 'tablet-1024x768', width: 1024, height: 768 },
  { name: 'tablet-768x1024', width: 768, height: 1024 },
  { name: 'mobile-390x844', width: 390, height: 844 },
];
const themes = ['light', 'dark'];
const port = Number(process.env.OPENAGENT_E2E_PORT || 8899);
const baseURL = process.env.OPENAGENT_E2E_BASE_URL || `http://127.0.0.1:${port}`;
const output = path.resolve(process.env.OPENAGENT_UI_EVIDENCE_DIR || '.tmp-ui-evidence-aion');
const browserCandidates = [
  process.env.OPENAGENT_E2E_BROWSER_PATH,
  'C:/Program Files/Google/Chrome/Application/chrome.exe',
  'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',
].filter(Boolean);
const executablePath = browserCandidates.find(candidate => existsSync(candidate));
const browser = await chromium.launch({ headless: true, ...(executablePath ? { executablePath } : {}) });
const evidence = [];
try {
  for (const theme of themes) {
    for (const viewport of viewports) {
      const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height }, colorScheme: theme });
      const page = await context.newPage();
      const pageErrors = [];
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.addInitScript(mode => localStorage.setItem('openagent.ui.v2.theme', mode), theme);
      // Keep evidence deterministic and private: render empty API-backed states
      // rather than screenshotting the current user's saved conversations/data.
      await page.route('**/api/**', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ok: true, status: 'ok', projects: [], runs: [], sessions: [], items: [], documents: [], entries: [],
          providers: [
            { id: 'local', label: 'Local Runtime', local: true, network: false, available: true, default_model: 'runtime-default', installed_models: [] },
            { id: 'ollama', label: 'Ollama', local: true, network: false, available: true, default_model: 'llama3.2', installed_models: ['llama3.2'] },
          ], models: [], agents: [
            { id: 'executive', name: 'Executive Agent', description: 'Turns goals into local-first plans, milestones, and next actions.' },
            { id: 'research', name: 'Research Agent', description: 'Collects local research context and prepares sourced investigation plans.' },
            { id: 'coding', name: 'Coding Agent', description: 'Plans implementation work, tests, and repository changes.' },
            { id: 'writing', name: 'Writing Agent', description: 'Drafts, edits, and synthesizes local knowledge into useful writing.' },
            { id: 'planning', name: 'Planning Agent', description: 'Breaks goals into tasks, dependencies, and milestones.' },
            { id: 'automation', name: 'Automation Agent', description: 'Designs safe workflows with triggers, actions, and approvals.' },
            { id: 'knowledge', name: 'Knowledge Agent', description: 'Searches local memory and summarizes what OpenAgent already knows.' },
            { id: 'voice', name: 'Voice Agent', description: 'Prepares voice interaction, transcription, and speech workflows.' },
            { id: 'communication', name: 'Communication Agent', description: 'Manages inbox context, drafting, replies, and follow-ups.' },
          ], settings: {}, events: [], modules: [], requests: [], policies: [], secrets: [],
          grants: [], reports: [], calendars: [], jobs: [], workflows: [], applications: [], tools: [], adapters: [],
          templates: [], backups: [], conversations: [], messages: [], approvals: [], audit: [], results: [],
        }),
      }));
      for (const view of views) {
        const query = new URLSearchParams({ view, project: 'default' });
        if (view === 'dashboard') query.set('activity', '1');
        await page.goto(`${baseURL}/?${query}`, { waitUntil: 'domcontentloaded' });
        await page.locator('#oa-main-content').waitFor({ state: 'visible' });
        await page.waitForTimeout(300);
        // Capture the settled API-backed state. A fixed delay alone can freeze
        // transient loading UI (notably the workspace files pane) on a busy host.
        let loadingAtCapture = false;
        try {
          await page.waitForFunction(
            () => !document.querySelector('[aria-busy="true"]'),
            undefined,
            { timeout: 5000 },
          );
        } catch {
          loadingAtCapture = await page.locator('[aria-busy="true"]').count() > 0;
        }
        const file = path.join(output, theme, viewport.name, `${view}.png`);
        mkdirSync(path.dirname(file), { recursive: true });
        await page.screenshot({ path: file, fullPage: false, animations: 'disabled' });
        const title = view === 'chat'
          ? await page.locator('.oa-page-title').textContent().catch(() => '')
          : await page.locator('#oa-main-content h1, #oa-main-content h2').first().textContent().catch(() => '');
        const horizontalOverflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
        evidence.push({ view, theme, viewport: viewport.name, file, title: title?.trim() || '', loadingAtCapture, horizontalOverflow, pageErrors: [...pageErrors] });
      }
      await context.close();
    }
  }
} finally {
  await browser.close();
}
mkdirSync(output, { recursive: true });
writeFileSync(path.join(output, 'manifest.json'), JSON.stringify({ baseURL, views: views.length, themes, viewports, evidence }, null, 2));
const failures = evidence.filter(item => !item.title || item.pageErrors.length > 0 || item.loadingAtCapture || item.horizontalOverflow);
console.log(`Captured ${evidence.length} screenshots at ${output}; ${failures.length} missing headings, unsettled loading states, horizontal overflow, or runtime errors.`);
if (failures.length) {
  console.error(JSON.stringify(failures, null, 2));
  process.exitCode = 1;
}
