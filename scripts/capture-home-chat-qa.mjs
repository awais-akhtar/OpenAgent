import { mkdirSync, writeFileSync } from 'node:fs';
import { chromium } from '@playwright/test';

// Screenshot-only fixtures. These are intercepted in Playwright and never bundled into the product.
const conversation = { id: 'visual-conversation', title: 'Reference review conversation', updated_at: '2026-10-14T10:24:00Z' };
const messages = [
  { id: 'visual-user', role: 'user', content: 'I am reviewing the current workspace design. Can you summarize the real conversation context and attached files?', created_at: '2026-10-14T10:24:00Z', attachments: [{ id: 'visual-attachment', name: 'design-notes.md', content_type: 'text/markdown' }] },
  { id: 'visual-assistant', role: 'assistant', content: 'I can work with the messages and files attached to this conversation. The response preview and context panel show this conversation’s actual returned content.', created_at: '2026-10-14T10:24:31Z' },
];
const apiFixtures = {
  '/api/health': { status: 'ok' },
  '/api/system/metrics': { cpu: { usage_percent: 18, count: 8 }, memory: { used_percent: 42, used_gb: 6, total_gb: 16 } },
  '/api/modules/health': { modules: [{ id: 'chat', status: 'ok' }] },
  '/api/inference/status': { status: 'ready', model: 'local-small' },
  '/api/local-ai/status': { hardware: { system: 'Local runtime' } },
  '/api/audit': { events: [{ id: 'activity-1', action: 'agent.run.completed', resource_type: 'agent run', status: 'completed', created_at: '2026-10-14T10:22:00Z' }, { id: 'activity-2', action: 'research.session.saved', resource_type: 'research session', status: 'saved', created_at: '2026-10-14T09:55:00Z' }] },
  '/api/agents/approvals': { approvals: [{ id: 'approval-1', title: 'Review access to project files', requested_by: 'Workspace agent', reason: 'Read access request', status: 'pending', created_at: '2026-10-14T10:20:00Z' }] },
  '/api/worker/jobs': { jobs: [{ id: 'job-1', kind: 'Agent run', state: 'Completed', created_at: '2026-10-14T10:22:00Z' }] },
  '/api/conversations/organized': { conversations: [conversation, { id: 'visual-conversation-2', title: 'Research planning', updated_at: '2026-10-14T09:55:00Z' }] },
  '/api/agents': { agents: [{ id: 'planner', name: 'Planner', description: 'Breaks a request into reviewable steps' }, { id: 'researcher', name: 'Researcher', description: 'Collects and cites research sources' }, { id: 'developer', name: 'Developer', description: 'Works within a protected code workspace' }] },
  '/api/models': { providers: [{ id: 'local', label: 'Local provider', available: true, default_model: 'local-small', installed_models: ['local-small'] }] },
};
const output = '.tmp-home-chat-reference';
mkdirSync(output, { recursive: true });
const browser = await chromium.launch({ headless: true, executablePath: process.env.OPENAGENT_E2E_BROWSER_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
const evidence = [];
try {
  for (const viewport of [{ name: 'desktop-1440x900', width: 1440, height: 900 }, { name: 'mobile-390x844', width: 390, height: 844 }]) {
    const context = await browser.newContext({ viewport: { width: viewport.width, height: viewport.height } });
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(error.message));
    await page.addInitScript(() => localStorage.setItem('openagent.ui.v2.theme', 'light'));
    await page.route('**/api/**', async route => {
      const url = new URL(route.request().url());
      const key = url.pathname;
      if (key === '/api/conversation/messages') return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ messages }) });
      const fixtureKey = Object.keys(apiFixtures).find(candidate => key === candidate || key.startsWith(`${candidate}/`));
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(apiFixtures[fixtureKey] ?? {}) });
    });
    for (const [view, query, name] of [
      ['dashboard', '?view=dashboard&project=visual-review', 'home'],
      ['chat', '?view=chat&project=visual-review&conversation_id=visual-conversation', 'chat'],
    ]) {
      await page.goto(`http://127.0.0.1:5174/${query}`, { waitUntil: 'domcontentloaded' });
      await page.locator('#oa-main-content').waitFor({ state: 'visible' });
      await page.waitForTimeout(350);
      const file = `${output}/${viewport.name}-${name}.png`;
      await page.screenshot({ path: file, fullPage: false, animations: 'disabled' });
      evidence.push({ view, viewport: viewport.name, file, pageErrors: [...pageErrors] });
    }
    await context.close();
  }
} finally { await browser.close(); }
writeFileSync(`${output}/manifest.json`, JSON.stringify({ note: 'Screenshot-only, intercepted fixtures; no fixture is used by production application code.', evidence }, null, 2));
console.log(`Captured ${evidence.length} Home/Chat comparison screenshots into ${output}.`);
if (evidence.some(item => item.pageErrors.length)) process.exitCode = 1;
