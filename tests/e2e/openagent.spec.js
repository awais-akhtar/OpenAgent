import { test, expect, request as playwrightRequest } from '@playwright/test';

const views = [
  'dashboard', 'chat', 'codebase', 'voice', 'research', 'agents', 'knowledge', 'workspace', 'automation', 'assistant', 'communications', 'models', 'security', 'developer', 'settings', 'engine', 'science', 'calendar', 'twin', 'vault', 'scraper', 'finetune',
];

// The Python server serves the SPA at `/`; the React shell reads query-based
// view links and normalizes them back to `/`. `/tools/*` is not a server route.
const routeFor = view => view === 'dashboard' ? '/?view=dashboard&activity=1' : `/?view=${view}`;

async function mockBackend(page, calls = []) {
  let archived = false;
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    let payload;
    try { payload = request.postDataJSON(); } catch { payload = undefined; }
    calls.push({ method: request.method(), path, url, payload });

    if (path === '/api/conversations/organized') {
      const row = { id: 'conversation-fixture', title: 'Find the invoice notes', project_id: 'default', pinned: false };
      const rows = url.searchParams.get('archived_only') === 'true' ? (archived ? [row] : []) : (archived ? [] : [row]);
      await route.fulfill({ json: { conversations: rows } }); return;
    }
    if (path === '/api/models' && request.method() === 'GET') {
      await route.fulfill({ json: { providers: [{ id: 'local-provider', label: 'Local provider', default_model: 'small-local', installed_models: ['small-local'] }] } }); return;
    }
    if (path === '/api/agents' && request.method() === 'GET') {
      await route.fulfill({ json: { agents: [{ id: 'executive', name: 'Executive Agent', description: 'Plans a goal using the registered local agent.' }] } }); return;
    }
    if (path === '/api/agents/run-checked' && request.method() === 'POST') {
      await route.fulfill({ status: 201, json: { run: { response: 'Checked agent plan.' }, verification: { status: 'verified' } } }); return;
    }
    if (path === '/api/chat/search') {
      const query = (url.searchParams.get('q') || url.searchParams.get('query') || '').toLowerCase();
      await route.fulfill({ json: { results: query.includes('invoice') ? [{ conversation_id: 'conversation-fixture', conversation_title: 'Find the invoice notes', content: 'Invoice reference' }] : [] } }); return;
    }
    if (path === '/api/conversation/messages') {
      const conversationId = url.searchParams.get('conversation_id');
      const messages = conversationId === 'conversation-created'
        ? [{ id: 'message-created', role: 'assistant', content: 'Fixture answer' }]
        : [{ id: 'message-fixture', role: 'assistant', content: 'Invoice reference saved in this conversation.' }];
      await route.fulfill({ json: { messages } }); return;
    }
    if (path === '/api/conversations/bulk') {
      archived = payload?.action === 'archive';
      await route.fulfill({ json: { updated: payload?.conversation_ids?.length ?? 0 } }); return;
    }
    if (path === '/api/chat/stream' && request.method() === 'POST') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"conversation","conversation":{"id":"conversation-created"}}\n\ndata: {"type":"chunk","content":"Fixture answer"}\n\n' }); return;
    }
    if (path === '/api/codebase/projects' && request.method() === 'GET') {
      await route.fulfill({ json: { projects: [{ id: 'repo-fixture', name: 'Fixture repository', status: 'ready' }] } }); return;
    }
    if (path === '/api/codebase/preflight' && request.method() === 'POST') {
      await route.fulfill({ json: { preflight: { summary: 'Read-only preflight complete.' } } }); return;
    }
    if (path === '/api/research/sessions' && request.method() === 'POST') {
      await route.fulfill({ json: { session: { id: 'research-fixture', query: payload?.query, status: 'queued' } } }); return;
    }
    if (path === '/api/research/sessions/research-fixture') {
      await route.fulfill({ json: { session: { id: 'research-fixture', query: 'Local research fixture', status: 'completed' } } }); return;
    }
    if (path === '/api/voice/sessions' && request.method() === 'POST') {
      await route.fulfill({ json: { session: { id: 'voice-fixture', status: 'created' } } }); return;
    }
    if (path === '/api/knowledge/hybrid-search' && request.method() === 'POST') {
      await route.fulfill({ json: { results: [{ id: 'source-fixture', title: 'Local source', content: 'A local source excerpt.' }] } }); return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, projects: [], runs: [], sessions: [], items: [], documents: [], entries: [], providers: [], models: [], settings: {}, events: [], modules: [], requests: [], policies: [], secrets: [], grants: [], reports: [], calendars: [], jobs: [], workflows: [], applications: [], tools: [], adapters: [], templates: [], backups: [], status: 'ok' }) });
  });
}

test('all 22 active views open from deep links and remain discoverable in the Tools launcher', async ({ page }) => {
  await mockBackend(page);
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?view=dashboard');
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
  await expect(page).toHaveURL(/view=chat/); // Legacy dashboard links intentionally resolve to the new conversation home.

  for (const view of views) {
    await page.goto(routeFor(view));
    await expect(page.locator('#oa-main-content')).toBeVisible();
    const screenTitle = view === 'chat' ? page.locator('.oa-page-title') : page.locator('#oa-main-content h1, #oa-main-content h2').first();
    await expect(screenTitle).toBeVisible();
    if (view === 'chat') await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
  }

  await page.getByRole('button', { name: 'Tools' }).click();
  const palette = page.getByRole('dialog', { name: 'Search OpenAgent' });
  await expect(palette).toBeVisible();
  await palette.getByRole('button', { name: 'Activity' }).click();
  await expect(page).toHaveURL(/view=dashboard.*activity=1|activity=1.*view=dashboard/);
  await expect(page.getByRole('heading', { name: 'Activity' })).toBeVisible();
  await page.goto('/?view=phd');
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
  await expect(page).toHaveURL(/view=chat/);
  await page.getByRole('button', { name: 'Tools' }).click();
  const retiredPalette = page.getByRole('dialog', { name: 'Search OpenAgent' });
  await expect(retiredPalette).toBeVisible();
  await retiredPalette.getByRole('textbox', { name: 'Search conversations and tools' }).fill('Application Studio');
  await expect(retiredPalette.locator('.oa-command-row')).toHaveCount(0);
  expect(pageErrors).toEqual([]);
});

test('command palette filters tools, opens a screen by keyboard and closes with Escape', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.keyboard.press('Control+k');
  const dialog = page.getByRole('dialog', { name: 'Search OpenAgent' });
  await expect(dialog).toBeVisible();
  const input = page.getByRole('textbox', { name: 'Search conversations and tools' });
  await input.fill('knowledge');
  const knowledge = dialog.locator('.oa-command-row').filter({ hasText: 'Knowledge' });
  await expect(knowledge).toBeVisible();
  await expect(dialog.locator('.oa-command-row')).toHaveCount(1);
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/view=knowledge/);
  await expect(page.locator('main.page h1')).toHaveText('Knowledge');
  await page.keyboard.press('Control+k');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog', { name: 'Search OpenAgent' })).toHaveCount(0);
});

test('new conversation sends a project-scoped, local-first chat request', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);
  await page.goto('/?view=chat&project=fixture-project');
  await page.getByRole('textbox', { name: 'Message' }).fill('A bounded fixture request');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/chat/stream')).toBe(true);
  const call = calls.find(item => item.method === 'POST' && item.path === '/api/chat/stream');
  expect(call.payload).toMatchObject({ message: 'A bounded fixture request', project: 'fixture-project', provider: 'auto', model: null, allow_external: false, conversation_id: null, attachments: [] });
  await expect(page.getByText('Fixture answer')).toBeVisible();
});

test('chat home provider and assistant shortcuts apply live selections and actions', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);
  await page.goto('/?view=chat&project=fixture-project');
  await page.getByRole('button', { name: 'Local provider' }).click();
  await expect(page.getByRole('button', { name: 'Local provider' })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('textbox', { name: 'Message' }).fill('Use selected provider');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect.poll(() => calls.find(call => call.method === 'POST' && call.path === '/api/chat/stream')?.payload?.provider).toBe('local-provider');

  await page.goto('/?view=chat&project=fixture-project');
  await page.getByRole('button', { name: /Executive Agent/ }).click();
  await expect(page.getByRole('button', { name: /Executive Agent/ })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('textbox', { name: 'Message' }).fill('Plan this local project');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByText('Checked agent plan.')).toBeVisible();
  const agentCall = calls.find(call => call.method === 'POST' && call.path === '/api/agents/run-checked');
  expect(agentCall?.payload).toMatchObject({ agent: 'executive', goal: 'Plan this local project', project: 'fixture-project', verify: true });
});

test('conversation search opens a saved result and archive can be restored from history', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);
  await page.goto('/');
  await page.keyboard.press('Control+k');
  const search = page.getByRole('textbox', { name: 'Search conversations and tools' });
  await search.fill('invoice');
  await expect(page.getByRole('button', { name: /Find the invoice notes/ })).toBeVisible();
  await page.getByRole('dialog', { name: 'Search OpenAgent' }).getByRole('button', { name: /Find the invoice notes/ }).click();
  await expect(page).toHaveURL(/conversation_id=conversation-fixture/);
  await expect(page.getByText('Invoice reference saved in this conversation.')).toBeVisible();

  const conversationRow = page.locator('.oa-conversation-entry').filter({ hasText: 'Find the invoice notes' });
  await conversationRow.hover();
  await conversationRow.locator('summary[aria-label="Actions for Find the invoice notes"]').click();
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('menuitem', { name: 'Archive conversation' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/conversations/bulk' && call.payload?.action === 'archive')).toBe(true);
  await page.getByRole('button', { name: 'Show archived conversations' }).click();
  await page.locator('.oa-conversation-entry').filter({ hasText: 'Find the invoice notes' }).hover();
  await page.locator('.oa-conversation-entry').filter({ hasText: 'Find the invoice notes' }).locator('summary[aria-label="Actions for Find the invoice notes"]').click();
  await expect(page.getByRole('menuitem', { name: 'Restore conversation' })).toBeVisible();
  await page.getByRole('menuitem', { name: 'Restore conversation' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/conversations/bulk' && call.payload?.action === 'unarchive')).toBe(true);
});

test('knowledge sections use accessible tabs and arrow-key navigation', async ({ page }) => {
  await mockBackend(page);
  await page.goto(routeFor('knowledge'));
  const searchTab = page.getByRole('tab', { name: 'Search' });
  await expect(searchTab).toHaveAttribute('aria-selected', 'true');
  await searchTab.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Library' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel')).toBeVisible();
});

test('dark and light themes update the document and persist', async ({ page }) => {
  await mockBackend(page);
  await page.goto(routeFor('settings'));
  await page.evaluate(() => localStorage.setItem('openagent.ui.v2.theme', 'dark'));
  await page.reload();
  await page.getByRole('button', { name: 'Use light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'Use dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('responsive shell and all representative workspaces avoid horizontal document overflow', async ({ page }) => {
  await mockBackend(page);
  for (const width of [1440, 1024, 768, 600, 390, 360, 320]) {
    await page.setViewportSize({ width, height: 850 });
    for (const view of ['dashboard', 'chat', 'research', 'codebase', 'settings']) {
      await page.goto(routeFor(view));
      await expect(page.locator('.activity-header h1, main.page h1, .oa-page-title').first()).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${view} at ${width}px`).toBeLessThanOrEqual(1);
    }
  }
});

test('workspace document title gets a readable mobile toolbar row', async ({ page }) => {
  await mockBackend(page);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(routeFor('workspace'));
    const toolbar = page.locator('.oa-file-toolbar');
    const title = page.getByLabel('Document title');
    const actions = page.locator('.oa-file-actions');
    await expect(title).toBeVisible();
    const titleBox = await title.boundingBox();
    const actionsBox = await actions.boundingBox();
    const toolbarBox = await toolbar.boundingBox();
    expect(titleBox?.width ?? 0, `title width at ${width}px`).toBeGreaterThan(180);
    expect(actionsBox?.y ?? 0, `actions row below title at ${width}px`).toBeGreaterThan(titleBox?.y ?? 0);
    expect((actionsBox?.x ?? 0) + (actionsBox?.width ?? 0), `actions inside toolbar at ${width}px`).toBeLessThanOrEqual((toolbarBox?.x ?? 0) + (toolbarBox?.width ?? 0) + 1);
  }
});

test('mobile navigation opens, navigates to a tool and closes after selection', async ({ page }) => {
  await mockBackend(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByRole('button', { name: 'Open navigation' }).click();
  await expect(page.locator('.oa-sidebar')).toHaveClass(/mobile-open/);
  const sidebar = page.locator('.oa-sidebar');
  const firstFocusable = sidebar.locator('button:not([disabled]),select:not([disabled])').first();
  const lastFocusable = sidebar.locator('button:not([disabled]),select:not([disabled])').last();
  await lastFocusable.focus();
  await page.keyboard.press('Tab');
  await expect(firstFocusable).toBeFocused();
  await firstFocusable.focus();
  await page.keyboard.press('Shift+Tab');
  await expect(lastFocusable).toBeFocused();
  await page.getByRole('button', { name: 'Tools' }).click();
  await page.getByRole('dialog', { name: 'Search OpenAgent' }).locator('.oa-command-row').filter({ hasText: 'Knowledge' }).click();
  await expect(page).toHaveURL(/view=knowledge/);
  await expect(page.locator('.oa-sidebar')).not.toHaveClass(/mobile-open/);
});

test('collapsed navigation preserves names and keyboard focusability', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Close sidebar' }).click();
  await expect(page.locator('.oa-app-shell')).toHaveClass(/sidebar-collapsed/);
  const tools = page.getByRole('button', { name: 'Tools' });
  await expect(tools).toBeVisible();
  await tools.focus();
  await expect(tools).toBeFocused();
  await expect(tools).toHaveAttribute('title', 'Tools');
  for (const label of ['New chat', 'Search', 'Assistants', 'Scheduled', 'Tools', 'Settings']) {
    const button = page.getByRole('button', { name: label, exact: true });
    await expect(button).toHaveAttribute('aria-label', label);
    await expect(button).toHaveAttribute('title', label);
  }
});

test('representative screen actions preserve API contracts and safety gates', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);

  await page.goto(`${routeFor('codebase')}&project=fixture-project`);
  await page.getByRole('button', { name: 'Fixture repository' }).click();
  await page.getByLabel('Task request').fill('Inspect only; do not modify files.');
  await page.getByRole('button', { name: 'Review task preflight' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/codebase/preflight')).toBe(true);
  expect(calls.find(call => call.path === '/api/codebase/preflight')?.payload).toMatchObject({ codebase_project_id: 'repo-fixture', goal: 'Inspect only; do not modify files.', acceptance_criteria: [] });
  await expect(page.getByRole('button', { name: 'Start run after review' })).toBeVisible();
  expect(calls.some(call => call.method === 'POST' && call.path === '/api/codebase/runs')).toBe(false); // A run remains behind the explicit reviewed preflight.

  await page.goto(`${routeFor('research')}&project=fixture-project`);
  await page.getByLabel('Research question').fill('Local research fixture');
  await page.getByRole('button', { name: 'Start research' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/research/sessions')).toBe(true);
  expect(calls.find(call => call.method === 'POST' && call.path === '/api/research/sessions')?.payload).toMatchObject({ query: 'Local research fixture', project: 'fixture-project', allow_external: false });

  await page.goto(`${routeFor('knowledge')}&project=fixture-project`);
  await page.getByLabel('Question or search phrase').fill('Search within project');
  await page.getByRole('button', { name: 'Search sources' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/knowledge/hybrid-search')).toBe(true);
  expect(calls.find(call => call.path === '/api/knowledge/hybrid-search')?.payload).toMatchObject({ query: 'Search within project', project: 'fixture-project' });

  await page.goto(`${routeFor('security')}&project=fixture-project`);
  await page.getByRole('button', { name: 'Egress requests' }).click();
  await page.getByLabel('Purpose').fill('Request external access for review');
  await page.getByLabel('Destination').fill('https://example.invalid');
  await page.getByRole('button', { name: 'Submit for review' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/security/egress/request')).toBe(true);
  expect(calls.find(call => call.path === '/api/security/egress/request')?.payload).toMatchObject({ requester: 'user', purpose: 'Request external access for review', destination: 'https://example.invalid', resource_type: 'unknown', resource_id: '', risk_level: 'medium', metadata: {} });
  expect(calls.some(call => call.method === 'POST' && call.path === '/api/security/egress/resolve')).toBe(false); // Filing a request cannot approve egress.

  await page.goto(`${routeFor('vault')}&project=fixture-project`);
  await page.getByLabel('Label', { exact: true }).fill('Fixture credential');
  await page.getByLabel('Secret value').fill('fixture-only-secret');
  await page.getByLabel('Vault passphrase').fill('fixture-only-passphrase');
  await page.getByRole('button', { name: 'Encrypt and store' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/vault/store')).toBe(true);
  const vaultCall = calls.find(call => call.path === '/api/vault/store');
  expect(vaultCall?.payload).toMatchObject({ label: 'Fixture credential', secret: 'fixture-only-secret', passphrase: 'fixture-only-passphrase', project: 'fixture-project' });
  expect(vaultCall?.payload).not.toHaveProperty('value');
});

test('Python server exposes live health, capability and conversation contracts', async () => {
  const port = Number(process.env.OPENAGENT_E2E_PORT || 8899);
  const client = await playwrightRequest.newContext({ baseURL: `http://127.0.0.1:${port}` });
  try {
    const health = await client.get('/api/health');
    expect(health.ok(), `health status ${health.status()}`).toBeTruthy();
    expect((await health.json()).status).toBe('ok');
    const capabilities = await client.get('/api/capabilities');
    expect(capabilities.ok(), `capabilities status ${capabilities.status()}`).toBeTruthy();
    expect(await capabilities.json()).toBeTruthy();
    const conversations = await client.get('/api/conversations?project=default&limit=3');
    expect(conversations.ok(), `conversation status ${conversations.status()}`).toBeTruthy();
    expect(Array.isArray((await conversations.json()).conversations)).toBeTruthy();
  } finally { await client.dispose(); }
});
