import { test, expect, request as playwrightRequest } from '@playwright/test';

const views = [
  'dashboard', 'chat', 'codebase', 'voice', 'research', 'agents', 'knowledge', 'workspace', 'automation', 'assistant', 'communications', 'models', 'security', 'developer', 'settings', 'engine', 'science', 'calendar', 'twin', 'vault', 'scraper', 'finetune',
];

// The Python server serves the SPA at `/`; the React shell reads query-based
// view links and normalizes them back to `/`. `/tools/*` is not a server route.
const routeFor = view => `/?view=${view}`;

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
    if (path === '/api/knowledge/upload' && request.method() === 'POST') {
      const upload = calls.at(-1);
      if (upload) {
        upload.contentType = request.headers()['content-type'];
        upload.body = request.postDataBuffer()?.toString('latin1') ?? '';
      }
      await route.fulfill({ status: 201, json: { attachment: { id: 'attachment-fixture', name: 'notes.txt' } } }); return;
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

test('all 22 active views open from deep links and remain discoverable in the More tools launcher', async ({ page }) => {
  await mockBackend(page);
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));
  await page.goto('/?view=dashboard');
  // Home leads with the workspace and primary actions; Activity remains a
  // discoverable destination in the launcher.
  await expect(page.getByRole('heading', { name: 'Welcome to OpenAgent' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Start chat' })).toBeVisible();
  await expect(page.locator('.home-prompt-tools')).toContainText('CodeQ');

  for (const view of views) {
    await page.goto(routeFor(view));
    await expect(page.locator('#oa-main-content')).toBeVisible();
    const screenTitle = page.locator('.oa-page-title');
    await expect(screenTitle, `${view} should expose a visible screen title`).toBeVisible();
    if (view === 'chat') await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
  }

  await page.getByRole('button', { name: 'More tools' }).click();
  const palette = page.getByRole('dialog', { name: 'Search OpenAgent' });
  await expect(palette).toBeVisible();
  await palette.getByRole('button', { name: 'Activity' }).click();
  await expect(page).toHaveURL(/view=dashboard.*activity=1|activity=1.*view=dashboard/);
  await expect(page.getByRole('heading', { name: 'Welcome to OpenAgent' })).toBeVisible();
  await page.goto('/?view=phd');
  await expect(page.getByRole('textbox', { name: 'Message' })).toBeVisible();
  await expect(page).toHaveURL(/view=chat/);
  await page.getByRole('button', { name: 'More tools' }).click();
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

test('chat fills the page stage and keeps the welcome and composer visible at desktop and mobile sizes', async ({ page }) => {
  await mockBackend(page);
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    await page.goto('/?view=chat&project=fixture-project');
    const stage = await page.locator('.oa-page-stage').boundingBox();
    const chat = await page.locator('.oa-chat-page').boundingBox();
    const surface = await page.locator('.oa-chat-surface').boundingBox();
    const timeline = await page.locator('.oa-chat-timeline').boundingBox();
    const welcome = await page.locator('.oa-chat-welcome').boundingBox();
    const composer = await page.locator('.oa-chat-composer').boundingBox();
    expect(stage).toBeTruthy();
    expect(chat?.height ?? 0, `chat height at ${viewport.width}px`).toBeGreaterThanOrEqual((stage?.height ?? 0) - 1);
    expect(surface?.height ?? 0, `surface height at ${viewport.width}px`).toBeGreaterThanOrEqual((stage?.height ?? 0) - 1);
    expect(timeline?.height ?? 0, `timeline height at ${viewport.width}px`).toBeGreaterThan((stage?.height ?? 0) * 0.85);
    for (const [name, box] of [['welcome', welcome], ['composer', composer]]) {
      expect(box, `${name} rendered at ${viewport.width}px`).toBeTruthy();
      expect(box?.y ?? -1, `${name} top at ${viewport.width}px`).toBeGreaterThanOrEqual(stage?.y ?? 0);
      expect((box?.y ?? Infinity) + (box?.height ?? 0), `${name} bottom at ${viewport.width}px`).toBeLessThanOrEqual((stage?.y ?? 0) + (stage?.height ?? 0) + 1);
    }
  }
});

test('chat provider and assistant selectors apply live selections and actions', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);
  await page.goto('/?view=chat&project=fixture-project');
  await page.getByLabel('Chat provider').selectOption('local-provider');
  await expect(page.getByLabel('Chat provider')).toHaveValue('local-provider');
  await page.getByRole('textbox', { name: 'Message' }).fill('Use selected provider');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect.poll(() => calls.find(call => call.method === 'POST' && call.path === '/api/chat/stream')?.payload?.provider).toBe('local-provider');

  await page.goto('/?view=chat&project=fixture-project');
  await page.getByLabel('Assistant').selectOption('executive');
  await expect(page.getByText('Checked run')).toBeVisible();
  await page.getByRole('textbox', { name: 'Message' }).fill('Plan this local project');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByText('Checked agent plan.')).toBeVisible();
  const agentCall = calls.find(call => call.method === 'POST' && call.path === '/api/agents/run-checked');
  expect(agentCall?.payload).toMatchObject({ agent: 'executive', goal: 'Plan this local project', project: 'fixture-project', verify: true });
});

test('chat accepts dropped files only after the upload endpoint receives the file', async ({ page }) => {
  const calls = [];
  await mockBackend(page, calls);
  await page.goto('/?view=chat&project=fixture-project');

  const transfer = await page.evaluateHandle(() => {
    const data = new DataTransfer();
    data.items.add(new File(['fixture attachment contents'], 'notes.txt', { type: 'text/plain' }));
    return data;
  });
  const composer = page.locator('.oa-chat-composer');
  await composer.evaluate((element, data) => {
    element.dispatchEvent(new DragEvent('dragenter', { bubbles: true, cancelable: true, dataTransfer: data }));
  }, transfer);
  await expect(page.getByText('Drop files to attach')).toBeVisible();
  await composer.evaluate((element, data) => {
    element.dispatchEvent(new DragEvent('drop', { bubbles: true, cancelable: true, dataTransfer: data }));
  }, transfer);

  await expect.poll(() => calls.find(call => call.method === 'POST' && call.path === '/api/knowledge/upload')).toBeTruthy();
  const upload = calls.find(call => call.method === 'POST' && call.path === '/api/knowledge/upload');
  expect(upload?.contentType).toMatch(/^multipart\/form-data; boundary=/);
  expect(upload?.body).toContain('filename="notes.txt"');
  expect(upload?.body).toContain('name="project"\r\n\r\nfixture-project');
  await expect(page.locator('.oa-chat-attachment')).toContainText('notes.txt');

  await page.getByRole('textbox', { name: 'Message' }).fill('Use the uploaded notes');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect.poll(() => calls.find(call => call.method === 'POST' && call.path === '/api/chat/stream')?.payload?.attachments).toEqual([{ id: 'attachment-fixture', name: 'notes.txt' }]);
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
  await page.getByLabel('Appearance').selectOption('light');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByLabel('Appearance').selectOption('dark');
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

test('chat send and stop controls stay inside the composer on narrow screens', async ({ page }) => {
  await mockBackend(page);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(routeFor('chat'));
    const composer = page.locator('.oa-chat-composer');
    const send = page.locator('.oa-chat-send');
    await expect(composer).toBeVisible();
    await expect(send).toBeVisible();
    const composerBox = await composer.boundingBox();
    const sendBox = await send.boundingBox();
    const pageWidth = await page.evaluate(() => document.documentElement.clientWidth);
    expect(composerBox).toBeTruthy();
    expect(sendBox).toBeTruthy();
    expect(sendBox.x, `send control left edge at ${width}px`).toBeGreaterThanOrEqual(composerBox.x - 1);
    expect(sendBox.x + sendBox.width, `send control right edge within composer at ${width}px`).toBeLessThanOrEqual(composerBox.x + composerBox.width + 1);
    expect(sendBox.x + sendBox.width, `send control right edge within viewport at ${width}px`).toBeLessThanOrEqual(pageWidth + 1);
  }
});

test('workspace document title gets a readable mobile toolbar row', async ({ page }) => {
  await mockBackend(page);
  for (const width of [390, 320]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto(routeFor('workspace'));
    // The redesigned empty state keeps the editor closed until the user asks
    // to create a document.
    await page.getByRole('button', { name: 'Create document' }).click();
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
  await page.getByRole('button', { name: 'More tools' }).click();
  await page.getByRole('dialog', { name: 'Search OpenAgent' }).locator('.oa-command-row').filter({ hasText: 'Knowledge' }).click();
  await expect(page).toHaveURL(/view=knowledge/);
  await expect(page.locator('.oa-sidebar')).not.toHaveClass(/mobile-open/);
});

test('collapsed navigation preserves names and keyboard focusability', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.getByRole('button', { name: 'Close sidebar' }).click();
  await expect(page.locator('.oa-app-shell')).toHaveClass(/sidebar-collapsed/);
  const tools = page.getByRole('button', { name: 'More tools' });
  await expect(tools).toBeVisible();
  await tools.focus();
  await expect(tools).toBeFocused();
  await expect(tools).toHaveAttribute('title', 'More tools');
  for (const label of ['New chat', 'Search', 'Agents', 'Automations', 'More tools', 'Settings']) {
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
  await page.getByLabel('Task').fill('Inspect only; do not modify files.');
  await page.getByRole('button', { name: 'Review task' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/codebase/preflight')).toBe(true);
  expect(calls.find(call => call.path === '/api/codebase/preflight')?.payload).toMatchObject({ codebase_project_id: 'repo-fixture', goal: 'Inspect only; do not modify files.', acceptance_criteria: [] });
  await expect(page.getByRole('button', { name: 'Start reviewed run' })).toBeVisible();
  expect(calls.some(call => call.method === 'POST' && call.path === '/api/codebase/runs')).toBe(false); // A run remains behind the explicit reviewed preflight.

  await page.goto(`${routeFor('research')}&project=fixture-project`);
  await page.getByLabel('Research question').fill('Local research fixture');
  await page.getByRole('button', { name: 'Start research' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/research/sessions')).toBe(true);
  expect(calls.find(call => call.method === 'POST' && call.path === '/api/research/sessions')?.payload).toMatchObject({ query: 'Local research fixture', project: 'fixture-project', allow_external: false });

  await page.goto(`${routeFor('knowledge')}&project=fixture-project`);
  await page.getByLabel('Search project knowledge').fill('Search within project');
  await page.locator('.knowledge-query').getByRole('button', { name: 'Search', exact: true }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/knowledge/hybrid-search')).toBe(true);
  expect(calls.find(call => call.path === '/api/knowledge/hybrid-search')?.payload).toMatchObject({ query: 'Search within project', project: 'fixture-project' });

  await page.goto(`${routeFor('security')}&project=fixture-project`);
  await page.getByRole('button', { name: 'Access', exact: true }).click();
  await page.getByRole('button', { name: 'Egress requests' }).click();
  await page.getByLabel('Purpose').fill('Request external access for review');
  await page.getByLabel('Destination').fill('https://example.invalid');
  await page.getByRole('button', { name: 'Submit for review' }).click();
  await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === '/api/security/egress/request')).toBe(true);
  expect(calls.find(call => call.path === '/api/security/egress/request')?.payload).toMatchObject({ requester: 'user', purpose: 'Request external access for review', destination: 'https://example.invalid', resource_type: 'unknown', resource_id: '', risk_level: 'medium', metadata: {} });
  expect(calls.some(call => call.method === 'POST' && call.path === '/api/security/egress/resolve')).toBe(false); // Filing a request cannot approve egress.

  await page.goto(`${routeFor('vault')}&project=fixture-project`);
  await page.getByRole('button', { name: 'Add your first entry' }).click();
  await page.getByLabel('Label', { exact: true }).fill('Fixture credential');
  await page.getByLabel('Secret value').fill('fixture-only-secret');
  await page.getByRole('dialog').getByLabel('Vault passphrase').fill('fixture-only-passphrase');
  await page.getByRole('dialog').getByRole('button', { name: 'Encrypt and store' }).click();
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
