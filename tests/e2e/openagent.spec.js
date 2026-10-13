import { test, expect, request as playwrightRequest } from '@playwright/test';

const views = [
  'dashboard', 'chat', 'codebase', 'voice', 'research', 'phd', 'agents', 'knowledge', 'workspace', 'automation', 'assistant', 'communications', 'models', 'security', 'developer', 'settings', 'engine', 'science', 'calendar', 'twin', 'vault', 'scraper', 'finetune',
];

async function mockBackend(page) {
  await page.route('**/api/**', async route => {
    const path = new URL(route.request().url()).pathname;
    const data = path === '/api/health'
      ? { status: 'ok', version: 'test' }
      : path === '/api/system/metrics'
        ? { cpu: { usage_percent: 0 }, memory: { used_percent: 0 }, disk: { used_percent: 0 } }
        : path === '/api/modules/health'
          ? { modules: [] }
          : path === '/api/audit'
            ? { events: [] }
            : path === '/api/conversations'
              ? { conversations: [] }
              : path === '/api/models'
                ? { providers: [] }
                : {};
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(data) });
  });
}

test('all original views deep-link, render, and keep route history in sync', async ({ page }) => {
  const errors = [];
  await mockBackend(page);
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
  await page.goto('/?view=dashboard');
  await expect(page.locator('.console')).toBeVisible();
  await expect(page.locator('.nav-item[aria-current="page"]')).toHaveCount(1);
  for (const view of views) {
    await page.goto(`/?view=${view}`);
    await expect(page.locator('#main-content h1').first(), `${view} heading`).toBeVisible();
    await expect(page.locator('.nav-item[aria-current="page"]')).toHaveCount(1);
    expect(new URL(page.url()).searchParams.get('view')).toBe(view);
  }
  await page.goto('/?view=dashboard');
  await page.locator('.nav-scroll .nav-item').nth(1).click();
  await expect(page).toHaveURL(/view=chat/);
  await page.goBack();
  await expect(page).toHaveURL(/view=dashboard/);
  expect(errors).toEqual([]);
});

test('command search supports filtering, keyboard navigation and Escape', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/');
  await page.keyboard.press('Control+k');
  const dialog = page.getByRole('dialog', { name: 'Command search' });
  await expect(dialog).toBeVisible();
  await page.getByRole('textbox', { name: 'Search screens' }).fill('knowledge');
  await expect(dialog.getByRole('option')).toHaveCount(1);
  await page.keyboard.press('Enter');
  await expect(page).toHaveURL(/view=knowledge/);
  await page.keyboard.press('Control+k');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog', { name: 'Command search' })).toHaveCount(0);
});

test('knowledge sections use accessible Radix tabs and arrow-key navigation', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/?view=knowledge');
  const searchTab = page.getByRole('tab', { name: 'Search' });
  await expect(searchTab).toHaveAttribute('aria-selected', 'true');
  await searchTab.focus();
  await page.keyboard.press('ArrowRight');
  await expect(page.getByRole('tab', { name: 'Library' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('tabpanel')).toBeVisible();
});

test('dark, light, and system themes update the document and persist', async ({ page }) => {
  await mockBackend(page);
  await page.emulateMedia({ colorScheme: 'light' });
  await page.goto('/?view=settings');
  await page.getByRole('button', { name: 'Light theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.reload();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.getByRole('button', { name: 'System theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.emulateMedia({ colorScheme: 'dark' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});

test('shell and representative workspaces have no document overflow at supported widths', async ({ page }) => {
  await mockBackend(page);
  for (const width of [1440, 1024, 768, 600, 390, 360, 320]) {
    await page.setViewportSize({ width, height: 850 });
    for (const view of ['dashboard', 'chat', 'research', 'codebase', 'settings']) {
      await page.goto(`/?view=${view}`);
      await expect(page.locator('#main-content h1').first()).toBeVisible();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
      expect(overflow, `${view} at ${width}px`).toBeLessThanOrEqual(1);
    }
  }
});

test('mobile navigation opens, changes route and closes after selection', async ({ page }) => {
  await mockBackend(page);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/?view=dashboard');
  await page.getByRole('button', { name: 'Open navigation' }).click();
  await expect(page.locator('.sidebar')).toHaveClass(/mobile-open/);
  await page.locator('.nav-scroll .nav-item').nth(1).click();
  await expect(page).toHaveURL(/view=chat/);
  await expect(page.locator('.sidebar')).not.toHaveClass(/mobile-open/);
});

test('collapsed navigation keeps accessible labels and reveals Radix tooltips', async ({ page }) => {
  await mockBackend(page);
  await page.goto('/?view=dashboard');
  await page.getByRole('button', { name: 'Collapse sidebar' }).click();
  const chatLink = page.getByRole('button', { name: 'Chat', exact: true });
  await expect(chatLink).toBeVisible();
  await chatLink.hover();
  await expect(page.getByRole('tooltip', { name: 'Chat' })).toBeVisible();
});

test('primary safe action on every view uses the expected same-origin backend contract', async ({ page }) => {
  const calls = [];
  await page.route('**/api/**', async route => {
    const request = route.request(); const url = new URL(request.url());
    let payload;
    try { payload = request.postDataJSON(); } catch { payload = undefined; }
    calls.push({ method: request.method(), path: url.pathname, payload });
    if (request.method() === 'POST' && url.pathname === '/api/chat/stream') {
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"chunk","content":"test response"}\n\n' });
      return;
    }
    if (request.method() === 'POST' && url.pathname === '/api/voice/sessions') {
      await route.fulfill({ json: { session: { id: 'session-fixture', status: 'created' } } }); return;
    }
    if (request.method() === 'POST' && url.pathname === '/api/research/sessions') {
      await route.fulfill({ json: { session: { id: 'research-fixture', query: payload?.query, status: 'queued' } } }); return;
    }
    if (url.pathname === '/api/codebase/projects' && request.method() === 'GET') {
      await route.fulfill({ json: { projects: [{ id: 'project-fixture', name: 'Fixture repository', status: 'ready' }] } }); return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, projects: [], runs: [], sessions: [], items: [], documents: [], entries: [], providers: [], models: [], settings: {}, events: [], modules: [] }) });
  });

  const generic = {
    agents: { path: '/api/agents/run-checked', fields: ['agent', 'goal', 'project', 'require_approval', 'verify'] },
    automation: { path: '/api/workflows', fields: ['name', 'project', 'trigger_type', 'steps'] },
    assistant: { path: '/api/assistant/tasks', fields: ['title', 'priority', 'project'] },
    communications: { path: '/api/communication/drafts', fields: ['platform', 'recipients', 'body', 'project'] },
    phd: { path: '/api/phd/applications', fields: ['project', 'application'] },
    developer: { path: '/api/developer/webhooks', fields: ['event', 'target_url', 'enabled', 'metadata'] },
    engine: { path: '/api/adaptive-engine/inspect', fields: ['path', 'include_tensors', 'project'] },
    finetune: { path: '/api/finetune/datasets', fields: ['name', 'format', 'project'] },
    science: { path: '/api/science/memory', fields: ['category', 'name', 'value', 'source', 'project'] },
    scraper: { path: '/api/scraper/schedule', fields: ['url', 'project', 'interval_hours', 'allow_external'] },
    twin: { path: '/api/twin/goals', fields: ['title', 'project'] },
    calendar: { path: '/api/calendar/events', fields: ['title', 'starts_at', 'ends_at', 'project'] },
  };
  const custom = {
    dashboard: { method: 'GET', path: '/api/health', required: [] },
    chat: { method: 'POST', path: '/api/chat/stream', required: ['message', 'project', 'provider', 'model', 'allow_external', 'conversation_id', 'attachments'] },
    codebase: { method: 'POST', path: '/api/codebase/runs', required: ['codebase_project_id', 'prompt', 'provider', 'allow_external'] },
    voice: { method: 'POST', path: '/api/voice/sessions', required: ['project', 'mode', 'language', 'always_listening'] },
    research: { method: 'POST', path: '/api/research/sessions', required: ['query', 'project', 'max_iterations', 'provider', 'allow_external'] },
    knowledge: { method: 'POST', path: '/api/knowledge/hybrid-search', required: ['query', 'project'] },
    workspace: { method: 'POST', path: '/api/document', required: ['title', 'content', 'format', 'project'] },
    models: { method: 'POST', path: '/api/models/discover', required: [] },
    security: { method: 'POST', path: '/api/security/egress/request', required: ['requester', 'purpose', 'destination', 'resource_type', 'resource_id', 'risk_level', 'metadata'] },
    settings: { method: 'POST', path: '/api/settings/runtime', required: ['settings'] },
    vault: { method: 'POST', path: '/api/vault/store', required: ['label', 'category', 'secret', 'passphrase', 'project'] },
  };

  for (const [view, contract] of Object.entries(custom)) {
    calls.length = 0;
    await page.goto(`/?view=${view}&project=fixture-project`);
    await expect(page.locator('#main-content h1').first()).toBeVisible();
    if (view === 'dashboard') {
      await page.getByRole('button', { name: 'Refresh overview' }).click();
    } else if (view === 'chat') {
      await page.getByRole('textbox', { name: 'Message' }).fill('Contract fixture prompt');
      await page.getByRole('button', { name: 'Send message' }).click();
    } else if (view === 'codebase') {
      await page.getByRole('button', { name: 'Fixture repository' }).click();
      await page.getByLabel('Task request').fill('Inspect only; do not modify files.');
      await page.getByRole('button', { name: 'Review task preflight' }).click();
      await page.getByRole('button', { name: 'Start run after review' }).click();
    } else if (view === 'voice') {
      await page.getByRole('button', { name: 'New session' }).click();
    } else if (view === 'research') {
      await page.getByLabel('Research question').fill('Local fixture research question');
      await page.getByRole('button', { name: 'Start research' }).click();
    } else if (view === 'knowledge') {
      await page.getByLabel('Question or search phrase').fill('fixture query');
      await page.getByRole('button', { name: 'Search sources' }).click();
    } else if (view === 'workspace') {
      await page.getByLabel('Title', { exact: true }).fill('Fixture document');
      await page.getByLabel('Content', { exact: true }).fill('Fixture content');
      await page.getByRole('button', { name: 'Create document' }).click();
    } else if (view === 'models') {
      await page.getByRole('button', { name: 'Discover models' }).click();
    } else if (view === 'security') {
      await page.getByRole('button', { name: 'Egress requests' }).click();
      await page.getByLabel('Purpose').fill('Request access for a contract fixture');
      await page.getByLabel('Destination').fill('https://example.invalid');
      await page.getByRole('button', { name: 'Submit for review' }).click();
    } else if (view === 'settings') {
      await page.getByRole('button', { name: 'Save runtime settings' }).click();
    } else if (view === 'vault') {
      await page.getByLabel('Label', { exact: true }).fill('Fixture credential');
      await page.getByLabel('Secret value').fill('fixture-only-secret');
      await page.getByLabel('Vault passphrase').fill('fixture-only-passphrase');
      await page.getByRole('button', { name: 'Encrypt and store' }).click();
    }
    await expect.poll(() => calls.some(call => call.method === contract.method && call.path === contract.path), { message: `${view}: expected ${contract.method} ${contract.path}` }).toBe(true);
    const call = calls.find(item => item.method === contract.method && item.path === contract.path);
    expect(call, `${view} should send its primary action`).toBeTruthy();
    for (const key of contract.required) expect(call?.payload, `${view} payload field ${key}`).toHaveProperty(key);
  }

  for (const [view, contract] of Object.entries(generic)) {
    calls.length = 0;
    await page.goto(`/?view=${view}&project=fixture-project`);
    const form = page.locator('.action-panel').first();
    await expect(form, `${view} resource action form`).toBeVisible({ timeout: 4000 });
    const requiredInputs = form.locator('input[required], textarea[required], select[required]');
    for (let index = 0; index < await requiredInputs.count(); index++) {
      const input = requiredInputs.nth(index); const type = await input.getAttribute('type');
      const value = type === 'number' ? '2' : view === 'communications' && await input.getAttribute('name') === 'recipients' ? 'person@example.invalid' : 'fixture value';
      try { await input.fill(value, { timeout: 3000 }); } catch (error) { throw new Error(`${view} required field could not be filled: ${error.message}`); }
    }
    const actionButton = view === 'agents'
      ? page.getByRole('button', { name: 'Run a checked agent' })
      : page.locator('.action-panel .submit-button').first();
    await expect(actionButton, `${view} primary action button`).toBeVisible({ timeout: 4000 });
    await actionButton.click({ timeout: 4000 });
    await expect.poll(() => calls.some(call => call.method === 'POST' && call.path === contract.path), { message: `${view}: expected POST ${contract.path}` }).toBe(true);
    const call = calls.find(item => item.method === 'POST' && item.path === contract.path);
    for (const key of contract.fields) expect(call?.payload, `${view} payload field ${key}`).toHaveProperty(key);
  }
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
