import { expect, test } from '@playwright/test';

const screens = [
  'home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'models',
  'knowledge', 'voice', 'assistant', 'communications', 'security', 'developer', 'engine', 'science',
  'twin', 'vault', 'scraper', 'finetune', 'settings', 'tools',
];

test('all 23 reference routes render from deep links', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  for (const route of screens) {
    await page.goto(`/?view=${route}`);
    await expect(page.locator('#content')).toBeVisible();
    // Wait for the lazy-loaded screen itself. The Suspense fallback also uses
    // `.page`, so asserting that class alone can pass before the route renders.
    if (route === 'chat') {
      await expect(page.locator('#content .split-layout')).toBeVisible();
    } else if (route === 'codeq') {
      await expect(page.locator('#content .codeq-root')).toBeVisible();
    } else {
      await expect(page.locator('#content .reference-screen-motion h1').first()).toBeVisible();
    }
  }
  expect(errors).toEqual([]);
});

test('sidebar routing, appearance, and command search work', async ({ page }) => {
  await page.goto('/?view=home');
  await page.getByRole('button', { name: 'CodeQ' }).first().click();
  await expect(page).toHaveURL(/view=codeq/);
  await page.getByRole('button', { name: 'Switch to dark theme' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.getByRole('button', { name: 'Open command palette' }).click();
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByPlaceholder('Search workspaces').fill('vault');
  await page.getByRole('dialog').getByRole('button', { name: 'Vault' }).click();
  await expect(page).toHaveURL(/view=vault/);
});

test('representative screens fit desktop and mobile widths without horizontal page overflow', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    for (const route of ['home', 'chat', 'codeq', 'agents', 'research', 'files', 'automations', 'calendar', 'vault']) {
      await page.goto(`/?view=${route}`);
      await page.locator('#content').waitFor();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
      expect(overflow, `${route} at ${viewport.width}px`).toBe(false);
    }
  }
});

test('every screen remains within the viewport at both supported desktop sizes', async ({ page }) => {
  for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
    await page.setViewportSize(viewport);
    for (const route of screens) {
      await page.goto(`/?view=${route}`);
      await page.locator('#content').waitFor();
      // Wait for the corresponding lazy route rather than measuring its fallback.
      await expect(page.locator('#content .reference-screen-motion').first()).toBeVisible();
      const dimensions = await page.evaluate(() => ({
        docWidth: document.documentElement.scrollWidth,
        viewWidth: document.documentElement.clientWidth,
        frameWidth: document.querySelector('.app-frame')?.getBoundingClientRect().width ?? 0,
      }));
      expect(dimensions.docWidth, `${route} document at ${viewport.width}px`).toBeLessThanOrEqual(dimensions.viewWidth);
      expect(dimensions.frameWidth, `${route} frame at ${viewport.width}px`).toBeGreaterThan(0);
    }
  }
});

test('keyboard navigation, sidebar controls, and mobile drawer are operable', async ({ page }) => {
  await page.goto('/?view=home');
  await page.keyboard.press('Control+k');
  const dialog = page.getByRole('dialog');
  await expect(dialog).toBeVisible();
  await page.getByPlaceholder('Search workspaces').fill('vault');
  await expect(dialog.getByRole('button', { name: 'Vault' })).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();

  await page.getByRole('button', { name: 'Collapse sidebar' }).first().click();
  await expect(page.locator('#sidebar')).toHaveClass(/collapsed/);
  await page.getByRole('button', { name: 'Expand sidebar' }).first().click();
  await expect(page.locator('#sidebar')).not.toHaveClass(/collapsed/);

});

test('workspace forms and panels use backend-backed chat and keep unsupported panels honest', async ({ page }) => {
  await page.goto('/?view=home');
  await page.getByRole('textbox', { name: 'Ask OpenAgent' }).fill('Plan a local task');
  await page.getByRole('button', { name: 'Open prompt in chat' }).click();
  await expect(page).toHaveURL(/view=chat/);
  await expect(page.getByRole('textbox', { name: 'Message OpenAgent' })).toHaveValue('Plan a local task');
  await page.route('**/api/chat/stream', async route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: 'data: {"type":"conversation","conversation":{"id":"e2e-chat-1"}}\n\ndata: {"type":"chunk","content":"Connected reply"}\n\ndata: {"type":"done","done":true}\n\n' }));
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.getByRole('paragraph').filter({ hasText: 'Plan a local task' })).toBeVisible();
  await expect(page.getByText('Connected reply')).toBeVisible();

  await page.goto('/?view=research');
  await page.getByRole('tab', { name: 'Notes' }).first().click();
  await page.getByRole('textbox', { name: 'Research note draft' }).fill('Local review note');
  await expect(page.getByText(/not sent or persisted/i)).toBeVisible();

  await page.goto('/?view=codeq');
  const codeqGeometry = await page.locator('.codeq-work').evaluate(element => {
    const rect = (selector: string) => {
      const node = element.querySelector<HTMLElement>(selector);
      if (!node) return null;
      const { x, y, width, height } = node.getBoundingClientRect();
      return { x: Math.round(x), y: Math.round(y), width: Math.round(width), height: Math.round(height) };
    };
    return {
      toolbar: rect('header.codeq-toolbar'),
      panelTabs: rect('.codeq-toolbar:not(header)'),
      fileTabs: rect('.codeq-tabbar'),
      previewHeader: rect('.codeq-preview-head'),
      status: rect('.codeq-statusbar'),
    };
  });
  expect(codeqGeometry.toolbar?.height).toBe(43);
  expect(codeqGeometry.panelTabs?.height).toBe(43);
  expect(codeqGeometry.fileTabs?.height).toBe(38);
  // The approved standalone CodeQ preview header measures 42 CSS pixels.
  expect(codeqGeometry.previewHeader?.height).toBe(42);
  expect(codeqGeometry.status?.height).toBe(27);
  await expect(page.locator('.codeq-assistant > form.chat-composer-outer')).toBeVisible();
  const assistantComposerBottomGap = await page.locator('.codeq-assistant').evaluate(element => {
    const composer = element.querySelector<HTMLElement>(':scope > .chat-composer-outer');
    if (!composer) return Number.NaN;
    return Math.abs(Math.round(element.getBoundingClientRect().bottom - composer.getBoundingClientRect().bottom));
  });
  expect(assistantComposerBottomGap).toBe(0);
  await page.getByRole('tab', { name: 'Preview' }).click();
  await expect(page.getByRole('heading', { name: 'Open a project to preview' })).toBeVisible();
  await page.locator('.codeq-assistant').getByRole('button', { name: 'Close assistant' }).click();
  await expect(page.locator('.codeq-assistant')).toHaveCount(0);
  await page.getByRole('button', { name: 'Open assistant' }).click();
  await expect(page.locator('.codeq-assistant')).toBeVisible();
});

test('calendar period controls match Day, Week, and Month views', async ({ page }) => {
  await page.goto('/?view=calendar');
  const dateHeading = page.locator('.calendar-heading').locator('strong').last();

  await page.getByRole('button', { name: 'Day', exact: true }).click();
  await expect(page.locator('.calendar-days .day-head')).toHaveCount(1);
  const initialDay = await dateHeading.textContent();
  await page.getByRole('button', { name: 'Next period' }).click();
  await expect(dateHeading).not.toHaveText(initialDay ?? '');

  await page.getByRole('button', { name: 'Month', exact: true }).click();
  await expect(page.locator('.calendar-days .day-head')).toHaveCount(7);
  await expect(page.locator('.calendar-month-day')).toHaveCount(42);
  const initialMonth = await dateHeading.textContent();
  await page.getByRole('button', { name: 'Next period' }).click();
  await expect(dateHeading).not.toHaveText(initialMonth ?? '');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/?view=calendar');
  const calendarHeading = page.locator('.calendar-heading');
  const newEvent = page.getByRole('button', { name: /^New$/ });
  await expect(calendarHeading).toBeVisible();
  await expect(newEvent).toBeVisible();
  const controlsFit = await page.evaluate(() => {
    const container = document.querySelector('.calendar-heading')?.getBoundingClientRect();
    const control = [...document.querySelectorAll('.calendar-heading button')].find(button => button.textContent?.includes('New'))?.getBoundingClientRect();
    return { container: container && { left: container.left, right: container.right }, control: control && { left: control.left, right: control.right } };
  });
  expect(controlsFit && controlsFit.container && controlsFit.control && controlsFit.control.left >= controlsFit.container.left && controlsFit.control.right <= controlsFit.container.right, `calendar action buttons fit inside the mobile heading: ${JSON.stringify(controlsFit)}`).toBe(true);
});

test('chat detail tabs support keyboard navigation and unavailable file actions are disabled', async ({ page }) => {
  await page.goto('/?view=chat');
  const tablist = page.getByRole('tablist', { name: 'Conversation details' });
  const preview = tablist.getByRole('tab', { name: 'Preview' });
  await preview.focus();
  await page.keyboard.press('ArrowLeft');
  await expect(tablist.getByRole('tab', { name: 'Files' })).toHaveAttribute('aria-selected', 'true');
  const addFiles = page.getByRole('button', { name: 'Add files', exact: true });
  await expect(addFiles).toBeDisabled();
  await expect(addFiles).toHaveAttribute('title', 'Adding files requires a connected workspace');
});

test('Chat demo opens cleanly without replaying the HTML showcase conversation', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.goto('/?view=chat');

  await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Conversation' }).locator('option')).toHaveCount(1);
  await expect(page.locator('.chat-thread .chat-message')).toHaveCount(0);
  await expect(page.getByText('I’m working on a frontend project and would like to improve its structure and visual design.')).toHaveCount(0);
  await expect(page.getByRole('button', { name: /GPT-4o/ })).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Home recent-chat fixtures stay out of Chat conversation state', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  const apiRequests: string[] = [];
  const keyWarnings: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  page.on('console', message => { if (message.type() === 'error' && /unique ["']key["'] prop/i.test(message.text())) keyWarnings.push(message.text()); });
  await page.goto('/?view=home');
  await page.waitForResponse(response => new URL(response.url()).pathname === '/demo/approved-shell-fixtures.json');
  await page.getByRole('button', { name: 'Chat', exact: true }).first().click();
  await expect(page).toHaveURL(/view=chat/);
  await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
  await expect(page.getByRole('combobox', { name: 'Conversation' }).locator('option')).toHaveCount(1);
  expect(apiRequests).toEqual([]);
  expect(keyWarnings).toEqual([]);
});

test('sidebar recent chats load backend records and open the selected conversation', async ({ page }) => {
  await page.route('**/api/health', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ status: 'ok' }) }));
  await page.route('**/api/conversations**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ conversations: [{ id: 'conv-1', title: 'Planning notes' }] }) }));
  await page.route('**/api/messages**', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ messages: [] }) }));
  await page.route('**/api/models', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ providers: [] }) }));

  await page.goto('/?view=home');
  const recentChat = page.locator('#sidebar').getByRole('button', { name: 'Planning notes' });
  await expect(recentChat).toBeVisible();
  await recentChat.click();
  await expect(page).toHaveURL(/view=chat/);
  await expect(page.getByRole('combobox', { name: 'Conversation' })).toHaveValue('conv-1');
});

test('automation preview is a backend dry-run and live confirmation keeps network disabled by default', async ({ page }) => {
  const workflow = { id: 'wf-e2e', name: 'Fetch reference', description: 'Review a remote page', project: 'default', enabled: true, trigger_type: 'manual', steps: [{ action: 'browser_fetch', url: 'https://example.test' }] };
  const runBodies: Record<string, unknown>[] = [];
  await page.route('**/api/workflows/builder', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ actions: { browser_fetch: { label: 'Fetch Web Page', category: 'browser', requires_network: true } } }) }));
  await page.route('**/api/workflows?project=default*', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ workflows: [workflow] }) }));
  await page.route('**/api/workflow-runs?project=default*', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ runs: [] }) }));
  await page.route('**/api/workflows/wf-e2e/run', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    runBodies.push(body);
    const dryRun = body.dry_run === true;
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ run: { id: dryRun ? 'preview-1' : 'live-1', workflow_id: 'wf-e2e', dry_run: dryRun, status: dryRun ? 'preview' : 'completed', result: { dry_run: dryRun, steps: [{ action: 'browser_fetch', status: dryRun ? 'planned' : 'completed' }] } } }) });
  });

  await page.goto('/?view=automations');
  await page.getByRole('button', { name: /Fetch reference/ }).click();
  await page.getByRole('button', { name: 'Preview run' }).click();
  await expect(page.getByText('Preview result')).toBeVisible();
  await expect(page.getByText('planned')).toBeVisible();
  expect(runBodies).toHaveLength(1);
  expect(runBodies[0]).toMatchObject({ dry_run: true, allow_network: false, project: 'default' });

  const networkConsent = page.getByRole('checkbox', { name: 'Allow network access for this live run' });
  await expect(networkConsent).not.toBeChecked();
  let confirmation = '';
  page.once('dialog', async dialog => { confirmation = dialog.message(); await dialog.accept(); });
  await page.getByRole('button', { name: 'Confirm live run' }).click();
  await expect.poll(() => runBodies).toHaveLength(2);
  expect(confirmation.includes('network access disabled')).toBe(true);
  expect(runBodies[1]).toMatchObject({ dry_run: false, allow_network: false, project: 'default' });
  await expect(page.getByText('Run result')).toBeVisible();
});

test('research run sends external-source access disabled when consent remains off', async ({ page }) => {
  const submissions: Record<string, unknown>[] = [];
  await page.route('**/api/research/sessions?project=default*', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) }));
  await page.route('**/api/research/sessions', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    submissions.push(body);
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ session: { id: 'research-local', query: body.query, status: 'completed', allow_external: body.allow_external } }) });
  });

  await page.goto('/?view=research');
  await page.getByText('New research', { exact: true }).click();
  const externalConsent = page.getByRole('checkbox', { name: 'Allow external research sources' });
  await expect(externalConsent).not.toBeChecked();
  await page.getByRole('textbox', { name: 'Research question' }).fill('Local-only question');
  await page.getByRole('button', { name: 'Run research' }).click();
  await expect.poll(() => submissions).toHaveLength(1);
  expect(submissions[0]).toMatchObject({ query: 'Local-only question', project: 'default', allow_external: false });
});

test('research run sends external-source access only after explicit per-run consent', async ({ page }) => {
  const submissions: Record<string, unknown>[] = [];
  await page.route('**/api/research/sessions?project=default*', route => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ sessions: [] }) }));
  await page.route('**/api/research/sessions', async route => {
    const body = route.request().postDataJSON() as Record<string, unknown>;
    submissions.push(body);
    await route.fulfill({ status: 201, contentType: 'application/json', body: JSON.stringify({ session: { id: 'research-external', query: body.query, status: 'completed', allow_external: body.allow_external } }) });
  });
  await page.goto('/?view=research');
  await page.getByText('New research', { exact: true }).click();
  const externalConsent = page.getByRole('checkbox', { name: 'Allow external research sources' });
  await expect(externalConsent).not.toBeChecked();
  await externalConsent.check();
  await page.getByRole('textbox', { name: 'Research question' }).fill('External-source question');
  await page.getByRole('button', { name: 'Run research' }).click();
  await expect.poll(() => submissions).toHaveLength(1);
  expect(submissions[0]).toMatchObject({ query: 'External-source question', project: 'default', allow_external: true });
});
