import { expect, test } from '@playwright/test';

test('Design Preview tutorial is discoverable only from Developer in Demo mode and hidden in Live mode', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  for (const route of ['agents', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'assistant']) {
    await page.goto(`/?view=${route}`);
    await expect(page.getByRole('button', { name: 'Open Design Preview sandbox' })).toHaveCount(0);
  }
  await page.goto('/?view=developer');
  await expect(page.getByRole('button', { name: 'Open Design Preview sandbox' })).toBeVisible();
  expect(apiRequests).toEqual([]);

  await page.getByRole('button', { name: /Data mode: demo/ }).click();
  await expect(page.getByRole('button', { name: 'Open Design Preview sandbox' })).toHaveCount(0);
});

test('Advanced integration guide matches the approved local demo workflow and exports example records', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.goto('/?view=voice');
  await page.getByRole('button', { name: 'Integration guide' }).click();
  const dialog = page.getByRole('dialog', { name: 'Demo and integration guide' });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByText(/Every interaction.*local example records/i)).toBeVisible();
  await expect(dialog.getByText('Current screen').locator('..')).toContainText('voice');
  await expect(dialog.getByText('Record collection').locator('..')).toContainText('recordings');
  const downloadPromise = page.waitForEvent('download');
  await dialog.getByRole('button', { name: 'Export sample data' }).click();
  expect((await downloadPromise).suggestedFilename()).toBe('openagent-demo-voice.json');
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  expect(apiRequests).toEqual([]);
});

test('Advanced integration guide is truthful and available in Live without demo export or fixtures', async ({ page }) => {
  const fixtureRequests: string[] = [];
  page.on('request', request => {
    if (new URL(request.url()).pathname.startsWith('/demo/')) fixtureRequests.push(request.url());
  });
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'live'));
  await page.route('**/*', route => {
    if (new URL(route.request().url()).pathname.startsWith('/api/')) {
      return route.fulfill({
        status: 503,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Offline Live test fixture' }),
      });
    }
    return route.continue();
  });
  await page.goto('/?view=voice');
  await expect(page.locator('.p-workspace--voice')).toBeVisible({ timeout: 15_000 });
  const trigger = page.getByRole('button', { name: 'Integration guide' });
  await trigger.click();
  const guide = page.getByRole('dialog', { name: 'Demo and integration guide' });
  await expect(guide).toBeVisible();
  await expect(guide).toContainText('Backend permissions apply');
  await expect(guide).toContainText('Live export is unavailable here. No backend export route is configured for this guide.');
  await expect(guide).not.toContainText('Only example data');
  await expect(guide.getByRole('button', { name: 'Export sample data' })).toHaveCount(0);
  await page.keyboard.press('Escape');
  await expect(guide).toBeHidden();
  await expect(trigger).toBeFocused();
  expect(fixtureRequests).toEqual([]);
});

test('Developer connections demo searches and saves locally without API calls', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=developer');
  await expect(page.locator('.p-workspace--developer')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Connect your tools safely' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('GitHub', { exact: true })).toBeVisible();
  await expect(page.getByText('MCP filesystem', { exact: true })).toBeVisible();
  await expect(page.getByText('Slack', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await page.getByLabel('Connection name').fill('Local fixture connector');
  await page.getByLabel('Description').fill('No external service');
  await page.getByRole('button', { name: 'Save local connection' }).click();
  await expect(page.getByText('Local fixture connector', { exact: true })).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Security overview shows canonical demo counts, clearly labeled as simulated', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=security');
  await expect(page.locator('.p-workspace--security')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Clear visibility and control' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('heading', { name: 'Security summary' })).toBeVisible();
  const stats = page.locator('.demo-stats-grid');
  await expect(stats.getByText('Active policies', { exact: true })).toBeVisible();
  await expect(stats.getByText('Sessions', { exact: true })).toBeVisible();
  await expect(stats.getByText('Audit entries', { exact: true })).toBeVisible();
  await expect(page.getByText(/no real authorisation/i)).toBeVisible();
});

test('Personal context demo uses local preferences and supports add, search, and toggle', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=twin');
  await expect(page.getByRole('heading', { name: 'Your context, your choice' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Context profile' })).toBeVisible();
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await page.getByLabel('Preference name').fill('Local preference');
  await page.getByLabel('Description').fill('Stored only in the demo fixture');
  await page.getByRole('button', { name: 'Save preference' }).click();
  await expect(page.locator('.twin-demo-preferences').getByText('Local preference', { exact: true }).first()).toBeVisible();
  const responsePreference = page.locator('.twin-demo-preferences .p-item').filter({ hasText: 'Response style' }).locator('.twin-demo-toggle');
  await responsePreference.click();
  await expect(responsePreference).toHaveAttribute('aria-pressed', 'false');
  await page.getByRole('textbox', { name: 'Search preferences' }).fill('Preferred tools');
  await expect(page.locator('.twin-demo-preferences').getByText('Response style', { exact: true })).toHaveCount(0);
  await expect(page.locator('.twin-demo-preferences').getByText('Preferred tools', { exact: true }).first()).toBeVisible();
  expect(apiRequests).toEqual([]);
});

test('Vault demo shows synthetic masked credentials and filters locally', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=vault');
  await expect(page.getByRole('heading', { name: 'Security without friction' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Masked credentials' })).toBeVisible();
  await expect(page.getByText(/synthetic placeholders/i)).toBeVisible();
  await page.getByRole('textbox', { name: 'Filter vault entries' }).fill('GitHub');
  await expect(page.getByText('GitHub integration token', { exact: true })).toBeVisible();
  await expect(page.getByText('OpenAI development key', { exact: true })).toHaveCount(0);
  expect(apiRequests).toEqual([]);
});

test('Vault demo adds a masked example locally without posting secret data', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => { localStorage.setItem('oa-second-data-mode', 'demo'); localStorage.removeItem('oa-second-demo-vault-entries'); });
  await page.goto('/?view=vault');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const dialog = page.getByRole('dialog', { name: 'Add secret' });
  await expect(dialog).toBeVisible();
  await dialog.getByLabel('Name').fill('Local sample credential');
  await dialog.getByLabel('Description').fill('Example only');
  await dialog.getByLabel('Scope').fill('Read-only demo scope');
  await dialog.getByRole('button', { name: 'Add secret', exact: true }).click();
  await expect(dialog).toBeHidden();
  await expect(page.getByText('Local sample credential', { exact: true })).toBeVisible();
  const saved = await page.evaluate(() => JSON.parse(localStorage.getItem('oa-second-demo-vault-entries') ?? '[]'));
  expect(saved).toHaveLength(1);
  expect(saved[0]).toMatchObject({ label: 'Local sample credential', value: '••••••••', demo: true });
  expect(apiRequests).toEqual([]);
});

test('Personal Assistant demo renders approved tasks and supports local planning actions', async ({ page }) => {
  const apiRequests: string[] = [];
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'demo'));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto('/?view=assistant');
  await expect(page.getByRole('heading', { name: 'Stay on top of your work' })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Make progress on the important things.' })).toBeVisible();
  await expect(page.getByText('Review CodeQ permissions', { exact: true }).first()).toBeVisible();
  await page.getByRole('button', { name: 'Add task', exact: true }).click();
  await page.getByRole('textbox', { name: 'Task', exact: true }).fill('Prepare a parity review');
  await page.getByRole('button', { name: 'Save locally' }).click();
  await expect(page.locator('.assistant-demo-list').getByText('Prepare a parity review', { exact: true })).toBeVisible();
  // The assistant workspace has no activity-event provider, so it must not
  // promise a feed that does not exist. Design Preview is discoverable from
  // Developer, separate from workspace activity.
  await expect(page.locator('.advanced-demo-activity')).toHaveCount(0);
  await page.goto('/?view=developer');
  await expect(page.getByRole('heading', { name: 'Developer' })).toBeVisible();
  await page.getByRole('button', { name: 'Open Design Preview sandbox' }).click();
  await expect(page.getByRole('region', { name: 'Demo tutorial sandbox' })).toBeVisible();
  await page.getByRole('button', { name: 'Close demo tutorial sandbox', exact: true }).click();
  await page.goto('/?view=assistant');
  await expect(page.getByRole('heading', { name: 'Stay on top of your work' })).toBeVisible();
  await page.getByRole('button', { name: 'Mark Review CodeQ permissions complete' }).click();
  await expect(page.locator('.assistant-demo-row').filter({ hasText: 'Review CodeQ permissions' }).getByText('Done', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Generate brief' }).click();
  await expect(page.getByRole('tab', { name: 'Daily brief' })).toHaveAttribute('aria-selected', 'true');
  await page.getByRole('tab', { name: 'Goals' }).click();
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  await expect(page.getByRole('spinbutton', { name: 'Progress' })).toHaveValue('65');
  const createGoal = page.getByRole('button', { name: 'Create item' });
  await expect(createGoal).toBeEnabled();
  await createGoal.click();
  await expect(page.getByRole('textbox', { name: 'Name' })).toHaveJSProperty('validity.valid', false);
  await expect(page.getByRole('dialog')).toBeVisible();
  await page.getByRole('textbox', { name: 'Name' }).fill('Complete desktop acceptance');
  await page.getByRole('textbox', { name: 'Description' }).fill('Review paired canonical and React captures.');
  await page.getByRole('spinbutton', { name: 'Progress' }).fill('65');
  await page.getByRole('button', { name: 'Create item' }).click();
  await expect(page.locator('.assistant-demo-list').getByText('Complete desktop acceptance', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Edit Complete desktop acceptance' }).click();
  await expect(page.getByRole('dialog')).toContainText('Edit goal');
  await page.getByRole('spinbutton', { name: 'Progress' }).fill('77');
  await page.getByRole('button', { name: 'Save changes' }).click();
  await page.getByRole('button', { name: 'More actions for Complete desktop acceptance' }).click();
  await expect(page.locator('.assistant-demo-detail')).toContainText('77% complete');
  expect(apiRequests).toEqual([]);
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Personal Assistant local activity works and persists (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const apiRequests: string[] = [];
      page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'demo');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      await page.goto('/?view=assistant');
      await page.getByRole('tab', { name: 'Goals' }).click();

      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('New actions you take in this demo will appear here.');
      await expect(page.locator('.assistant-demo-rail .assistant-demo-metric').nth(1).locator('strong')).toHaveText('0');
      await page.getByRole('button', { name: 'Add', exact: true }).click();
      const goalDialog = page.getByRole('dialog', { name: 'New goal' });
      await goalDialog.getByLabel('Name').fill('Quarterly launch goal');
      await goalDialog.getByRole('button', { name: 'Create item' }).click();
      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('Goal created');
      await expect(page.locator('.assistant-demo-rail .assistant-demo-metric').nth(1).locator('strong')).toHaveText('1');
      const activityRegion = page.getByRole('region', { name: 'Personal assistant recent activity' });
      const activityDialog = page.getByRole('dialog', { name: 'Demo activity' });
      await activityRegion.getByRole('button', { name: 'All activity' }).click();
      await expect(activityDialog).toBeVisible();
      await expect(activityDialog).toContainText('Actions completed in this browser, not from a real backend.');
      await expect(activityDialog).toContainText('Quarterly launch goal');
      await expect(activityDialog).toContainText('Saved locally · demo');
      await page.keyboard.press('Escape');

      await page.getByRole('button', { name: 'Edit Quarterly launch goal' }).click();
      await page.getByLabel('Description').fill('Updated only in this local demo.');
      await page.getByRole('button', { name: 'Save changes' }).click();
      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('Goal edited');
      await activityRegion.getByRole('button', { name: 'All activity' }).click();
      await expect(activityDialog).toBeVisible();
      await expect(activityDialog.getByText('Goal edited')).toBeVisible();
      await page.keyboard.press('Escape');

      await page.getByRole('tab', { name: 'Tasks' }).click();
      await page.getByRole('button', { name: 'Mark Review CodeQ permissions complete' }).click();
      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('Task completed');
      await page.reload();
      await page.getByRole('tab', { name: 'Tasks' }).click();
      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('Task completed');
      await activityRegion.getByRole('button', { name: 'All activity' }).click();
      await expect(activityDialog).toBeVisible();
      await expect(activityDialog).toContainText('Quarterly launch goal');
      await expect(activityDialog).toContainText('Review CodeQ permissions');
      await page.keyboard.press('Escape');
      await page.getByRole('button', { name: 'Reset this area' }).click();
      await expect(page.locator('.assistant-demo-rail .assistant-demo-metric').nth(1).locator('strong')).toHaveText('0');
      await expect(page.getByRole('region', { name: 'Personal assistant recent activity' })).toContainText('New actions you take in this demo will appear here.');
      expect(apiRequests).toEqual([]);
    });
  }
}

test('Live Security Audit displays the backend denial when the token lacks security:admin', async ({ page }) => {
  await page.addInitScript(() => localStorage.setItem('oa-second-data-mode', 'live'));
  await page.route(url => new URL(url).pathname.startsWith('/api/'), async route => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/audit') {
      await route.fulfill({ status: 403, contentType: 'application/json', body: JSON.stringify({ error: 'This account does not have permission for this action.' }) });
      return;
    }
    await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
  });
  await page.goto('/?view=security');
  await page.getByRole('tab', { name: 'Audit' }).click();
  await expect(page.getByRole('heading', { name: 'Security service unavailable' })).toBeVisible();
  await expect(page.getByText('This account does not have permission for this action.')).toBeVisible();
});

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live Personal Assistant loads only its scoped activity feed (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const auditRequests: string[] = [];
      page.on('request', request => {
        if (new URL(request.url()).pathname === '/api/audit') auditRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      await page.route(url => new URL(url).pathname.startsWith('/api/'), async route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api/audit') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ events: [{ id: 'assistant-event-1', actor: 'owner', action: 'assistant.task.completed', scope: 'assistant', project: 'default', created_at: '2026-10-25T10:00:00Z', metadata: { private_marker: 'must-not-render' } }] }) });
          return;
        }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
      });

      await page.goto('/?view=assistant');
      await expect(page.getByRole('heading', { name: 'Personal assistant', exact: true })).toBeVisible();
      await expect(page.getByText('assistant.task.completed')).toBeVisible();
      await expect(page.getByText('must-not-render')).toHaveCount(0);
      expect(auditRequests.length).toBeGreaterThan(0);
      for (const request of auditRequests) {
        const url = new URL(request);
        expect(url.searchParams.get('project')).toBe('default');
        expect(url.searchParams.get('scope')).toBe('assistant');
        expect(url.searchParams.get('limit')).toBe('20');
      }
    });
  }
}

for (const viewport of [{ width: 1440, height: 900 }, { width: 1280, height: 800 }]) {
  for (const theme of ['light', 'dark'] as const) {
    test(`Live Security Audit loads only the explicitly scoped event list (${viewport.width}x${viewport.height}, ${theme})`, async ({ page }) => {
      const auditListRequests: Array<{ project: string | null; limit: string | null }> = [];
      const chainVerifyRequests: string[] = [];
      page.on('request', request => {
        const url = new URL(request.url());
        if (url.pathname === '/api/audit') auditListRequests.push({ project: url.searchParams.get('project'), limit: url.searchParams.get('limit') });
        if (url.pathname === '/api/security/audit-chain/verify') chainVerifyRequests.push(request.url());
      });
      await page.setViewportSize(viewport);
      await page.addInitScript(({ selectedTheme }) => {
        localStorage.setItem('oa-second-data-mode', 'live');
        localStorage.setItem('oa-second-theme', selectedTheme);
      }, { selectedTheme: theme });
      await page.route(url => new URL(url).pathname.startsWith('/api/'), async route => {
        const url = new URL(route.request().url());
        if (url.pathname === '/api/security/audit-chain/verify') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ ok: true, sealed: 12, issues: [] }) });
          return;
        }
        if (url.pathname === '/api/audit') {
          await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ events: [{ id: 'scoped-audit-1', actor: 'owner', action: 'file.updated', resource: 'readme.md', project: 'default', status: 'success', created_at: '2026-10-24T20:00:00Z', metadata: { private_marker: 'must-not-render' } }] }) });
          return;
        }
        await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({}) });
      });

      await page.goto('/?view=security');
      await expect(page.getByRole('heading', { name: 'Security', exact: true })).toBeVisible();
      await page.getByRole('tab', { name: 'Audit' }).click();
      await expect(page.getByText('file.updated')).toBeVisible();
      await expect(page.getByText('must-not-render')).toHaveCount(0);
      await expect(page.getByText(/read-only events for project/i)).toBeVisible();
      expect(auditListRequests).toEqual([{ project: 'default', limit: '100' }]);

      await page.getByRole('button', { name: 'Verify chain' }).click();
      await expect(page.getByText('Audit-chain verification completed.')).toBeVisible();
      expect(chainVerifyRequests).toHaveLength(1);
      expect(auditListRequests).toEqual([
        { project: 'default', limit: '100' },
        { project: 'default', limit: '100' },
      ]);
    });
  }
}
