import { expect, test, type Page, type Route } from '@playwright/test';

const desktop = [
  { width: 1440, height: 900, name: '1440x900' },
  { width: 1280, height: 800, name: '1280x800' },
];
const themes = ['light', 'dark'] as const;
type Reply = (route: Route) => Promise<void>;
type ConfigureOptions = { delayConversationRefresh?: () => Promise<void> };

async function configure(page: Page, mode: 'demo' | 'live', theme: string, size: { width: number; height: number }, reply?: Reply, options: ConfigureOptions = {}) {
  await page.setViewportSize(size);
  await page.addInitScript(({ dataMode, colorTheme }) => {
    localStorage.setItem('oa-second-data-mode', dataMode);
    localStorage.setItem('oa-second-theme', colorTheme);
    sessionStorage.removeItem('oa-second-conversation-id');
  }, { dataMode: mode, colorTheme: theme });
  let conversationReads = 0;
  await page.route(/^https?:\/\/[^/]+\/api\/.*/, async route => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === 'POST' && path === '/api/chat/stream' && reply) return reply(route);
    if (request.method() !== 'GET') return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
    if (path === '/api/conversations' || path === '/api/conversations/organized') {
      conversationReads += 1;
      if (options.delayConversationRefresh && conversationReads > 1) await options.delayConversationRefresh();
      return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ conversations: [] }) });
    }
    if (path === '/api/messages') return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ messages: [] }) });
    if (path === '/api/models') return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ providers: [{ id: 'auto', label: 'Auto route', local: true }], models: [] }) });
    return route.fulfill({ status: 200, contentType: 'application/json', body: '{}' });
  });
  await page.goto('/?view=chat', { waitUntil: 'domcontentloaded' });
  await expect(page.getByRole('textbox', { name: 'Message OpenAgent' })).toBeVisible({ timeout: 20_000 });
}

function sse(events: Record<string, unknown>[]) {
  return events.map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
}

const shortReply = sse([
  { type: 'conversation', conversation: { id: 'runtime-chat-short' } },
  { type: 'status', status: 'started' },
  { type: 'chunk', content: 'Hello.' },
  { type: 'message', message: { id: 'runtime-message-short', content: 'Hello.' } },
  { type: 'done' },
]);

const emptyReply = sse([
  { type: 'conversation', conversation: { id: 'runtime-chat-empty' } },
  { type: 'status', status: 'started' },
  { type: 'done' },
]);

const progressiveReply = sse([
  { type: 'conversation', conversation: { id: 'runtime-chat-progressive' } },
  { type: 'status', status: 'started' },
  { type: 'chunk', content: 'First ' },
  { type: 'chunk', content: 'second ' },
  { type: 'chunk', content: 'third.' },
  { type: 'message', message: { id: 'runtime-message-progressive', content: 'First second third.' } },
  { type: 'done' },
]);

for (const size of desktop) for (const theme of themes) {
  test(`Chat idle is clean and has no sample trace (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size);
    await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
    await expect(page.locator('[data-approved-assistant-message]')).toHaveCount(0);
    await expect(page.getByText('Frontend redesign', { exact: true })).toHaveCount(0);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
  });

  test(`Chat short stream shows the real answer without a timeline (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size, async route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }));
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Say hello briefly.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByText('Hello.', { exact: true })).toBeVisible();
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
    await expect(page.getByRole('status').filter({ hasText: 'Completed' })).toHaveCount(0);
  });

  test(`Chat immediate reply never flashes the waiting indicator (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size, route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }));
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Reply immediately.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByText('Hello.', { exact: true })).toBeVisible();
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
  });

  test(`Chat delayed stream uses only a compact pending indicator (${size.name}, ${theme})`, async ({ page }) => {
    let releaseResponse!: () => void;
    let requestStarted!: () => void;
    const responseGate = new Promise<void>(resolve => { releaseResponse = resolve; });
    const started = new Promise<void>(resolve => { requestStarted = resolve; });
    await configure(page, 'live', theme, size, async route => {
      requestStarted();
      await responseGate;
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply });
    });
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Wait for a short response.');
    await page.getByRole('button', { name: 'Send message' }).click();
    const indicator = page.getByRole('status').filter({ hasText: 'Working…' });
    try {
      await requestStarted;
      // Keep the actual response withheld until the delayed indicator appears.
      // This verifies the product's pending state without racing a fixed timer.
      await expect(indicator).toBeVisible({ timeout: 10_000 });
      await expect(page.locator('.chat-generation-indicator')).toHaveCSS('font-size', '12px');
      await expect(page.locator('#content .trace-mount')).toHaveCount(0);
      await expect(page.getByText(/reasoning|sample stage|tool call|\d+%/i)).toHaveCount(0);
    } finally {
      releaseResponse();
    }
    await expect(page.getByText('Hello.', { exact: true })).toBeVisible();
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
  });

  test(`Chat hides waiting state as soon as progressive content arrives (${size.name}, ${theme})`, async ({ page }) => {
    let releaseStream!: () => void;
    let streamStarted!: () => void;
    const gate = new Promise<void>(resolve => { releaseStream = resolve; });
    const started = new Promise<void>(resolve => { streamStarted = resolve; });
    await configure(page, 'live', theme, size, async route => {
      streamStarted();
      await gate;
      await route.fulfill({ status: 200, contentType: 'text/event-stream', body: progressiveReply });
    });
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Return a progressive response.');
    await page.getByRole('button', { name: 'Send message' }).click();
    const indicator = page.locator('.chat-generation-indicator');
    await expect(indicator).toBeVisible();
    await started;
    const assistantRow = await page.locator('.chat-thread .chat-message').nth(1).elementHandle();
    releaseStream();
    await expect(page.locator('.chat-thread .chat-message').nth(1).locator('.message-body')).toContainText('First second third.');
    await expect(indicator).toHaveCount(0);
    expect(await assistantRow?.evaluate(element => element.isConnected)).toBe(true);
    await expect(page.locator('.chat-thread .chat-message')).toHaveCount(2);
  });

  test(`Chat empty completion shows a neutral notice and no synthetic answer (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size, route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: emptyReply }));
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Complete without content.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByRole('status').filter({ hasText: 'The backend completed without returning a response.' })).toBeVisible();
    await expect(page.locator('.chat-thread .chat-message')).toHaveCount(1);
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
  });

  test(`Chat completion is immediate even if history refresh is delayed (${size.name}, ${theme})`, async ({ page }) => {
    let releaseHistory!: () => void;
    let historyRefreshStarted!: () => void;
    const historyGate = new Promise<void>(resolve => { releaseHistory = resolve; });
    const historyStarted = new Promise<void>(resolve => { historyRefreshStarted = resolve; });
    await configure(page, 'live', theme, size, route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }), {
      delayConversationRefresh: async () => { historyRefreshStarted(); await historyGate; },
    });
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Do not wait on history refresh.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await historyStarted;
    await expect(page.getByText('Hello.', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Cancel', exact: true })).toHaveCount(0);
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
    releaseHistory();
  });

  test(`Chat failure removes pending activity and shows the backend error (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size, route => route.fulfill({ status: 503, contentType: 'application/json', body: JSON.stringify({ error: 'Chat provider unavailable for this test.' }) }));
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('This request should fail.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByRole('alert').filter({ hasText: 'Chat provider unavailable for this test.' })).toBeVisible();
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
    await expect(page.getByRole('status').filter({ hasText: 'Completed' })).toHaveCount(0);
  });

  test(`Chat cancellation clears the real pending indicator (${size.name}, ${theme})`, async ({ page }) => {
    let releaseResponse!: () => void;
    let requestStarted!: () => void;
    const gate = new Promise<void>(resolve => { releaseResponse = resolve; });
    const started = new Promise<void>(resolve => { requestStarted = resolve; });
    await configure(page, 'live', theme, size, async route => {
      requestStarted();
      await gate;
      try { await route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }); } catch { /* The browser may abort the intercepted stream when Cancel is selected. */ }
    });
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Cancel this request.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await started;
    await expect(page.getByRole('status').filter({ hasText: 'Working…' })).toBeVisible();
    await page.getByRole('button', { name: 'Cancel', exact: true }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Response cancelled.' })).toBeVisible();
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
    releaseResponse();
    await expect(page.getByText('Hello.', { exact: true })).toHaveCount(0);
  });

  test(`Chat conversation switching ignores a late response from the previous thread (${size.name}, ${theme})`, async ({ page }) => {
    let releaseResponse!: () => void;
    let requestStarted!: () => void;
    const gate = new Promise<void>(resolve => { releaseResponse = resolve; });
    const started = new Promise<void>(resolve => { requestStarted = resolve; });
    await configure(page, 'live', theme, size, async route => {
      requestStarted();
      await gate;
      try { await route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }); } catch { /* Selecting another conversation aborts this request. */ }
    });
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('This answer belongs to the previous thread.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await started;
    await page.evaluate(() => window.dispatchEvent(new CustomEvent('openagent:select-conversation', { detail: 'other-chat' })));
    await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
    releaseResponse();
    await expect(page.getByText('Hello.', { exact: true })).toHaveCount(0);
    await expect(page.locator('.chat-thread .chat-message')).toHaveCount(0);
  });

  test(`Chat completion leaves only the backend final answer (${size.name}, ${theme})`, async ({ page }) => {
    await configure(page, 'live', theme, size, route => route.fulfill({ status: 200, contentType: 'text/event-stream', body: shortReply }));
    await page.getByRole('textbox', { name: 'Message OpenAgent' }).fill('Complete normally.');
    await page.getByRole('button', { name: 'Send message' }).click();
    await expect(page.getByText('Hello.', { exact: true })).toBeVisible();
    await expect(page.locator('.chat-generation-indicator')).toHaveCount(0);
    await expect(page.getByRole('status').filter({ hasText: 'Completed' })).toHaveCount(0);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
  });

  test(`Chat Demo idle is empty and makes no backend requests (${size.name}, ${theme})`, async ({ page }) => {
    const apiCalls: string[] = [];
    page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiCalls.push(request.url()); });
    await configure(page, 'demo', theme, size);
    await expect(page.getByRole('heading', { name: 'Start a conversation' })).toBeVisible();
    await expect(page.getByText('Frontend redesign', { exact: true })).toHaveCount(0);
    await expect(page.locator('#content .trace-mount')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Send message' })).toBeDisabled();
    expect(apiCalls).toEqual([]);
  });
}
