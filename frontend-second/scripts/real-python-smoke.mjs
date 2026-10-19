import { spawn } from 'node:child_process';
import { writeFile } from 'node:fs/promises';
import { request as httpRequest } from 'node:http';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = path.resolve(frontendRoot, '..');
const python = process.env.OPENAGENT_TEST_PYTHON || path.join(repositoryRoot, '.venv', 'Scripts', 'python.exe');
const chrome = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const results = [];
let backend;
let vite;
let browser;
let previewCleanup;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitFor(predicate, label, timeoutMs = 30_000, intervalMs = 250) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { const value = await predicate(); if (value) return value; } catch { /* process is still starting */ }
    await delay(intervalMs);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

function startBackend() {
  return new Promise((resolve, reject) => {
    backend = spawn(python, ['tests/support/real_integration_server.py'], {
      cwd: repositoryRoot, stdio: ['pipe', 'pipe', 'inherit'], windowsHide: true,
    });
    let buffered = '';
    const timer = setTimeout(() => reject(new Error('Timed out starting the isolated Python server')), 60_000);
    backend.once('error', reject);
    backend.stdout.on('data', chunk => {
      buffered += chunk.toString();
      const newline = buffered.indexOf('\n');
      if (newline < 0) return;
      clearTimeout(timer);
      try { resolve(JSON.parse(buffered.slice(0, newline))); }
      catch (error) { reject(error); }
    });
  });
}

async function requestJson(base, route, token, { method = 'GET', body, bootstrapToken = '' } = {}) {
  const url = new URL(route, base);
  const requestBody = body ? Buffer.from(JSON.stringify(body)) : undefined;
  const headers = {
    Host: 'remote.test',
    'X-Forwarded-Proto': 'https',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(bootstrapToken ? { 'X-OpenAgent-Bootstrap-Token': bootstrapToken } : {}),
    ...(requestBody ? { 'Content-Type': 'application/json', 'Content-Length': String(requestBody.length) } : {}),
  };
  const rawResponse = await new Promise((resolve, reject) => {
    const request = httpRequest({
      protocol: url.protocol, hostname: url.hostname, port: url.port, method,
      path: `${url.pathname}${url.search}`, headers,
    }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(Buffer.from(chunk)));
      response.on('end', () => resolve({
        status: response.statusCode ?? 0,
        headers: response.headers,
        text: Buffer.concat(chunks).toString('utf8'),
      }));
    });
    request.on('error', reject);
    if (requestBody) request.write(requestBody);
    request.end();
  });
  let payload = {};
  try { payload = JSON.parse(rawResponse.text); } catch { /* keep status evidence */ }
  const response = {
    status: rawResponse.status,
    ok: rawResponse.status >= 200 && rawResponse.status < 300,
    json: async () => payload,
    text: async () => rawResponse.text,
  };
  return { response, payload };
}

function finishResults(metadata = {}) {
  const blockers = results.filter(entry => entry.result !== 'PASS');
  console.log(JSON.stringify({
    kind: 'real-react-python-integration',
    overallStatus: blockers.length > 0 ? 'BLOCKED' : 'PASS',
    blockedWorkflows: blockers.map(entry => entry.workflow),
    authEnforced: true,
    externalProvidersEnabled: false,
    ...metadata,
    results,
  }, null, 2));
  if (blockers.length > 0) {
    throw new Error(`Integration/security acceptance BLOCKED: ${blockers.map(entry => entry.workflow).join(', ')}`);
  }
}

async function run() {
  const fixture = await startBackend();
  console.log('isolated Python server ready');
  await waitFor(async () => (await requestJson(fixture.baseUrl, '/api/health', '')).response.ok, 'Python health route');
  const secondRegistration = await requestJson(fixture.baseUrl, '/api/auth/register', fixture.tokens.agents, {
    method: 'POST',
    bootstrapToken: 'batch2b-integration-bootstrap-secret-32-chars',
    body: { username: 'second-account', email: 'second@example.test', password: 'temporary-second-password' },
  });
  assert(secondRegistration.response.status === 403, `personal_remote accepted second account registration (${secondRegistration.response.status})`);
  results.push({ workflow: 'Single owner registration boundary', result: 'PASS', evidence: 'An owner credential and bootstrap header cannot create another account in the installation.' });
  await writeFile(path.join(fixture.repoPath, 'preview-server.mjs'), "import { createServer } from 'node:http';\nconst server = createServer((_request, response) => { response.writeHead(200, { 'content-type': 'text/plain' }); response.end('Disposable CodeQ preview is ready.'); });\nserver.listen(0, '127.0.0.1', () => console.log(`http://127.0.0.1:${server.address().port}/`));\n");
  const fixtureRepo = await requestJson(fixture.baseUrl, '/api/codebase/projects', fixture.tokens.codeq, {
    method: 'POST',
    body: { name: 'Disposable React integration', source_path: fixture.repoPath, project: 'default', backend: 'local' },
  });
  assert(fixtureRepo.response.status === 201, `CodeQ fixture project creation returned ${fixtureRepo.response.status}`);
  console.log('disposable project created');
  const codeProjectId = fixtureRepo.payload.project?.id;
  assert(codeProjectId, 'CodeQ fixture project did not return an ID');

  vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5180', '--strictPort'], {
    cwd: frontendRoot,
    env: { ...process.env, OPENAGENT_API_TARGET: fixture.baseUrl, OPENAGENT_TEST_PERSONAL_REMOTE_PROXY: '1' },
    stdio: 'ignore', windowsHide: true,
  });
  const origin = 'http://127.0.0.1:5180';
  await waitFor(async () => (await fetch(origin)).ok, 'React Vite app');

  browser = await chromium.launch({ headless: true, executablePath: chrome });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const apiResponses = [];
  const apiRequests = [];
  const onlyAgents = process.env.OPENAGENT_SMOKE_ONLY === 'agents';
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request); });
  page.on('response', response => {
    if (new URL(response.url()).pathname.startsWith('/api/')) apiResponses.push({ path: new URL(response.url()).pathname, status: response.status() });
  });

  async function open(view, token) {
    await page.goto(origin);
    await page.evaluate(({ token: value }) => {
      localStorage.setItem('oa-second-data-mode', 'live');
      localStorage.setItem('oa-project', 'default');
      sessionStorage.setItem('oa-second-api-token', value);
    }, { token });
    await page.goto(`${origin}/?view=${encodeURIComponent(view)}`);
    await page.locator('#content').waitFor({ state: 'attached' });
  }

  // Prove the remote owner can authenticate, renew and end a real session
  // through the React UI before testing workspace APIs with scoped tokens.
  const authPage = await context.newPage();
  await authPage.goto(origin);
  await authPage.evaluate(() => localStorage.setItem('oa-second-data-mode', 'live'));
  await authPage.goto(`${origin}/?view=settings`);
  await authPage.getByRole('tab', { name: 'Advanced' }).click();
  await authPage.getByRole('heading', { name: 'Owner sign-in' }).waitFor({ state: 'visible' });
  await authPage.getByLabel('Username or email').fill(fixture.ownerUsername);
  await authPage.getByLabel('Password').fill(fixture.ownerPassword);
  const loginResponsePromise = authPage.waitForResponse(response => response.url().endsWith('/api/auth/login') && response.request().method() === 'POST');
  await authPage.getByRole('button', { name: 'Sign in', exact: true }).click();
  const loginResponse = await loginResponsePromise;
  assert(loginResponse.status() === 200, `React owner sign-in returned HTTP ${loginResponse.status()}`);
  await expect(authPage.locator('[aria-label="Owner session"]')).toContainText('Authenticated owner session');
  const refreshResponsePromise = authPage.waitForResponse(response => response.url().endsWith('/api/auth/refresh') && response.request().method() === 'POST');
  await authPage.getByRole('button', { name: 'Renew session' }).click();
  const refreshResponse = await refreshResponsePromise;
  assert(refreshResponse.status() === 200, `React owner session renewal returned HTTP ${refreshResponse.status()}`);
  await expect(authPage.locator('[aria-label="Owner session"]')).toContainText('Session renewed.');
  await authPage.getByRole('button', { name: 'Sign out', exact: true }).click();
  await expect(authPage.locator('[aria-label="Owner session"]')).toContainText('Signed out of this browser tab.');
  assert(await authPage.evaluate(() => sessionStorage.getItem('oa-second-api-token')) === null,
    'React sign-out did not clear the tab token');
  const signedOutCheck = await requestJson(fixture.baseUrl, '/api/auth/me', '');
  assert(signedOutCheck.response.status === 401, `Signed-out private identity read returned HTTP ${signedOutCheck.response.status}`);
  results.push({ workflow: 'Personal remote owner sign-in/out', result: 'PASS', evidence: 'React signed in with owner credentials, renewed the authenticated session, signed out, cleared tab storage, and the private identity endpoint returned 401 without credentials.' });
  await authPage.close();

  if (!onlyAgents) {
  // Chat: the actual React composer submits over authenticated Python SSE and
  // the resulting conversation/messages must persist in the isolated SQLite DB.
  const beforeChat = apiResponses.length;
  await open('chat', fixture.tokens.chat);
  const chatInput = page.getByLabel('Message OpenAgent');
  await chatInput.waitFor({ state: 'visible' });
  await chatInput.fill('Reply with a short local integration confirmation.');
  const chatStreamResponse = page.waitForResponse(response => response.url().includes('/api/chat/stream') && response.request().method() === 'POST', { timeout: 30_000 });
  await page.getByRole('button', { name: 'Send message' }).click();
  const chatStream = await chatStreamResponse;
  console.log('React Chat SSE response headers received:', chatStream.status());
  assert(chatStream.status() === 200, `Chat SSE returned ${chatStream.status()}`);
  await waitFor(async () => await page.locator('.chat-message').count() >= 2, 'React chat assistant response', 30_000);
  console.log('React Chat rendered user and assistant rows');
  const chatState = await requestJson(fixture.baseUrl, '/api/conversations?project=default&limit=10', fixture.tokens.chat);
  assert(chatState.response.ok && (chatState.payload.conversations?.length ?? 0) > 0, 'Chat conversation was not persisted');
  const conversationId = chatState.payload.conversations[0].id;
  let messages;
  await waitFor(async () => {
    messages = await requestJson(fixture.baseUrl, `/api/messages?conversation_id=${encodeURIComponent(conversationId)}`, fixture.tokens.chat);
    return messages.response.ok && (messages.payload.messages?.length ?? 0) >= 2;
  }, 'persisted Chat assistant response', 15_000);
  console.log('Chat messages persisted');
  assert(messages.response.ok && (messages.payload.messages?.length ?? 0) >= 2,
    `Chat messages were not persisted (HTTP ${messages.response.status}: ${JSON.stringify(messages.payload)})`);
  results.push({ workflow: 'Chat', result: 'PASS', evidence: 'React POST /api/chat/stream returned 200; Python persisted conversation and user/assistant messages.' });

  await chatInput.fill('Trigger the controlled local provider failure [integration-fail].');
  await page.getByRole('button', { name: 'Send message' }).click();
  await expect(page.locator('#content [role="alert"]').filter({ hasText: 'Synthetic local integration provider failure.' })).toBeVisible({ timeout: 15_000 });
  results.push({ workflow: 'Chat failure', result: 'PASS', evidence: 'React displayed the actual Python SSE provider error; no assistant success message was fabricated.' });

  await chatInput.fill('Wait before responding so I can cancel [integration-delay].');
  await page.getByRole('button', { name: 'Send message' }).click();
  const cancelChat = page.getByRole('button', { name: 'Cancel', exact: true });
  await cancelChat.waitFor({ state: 'visible', timeout: 10_000 });
  await cancelChat.click();
  await expect(page.getByRole('status').filter({ hasText: 'Response cancelled.' })).toBeVisible({ timeout: 10_000 });
  results.push({ workflow: 'Chat cancellation', result: 'PASS', evidence: 'React aborted the authenticated SSE request and showed its cancelled state; the fixture gateway is local and has no external side effects.' });

  // CodeQ: actual React project discovery and run lifecycle against a copied
  // disposable workspace. External providers remain disabled.
  await open('codeq', fixture.tokens.codeq);
  await page.getByLabel('CodeQ project').waitFor({ state: 'visible' });
  await page.getByLabel('CodeQ project').selectOption(String(codeProjectId));
  await page.getByRole('textbox', { name: 'Preview command' }).fill('node preview-server.mjs');
  await page.getByRole('checkbox', { name: /I reviewed this command/ }).check();
  page.once('dialog', dialog => dialog.accept());
  const previewStartPromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').tool === 'start_process', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Start preview' }).click();
  const previewStartResponse = await previewStartPromise;
  assert(previewStartResponse.ok(), `CodeQ preview start returned HTTP ${previewStartResponse.status()}`);
  const previewStartPayload = await previewStartResponse.json();
  const previewProcessId = String(previewStartPayload.result?.id ?? previewStartPayload.result?.process_id ?? '');
  assert(previewProcessId, 'CodeQ preview did not return a managed process ID');
  const previewToolsPath = `/api/codebase/projects/${codeProjectId}/tools`;
  previewCleanup = () => requestJson(fixture.baseUrl, previewToolsPath, fixture.tokens.codeq, { method: 'POST', body: { tool: 'stop_process', arguments: { process_id: previewProcessId }, approved: true } });
  const previewLogsResponse = await requestJson(fixture.baseUrl, previewToolsPath, fixture.tokens.codeq, { method: 'POST', body: { tool: 'inspect_logs', arguments: { process_id: previewProcessId, limit: 20000 }, approved: false } });
  assert(previewLogsResponse.response.ok, `CodeQ preview log read returned HTTP ${previewLogsResponse.response.status}: ${JSON.stringify(previewLogsResponse.payload)}`);
  const previewOutput = String(previewLogsResponse.payload.result?.stdout ?? '') + String(previewLogsResponse.payload.result?.stderr ?? '');
  const previewUrl = previewOutput.match(/http:\/\/127\.0\.0\.1:\d+\//)?.[0];
  assert(previewUrl, `CodeQ preview output did not report a loopback URL: ${previewOutput}`);
  const previewResponse = await fetch(previewUrl);
  assert(previewResponse.ok && (await previewResponse.text()).includes('Disposable CodeQ preview is ready.'), 'CodeQ preview URL did not serve the disposable fixture response');
  await expect(page.getByRole('link', { name: previewUrl })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('region', { name: 'Live CodeQ preview session' })).toContainText(`Preview URL reported by the process: ${previewUrl}`, { timeout: 15_000 });
  await page.getByRole('button', { name: 'Refresh preview status' }).click();
  await waitFor(async () => apiRequests.some(request => request.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && JSON.parse(request.postData() ?? '{}').tool === 'process_status'), 'CodeQ preview status refresh', 10_000);
  const previewStopPromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').tool === 'stop_process', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Stop preview' }).click();
  const previewStopResponse = await previewStopPromise;
  assert(previewStopResponse.ok(), `CodeQ preview stop returned HTTP ${previewStopResponse.status()}`);
  assert(String((await previewStopResponse.json()).result?.status ?? '') === 'exited', 'CodeQ preview process did not report its stopped state');
  results.push({ workflow: 'CodeQ preview lifecycle', result: 'PASS', evidence: `React used the authenticated project tools contract to start process ${previewProcessId}; Python output reported ${previewUrl}, the URL served the disposable fixture response, and status refresh and stop were accepted by the backend.` });
  await page.getByLabel('Ask CodeQ').fill('Inspect the fixture and report one safe observation [integration-delay].');
  const codeRunResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/codebase/runs') && response.request().method() === 'POST', { timeout: 30_000 });
  await page.getByRole('button', { name: /^Run$/ }).first().click();
  const codeRunResponse = await codeRunResponsePromise;
  assert([202, 400, 500].includes(codeRunResponse.status()), `Unexpected CodeQ create-run status ${codeRunResponse.status()}`);
  if (codeRunResponse.status() === 202) {
    const codeRun = (await codeRunResponse.json()).run;
    assert(codeRun?.id, 'CodeQ accepted run did not include a run ID');
    const codeEvents = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(codeRun.id)}/events?after=0`, fixture.tokens.codeq);
    assert(codeEvents.response.ok && Array.isArray(codeEvents.payload.events), 'CodeQ event cursor read failed');
    const streamRequest = await waitFor(async () => apiRequests.find(request => request.url().includes(`/api/codebase/runs/${codeRun.id}/events/stream`)), 'CodeQ authenticated event stream request', 10_000);
    const headers = await streamRequest.allHeaders();
    assert(Boolean(headers.authorization), 'CodeQ event stream omitted the API bearer token');
    await expect(page.locator('.terminal-window').getByText(/run\.created/)).toBeVisible({ timeout: 10_000 });
    // The isolated Python fixture closes this marked SSE connection after its
    // first real event. React must resubscribe to the same route from cursor 0.
    const reconnectRunResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/codebase/runs') && response.request().method() === 'POST', { timeout: 30_000 });
    await page.getByLabel('Ask CodeQ').fill('Continue after a browser network interruption [integration-reconnect].');
    await page.getByRole('button', { name: /^Run$/ }).first().click();
    const reconnectRunResponse = await reconnectRunResponsePromise;
    assert(reconnectRunResponse.status() === 202, `CodeQ reconnect fixture creation returned ${reconnectRunResponse.status()}`);
    const reconnectRun = (await reconnectRunResponse.json()).run;
    assert(reconnectRun?.id, 'CodeQ reconnect fixture did not return a run ID');
    await expect(page.locator('.terminal-window').getByText(/run\.created/)).toBeVisible({ timeout: 10_000 });
    await waitFor(async () => {
      const state = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}`, fixture.tokens.codeq);
      return ['queued', 'running', 'analysing', 'planning', 'executing', 'validating', 'waiting_approval', 'waiting'].includes(String(state.payload.run?.status ?? '').toLowerCase());
    }, 'active CodeQ reconnect fixture state', 15_000);
    const reconnectPath = `/api/codebase/runs/${reconnectRun.id}/events/stream`;
    const resumedStreamRequest = await waitFor(() => {
      const requests = apiRequests.filter(request => request.url().includes(reconnectPath));
      return requests.length >= 2 ? requests.at(-1) : null;
    }, 'CodeQ SSE resubscription after the isolated transport close', 15_000);
    const resumedAfter = Number(new URL(resumedStreamRequest.url()).searchParams.get('after') ?? -1);
    assert(resumedAfter > 0, `CodeQ resumed from invalid cursor ${resumedAfter}; expected the delivered event sequence`);
    await expect(page.locator('.terminal-window').getByText(/run\.created/)).toHaveCount(1, { timeout: 10_000 });
    const resumedEvents = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}/events?after=0`, fixture.tokens.codeq);
    assert(resumedEvents.response.ok && resumedEvents.payload.events.length > 0, 'CodeQ event list failed after reconnect');
    const resumedSequences = resumedEvents.payload.events.map(event => Number(event.sequence ?? 0));
    assert(new Set(resumedSequences).size === resumedSequences.length, 'CodeQ backend event cursor returned duplicate sequence values');
    const reconnectedRunState = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}`, fixture.tokens.codeq);
    assert(reconnectedRunState.response.ok, 'CodeQ run could not be read after stream reconnection');
    const current = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}`, fixture.tokens.codeq);
    const currentCodeStatus = String(current.payload.run?.status ?? current.payload.status ?? '').toLowerCase();
    console.log('CodeQ run status at cancellation check:', currentCodeStatus || 'missing');
    let codeqCancellation = 'not exercised; run was no longer active';
    if (current.response.ok && ['queued', 'running', 'analysing', 'planning', 'executing', 'validating', 'waiting_approval', 'waiting'].includes(currentCodeStatus)) {
      const cancelButton = page.getByRole('button', { name: 'Cancel run' });
      await cancelButton.waitFor({ state: 'visible', timeout: 10_000 }).catch(() => undefined);
      if (await cancelButton.isVisible().catch(() => false)) {
        const cancelResponse = page.waitForResponse(response => response.url().endsWith(`/api/codebase/runs/${reconnectRun.id}/cancel`) && response.request().method() === 'POST');
        await cancelButton.click();
        const cancelled = await cancelResponse;
        assert(cancelled.ok(), `CodeQ cancellation failed through React control (${cancelled.status()})`);
        await waitFor(async () => {
          const state = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}`, fixture.tokens.codeq);
          return ['cancelled', 'canceled'].includes(String(state.payload.run?.status).toLowerCase());
        }, 'persisted CodeQ cancellation', 15_000);
        await delay(8_500); // let the local fixture provider release the cancelled worker before its temp DB is removed
        codeqCancellation = 'cancelled through React control and confirmed by the Python run record';
      }
    }
    assert(codeqCancellation.startsWith('cancelled'), `CodeQ run could not be cancelled through the UI: ${codeqCancellation}`);

    // Exercise Resume after a controlled local provider failure.
    await page.getByLabel('Ask CodeQ').fill('Resume this disposable run after a controlled failure [integration-resume-fail].');
    const failedRunCreate = page.waitForResponse(response => response.url().endsWith('/api/codebase/runs') && response.request().method() === 'POST', { timeout: 30_000 });
    await page.getByRole('button', { name: /^Run$/ }).first().click();
    const failedRunResponse = await failedRunCreate;
    assert(failedRunResponse.status() === 202, `Controlled failure CodeQ run creation returned ${failedRunResponse.status()}`);
    const failedRun = (await failedRunResponse.json()).run;
    assert(failedRun?.id, 'Controlled failure CodeQ run did not include an ID');
    await expect(page.getByRole('button', { name: 'Resume run' })).toBeVisible({ timeout: 30_000 });
    const failedState = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(failedRun.id)}`, fixture.tokens.codeq);
    assert(String(failedState.payload.run?.status).toLowerCase() === 'failed', 'CodeQ controlled run did not reach a persisted failed state');
    const resumeResponsePromise = page.waitForResponse(response => response.url().endsWith(`/api/codebase/runs/${failedRun.id}/resume`) && response.request().method() === 'POST', { timeout: 15_000 });
    await page.getByRole('button', { name: 'Resume run' }).click();
    const resumeResponse = await resumeResponsePromise;
    assert(resumeResponse.status() === 202, `React CodeQ resume returned ${resumeResponse.status()}`);
    const resumedRun = (await resumeResponse.json()).run;
    assert(resumedRun?.id === failedRun.id, 'Resume returned a different CodeQ run');
    await waitFor(async () => {
      const state = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(failedRun.id)}`, fixture.tokens.codeq);
      return ['queued', 'running', 'analysing', 'planning', 'executing', 'validating', 'failed'].includes(String(state.payload.run?.status).toLowerCase());
    }, 'persisted CodeQ resumed run state', 15_000);
    results.push({ workflow: 'CodeQ', result: 'PASS', evidence: `React project/run; authenticated SSE event and cursor; isolated Python fixture reset the live SSE connection after the first event, React resubscribed with after=${resumedAfter}, and the UI showed one run.created row; cancellation persisted; controlled failure and Resume returned 202 for the same run.` });
  } else {
    results.push({ workflow: 'CodeQ', result: 'BLOCKED', evidence: `Real Python rejected isolated local run with HTTP ${codeRunResponse.status()}; project read and API auth were exercised.` });
  }

  // Research: create an isolated, external-network-disabled session through
  // the React form, then verify its real persisted record, report and events.
  await open('research', fixture.tokens.research);
  await page.getByText('New research', { exact: true }).click();
  await page.getByLabel('Research question').fill('Summarize the disposable local integration fixture.');
  const researchCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/research/sessions') && response.request().method() === 'POST', { timeout: 30_000 });
  await page.getByRole('button', { name: 'Run research' }).click();
  const researchResponse = await researchCreatePromise;
  assert(researchResponse.status() === 202, `Research session creation returned ${researchResponse.status()}`);
  const researchPayload = await researchResponse.json();
  const researchId = researchPayload.session?.id;
  assert(researchId, 'Research accepted response did not include a session ID');
  await page.locator('#research-session').waitFor({ state: 'visible' });
  const persistedResearch = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(researchId)}/report`, fixture.tokens.research);
  assert(persistedResearch.response.ok && persistedResearch.payload.session_id === researchId, 'Research report/session response did not match accepted session');
  const completedResearch = await waitFor(async () => {
    const current = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(researchId)}/report`, fixture.tokens.research);
    return current.response.ok && current.payload.status === 'completed' && String(current.payload.report ?? '').trim()
      ? current.payload
      : null;
  }, 'Research worker to persist the actual report', 30_000);
  const expectedReportText = String(completedResearch.report).trim();
  await expect(page.locator('.research-summary')).toContainText(expectedReportText.slice(0, Math.min(expectedReportText.length, 80)), { timeout: 30_000 });
  if (Array.isArray(completedResearch.sources) && completedResearch.sources.length) {
    await expect(page.locator('.research-side')).toContainText(String(completedResearch.sources[0].title ?? completedResearch.sources[0].url ?? ''));
  }
  const crossOwnerResearch = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(researchId)}/report`, fixture.tokens.otherResearchRead);
  if (crossOwnerResearch.response.status === 200 && crossOwnerResearch.payload.session_id === researchId) {
    const crossOwnerResearchList = await requestJson(fixture.baseUrl, '/api/research/sessions?project=default&limit=100', fixture.tokens.otherResearchRead);
    assert(crossOwnerResearchList.response.ok && crossOwnerResearchList.payload.sessions?.some(session => String(session.id) === String(researchId)), 'Cross-owner report read succeeded, but list leakage could not be reproduced');
    results.push({ workflow: 'Research installation isolation', result: 'BLOCKED', evidence: 'A foreign installation credential read owner Research data (HTTP 200); personal_remote isolation failed.' });
  } else {
    assert([401, 403].includes(crossOwnerResearch.response.status), `Foreign installation Research report returned unexpected ${crossOwnerResearch.response.status}`);
    const foreignResearchList = await requestJson(fixture.baseUrl, '/api/research/sessions?project=default&limit=100', fixture.tokens.otherResearchRead);
    assert([401, 403].includes(foreignResearchList.response.status), `Foreign installation Research list returned unexpected ${foreignResearchList.response.status}`);
    results.push({ workflow: 'Research installation isolation', result: 'PASS', evidence: `A credential issued by a separate installation was rejected for Research report and list requests (HTTP ${crossOwnerResearch.response.status}/${foreignResearchList.response.status}).` });
  }
  const checkpoints = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(researchId)}/checkpoints`, fixture.tokens.research);
  assert(checkpoints.response.ok && Array.isArray(checkpoints.payload.checkpoints), 'Research checkpoint response schema is invalid');
  const researchStreamRequest = await waitFor(async () => apiRequests.find(request => request.url().includes(`/api/research/sessions/${researchId}/stream`)), 'React Research event stream request', 10_000);
  assert(Boolean((await researchStreamRequest.allHeaders()).authorization), 'Research event stream omitted the API bearer token');
  await page.locator('.research-side').getByRole('tab', { name: 'Activity', exact: true }).click();
  await expect(page.locator('.research-side [role="log"]')).toContainText(/queued|planning/i, { timeout: 10_000 });

  // Exercise the real cancellation control on a second in-flight session.
  const researchControls = page.locator('details.research-run-controls');
  const researchQuestion = researchControls.getByLabel('Research question');
  if (!(await researchQuestion.isVisible().catch(() => false))) await researchControls.locator('summary').click();
  await researchControls.getByLabel('Research question').fill('Cancel this isolated session [integration-research-cancel].');
  const cancelResearchCreate = page.waitForResponse(response => response.url().endsWith('/api/research/sessions') && response.request().method() === 'POST', { timeout: 30_000 });
  await researchControls.getByRole('button', { name: 'Run research' }).click();
  const cancelResearchResponse = await cancelResearchCreate;
  assert(cancelResearchResponse.status() === 202, `Research cancellation fixture creation returned ${cancelResearchResponse.status()}`);
  const cancelResearchId = (await cancelResearchResponse.json()).session?.id;
  assert(cancelResearchId, 'Research cancellation fixture did not include a session ID');
  await waitFor(async () => await page.locator('#research-session').inputValue() === String(cancelResearchId), 'React selected the cancellation session', 10_000);
  const cancelResearchButton = page.getByRole('button', { name: 'Cancel run' });
  await cancelResearchButton.waitFor({ state: 'visible', timeout: 10_000 });
  const cancelHit = await cancelResearchButton.evaluate(button => {
    const rect = button.getBoundingClientRect();
    const hit = document.elementFromPoint(rect.left + rect.width / 2, rect.top + rect.height / 2);
    return { target: button.outerHTML, hit: hit?.outerHTML, hitIsButton: hit === button || Boolean(hit?.closest('button') === button), rect: { x: rect.x, y: rect.y, width: rect.width, height: rect.height } };
  });
  console.log('Research cancel pointer target:', JSON.stringify(cancelHit));
  assert(cancelHit.hitIsButton, `Research Cancel run is obscured by ${cancelHit.hit}`);
  const cancelResearchRequest = page.waitForResponse(response => response.url().endsWith(`/api/research/sessions/${cancelResearchId}/cancel`) && response.request().method() === 'POST', { timeout: 15_000 });
  await cancelResearchButton.click();
  const cancelResearchResult = await cancelResearchRequest;
  assert(cancelResearchResult.ok(), `React Research cancellation returned ${cancelResearchResult.status()}`);
  const cancelResponsePayload = await cancelResearchResult.json();
  assert(cancelResponsePayload.session?.cancel_requested === true, 'Research cancel response did not preserve the cancellation request');
  let cancelState;
  try { await waitFor(async () => {
    cancelState = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(cancelResearchId)}`, fixture.tokens.research);
    return cancelState.response.ok && cancelState.payload.session?.status === 'cancelled';
  }, 'Research worker to reach its actual cancelled terminal state', 30_000, 1_000); }
  catch (error) {
    const checkpoints = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(cancelResearchId)}/checkpoints`, fixture.tokens.research).catch(() => null);
    throw new Error(`${String(error)}. Last Research response=${JSON.stringify(cancelState ?? null)} checkpointsResponse=${JSON.stringify(checkpoints ?? null)}`);
  }
  assert(cancelState?.response.ok && cancelState.payload.session?.cancel_requested === true, 'Research cancel request did not persist in the session record');
  const cancelledCheckpoints = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(cancelResearchId)}/checkpoints`, fixture.tokens.research);
  assert(cancelledCheckpoints.response.ok && cancelledCheckpoints.payload.checkpoints.some(checkpoint => checkpoint.stage === 'cancel-requested' || checkpoint.stage === 'cancelled'), 'Research cancellation checkpoint was not persisted');
  await expect(page.locator('.research-side [role="log"]')).toContainText(/cancel-requested|cancelled/i, { timeout: 10_000 });
  results.push({ workflow: 'Research cancellation terminal state', result: 'PASS', evidence: 'React Cancel run persisted cancel_requested=true and the cancel-requested checkpoint; the test waited for the Python worker to finish its controlled local provider call, then verified the session reached cancelled status.' });
  results.push({ workflow: 'Research owner session workflow', result: crossOwnerResearch.response.status === 200 ? 'BLOCKED' : 'PASS', evidence: 'React created an external-network-disabled session; the local worker completed it; its persisted report text rendered in the React Summary and returned sources were checked when present; authenticated events/checkpoints were visible in Activity. Foreign-installation credentials were rejected by the personal_remote boundary.' });
  }

  // Exercise the exact Agents workspace control and its /run-checked contract,
  // then prove that its response is owner-bound and visible after reload.
  const forbiddenAgent = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.readonly);
  assert(forbiddenAgent.response.status === 403, `Read-only unrelated token unexpectedly received ${forbiddenAgent.response.status}`);
  const crossOwnerAgent = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.otherAgentRead);
  assert(crossOwnerAgent.response.status === 401, `Foreign-installation history read returned ${crossOwnerAgent.response.status}, expected 401`);
  const agentRunOnly = await requestJson(fixture.baseUrl, '/api/agents/run-checked', fixture.tokens.agentsRunOnly, {
    method: 'POST', body: { agent: 'executive', goal: 'Run-only scope fixture', project: 'run-only-project', verify: false },
  });
  assert(agentRunOnly.response.status === 201, `agents:run-only checked run returned ${agentRunOnly.response.status}`);
  const runOnlyHistory = await requestJson(fixture.baseUrl, '/api/agent-runs?project=run-only-project', fixture.tokens.agentsRunOnly);
  assert(runOnlyHistory.response.status === 403, `agents:run-only token read history status ${runOnlyHistory.response.status}; expected 403`);
  const readOnlyRun = await requestJson(fixture.baseUrl, '/api/agents/run-checked', fixture.tokens.agentsReadOnly, {
    method: 'POST', body: { agent: 'executive', goal: 'Should be denied', project: 'read-only-project' },
  });
  assert(readOnlyRun.response.status === 403, `agents:read-only token ran an agent (${readOnlyRun.response.status})`);
  const neitherRun = await requestJson(fixture.baseUrl, '/api/agents/run-checked', fixture.tokens.neitherAgentScope, {
    method: 'POST', body: { agent: 'executive', goal: 'Should be denied', project: 'no-scope-project' },
  });
  assert(neitherRun.response.status === 403, `unscoped token ran an agent (${neitherRun.response.status})`);
  await open('agents', fixture.tokens.agents);
  await waitFor(async () => apiResponses.some(item => item.path === '/api/agents' && item.status === 200), 'React authorized agent directory query');
  await page.locator('details.agent-run-panel > summary').click();
  const actualAgentGoal = 'History integration fixture from the Agents workspace';
  await page.getByLabel('Task goal').fill(actualAgentGoal);
  const actualAgentResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/agents/jobs') && response.request().method() === 'POST', { timeout: 15_000 });
  const runAgentButton = page.getByRole('button', { name: 'Run agent', exact: true });
  assert(await runAgentButton.isEnabled(), `Agents submit control is disabled; form=${JSON.stringify(await page.locator('.agent-run-form').innerText())}`);
  await runAgentButton.click();
  const actualAgentResponse = await actualAgentResponsePromise.catch(async error => {
    console.log('Agents checked-run diagnostics:', JSON.stringify({
      mode: await page.evaluate(() => localStorage.getItem('oa-second-data-mode')),
      requests: apiRequests.map(request => request.url()).filter(url => /\/api\/agents|agent-runs/.test(url)).slice(-12),
      form: await page.locator('.agent-run-form').innerText().catch(() => ''),
      alerts: await page.getByRole('alert').allTextContents().catch(() => []),
    }));
    throw error;
  });
  assert(actualAgentResponse.status() === 202, `Agents workspace lifecycle start returned ${actualAgentResponse.status()}`);
  const actualJob = (await actualAgentResponse.json()).job;
  assert(actualJob?.id, 'Agents workspace lifecycle start did not return a persisted job ID');
  const completedAgentJob = await waitFor(async () => {
    const snapshot = await requestJson(fixture.baseUrl, `/api/agents/jobs/${actualJob.id}`, fixture.tokens.agentsReadOnly);
    return snapshot.payload.job?.status === 'completed' ? snapshot.payload.job : null;
  }, 'the actual agent lifecycle job to complete', 15_000);
  const actualRun = completedAgentJob.result?.run;
  assert(actualRun?.id, 'Completed Agents lifecycle job did not contain a persisted run ID');
  await waitFor(async () => apiResponses.some(item => item.path === '/api/agent-runs' && item.status === 200), 'React authorized agent history query after project ownership was established');
  const persistedViaHistory = await waitFor(async () => {
    const history = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.agentsReadOnly);
    if (!history.response.ok) return null;
    return history.payload.runs?.find(run => String(run.id) === String(actualRun.id)) ?? null;
  }, 'the actual lifecycle result row in authorized history', 10_000).catch(async error => {
    const history = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.agentsReadOnly);
    console.log('Agent history diagnostic:', JSON.stringify({
      run: actualRun,
      historyStatus: history.response.status,
      history: history.payload,
    }));
    throw error;
  });
  assert(persistedViaHistory.goal === actualAgentGoal, 'The persisted checked-run history record has incorrect task data');
  await page.reload();
  await page.getByRole('heading', { name: actualAgentGoal, exact: true }).waitFor({ state: 'visible', timeout: 15_000 });
  await page.getByText('Run history · 1', { exact: true }).click();
  await page.getByRole('heading', { name: actualAgentGoal, exact: true }).waitFor({ state: 'visible' });
  // Exercise React cancellation immediately after a second persisted job is
  // accepted, then retry that same cancelled lifecycle record.
  await page.locator('details.agent-run-panel > summary').click();
  await page.getByLabel('Task goal').fill('Cancel and retry integration fixture [integration-agent-cancel]');
  const cancelJobStart = page.waitForResponse(response => response.url().endsWith('/api/agents/jobs') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Run agent', exact: true }).click();
  const cancelJobResponse = await cancelJobStart;
  const cancelJob = (await cancelJobResponse.json()).job;
  assert(cancelJob?.id, 'Cancellation lifecycle start did not return a job ID');
  const cancelRequest = page.waitForResponse(response => response.url().endsWith(`/api/agents/jobs/${cancelJob.id}/cancel`) && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Cancel run', exact: true }).click();
  assert((await cancelRequest).ok(), 'React agent cancellation request failed');
  const cancelledJob = await waitFor(async () => {
    const snapshot = await requestJson(fixture.baseUrl, `/api/agents/jobs/${cancelJob.id}`, fixture.tokens.agentsReadOnly);
    return snapshot.payload.job?.status === 'cancelled' ? snapshot.payload.job : null;
  }, 'agent lifecycle cancellation to settle', 15_000);
  assert(cancelledJob.cancel_requested === true && Object.keys(cancelledJob.result ?? {}).length === 0, 'Cancelled agent job exposed a late successful result');
  const retryRequest = page.waitForResponse(response => response.url().endsWith(`/api/agents/jobs/${cancelJob.id}/retry`) && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Retry run', exact: true }).click();
  const retriedJob = (await (await retryRequest).json()).job;
  assert(retriedJob?.retry_of === cancelJob.id && retriedJob?.attempt === 2, 'React retry did not preserve lifecycle lineage');
  await waitFor(async () => {
    const snapshot = await requestJson(fixture.baseUrl, `/api/agents/jobs/${retriedJob.id}`, fixture.tokens.agentsReadOnly);
    return snapshot.payload.job?.status === 'completed' ? snapshot.payload.job : null;
  }, 'retried agent lifecycle to complete', 15_000);
  // Approval request and denial use the existing administrative API scope;
  // no agent execution is started by this denied request.
  await open('agents', fixture.tokens.agentsAdmin);
  await page.locator('details.agent-run-panel > summary').click();
  const approvalGoal = 'Review a local approval request only';
  await page.getByLabel('Task goal').fill(approvalGoal);
  const approvalRequestPromise = page.waitForResponse(response => response.url().endsWith('/api/agents/approvals/request') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Request approval', exact: true }).click();
  const approvalRequest = await approvalRequestPromise;
  assert(approvalRequest.status() === 201, `Agents approval request returned ${approvalRequest.status()}`);
  const pendingApproval = (await approvalRequest.json()).approval;
  assert(pendingApproval?.id, 'Agents approval response omitted its ID');
  await page.getByRole('button', { name: 'Access & approvals' }).click();
  await expect(page.getByText(approvalGoal, { exact: false })).toBeVisible({ timeout: 10_000 });
  const rejectApprovalPromise = page.waitForResponse(response => response.url().endsWith('/api/agents/approvals/resolve') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Reject', exact: true }).click();
  const rejectedApproval = await rejectApprovalPromise;
  assert(rejectedApproval.ok(), `Agents approval rejection returned ${rejectedApproval.status()}`);
  const rejectedPayload = await rejectedApproval.json();
  assert(rejectedPayload.approval?.status === 'rejected', 'Agents approval denial did not return its real rejected status');
  results.push({ workflow: 'Agents', result: 'PASS', evidence: 'React start/completion/history reload passed. A second React job was cancelled, its late result remained empty, and React retry completed with attempt/retry lineage. Run-only, read-only and unscoped permissions were checked; a foreign-installation credential returned 401. The approval flow created and rejected a request.' });

  if (onlyAgents) {
    finishResults({ selectedWorkflows: ['Agents'] });
    return;
  }

  // Basic read-only workspace integrations are loaded by the real React screens.
  const readonlyChecks = [
    { view: 'models', route: '/api/models' },
    { view: 'files', route: '/api/documents/library' },
    { view: 'calendar', route: '/api/calendar/events' },
  ];
  for (const check of readonlyChecks) {
    await open(check.view, fixture.tokens.readonly);
    await waitFor(async () => apiResponses.some(item => item.path === check.route && item.status === 200), `React ${check.view} read route`, 20_000);
    results.push({ workflow: check.view, result: 'PASS', evidence: `React loaded ${check.route} from the isolated Python server with a limited read token.` });
  }

  // Files: import a deterministic text file through the actual React file
  // picker, then verify the authenticated document response and local download.
  await open('files', fixture.tokens.documents);
  await page.getByRole('button', { name: 'Document filters and actions' }).click();
  const importedText = 'Batch 2A synthetic document content for import and download verification.';
  const importResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/documents/import') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByLabel('Choose document to import').setInputFiles({
    name: 'batch2a-synthetic-source.md', mimeType: 'text/markdown', buffer: Buffer.from(importedText),
  });
  const importResponse = await importResponsePromise;
  assert(importResponse.status() === 201, `React document import returned ${importResponse.status()}`);
  const importedDoc = (await importResponse.json()).document;
  const importedDocId = String(importedDoc?.doc_id ?? importedDoc?.id ?? '');
  assert(importedDocId, 'React document import did not return a document identifier');
  const readImportedDoc = await requestJson(fixture.baseUrl, `/api/document/${encodeURIComponent(importedDocId)}`, fixture.tokens.documents);
  assert(readImportedDoc.response.ok && readImportedDoc.payload.document?.content === importedText, 'Imported document did not persist its exact text');
  const crossOwnerDocument = await requestJson(fixture.baseUrl, `/api/document/${encodeURIComponent(importedDocId)}`, fixture.tokens.otherDocumentsRead);
  assert(crossOwnerDocument.response.status === 401, `Foreign installation read the imported document (HTTP ${crossOwnerDocument.response.status})`);
  await page.getByRole('button', { name: 'Download document text' }).waitFor({ state: 'visible', timeout: 15_000 });
  const documentDownloadPromise = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download document text' }).click();
  const documentDownload = await documentDownloadPromise;
  assert(documentDownload.suggestedFilename() === 'batch2a-synthetic-source.txt', `Unexpected document download filename ${documentDownload.suggestedFilename()}`);
  await page.getByRole('tab', { name: 'Uploaded files', exact: true }).click();
  const mediaUploadPromise = page.waitForResponse(response => response.url().endsWith('/api/media/ingest') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByLabel('Choose uploaded file').setInputFiles({
    name: 'batch2a-synthetic-media.txt', mimeType: 'text/plain', buffer: Buffer.from('Synthetic local media upload for the Files workflow.'),
  });
  const mediaUpload = await mediaUploadPromise;
  assert(mediaUpload.status() === 201, `React media upload returned ${mediaUpload.status()}`);
  const uploadedAsset = (await mediaUpload.json()).asset;
  assert(uploadedAsset?.id, 'Media ingestion response omitted its asset identifier');
  const mediaAssets = await requestJson(fixture.baseUrl, '/api/media/assets?project=default&limit=100', fixture.tokens.documents);
  assert(mediaAssets.response.ok && mediaAssets.payload.assets?.some(asset => String(asset.id) === String(uploadedAsset.id)), 'Uploaded media asset was not persisted');
  const crossOwnerMedia = await requestJson(fixture.baseUrl, '/api/media/assets?project=default&limit=100', fixture.tokens.otherDocumentsRead);
  if (crossOwnerMedia.response.ok && crossOwnerMedia.payload.assets?.some(asset => String(asset.id) === String(uploadedAsset.id))) {
    results.push({ workflow: 'Media installation isolation', result: 'BLOCKED', evidence: 'A foreign installation credential listed owner media data (HTTP 200); personal_remote isolation failed.' });
  } else {
    assert([401, 403].includes(crossOwnerMedia.response.status), `Foreign installation media read returned unexpected ${crossOwnerMedia.response.status}`);
    results.push({ workflow: 'Media installation isolation', result: 'PASS', evidence: `A credential issued by a separate installation could not list the owner media asset (HTTP ${crossOwnerMedia.response.status}).` });
  }
  results.push({ workflow: 'Files', result: 'PASS', evidence: `React imported a deterministic Markdown file through /api/documents/import, read the persisted text, and triggered the text download; a foreign installation credential was denied with HTTP ${crossOwnerDocument.response.status}.` });

  // Knowledge: submit a local-only text source from the approved Ingestion
  // panel and verify that the existing backend indexed the source.
  await open('knowledge', fixture.tokens.knowledge);
  await page.getByRole('tab', { name: 'Ingestion', exact: true }).click();
  await page.locator('.p-card').getByRole('button', { name: 'Add source', exact: true }).click();
  const knowledgeTitle = 'Batch 2A synthetic knowledge source';
  await page.getByLabel('Title', { exact: true }).fill(knowledgeTitle);
  await page.getByLabel('Source text', { exact: true }).fill('Local fixture evidence only. No external research or network provider is used.');
  const knowledgeIngestPromise = page.waitForResponse(response => response.url().endsWith('/api/knowledge/ingest') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Index source', exact: true }).click();
  const knowledgeIngest = await knowledgeIngestPromise;
  assert(knowledgeIngest.status() === 201, `Knowledge source ingestion returned ${knowledgeIngest.status()}`);
  const persistedKnowledge = await requestJson(fixture.baseUrl, '/api/knowledge/documents?project=default&limit=100', fixture.tokens.knowledge);
  assert(persistedKnowledge.response.ok && persistedKnowledge.payload.documents?.some(document => document.title === knowledgeTitle), 'Knowledge source was not persisted in the project index');
  await page.getByRole('tab', { name: 'Search', exact: true }).click();
  const knowledgeUploadText = 'Batch 2A Knowledge upload: deterministic text extracted locally.';
  const knowledgeUploadPromise = page.waitForResponse(response => response.url().endsWith('/api/knowledge/upload') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByLabel('Upload knowledge document').setInputFiles({
    name: 'batch2a-knowledge-upload.txt', mimeType: 'text/plain', buffer: Buffer.from(knowledgeUploadText),
  });
  const knowledgeUpload = await knowledgeUploadPromise;
  assert(knowledgeUpload.status() === 201, `React Knowledge upload returned ${knowledgeUpload.status()}`);
  const knowledgeUploadPayload = await knowledgeUpload.json();
  assert(Number(knowledgeUploadPayload.chars_extracted) > 0, 'Knowledge upload did not extract deterministic text locally');
  const listedUploadedKnowledge = await requestJson(fixture.baseUrl, '/api/knowledge/documents?project=default&limit=100', fixture.tokens.knowledge);
  assert(listedUploadedKnowledge.response.ok && listedUploadedKnowledge.payload.documents?.some(document => document.title === 'batch2a-knowledge-upload.txt'), 'Uploaded Knowledge file was not indexed in the project list');
  const crossOwnerKnowledge = await requestJson(fixture.baseUrl, '/api/knowledge/documents?project=default&limit=100', fixture.tokens.otherKnowledgeRead);
  if (crossOwnerKnowledge.response.ok && crossOwnerKnowledge.payload.documents?.some(document => document.title === knowledgeTitle)) {
    results.push({ workflow: 'Knowledge installation isolation', result: 'BLOCKED', evidence: 'A foreign installation credential listed owner Knowledge data (HTTP 200); personal_remote isolation failed.' });
  } else {
    assert([401, 403].includes(crossOwnerKnowledge.response.status), `Foreign installation Knowledge list returned unexpected ${crossOwnerKnowledge.response.status}`);
    results.push({ workflow: 'Knowledge installation isolation', result: 'PASS', evidence: `A credential issued by a separate installation could not list owner Knowledge data (HTTP ${crossOwnerKnowledge.response.status}).` });
  }
  results.push({ workflow: 'Knowledge', result: 'PASS', evidence: 'React Ingestion POSTed /api/knowledge/ingest and the Documents form multipart-POSTed /api/knowledge/upload; the real local server returned extracted text and both records appeared in the project list.' });

  // Communications: seed one synthetic inbox row, draft a reply through the
  // React control and archive it. This does not send mail or contact a provider.
  const importedMessageResponse = await requestJson(fixture.baseUrl, '/api/communication/messages/import', fixture.tokens.communication, {
    method: 'POST', body: {
      platform: 'email', sender: 'batch2a@example.test', recipients: ['local-user@example.test'],
      subject: 'Batch 2A synthetic inbox message', body: 'Local-only message fixture for draft and archive checks.', project: 'default',
    },
  });
  assert(importedMessageResponse.response.status === 201, `Synthetic inbox fixture import returned ${importedMessageResponse.response.status}`);
  const importedMessage = importedMessageResponse.payload.message;
  await open('communications', fixture.tokens.communication);
  await page.getByRole('button', { name: /Batch 2A synthetic inbox message/ }).click();
  const replyDraftPromise = page.waitForResponse(response => response.url().endsWith('/api/communication/drafts/reply') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Draft reply', exact: true }).click();
  const replyDraftResponse = await replyDraftPromise;
  assert(replyDraftResponse.status() === 201, `React reply draft returned ${replyDraftResponse.status()}`);
  const replyDraft = (await replyDraftResponse.json()).draft;
  assert(replyDraft?.id, 'Reply draft response omitted its identifier');
  const draftsAfterReply = await requestJson(fixture.baseUrl, '/api/communication/drafts?project=default&limit=100', fixture.tokens.communication);
  assert(draftsAfterReply.response.ok && draftsAfterReply.payload.drafts?.some(draft => String(draft.id) === String(replyDraft.id)), 'Reply draft did not persist in communication drafts');
  const archiveMessagePromise = page.waitForResponse(response => response.url().endsWith('/api/communication/messages/state') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Archive', exact: true }).click();
  const archiveMessage = await archiveMessagePromise;
  assert(archiveMessage.ok(), `React message archive returned ${archiveMessage.status()}`);
  const inboxAfterArchive = await requestJson(fixture.baseUrl, '/api/communication/inbox?project=default', fixture.tokens.communication);
  assert(inboxAfterArchive.response.ok && inboxAfterArchive.payload.messages?.some(message => String(message.id) === String(importedMessage.id) && (message.archived === true || Number(message.archived) === 1 || message.folder === 'archive')), 'Communication archive state was not persisted');
  assert(!apiRequests.some(request => request.url().includes('/api/communication/send')), 'The local communications smoke unexpectedly invoked a send operation');
  results.push({ workflow: 'Communications', result: 'PASS', evidence: 'The actual React Inbox created a reply draft through the existing API, persisted it, archived the synthetic imported message, and issued no send request.' });

  // Calendar: create a local event in the selected React calendar and verify
  // persistence through the read contract; no sync provider is involved.
  await open('calendar', fixture.tokens.calendar);
  await page.getByRole('button', { name: 'New', exact: true }).click();
  const calendarTitle = 'Batch 2A local calendar fixture';
  await page.locator('[aria-labelledby="calendar-create-title"] input').first().fill(calendarTitle);
  const calendarStart = page.getByLabel('Starts', { exact: true });
  const calendarEnd = page.getByLabel('Ends', { exact: true });
  const existingStart = await calendarStart.inputValue();
  const existingEnd = await calendarEnd.inputValue();
  assert(existingStart && existingEnd, 'Calendar event form did not initialise its start and end fields');
  const calendarCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/calendar/events') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Create event', exact: true }).click();
  const calendarCreate = await calendarCreatePromise;
  assert(calendarCreate.status() === 201, `React calendar create returned ${calendarCreate.status()}`);
  const createdCalendarEvent = (await calendarCreate.json()).event;
  assert(createdCalendarEvent?.id, 'Calendar create response omitted its event ID');
  const calendarRead = await requestJson(fixture.baseUrl, '/api/calendar/events?project=default', fixture.tokens.calendar);
  assert(calendarRead.response.ok && calendarRead.payload.events?.some(event => String(event.id) === String(createdCalendarEvent.id)), 'Calendar event did not persist through its read endpoint');
  results.push({ workflow: 'Calendar', result: 'PASS', evidence: 'React created a local calendar event using the existing POST contract and the record was returned by the authenticated calendar read endpoint.' });

  const demoArtifactCalls = apiRequests.filter(request => request.url().includes('/demo/'));
  assert(demoArtifactCalls.length === 0, 'Live integration unexpectedly requested Demo fixture assets');
  finishResults({ demoFixtureRequests: demoArtifactCalls.length, apiResponses: apiResponses.length });
}

try {
  await run();
} finally {
  if (previewCleanup) await previewCleanup().catch(() => undefined);
  if (browser) await browser.close().catch(() => undefined);
  if (vite && !vite.killed) vite.kill();
  if (backend && backend.stdin.writable) {
    backend.stdin.write('STOP\n');
    await new Promise(resolve => {
      const timeout = setTimeout(resolve, 5000);
      backend.once('exit', () => { clearTimeout(timeout); resolve(); });
    });
  }
}
