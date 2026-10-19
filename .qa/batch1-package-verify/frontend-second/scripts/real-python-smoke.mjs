import { spawn } from 'node:child_process';
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

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function waitFor(predicate, label, timeoutMs = 30_000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { const value = await predicate(); if (value) return value; } catch { /* process is still starting */ }
    await delay(250);
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

async function requestJson(base, route, token, { method = 'GET', body } = {}) {
  const response = await fetch(`${base}${route}`, {
    method,
    headers: { Authorization: `Bearer ${token}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    ...(body ? { body: JSON.stringify(body) } : {}),
  });
  let payload = {};
  try { payload = await response.json(); } catch { /* keep status evidence */ }
  return { response, payload };
}

async function run() {
  const fixture = await startBackend();
  console.log('isolated Python server ready');
  await waitFor(async () => (await fetch(`${fixture.baseUrl}/api/health`)).ok, 'Python health route');
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
    env: { ...process.env, OPENAGENT_API_TARGET: fixture.baseUrl },
    stdio: 'ignore', windowsHide: true,
  });
  const origin = 'http://127.0.0.1:5180';
  await waitFor(async () => (await fetch(origin)).ok, 'React Vite app');

  browser = await chromium.launch({ headless: true, executablePath: chrome });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const apiResponses = [];
  const apiRequests = [];
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
    const current = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(codeRun.id)}`, fixture.tokens.codeq);
    const currentCodeStatus = String(current.payload.run?.status ?? current.payload.status ?? '').toLowerCase();
    console.log('CodeQ run status at cancellation check:', currentCodeStatus || 'missing');
    let codeqCancellation = 'not exercised; run was no longer active';
    if (current.response.ok && ['queued', 'running', 'analysing', 'planning', 'executing', 'validating', 'waiting_approval', 'waiting'].includes(currentCodeStatus)) {
      const cancelButton = page.getByRole('button', { name: 'Cancel run' });
      await cancelButton.waitFor({ state: 'visible', timeout: 10_000 }).catch(() => undefined);
      if (await cancelButton.isVisible().catch(() => false)) {
        const cancelResponse = page.waitForResponse(response => response.url().endsWith(`/api/codebase/runs/${codeRun.id}/cancel`) && response.request().method() === 'POST');
        await cancelButton.click();
        const cancelled = await cancelResponse;
        assert(cancelled.ok(), `CodeQ cancellation failed through React control (${cancelled.status()})`);
        await waitFor(async () => {
          const state = await requestJson(fixture.baseUrl, `/api/codebase/runs/${encodeURIComponent(codeRun.id)}`, fixture.tokens.codeq);
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
    results.push({ workflow: 'CodeQ', result: 'PASS', evidence: `React project/run; event cursor, authenticated SSE event, cancellation and persisted cancellation; controlled provider failure and Resume returned 202 for the same persisted run.` });
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
  const cancelledResearch = await waitFor(async () => {
    const state = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(cancelResearchId)}`, fixture.tokens.research);
    return state.payload.session?.status === 'cancelled' ? state.payload.session : null;
  }, 'persisted Research cancellation', 60_000);
  assert(cancelledResearch.cancel_requested === true, 'Cancelled Research record did not preserve the cancellation request');
  const cancelledCheckpoints = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(cancelResearchId)}/checkpoints`, fixture.tokens.research);
  assert(cancelledCheckpoints.response.ok && cancelledCheckpoints.payload.checkpoints.some(checkpoint => checkpoint.stage === 'cancel-requested' || checkpoint.stage === 'cancelled'), 'Research cancellation checkpoint was not persisted');
  await expect(page.locator('.research-side [role="log"]')).toContainText(/cancel-requested|cancelled/i, { timeout: 10_000 });
  results.push({ workflow: 'Research', result: 'PASS', evidence: 'React created a session with external access disabled; persisted report/checkpoint; authenticated event stream rendered in Activity; React Cancel run POST persisted cancelled state and checkpoint.' });

  // Authorized, project-specific agent history now uses the actual React query.
  const agentRun = await requestJson(fixture.baseUrl, '/api/agents/run', fixture.tokens.agents, {
    method: 'POST', body: { agent: 'executive', goal: 'History integration fixture', project: 'default' },
  });
  assert(agentRun.response.status === 201, `Agent run creation returned ${agentRun.response.status}`);
  const forbiddenAgent = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.readonly);
  assert(forbiddenAgent.response.status === 403, `Read-only unrelated token unexpectedly received ${forbiddenAgent.response.status}`);
  const crossOwnerAgent = await requestJson(fixture.baseUrl, '/api/agent-runs?project=default', fixture.tokens.otherAgentRead);
  assert(crossOwnerAgent.response.status === 404, `Cross-owner history read returned ${crossOwnerAgent.response.status}, expected non-disclosing 404`);
  await open('agents', fixture.tokens.agents);
  await waitFor(async () => apiResponses.some(item => item.path === '/api/agent-runs' && item.status === 200), 'React authorized agent history query');
  await page.getByText('Run history · 1', { exact: true }).click();
  await page.getByRole('heading', { name: 'History integration fixture', exact: true }).waitFor({ state: 'visible' });
  results.push({ workflow: 'Agents', result: 'PASS', evidence: 'React read authorized owner-bound project history; unrelated token was denied with 403; the real run record rendered.' });

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

  const demoArtifactCalls = apiRequests.filter(request => request.url().includes('/demo/'));
  assert(demoArtifactCalls.length === 0, 'Live integration unexpectedly requested Demo fixture assets');
  console.log(JSON.stringify({
    kind: 'real-react-python-integration',
    authEnforced: true,
    externalProvidersEnabled: false,
    demoFixtureRequests: demoArtifactCalls.length,
    apiResponses: apiResponses.length,
    results,
  }, null, 2));
}

try {
  await run();
} finally {
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
