import { spawn } from 'node:child_process';
import { access, writeFile } from 'node:fs/promises';
import { request as httpRequest } from 'node:http';
import { createServer as createNetServer } from 'node:net';
import { setTimeout as delay } from 'node:timers/promises';
import { chromium, expect } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const repositoryRoot = process.env.OPENAGENT_TEST_REPOSITORY_ROOT || path.resolve(frontendRoot, '..');
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

function syntheticSilenceWav() {
  const sampleRate = 8_000;
  const sampleCount = 800;
  const dataBytes = sampleCount * 2;
  const wav = Buffer.alloc(44 + dataBytes);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + dataBytes, 4); wav.write('WAVE', 8);
  wav.write('fmt ', 12); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20);
  wav.writeUInt16LE(1, 22); wav.writeUInt32LE(sampleRate, 24);
  wav.writeUInt32LE(sampleRate * 2, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34);
  wav.write('data', 36); wav.writeUInt32LE(dataBytes, 40);
  return wav;
}

async function waitFor(predicate, label, timeoutMs = 30_000, intervalMs = 250) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    try { const value = await predicate(); if (value) return value; } catch { /* process is still starting */ }
    await delay(intervalMs);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

async function reserveLocalPort() {
  const server = createNetServer();
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const address = server.address();
  if (!address || typeof address === 'string') {
    await new Promise(resolve => server.close(resolve));
    throw new Error('Could not reserve an isolated local port for the React smoke server');
  }
  const port = address.port;
  await new Promise((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  return port;
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

async function requestBytes(base, route, token) {
  const url = new URL(route, base);
  return new Promise((resolve, reject) => {
    const request = httpRequest({
      protocol: url.protocol, hostname: url.hostname, port: url.port, method: 'GET',
      path: `${url.pathname}${url.search}`,
      headers: { Host: 'remote.test', 'X-Forwarded-Proto': 'https', Authorization: `Bearer ${token}` },
    }, response => {
      const chunks = [];
      response.on('data', chunk => chunks.push(Buffer.from(chunk)));
      response.on('end', () => resolve({
        status: response.statusCode ?? 0, headers: response.headers,
        body: Buffer.concat(chunks),
      }));
    });
    request.on('error', reject);
    request.end();
  });
}

function finishResults(metadata = {}) {
  const blockers = results.filter(entry => entry.result !== 'PASS');
  console.log(JSON.stringify({
    kind: 'real-react-python-integration',
    overallStatus: blockers.length > 0 ? 'BLOCKED' : 'PASS',
    blockedWorkflows: blockers.map(entry => entry.workflow),
    authEnforced: true,
    externalProvidersEnabled: false,
    workflowCount: results.length,
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
  const directOwnerLogin = await requestJson(fixture.baseUrl, '/api/auth/login', '', {
    method: 'POST', body: { username_or_email: fixture.ownerUsername, password: fixture.ownerPassword },
  });
  assert(directOwnerLogin.response.status === 200, `Isolated Python owner login returned ${directOwnerLogin.response.status}: ${await directOwnerLogin.response.text()}`);
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

  const vitePort = await reserveLocalPort();
  vite = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', String(vitePort), '--strictPort'], {
    cwd: frontendRoot,
    env: { ...process.env, OPENAGENT_API_TARGET: fixture.baseUrl, OPENAGENT_TEST_PERSONAL_REMOTE_PROXY: '1' },
    stdio: 'ignore', windowsHide: true,
  });
  const origin = `http://127.0.0.1:${vitePort}`;
  await waitFor(async () => {
    if (vite.exitCode !== null) throw new Error(`Isolated Vite exited with ${vite.exitCode}`);
    return (await fetch(origin)).ok;
  }, 'isolated React Vite app');

  browser = await chromium.launch({ headless: true, executablePath: chrome });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();
  const apiResponses = [];
  const apiRequests = [];
  const onlyAgents = process.env.OPENAGENT_SMOKE_ONLY === 'agents';
  const onlyCalendar = process.env.OPENAGENT_SMOKE_ONLY === 'calendar';
  page.on('pageerror', error => console.error(`React page error: ${error.stack ?? error.message}`));
  page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request); });
  page.on('response', response => {
    if (new URL(response.url()).pathname.startsWith('/api/')) {
      apiResponses.push({ path: new URL(response.url()).pathname, status: response.status() });
      if (new URL(response.url()).pathname === '/api/integrations/preferences') console.log(`notification preference API ${response.request().method()} ${response.status()}`);
    }
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
  assert(loginResponse.status() === 200, `React owner sign-in returned HTTP ${loginResponse.status()}: ${await loginResponse.text()}`);
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

  if (!onlyAgents && !onlyCalendar) {
  // Fine-tuning job records: create a disposable queued record through the
  // real Python API, then cancel it from React and verify persisted status.
  const fineTuneDataset = await requestJson(fixture.baseUrl, '/api/finetune/datasets', fixture.tokens.finetune, {
    method: 'POST', body: { project: 'default', name: 'Disposable fine-tuning smoke dataset' },
  });
  assert(fineTuneDataset.response.status === 201, `Fine-tuning dataset creation returned ${fineTuneDataset.response.status}`);
  const fineTuneJob = await requestJson(fixture.baseUrl, '/api/finetune/jobs', fixture.tokens.finetune, {
    method: 'POST', body: { project: 'default', dataset_id: fineTuneDataset.payload.dataset.id, base_model: 'local-smoke-model', method: 'lora' },
  });
  assert(fineTuneJob.response.status === 201 && fineTuneJob.payload.job?.status === 'queued', 'Fine-tuning did not return a persisted queued job record');
  await open('finetune', fixture.tokens.finetune);
  await page.getByRole('tab', { name: 'Jobs' }).click();
  const fineTuneJobCard = page.locator('#content article.p-record-section').filter({ hasText: 'local-smoke-model' });
  await expect(fineTuneJobCard).toContainText('queued', { timeout: 15_000 });
  const cancelFineTuneResponse = page.waitForResponse(response =>
    new URL(response.url()).pathname.endsWith(`/api/finetune/jobs/${fineTuneJob.payload.job.id}/cancel`) &&
    response.request().method() === 'POST', { timeout: 15_000 });
  page.once('dialog', dialog => dialog.accept());
  await fineTuneJobCard.getByRole('button', { name: 'Cancel queued job' }).click();
  const fineTuneCancelResponse = await cancelFineTuneResponse;
  assert(fineTuneCancelResponse.status() === 200, `Fine-tuning queued cancellation returned ${fineTuneCancelResponse.status()}`);
  await expect(page.locator('#content')).toContainText('cancelled', { timeout: 15_000 });
  const persistedFineTuneJobs = await requestJson(fixture.baseUrl, '/api/finetune/jobs?project=default', fixture.tokens.finetune);
  assert(persistedFineTuneJobs.response.ok && persistedFineTuneJobs.payload.jobs?.some(job => job.id === fineTuneJob.payload.job.id && job.status === 'cancelled'), 'Cancelled Fine-tuning status did not persist in Python');
  results.push({ workflow: 'Fine-tuning queued job cancellation', result: 'PASS', evidence: 'A disposable job was persisted by the isolated Python server, then cancelled through the authenticated React UI. The project-scoped job list returned cancelled; no training process or checkpoint was claimed.' });

  // Image Studio uploads a disposable local image and exercises the existing
  // project-scoped upscale API through the real React controls.
  await open('gallery', fixture.tokens.documents);
  await page.getByRole('tab', { name: 'Upload' }).click();
  await page.locator('input[type="file"]').setInputFiles({
    name: 'smoke-gallery.png', mimeType: 'image/png',
    buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/G7sAAAAASUVORK5CYII=', 'base64'),
  });
  const galleryUploadPromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/gallery/upload' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Upload', exact: true }).click();
  const galleryUpload = await galleryUploadPromise;
  assert(galleryUpload.status() === 201, `Image Studio upload returned HTTP ${galleryUpload.status()}: ${await galleryUpload.text()}`);
  const galleryUploadPayload = await galleryUpload.json();
  const galleryAssetId = galleryUploadPayload.asset?.id;
  assert(galleryAssetId, 'Image Studio upload did not return a persisted asset id');
  await page.getByRole('tab', { name: 'Library' }).click();
  await page.getByRole('button', { name: /smoke-gallery\.png/ }).waitFor({ state: 'visible' });
  await page.getByRole('button', { name: /smoke-gallery\.png/ }).click();
  await page.getByRole('tab', { name: 'Operations' }).click();
  const galleryUpscalePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/gallery/upscale' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Upscale' }).click();
  const galleryUpscale = await galleryUpscalePromise;
  assert(galleryUpscale.ok, `Image Studio upscale returned HTTP ${galleryUpscale.status()}: ${await galleryUpscale.text()}`);
  const galleryUpscalePayload = await galleryUpscale.json();
  assert(galleryUpscalePayload.asset?.id, 'Image Studio upscale did not return the resulting backend asset');
  results.push({ workflow: 'Image Studio upload and upscale', result: 'PASS', evidence: `React uploaded a synthetic 1×1 PNG as asset ${galleryAssetId}, selected it in Library and called the real global upscale contract with the selected asset ID. Python returned the persisted derivative ${galleryUpscalePayload.asset.id}; no external image service was used.` });

  // Personal OS uses the backend's title/body schema and renders persisted
  // notes from the service response rather than a client-only example.
  await open('personalos', fixture.tokens.personalOS);
  await page.getByRole('tab', { name: 'Notes' }).click();
  await page.getByPlaceholder('Write a note…').fill('Synthetic Personal OS integration note.');
  const personalNotePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/personal-os/notes' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Save note' }).click();
  const personalNoteResponse = await personalNotePromise;
  assert(personalNoteResponse.status() === 201, `Personal OS note save returned HTTP ${personalNoteResponse.status()}`);
  const personalNotePayload = await personalNoteResponse.json();
  assert(personalNotePayload.note?.body === 'Synthetic Personal OS integration note.', 'Personal OS did not persist the note body');
  await expect(page.getByText('Synthetic Personal OS integration note.', { exact: true })).toBeVisible();
  const persistedPersonalNotes = await requestJson(fixture.baseUrl, '/api/personal-os/notes?project=default', fixture.tokens.personalOS);
  assert(persistedPersonalNotes.response.ok && persistedPersonalNotes.payload.notes?.some(note => note.id === personalNotePayload.note.id && note.body === 'Synthetic Personal OS integration note.'), 'Personal OS note was not returned from a fresh Python read');
  results.push({ workflow: 'Personal OS note persistence', result: 'PASS', evidence: 'React submitted the backend title/body note contract, rendered the returned record, and a fresh authenticated Python read returned the same persisted note.' });

  // Model Cookbook serves are explicitly plans in the current UI. The test
  // proves persisted host/session/log/stop behavior without launching a child.
  await open('cookbook', fixture.tokens.models);
  await page.getByPlaceholder('Host name').fill('Disposable cookbook host');
  await page.getByPlaceholder('Host address').fill('127.0.0.1');
  const hostCreatePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/model-cookbook/hosts' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Add', exact: true }).click();
  const hostCreateResponse = await hostCreatePromise;
  assert(hostCreateResponse.status() === 201, `Model Cookbook host create returned HTTP ${hostCreateResponse.status()}`);
  const hostPayload = await hostCreateResponse.json();
  assert(hostPayload.host?.label === 'Disposable cookbook host', 'Model Cookbook did not store the submitted host label');
  await page.getByRole('textbox', { name: 'Model reference to serve' }).fill('local-smoke-model');
  const servePlanPromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/model-cookbook/serve' && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Create serve plan' }).click();
  const servePlanResponse = await servePlanPromise;
  assert(servePlanResponse.status() === 201, `Model Cookbook serve plan returned HTTP ${servePlanResponse.status()}`);
  const servePlanPayload = await servePlanResponse.json();
  const serveSession = servePlanPayload.session;
  assert(serveSession?.status === 'planned' && serveSession?.mode === 'plan' && serveSession?.pid === 0, 'Model Cookbook started a process instead of saving a plan');
  const persistedCookbookSessions = await requestJson(fixture.baseUrl, '/api/model-cookbook/sessions?project=default', fixture.tokens.models);
  assert(persistedCookbookSessions.response.ok && persistedCookbookSessions.payload.sessions?.some(session => session.id === serveSession.id), 'Model Cookbook session history did not persist');
  await page.getByRole('tab', { name: 'Logs' }).click();
  await expect(page.getByText(/Serve session planned/)).toBeVisible();
  const closePlanPromise = page.waitForResponse(response => new URL(response.url()).pathname === `/api/model-cookbook/stop/${serveSession.id}` && response.request().method() === 'POST');
  await page.getByRole('tab', { name: 'Hosts' }).click();
  await page.getByRole('button', { name: 'Close plan' }).click();
  const closePlanResponse = await closePlanPromise;
  assert(closePlanResponse.ok, `Model Cookbook plan close returned HTTP ${closePlanResponse.status()}`);
  const closedSessionRead = await requestJson(fixture.baseUrl, '/api/model-cookbook/sessions?project=default', fixture.tokens.models);
  assert(closedSessionRead.payload.sessions?.some(session => session.id === serveSession.id && session.status === 'stopped'), 'Model Cookbook plan stop state did not persist');
  results.push({ workflow: 'Model Cookbook host, serve plan, logs and stop', result: 'PASS', evidence: `React persisted host ${hostPayload.host.id}, created planned session ${serveSession.id}, displayed its backend log, then closed the plan. The session list confirmed stopped status; allow_process=false and pid=0 prove no process was launched.` });

  // Scientific AI Reports: execute only the owner-approved bounded fixture
  // source through the existing local runner, then verify history persistence.
  await open('science', fixture.tokens.knowledge);
  await page.getByRole('tab', { name: 'Reports' }).click();
  await page.getByRole('textbox', { name: 'Python source' }).fill("print('isolated science verification')");
  await page.getByRole('checkbox', { name: /I reviewed this Python source/ }).check();
  const scienceRunResponsePromise = page.waitForResponse(response => response.url().endsWith('/api/science/execute') && response.request().method() === 'POST', { timeout: 20_000 });
  await page.getByRole('button', { name: 'Run approved code' }).click();
  const scienceRunResponse = await scienceRunResponsePromise;
  assert(scienceRunResponse.status() === 201, `Scientific code run returned ${scienceRunResponse.status()}: ${await scienceRunResponse.text()}`);
  const persistedScienceExecutions = await requestJson(fixture.baseUrl, '/api/science/executions?project=default', fixture.tokens.knowledge);
  const scienceExecutionRecord = persistedScienceExecutions.payload.executions?.[0];
  assert(persistedScienceExecutions.response.ok && scienceExecutionRecord?.stdout?.includes('isolated science verification'), 'Scientific execution output was not persisted in the project report history');
  assert(!Object.hasOwn(scienceExecutionRecord, 'final_code'), 'Scientific report stored executable source code instead of only a source hash');
  results.push({ workflow: 'Scientific AI bounded execution report', result: 'PASS', evidence: 'React required explicit approval, submitted a deterministic local-only script to the policy-checked bounded Python runner, and the project-scoped history returned the actual output and source hash without storing source code.' });

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

  // Delete only the disposable conversation created by this smoke run, using
  // the user's existing confirmation path and the existing scoped Live route.
  await page.getByRole('button', { name: 'Conversation actions' }).click();
  await page.getByRole('button', { name: 'Conversation history', exact: true }).click();
  const organizedChatHistory = await requestJson(fixture.baseUrl, '/api/conversations/organized?project=default&limit=100', fixture.tokens.chat);
  assert(organizedChatHistory.response.ok && organizedChatHistory.payload.conversations?.some(item => String(item.id) === String(conversationId)), 'Disposable Chat conversation was not present in organized history before delete');
  const disposableConversation = organizedChatHistory.payload.conversations.find(item => String(item.id) === String(conversationId));
  const conversationTitle = String(disposableConversation.title || conversationId);
  await page.getByRole('button', { name: conversationTitle, exact: true }).waitFor({ state: 'visible', timeout: 15_000 });
  page.once('dialog', dialog => dialog.accept());
  const deleteConversationPromise = page.waitForResponse(response => response.url().endsWith(`/api/conversations/${encodeURIComponent(conversationId)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('button', { name: `Delete ${conversationTitle}`, exact: true }).click();
  const deleteConversationResponse = await deleteConversationPromise;
  assert(deleteConversationResponse.ok(), `React conversation delete returned HTTP ${deleteConversationResponse.status()}`);
  const remainingConversations = await requestJson(fixture.baseUrl, '/api/conversations?project=default&limit=100', fixture.tokens.chat);
  assert(remainingConversations.response.ok && !remainingConversations.payload.conversations?.some(item => String(item.id) === String(conversationId)), 'Deleted Chat conversation remained in Python history');
  results.push({ workflow: 'Chat conversation deletion', result: 'PASS', evidence: 'React confirmed and deleted the disposable persisted conversation through authenticated DELETE /api/conversations/{id}; a subsequent Python history read no longer returned the conversation.' });

  // CodeQ: actual React project discovery and run lifecycle against a copied
  // disposable workspace. External providers remain disabled.
  await open('codeq', fixture.tokens.codeq);
  await page.getByLabel('CodeQ project').waitFor({ state: 'visible' });
  await page.getByLabel('CodeQ project').selectOption(String(codeProjectId));
  const smokeFilePath = `openagent-smoke-${Date.now()}.txt`;
  const renamedSmokeFilePath = smokeFilePath.replace('.txt', '-renamed.txt');
  page.once('dialog', dialog => dialog.accept(smokeFilePath));
  const createCodeQFilePromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/paths`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').action === 'create_file', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Create file' }).click();
  const createCodeQFileResponse = await createCodeQFilePromise;
  assert(createCodeQFileResponse.ok(), `React CodeQ create-file returned HTTP ${createCodeQFileResponse.status()}`);
  await expect(page.locator('.monaco-editor')).toBeVisible({ timeout: 15_000 });
  await page.locator('.monaco-editor .view-lines').click({ position: { x: 16, y: 10 } });
  await page.keyboard.insertText('OpenAgent CodeQ persisted file smoke.');
  await expect(page.locator('.monaco-editor .view-lines')).toContainText('OpenAgent CodeQ persisted file smoke.');
  await expect(page.getByRole('button', { name: 'Save file' })).toBeEnabled();
  const saveCodeQFilePromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/file`) && response.request().method() === 'POST', { timeout: 10_000 }).catch(async cause => {
    const calls = apiRequests.filter(request => request.url().includes(`/api/codebase/projects/${codeProjectId}/file`)).map(request => `${request.method()} ${request.url()}`);
    throw new Error(`CodeQ Save file produced no matching response; editor=${JSON.stringify((await page.locator('.monaco-editor .view-lines').innerText()).slice(0, 120))}; toolbar=${JSON.stringify(await page.locator('.codeq-tabbar').innerText())}; requests=${JSON.stringify(calls)}; cause=${String(cause)}`);
  });
  await page.getByRole('button', { name: 'Save file' }).click();
  const saveCodeQFileResponse = await saveCodeQFilePromise;
  assert(saveCodeQFileResponse.ok(), `React CodeQ save-file returned HTTP ${saveCodeQFileResponse.status()}`);
  const saveCodeQFilePayload = await saveCodeQFileResponse.json();
  const savedCodeQFile = await requestJson(fixture.baseUrl, `/api/codebase/projects/${encodeURIComponent(codeProjectId)}/file?path=${encodeURIComponent(smokeFilePath)}`, fixture.tokens.codeq);
  assert(savedCodeQFile.response.ok && String(savedCodeQFile.payload.file?.content ?? '').includes('OpenAgent CodeQ persisted file smoke.'), `CodeQ file content did not persist through the Python backend: UI response=${JSON.stringify(saveCodeQFilePayload)}; fresh read status=${savedCodeQFile.response.status}, payload=${JSON.stringify(savedCodeQFile.payload)}`);
  const renameDialogHandler = async dialog => {
    if (dialog.type() === 'confirm') await dialog.accept();
    else await dialog.accept(renamedSmokeFilePath);
  };
  page.on('dialog', renameDialogHandler);
  const renameCodeQFilePromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/paths`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').action === 'rename', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Rename selected path' }).click();
  const renameCodeQFileResponse = await renameCodeQFilePromise;
  page.off('dialog', renameDialogHandler);
  assert(renameCodeQFileResponse.ok(), `React CodeQ rename returned HTTP ${renameCodeQFileResponse.status()}`);
  const renamedCodeQFile = await requestJson(fixture.baseUrl, `/api/codebase/projects/${encodeURIComponent(codeProjectId)}/file?path=${encodeURIComponent(renamedSmokeFilePath)}`, fixture.tokens.codeq);
  assert(renamedCodeQFile.response.ok && String(renamedCodeQFile.payload.file?.content ?? '').includes('OpenAgent CodeQ persisted file smoke.'), 'CodeQ rename did not preserve the saved file content');
  page.once('dialog', dialog => dialog.accept());
  const deleteCodeQFilePromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/paths`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').action === 'delete', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Delete selected path' }).click();
  const deleteCodeQFileResponse = await deleteCodeQFilePromise;
  assert(deleteCodeQFileResponse.ok(), `React CodeQ delete returned HTTP ${deleteCodeQFileResponse.status()}`);
  const deletedCodeQFile = await requestJson(fixture.baseUrl, `/api/codebase/projects/${encodeURIComponent(codeProjectId)}/file?path=${encodeURIComponent(renamedSmokeFilePath)}`, fixture.tokens.codeq);
  assert(deletedCodeQFile.response.status === 404, `Deleted CodeQ file remained readable (HTTP ${deletedCodeQFile.response.status})`);
  results.push({ workflow: 'CodeQ file lifecycle', result: 'PASS', evidence: 'The actual React/Monaco editor created a file in the disposable repository, saved editor content through the authenticated CodeQ file operation, verified the content from a fresh Python read, renamed it without losing content, then deleted it and confirmed a fresh read returned 404.' });
  await page.getByRole('textbox', { name: 'Preview command' }).fill('node preview-server.mjs');
  await page.getByRole('checkbox', { name: /I reviewed this command/ }).check();
  page.once('dialog', dialog => dialog.accept());
  const previewProbePromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').tool === 'probe_local_url', { timeout: 15_000 });
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
  const previewProbeResponse = await previewProbePromise;
  assert(previewProbeResponse.ok(), `CodeQ preview readiness probe returned HTTP ${previewProbeResponse.status()}`);
  const previewProbePayload = await previewProbeResponse.json();
  assert(previewProbePayload.result?.reachable === true && previewProbePayload.result?.http_status === 200, `CodeQ preview readiness probe did not verify HTTP 200: ${JSON.stringify(previewProbePayload)}`);
  await expect(page.getByRole('region', { name: 'Live CodeQ preview session' })).toContainText(`Preview is ready · ${previewUrl}`, { timeout: 15_000 });
  const previewStartCountBeforeReload = apiRequests.filter(request => request.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && JSON.parse(request.postData() ?? '{}').tool === 'start_process').length;
  const restoredPreviewStatus = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').tool === 'process_status', { timeout: 15_000 });
  await page.reload();
  await expect(page.getByRole('link', { name: previewUrl })).toBeVisible({ timeout: 15_000 });
  assert((await restoredPreviewStatus).ok(), 'CodeQ preview process status could not be restored after a page reload');
  assert(apiRequests.filter(request => request.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && JSON.parse(request.postData() ?? '{}').tool === 'start_process').length === previewStartCountBeforeReload, 'Reloading CodeQ started a duplicate preview process');
  await page.getByRole('button', { name: 'Refresh preview status' }).click();
  await waitFor(async () => apiRequests.some(request => request.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && JSON.parse(request.postData() ?? '{}').tool === 'process_status'), 'CodeQ preview status refresh', 10_000);
  const previewStopPromise = page.waitForResponse(response => response.url().includes(`/api/codebase/projects/${codeProjectId}/tools`) && response.request().method() === 'POST' && JSON.parse(response.request().postData() ?? '{}').tool === 'stop_process', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Stop preview' }).click();
  const previewStopResponse = await previewStopPromise;
  assert(previewStopResponse.ok(), `CodeQ preview stop returned HTTP ${previewStopResponse.status()}`);
  assert(String((await previewStopResponse.json()).result?.status ?? '') === 'exited', 'CodeQ preview process did not report its stopped state');
  results.push({ workflow: 'CodeQ preview lifecycle', result: 'PASS', evidence: `React used the authenticated project tools contract to start process ${previewProcessId}; the loopback-only readiness tool verified HTTP 200 at ${previewUrl}. After page reload React restored that same process without starting a duplicate, and status refresh and stop were accepted by the backend.` });
  await page.getByRole('tab', { name: /Terminal/ }).click();
  await page.getByRole('textbox', { name: 'Terminal command' }).fill('python --version');
  const terminalStartPromise = page.waitForResponse(response => response.url().endsWith('/api/terminal/runs') && response.request().method() === 'POST');
  await page.getByRole('button', { name: 'Run command' }).click();
  const terminalStartResponse = await terminalStartPromise;
  assert(terminalStartResponse.status() === 202, `CodeQ terminal start returned HTTP ${terminalStartResponse.status()}`);
  const terminalStarted = await terminalStartResponse.json();
  const terminalRunId = String(terminalStarted.run?.id ?? '');
  assert(terminalRunId, 'CodeQ terminal start did not return a persisted run ID');
  await expect(page.getByRole('log', { name: 'Terminal run output' })).toContainText(/Python \d+\.\d+/, { timeout: 10_000 });
  const terminalStartRequest = apiRequests.find(request => request.url().endsWith('/api/terminal/runs') && request.method() === 'POST');
  assert(Boolean(terminalStartRequest && (await terminalStartRequest.allHeaders()).authorization), 'CodeQ terminal start omitted the API bearer token');
  results.push({ workflow: 'CodeQ incremental terminal', result: 'PASS', evidence: `React started authenticated run ${terminalRunId}, displayed output from the local Python executable through persisted status polling, and retained the existing safe-command policy.` });
  const terminalReadDenied = await requestJson(fixture.baseUrl, '/api/terminal/runs?project=default&limit=1', fixture.tokens.readonly);
  const terminalWriteDenied = await requestJson(fixture.baseUrl, '/api/terminal/runs', fixture.tokens.readonly, {
    method: 'POST', body: { project: 'default', command: ['python', '--version'], cwd: fixture.repoPath, approved: false },
  });
  assert(terminalReadDenied.response.status === 403 && terminalWriteDenied.response.status === 403,
    `terminal scopes were not enforced (read ${terminalReadDenied.response.status}, write ${terminalWriteDenied.response.status})`);
  results.push({ workflow: 'CodeQ terminal least-privilege scopes', result: 'PASS', evidence: 'The existing media/workspace read-only credential was denied terminal history and command execution with HTTP 403; no command was started.' });
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
        const cancelPath = `/api/codebase/runs/${encodeURIComponent(reconnectRun.id)}/cancel`;
        const cancelResponse = page.waitForResponse(response => new URL(response.url()).pathname === cancelPath && response.request().method() === 'POST');
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
    const resumePath = `/api/codebase/runs/${encodeURIComponent(failedRun.id)}/resume`;
    const resumeResponsePromise = page.waitForResponse(response => new URL(response.url()).pathname === resumePath && response.request().method() === 'POST', { timeout: 15_000 });
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
  // the React form, then verify its report, source review note and events.
  await open('research', fixture.tokens.research);
  await page.getByText('New research', { exact: true }).click();
  await page.getByLabel('Research question').fill('Review the Research source-note smoke fixture.');
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
    const sourceTitle = String(completedResearch.sources[0].title ?? completedResearch.sources[0].url ?? '');
    const sourceButton = page.locator('.research-side button.timeline-row').filter({ hasText: sourceTitle }).first();
    await sourceButton.click();
    await page.getByLabel('Local source note').fill('Verified against the temporary local integration source.');
    const researchNoteSave = page.waitForResponse(response => response.url().includes(`/api/research/sessions/${researchId}/source-notes/`) && response.request().method() === 'POST', { timeout: 15_000 });
    await page.getByRole('button', { name: 'Save source note' }).click();
    const researchNoteResponse = await researchNoteSave;
    assert(researchNoteResponse.status() === 200, `Research source-note save returned ${researchNoteResponse.status()}`);
    const researchNotePayload = await researchNoteResponse.json();
    assert(researchNotePayload.note?.content === 'Verified against the temporary local integration source.', 'Research source-note response omitted the saved note');
    const persistedNotes = await requestJson(fixture.baseUrl, `/api/research/sessions/${encodeURIComponent(researchId)}/source-notes?project=default`, fixture.tokens.research);
    assert(persistedNotes.response.ok && persistedNotes.payload.notes?.some(note => note.source_key === researchNotePayload.note.source_key && note.content === researchNotePayload.note.content), 'Research source note did not persist in Python storage');
  } else {
    throw new Error('Synthetic local Research context source was not returned; source-note React-to-Python workflow could not be exercised.');
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
  results.push({ workflow: 'Research owner session workflow', result: crossOwnerResearch.response.status === 200 ? 'BLOCKED' : 'PASS', evidence: 'React created an external-network-disabled session; the local worker completed it; the report rendered in React; a seeded temporary context source was selected and its note was saved through React, then read back from the Python session-scoped persistence route; authenticated events/checkpoints were visible in Activity. Foreign-installation credentials were rejected by the personal_remote boundary.' });
  }

  if (!onlyCalendar) {
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

  // Personal Assistant task and goal lifecycle uses the dedicated assistant
  // records and authenticated assistant:write permission.
  await open('assistant', fixture.tokens.assistant);
  await page.getByRole('tab', { name: 'Tasks', exact: true }).click();
  const assistantTaskTitle = 'Batch 2B synthetic assistant task';
  await page.getByLabel('Task', { exact: true }).fill(assistantTaskTitle);
  const assistantTaskCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/assistant/tasks') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Add task', exact: true }).click();
  const assistantTaskCreate = await assistantTaskCreatePromise;
  assert(assistantTaskCreate.status() === 201, `Assistant task create returned ${assistantTaskCreate.status()}`);
  const assistantTask = (await assistantTaskCreate.json()).task;
  assert(assistantTask?.id, 'Assistant task create returned no ID');
  const assistantTaskRow = page.locator('.p-record-section').filter({ hasText: assistantTaskTitle });
  await assistantTaskRow.getByRole('button', { name: 'Edit', exact: true }).click();
  await page.getByLabel('Edit task', { exact: true }).fill(`${assistantTaskTitle} updated`);
  const assistantTaskPatchPromise = page.waitForResponse(response => response.url().includes(`/api/assistant/tasks/${encodeURIComponent(assistantTask.id)}`) && response.request().method() === 'PATCH', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Save changes', exact: true }).click();
  const assistantTaskPatch = await assistantTaskPatchPromise;
  assert(assistantTaskPatch.status() === 200, `Assistant task update returned ${assistantTaskPatch.status()}`);
  await page.locator('.p-record-section').filter({ hasText: `${assistantTaskTitle} updated` }).getByRole('button', { name: 'Mark complete', exact: true }).click();
  await page.getByText(/Backend task status: completed/).waitFor({ state: 'visible', timeout: 10_000 });
  page.once('dialog', dialog => dialog.accept());
  const assistantTaskDeletePromise = page.waitForResponse(response => response.url().includes(`/api/assistant/tasks/${encodeURIComponent(assistantTask.id)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.locator('.p-record-section').filter({ hasText: `${assistantTaskTitle} updated` }).getByRole('button', { name: 'Delete', exact: true }).click();
  assert((await assistantTaskDeletePromise).status() === 200, 'Assistant task delete failed');
  const remainingTasks = await requestJson(fixture.baseUrl, '/api/assistant/tasks?project=default', fixture.tokens.assistant);
  assert(remainingTasks.response.ok && !remainingTasks.payload.tasks?.some(task => String(task.id) === String(assistantTask.id)), 'Assistant task deletion did not persist');

  await page.getByRole('tab', { name: 'Goals', exact: true }).click();
  // A separate disabled backend-action sample shares the accessible label.
  // Target the enabled workspace control inside the Goals card.
  await page.locator('.p-primary > .p-record-sections > .p-card .p-card-heading').getByRole('button', { name: 'Add goal', exact: true }).click();
  const goalDialog = page.getByRole('dialog', { name: 'New goal' });
  const assistantGoalTitle = 'Batch 2B synthetic assistant goal';
  await goalDialog.getByLabel('Name').fill(assistantGoalTitle);
  const assistantGoalCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/assistant/goals') && response.request().method() === 'POST', { timeout: 15_000 });
  await goalDialog.getByRole('button', { name: 'Create goal' }).click();
  const assistantGoalCreate = await assistantGoalCreatePromise;
  assert(assistantGoalCreate.status() === 201, `Assistant goal create returned ${assistantGoalCreate.status()}`);
  const assistantGoal = (await assistantGoalCreate.json()).goal;
  assert(assistantGoal?.id, 'Assistant goal create returned no ID');
  const assistantGoalRow = page.locator('.p-record-section').filter({ hasText: assistantGoalTitle });
  await assistantGoalRow.getByRole('button', { name: 'Edit', exact: true }).click();
  const editGoalDialog = page.getByRole('dialog', { name: 'Edit goal' });
  await editGoalDialog.getByLabel('Name').fill(`${assistantGoalTitle} updated`);
  const assistantGoalPatchPromise = page.waitForResponse(response => response.url().includes(`/api/assistant/goals/${encodeURIComponent(assistantGoal.id)}`) && response.request().method() === 'PATCH', { timeout: 15_000 });
  await editGoalDialog.getByRole('button', { name: 'Save changes' }).click();
  assert((await assistantGoalPatchPromise).status() === 200, 'Assistant goal update failed');
  page.once('dialog', dialog => dialog.accept());
  const assistantGoalDeletePromise = page.waitForResponse(response => response.url().includes(`/api/assistant/goals/${encodeURIComponent(assistantGoal.id)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.locator('.p-record-section').filter({ hasText: `${assistantGoalTitle} updated` }).getByRole('button', { name: 'Delete', exact: true }).click();
  assert((await assistantGoalDeletePromise).status() === 200, 'Assistant goal delete failed');
  const remainingGoals = await requestJson(fixture.baseUrl, '/api/assistant/goals?project=default', fixture.tokens.assistant);
  assert(remainingGoals.response.ok && !remainingGoals.payload.goals?.some(goal => String(goal.id) === String(assistantGoal.id)), 'Assistant goal deletion did not persist');
  results.push({ workflow: 'Personal Assistant task and goal lifecycle', result: 'PASS', evidence: 'Actual React created, edited, completed and deleted an assistant task, then created, edited and deleted a goal through authenticated assistant:write routes; final GET queries confirmed persistence.' });

  // Personal Context source attribution is restricted to extracted topic
  // signals. Seed one synthetic interaction through the real Python route,
  // then verify React renders its attributed preference signals without
  // exposing the captured interaction text or context.
  const twinSourceBody = {
    project: 'default', kind: 'research',
    content: 'Synthetic provenance materials review from the integration smoke.',
    context: { private_marker: 'must-not-be-rendered' },
  };
  const twinSourceWrite = await requestJson(fixture.baseUrl, '/api/twin/interaction', fixture.tokens.twin, { method: 'POST', body: twinSourceBody });
  assert(twinSourceWrite.response.status === 201, `Personal Context source seed returned HTTP ${twinSourceWrite.response.status}`);
  const twinSourceId = String(twinSourceWrite.payload.id ?? '');
  const twinSourceRead = await requestJson(fixture.baseUrl, '/api/twin/sources?project=default&limit=100', fixture.tokens.twin);
  const attributedSource = twinSourceRead.payload.sources?.find(source => String(source.id) === twinSourceId);
  assert(twinSourceRead.response.ok && attributedSource?.preference_signals?.some(signal => signal.key === 'materials'), 'Python did not return the recorded topic-to-preference signal association');
  await open('twin', fixture.tokens.twin);
  await page.getByRole('tab', { name: 'Sources' }).click();
  const attributedRow = page.locator(`.p-record-section[data-source-id="${twinSourceId}"]`);
  await expect(attributedRow).toContainText('Inferred preference signals:');
  await expect(attributedRow).toContainText('materials');
  await expect(attributedRow).not.toContainText('Synthetic provenance materials review');
  await expect(attributedRow).not.toContainText('must-not-be-rendered');
  results.push({ workflow: 'Personal Context preference-signal attribution', result: 'PASS', evidence: 'An authenticated synthetic interaction persisted in the isolated Python store; the existing source endpoint mapped extracted materials topic signals to the project preference learner, and React displayed that attribution without original content or context.' });

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

  // Security Audit uses its existing admin-only route, explicitly filtered to
  // the local workspace project. Only safe summary fields are rendered.
  await open('security', fixture.tokens.securityAdmin);
  const scopedAuditResponse = page.waitForResponse(response => {
    const url = new URL(response.url());
    return url.pathname === '/api/audit' && url.searchParams.get('project') === 'default';
  }, { timeout: 15_000 });
  await page.getByRole('tab', { name: 'Audit' }).click();
  const scopedAudit = await scopedAuditResponse;
  assert(scopedAudit.status() === 200, `React scoped Security Audit returned HTTP ${scopedAudit.status()}: ${await scopedAudit.text()}`);
  const scopedAuditPayload = await scopedAudit.json();
  assert(Array.isArray(scopedAuditPayload.events) && scopedAuditPayload.events.every(event => event.project === 'default'), 'React Security Audit returned events outside the explicitly selected project');
  await expect(page.locator('[data-audit-event-id="integration-audit-smoke"]')).toContainText('security.audit.smoke');
  await expect(page.getByText('must-not-be-rendered')).toHaveCount(0);
  results.push({ workflow: 'Security project-scoped audit list', result: 'PASS', evidence: 'The actual React Audit tab loaded GET /api/audit?project=default&limit=100 using security:admin, rendered the synthetic same-project event, and did not render its private metadata marker.' });

  await open('security', fixture.tokens.securityRead);
  const deniedAuditResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/audit', { timeout: 15_000 });
  await page.getByRole('tab', { name: 'Audit' }).click();
  const deniedAudit = await deniedAuditResponse;
  assert(deniedAudit.status() === 403, `A security:read token unexpectedly received HTTP ${deniedAudit.status()} from the Security Audit route`);
  await expect(page.getByText('This account does not have permission for this action.')).toBeVisible();
  results.push({ workflow: 'Security audit permission boundary', result: 'PASS', evidence: 'The actual React Audit tab received HTTP 403 with a security:read token and displayed the permission denial; it did not substitute demo records.' });

  // Settings notification preferences use the existing project-scoped
  // integration preferences contract. Verify the UI write survives a fresh
  // React load against the temporary Python database.
  await open('settings', fixture.tokens.integrations);
  await page.getByRole('tab', { name: 'Notifications' }).click();
  const taskCompletionPreference = page.getByRole('checkbox', { name: 'Task completion' });
  await expect(taskCompletionPreference).toBeVisible({ timeout: 15_000 });
  const preferenceWritePromise = page.waitForResponse(response => response.url().endsWith('/api/integrations/preferences') && response.request().method() === 'POST', { timeout: 15_000 });
  await taskCompletionPreference.click();
  const preferenceWrite = await preferenceWritePromise;
  assert(preferenceWrite.status() === 201, `React notification preference was not persisted by Python: HTTP ${preferenceWrite.status()} ${await preferenceWrite.text()}`);
  await expect(taskCompletionPreference).not.toBeChecked();
  const storedPreferences = await requestJson(fixture.baseUrl, '/api/integrations/preferences?project=default&scope=notifications', fixture.tokens.integrations);
  assert(storedPreferences.response.ok && storedPreferences.payload.preferences?.some(preference => preference.key === 'task_completion' && preference.value === false), 'Notification preference did not persist in the backend');
  await open('settings', fixture.tokens.integrations);
  await page.getByRole('tab', { name: 'Notifications' }).click();
  await expect(page.getByRole('tab', { name: 'Notifications' })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByRole('checkbox', { name: 'Task completion' })).toBeVisible({ timeout: 15_000 });
  await expect(page.getByRole('checkbox', { name: 'Task completion' })).not.toBeChecked();
  results.push({ workflow: 'Settings notification preference persistence', result: 'PASS', evidence: 'React loaded the project-scoped preference, saved a change through authenticated POST /api/integrations/preferences, and a fresh React load plus Python GET confirmed the persisted value.' });

  // Models: register metadata through the workspace's actual Live form. This
  // is metadata-only and must not fetch a source URL or create model files.
  await open('models', fixture.tokens.models);
  await page.locator('.models-advanced-tools > summary').click();
  await page.getByRole('tab', { name: 'Marketplace & runtime' }).click();
  const registeredModelName = 'Batch 2A local metadata fixture';
  await page.getByLabel('Model name').fill(registeredModelName);
  const registerModelPromise = page.waitForResponse(response => response.url().endsWith('/api/local-ai/models/register') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Register model', exact: true }).click();
  const registerModelResponse = await registerModelPromise;
  assert(registerModelResponse.status() === 201, `React local model metadata registration returned ${registerModelResponse.status()}: ${await registerModelResponse.text()}`);
  const registeredModel = (await registerModelResponse.json()).model;
  assert(registeredModel?.id && registeredModel.name === registeredModelName, 'Registered model response omitted expected metadata');
  assert(!registeredModel.local_path && !registeredModel.source_url, 'Metadata-only registration unexpectedly configured a model file or external source');
  const modelsAfterRegister = await requestJson(fixture.baseUrl, '/api/local-ai/models', fixture.tokens.models);
  assert(modelsAfterRegister.response.ok && modelsAfterRegister.payload.models?.some(model => String(model.id) === String(registeredModel.id)), 'Local model metadata did not persist after the React form submission');
  assert(!apiRequests.some(request => request.url().includes('/api/local-ai/models/download') || request.url().includes('/api/local-ai/ollama/pull')), 'Metadata-only model registration unexpectedly started a download');
  await page.getByRole('tab', { name: 'Setup activity' }).click();
  await expect(page.getByRole('heading', { name: 'No benchmark history' })).toBeVisible({ timeout: 15_000 });
  const benchmarkHistoryRead = apiRequests.find(request => request.url().includes('/api/local-ai/benchmarks?limit=50') && request.method() === 'GET');
  assert(benchmarkHistoryRead, 'React did not load benchmark history from the authenticated Python API');
  const benchmarkHistoryResponse = await requestJson(fixture.baseUrl, '/api/local-ai/benchmarks?limit=50', fixture.tokens.readonly);
  assert(benchmarkHistoryResponse.response.ok && Array.isArray(benchmarkHistoryResponse.payload.benchmarks), 'Read-only local_ai:read credential could not read the benchmark history contract');
  results.push({ workflow: 'Models metadata and benchmark history', result: 'PASS', evidence: 'React registered metadata and read the truthful empty benchmark history from Python; a least-privilege local_ai:read credential also read the history route. No benchmark was run, no model downloaded, and no provider pull was requested.' });

  // Automations: exercise definition CRUD only. The workflow contains an
  // add_event step but is never run, so no event or external action is created.
  await open('automations', fixture.tokens.workflows);
  const automationName = 'Batch 2A disposable workflow';
  await page.getByRole('button', { name: 'New', exact: true }).click();
  await page.getByLabel('Name', { exact: true }).fill(automationName);
  await page.getByLabel('New step 1 Title').fill('Synthetic event title');
  await page.getByLabel('New step 1 Content').fill('Definition-only fixture; this workflow will not be run.');
  const workflowCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/workflows') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Save automation' }).click();
  const workflowCreate = await workflowCreatePromise;
  assert(workflowCreate.status() === 201, `React automation creation returned ${workflowCreate.status()}: ${await workflowCreate.text()}`);
  const createdWorkflow = (await workflowCreate.json()).workflow;
  assert(createdWorkflow?.id && createdWorkflow.name === automationName, 'Automation create response omitted the saved workflow');
  const workflowPreviewPromise = page.waitForResponse(response => response.url().endsWith(`/api/workflows/${createdWorkflow.id}/run`) && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Preview run' }).click();
  const workflowPreviewResponse = await workflowPreviewPromise;
  assert(workflowPreviewResponse.status() === 201, `React automation preview returned ${workflowPreviewResponse.status()}: ${await workflowPreviewResponse.text()}`);
  const workflowPreview = (await workflowPreviewResponse.json()).run;
  assert(workflowPreview?.dry_run === true || workflowPreview?.result?.dry_run === true, 'Automation preview response did not identify itself as a dry run');
  const workflowRunHistory = await requestJson(fixture.baseUrl, '/api/workflow-runs?project=default&limit=100', fixture.tokens.workflows);
  assert(workflowRunHistory.response.ok && workflowRunHistory.payload.runs?.some(run => String(run.id) === String(workflowPreview.id) && (run.dry_run === true || run.result?.dry_run === true)), 'Automation dry-run was not recorded as a preview run');
  const workflowEditName = `${automationName} edited`;
  await page.getByRole('tab', { name: 'Configuration' }).click();
  await page.getByLabel('Name', { exact: true }).fill(workflowEditName);
  const workflowUpdatePromise = page.waitForResponse(response => response.url().endsWith(`/api/workflows/${createdWorkflow.id}`) && response.request().method() === 'PUT', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Save configuration' }).click();
  const workflowUpdate = await workflowUpdatePromise;
  assert(workflowUpdate.status() === 200, `React automation update returned ${workflowUpdate.status()}: ${await workflowUpdate.text()}`);
  const updatedWorkflow = (await workflowUpdate.json()).workflow;
  assert(updatedWorkflow?.name === workflowEditName, 'Automation update response did not include the revised name');
  const workflowsAfterUpdate = await requestJson(fixture.baseUrl, '/api/workflows?project=default&limit=100', fixture.tokens.workflows);
  assert(workflowsAfterUpdate.response.ok && workflowsAfterUpdate.payload.workflows?.some(workflow => String(workflow.id) === String(createdWorkflow.id) && workflow.name === workflowEditName), 'Automation update did not persist in the backend');
  await page.getByRole('button', { name: 'Automation options' }).click();
  page.once('dialog', dialog => dialog.accept());
  const workflowDeletePromise = page.waitForResponse(response => response.url().endsWith(`/api/workflows/${createdWorkflow.id}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Delete automation' }).click();
  const workflowDelete = await workflowDeletePromise;
  assert(workflowDelete.ok(), `React automation delete returned ${workflowDelete.status()}: ${await workflowDelete.text()}`);
  const workflowsAfterDelete = await requestJson(fixture.baseUrl, '/api/workflows?project=default&limit=100', fixture.tokens.workflows);
  assert(workflowsAfterDelete.response.ok && !workflowsAfterDelete.payload.workflows?.some(workflow => String(workflow.id) === String(createdWorkflow.id)), 'Automation delete did not persist in the backend');
  const executedWorkflowRequests = apiRequests.filter(request => request.url().includes(`/api/workflows/${createdWorkflow.id}/run`));
  assert(executedWorkflowRequests.length === 1, `Expected exactly one explicit dry-run preview, got ${executedWorkflowRequests.length} workflow run requests`);
  results.push({ workflow: 'Automations definition and preview lifecycle', result: 'PASS', evidence: 'React created, previewed in backend dry-run mode, edited and deleted a disposable project workflow through authenticated APIs; Python reads confirmed definition and preview persistence, and no live workflow execution was requested.' });

  // Web Scraper: save a recurring job without approving outbound access.
  // Scheduling validates locally; it must not resolve DNS or start a fetch.
  await open('scraper', fixture.tokens.scraper);
  await page.getByRole('tab', { name: 'Jobs', exact: true }).click();
  const scraperJobLabel = 'Disposable offline scheduling smoke';
  await page.getByLabel('Website URL').fill('https://example.org/offline-scheduling-smoke');
  await page.getByLabel('Job label').fill(scraperJobLabel);
  await page.getByLabel('Repeat interval (hours)').fill('24');
  await expect(page.getByRole('checkbox', { name: /I approve recurring external network requests/ })).not.toBeChecked();
  page.once('dialog', dialog => dialog.accept());
  const scraperSchedulePromise = page.waitForResponse(response => response.url().endsWith('/api/scraper/schedule') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Schedule extraction' }).click();
  const scraperSchedule = await scraperSchedulePromise;
  assert(scraperSchedule.status() === 201, `React offline scraper scheduling returned ${scraperSchedule.status()}: ${await scraperSchedule.text()}`);
  const scheduledScraperJob = (await scraperSchedule.json()).job;
  assert(scheduledScraperJob?.job_id && scheduledScraperJob.status === 'awaiting_approval', 'Scraper schedule response omitted the job identity or incorrectly implied outbound approval');
  const scheduledScraperJobs = await requestJson(fixture.baseUrl, '/api/scraper/jobs?project=default', fixture.tokens.scraper);
  const persistedScraperJob = scheduledScraperJobs.payload.jobs?.find(job => String(job.id) === String(scheduledScraperJob.job_id));
  assert(scheduledScraperJobs.response.ok && persistedScraperJob?.metadata?.label === scraperJobLabel && persistedScraperJob.status === 'awaiting_approval', 'Scheduled scraper job was not persisted in the isolated Python database with its label and approval state');
  assert(!apiRequests.some(request => request.url().includes('/api/scraper/scrape')), 'Scheduling a scraper job unexpectedly started a fetch');
  await page.getByRole('tab', { name: 'Jobs' }).click();
  const cancelScraperJob = page.getByRole('button', { name: `Cancel scheduled extraction for ${scraperJobLabel}` });
  await expect(cancelScraperJob).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  const scraperCancelPromise = page.waitForResponse(response => response.url().endsWith(`/api/scraper/jobs/${scheduledScraperJob.job_id}/cancel`) && response.request().method() === 'POST', { timeout: 15_000 });
  await cancelScraperJob.click();
  const scraperCancel = await scraperCancelPromise;
  assert(scraperCancel.status() === 200, `React scraper cancellation returned ${scraperCancel.status()}: ${await scraperCancel.text()}`);
  assert((await scraperCancel.json()).cancelled === true, 'Scraper cancellation response did not confirm a pending job cancellation');
  const scraperCancellationMessage = page.getByRole('status').getByText('Schedule cancelled. The saved job history has been retained.');
  await expect(scraperCancellationMessage).toBeVisible();
  const scraperJobsAfterCancel = await requestJson(fixture.baseUrl, '/api/scraper/jobs?project=default', fixture.tokens.scraper);
  const retainedCancelledScraperJob = scraperJobsAfterCancel.payload.jobs?.find(job => String(job.id) === String(scheduledScraperJob.job_id));
  assert(scraperJobsAfterCancel.response.ok && retainedCancelledScraperJob?.status === 'cancelled', 'Scraper cancellation was not persisted while retaining the job history row');
  assert(!apiRequests.some(request => request.url().includes('/api/scraper/scrape')), 'Schedule creation or cancellation unexpectedly started a scraper fetch');
  results.push({ workflow: 'Web Scraper schedule, cancellation and history', result: 'PASS', evidence: 'React created a disposable schedule with external access approval unchecked, Python confirmed its awaiting-approval state, React cancelled it through the authenticated project-scoped route, and a fresh Python GET returned the retained row with status cancelled. No fetch route was called.' });

  // Files: import a deterministic text file through the actual React file
  // picker, then verify the authenticated document response and local download.
  await open('files', fixture.tokens.documents);
  await page.getByRole('button', { name: 'Document filters and actions' }).click();
  const requestsBeforeLegacyOffice = apiRequests.length;
  await page.getByLabel('Choose document to import').setInputFiles({
    name: 'unsupported-legacy.doc', mimeType: 'application/msword', buffer: Buffer.from([0, 1, 255, 0]),
  });
  await expect(page.getByRole('alert').getByText(/Legacy Office files cannot be safely extracted/)).toBeVisible();
  assert(!apiRequests.slice(requestsBeforeLegacyOffice).some(request => /\/api\/documents\/import(?:\?|$)/.test(new URL(request.url()).pathname + new URL(request.url()).search)), 'React sent an unsupported legacy Office binary to the document importer');
  results.push({ workflow: 'Files unsupported legacy Office input', result: 'PASS', evidence: 'React rejected a synthetic binary .doc with a clear conversion message and made no document-import request; the Python processing service separately rejects legacy .doc/.ppt/.xls before creating a record.' });
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
  const deniedMediaDelete = await requestJson(fixture.baseUrl, `/api/media/assets/${encodeURIComponent(uploadedAsset.id)}?project=default`, fixture.tokens.readonly, { method: 'DELETE' });
  assert(deniedMediaDelete.response.status === 403, `media:read token deleted stored media (HTTP ${deniedMediaDelete.response.status})`);
  await page.once('dialog', dialog => dialog.accept());
  const mediaDeletePromise = page.waitForResponse(response => response.url().includes(`/api/media/assets/${encodeURIComponent(uploadedAsset.id)}?project=default`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('button', { name: `Delete ${uploadedAsset.title}` }).click();
  const mediaDeleteResponse = await mediaDeletePromise;
  assert(mediaDeleteResponse.status() === 200, `React stored media deletion returned ${mediaDeleteResponse.status()}: ${await mediaDeleteResponse.text()}`);
  const mediaDeleteResult = await mediaDeleteResponse.json();
  assert(mediaDeleteResult.deleted === true && mediaDeleteResult.stored_bytes_removed === true, 'Media delete did not confirm removal of its stored copy');
  const mediaAssetsAfterDelete = await requestJson(fixture.baseUrl, '/api/media/assets?project=default&limit=100', fixture.tokens.documents);
  assert(mediaAssetsAfterDelete.response.ok && !mediaAssetsAfterDelete.payload.assets?.some(asset => String(asset.id) === String(uploadedAsset.id)), 'Deleted media remains in the persisted Files list');
  let uploadedBytesRemain = false;
  try { await access(uploadedAsset.stored_path); uploadedBytesRemain = true; } catch { /* deleted in the isolated media store */ }
  assert(!uploadedBytesRemain, 'Deleted media bytes remain in the temporary data directory');
  results.push({ workflow: 'Files', result: 'PASS', evidence: `React imported and downloaded a deterministic Markdown document, uploaded a synthetic media file, and deleted its stored copy through project-scoped DELETE. A media:read-only credential received 403; the deleted asset and stored bytes were absent on fresh Python reads. Foreign installation document/media reads were denied (document HTTP ${crossOwnerDocument.response.status}, media HTTP ${crossOwnerMedia.response.status}).` });

  // Workspace Notes are separate from the document library. Create one via
  // the actual React form, then confirm and delete it through its project-scoped
  // lifecycle route; no user documents outside this disposable fixture are used.
  const workspaceNoteTitle = 'Batch 2A disposable workspace note';
  const documentsTokenWorkspaceRead = await requestJson(fixture.baseUrl, '/api/workspace/items?project=default&kind=note&limit=100', fixture.tokens.documents);
  assert([401, 403].includes(documentsTokenWorkspaceRead.response.status), `Document-only token unexpectedly read workspace items (HTTP ${documentsTokenWorkspaceRead.response.status})`);
  await open('files', fixture.tokens.workspace);
  await page.getByRole('tab', { name: 'Notes', exact: true }).click();
  await page.getByRole('button', { name: 'Add note', exact: true }).click();
  const workspaceNoteDialog = page.getByRole('dialog', { name: 'New workspace note' });
  await workspaceNoteDialog.getByLabel('Note title', { exact: true }).fill(workspaceNoteTitle);
  await workspaceNoteDialog.getByLabel('Note', { exact: true }).fill('Disposable note used to verify authenticated workspace item deletion.');
  const workspaceCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/workspace/items') && response.request().method() === 'POST', { timeout: 15_000 });
  await workspaceNoteDialog.getByRole('button', { name: 'Create note', exact: true }).click();
  const workspaceCreate = await workspaceCreatePromise;
  assert(workspaceCreate.status() === 201, `Workspace note create returned ${workspaceCreate.status()}`);
  const workspaceItem = (await workspaceCreate.json()).item;
  assert(workspaceItem?.id, 'Workspace note create omitted the item id');
  await expect(page.getByRole('button', { name: new RegExp(workspaceNoteTitle) })).toBeVisible();
  const updatedWorkspaceTitle = `${workspaceNoteTitle} edited`;
  await page.getByLabel('Note title', { exact: true }).fill(updatedWorkspaceTitle);
  await page.getByLabel('Note content', { exact: true }).fill('Updated disposable note content for project-scope and persistence verification.');
  const workspaceUpdatePromise = page.waitForResponse(response => response.url().endsWith('/api/workspace/items/update') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Save note', exact: true }).click();
  const workspaceUpdate = await workspaceUpdatePromise;
  assert(workspaceUpdate.status() === 200, `Workspace note update returned ${workspaceUpdate.status()}`);
  const workspaceUpdatedItem = (await workspaceUpdate.json()).item;
  assert(String(workspaceUpdatedItem?.title) === updatedWorkspaceTitle && workspaceUpdatedItem?.project === 'default', 'Workspace note update did not persist under the requested project');
  const updatedWorkspaceList = await requestJson(fixture.baseUrl, '/api/workspace/items?project=default&kind=note&limit=100', fixture.tokens.workspace);
  assert(updatedWorkspaceList.response.ok && updatedWorkspaceList.payload.items?.some(item => String(item.id) === String(workspaceItem.id) && String(item.title) === updatedWorkspaceTitle), 'Updated workspace note was not persisted in Python project history');
  await expect(page.getByRole('button', { name: new RegExp(updatedWorkspaceTitle) })).toBeVisible();
  page.once('dialog', dialog => dialog.accept());
  const workspaceDeletePromise = page.waitForResponse(response => response.url().includes(`/api/workspace/items/${encodeURIComponent(workspaceItem.id)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Delete note', exact: true }).click();
  const workspaceDelete = await workspaceDeletePromise;
  assert(workspaceDelete.status() === 200, `Workspace note delete returned ${workspaceDelete.status()}`);
  const remainingWorkspaceNotes = await requestJson(fixture.baseUrl, '/api/workspace/items?project=default&kind=note&limit=100', fixture.tokens.workspace);
  assert(remainingWorkspaceNotes.response.ok && !remainingWorkspaceNotes.payload.items?.some(item => String(item.id) === String(workspaceItem.id)), 'Deleted workspace note remained in the project item list');
  results.push({ workflow: 'Files workspace notes', result: 'PASS', evidence: 'Actual React created a disposable note, updated its title/content with the explicit project in POST /api/workspace/items/update, verified the returned record and a fresh Python list response, then confirmed deletion through the project-scoped DELETE route and verified the row was absent.' });

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
  const indexedKnowledge = persistedKnowledge.payload.documents?.find(document => document.title === knowledgeTitle);
  assert(persistedKnowledge.response.ok && indexedKnowledge, 'Knowledge source was not persisted in the project index');
  const knowledgeDetail = await requestJson(fixture.baseUrl, `/api/knowledge/documents/${encodeURIComponent(indexedKnowledge.id)}?project=default`, fixture.tokens.knowledge);
  assert(knowledgeDetail.response.ok && knowledgeDetail.payload.document?.content.includes('Local fixture evidence only.'), 'Knowledge detail route did not return the persisted source content');
  const knowledgeUpdatedTitle = 'Batch 2A revised synthetic knowledge source';
  await page.getByRole('tab', { name: 'Ingestion', exact: true }).click();
  await page.getByRole('button', { name: `Open ${knowledgeTitle}`, exact: true }).click();
  await page.getByRole('dialog').getByText('Local fixture evidence only.', { exact: false }).waitFor({ state: 'visible', timeout: 15_000 });
  await page.getByRole('button', { name: 'Edit source', exact: true }).click();
  const knowledgeEditDialog = page.getByRole('dialog', { name: 'Edit knowledge source' });
  await knowledgeEditDialog.getByRole('textbox', { name: 'Knowledge source title' }).fill(knowledgeUpdatedTitle);
  await knowledgeEditDialog.getByRole('textbox', { name: 'Knowledge source text' }).fill('Updated local fixture evidence only. No external research or network provider is used.');
  const knowledgeUpdatePromise = page.waitForResponse(response => response.url().endsWith(`/api/knowledge/documents/${encodeURIComponent(indexedKnowledge.id)}`) && response.request().method() === 'PUT', { timeout: 15_000 });
  await knowledgeEditDialog.getByRole('button', { name: 'Save source', exact: true }).click();
  const knowledgeUpdate = await knowledgeUpdatePromise;
  assert(knowledgeUpdate.status() === 200, `Knowledge source update returned ${knowledgeUpdate.status()}`);
  const updatedKnowledgeDetail = await requestJson(fixture.baseUrl, `/api/knowledge/documents/${encodeURIComponent(indexedKnowledge.id)}?project=default`, fixture.tokens.knowledge);
  assert(updatedKnowledgeDetail.response.ok && updatedKnowledgeDetail.payload.document?.title === knowledgeUpdatedTitle && updatedKnowledgeDetail.payload.document?.content.includes('Updated local fixture evidence only.'), 'Knowledge update did not preserve the source ID and persist its revised title/content');
  await page.getByRole('dialog', { name: knowledgeUpdatedTitle }).getByRole('button', { name: 'Close source details' }).click();
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
  const richTextBuffer = Buffer.from('{\\rtf1\\ansi Local rich text extraction\\par persisted from RTF upload.}');
  const rtfUploadPromise = page.waitForResponse(response => response.url().endsWith('/api/knowledge/upload') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByLabel('Upload knowledge document').setInputFiles({ name: 'batch2a-knowledge-upload.rtf', mimeType: 'application/rtf', buffer: richTextBuffer });
  const rtfUpload = await rtfUploadPromise;
  const rtfUploadPayload = await rtfUpload.json();
  assert(rtfUpload.status() === 201 && String(rtfUploadPayload.attachment?.extracted_text ?? '').includes('Local rich text extraction'), `Knowledge RTF upload did not extract readable text: ${JSON.stringify(rtfUploadPayload)}`);
  const crossOwnerKnowledge = await requestJson(fixture.baseUrl, '/api/knowledge/documents?project=default&limit=100', fixture.tokens.otherKnowledgeRead);
  if (crossOwnerKnowledge.response.ok && crossOwnerKnowledge.payload.documents?.some(document => document.title === knowledgeTitle)) {
    results.push({ workflow: 'Knowledge installation isolation', result: 'BLOCKED', evidence: 'A foreign installation credential listed owner Knowledge data (HTTP 200); personal_remote isolation failed.' });
  } else {
    assert([401, 403].includes(crossOwnerKnowledge.response.status), `Foreign installation Knowledge list returned unexpected ${crossOwnerKnowledge.response.status}`);
    results.push({ workflow: 'Knowledge installation isolation', result: 'PASS', evidence: `A credential issued by a separate installation could not list owner Knowledge data (HTTP ${crossOwnerKnowledge.response.status}).` });
  }
  await page.getByRole('tab', { name: 'Ingestion', exact: true }).click();
  await page.getByRole('button', { name: `Open ${knowledgeUpdatedTitle}`, exact: true }).click();
  await page.getByRole('dialog').getByText('Updated local fixture evidence only.', { exact: false }).waitFor({ state: 'visible', timeout: 15_000 });
  page.once('dialog', dialog => dialog.accept());
  const knowledgeDeletePromise = page.waitForResponse(response => response.url().includes(`/api/knowledge/documents/${encodeURIComponent(indexedKnowledge.id)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Delete source', exact: true }).click();
  const knowledgeDelete = await knowledgeDeletePromise;
  assert(knowledgeDelete.status() === 200, `Knowledge source delete returned ${knowledgeDelete.status()}`);
  const afterKnowledgeDelete = await requestJson(fixture.baseUrl, '/api/knowledge/documents?project=default&limit=100', fixture.tokens.knowledge);
  assert(afterKnowledgeDelete.response.ok && !afterKnowledgeDelete.payload.documents?.some(document => String(document.id) === String(indexedKnowledge.id)), 'Deleted Knowledge source remained in project index');
  results.push({ workflow: 'Knowledge', result: 'PASS', evidence: 'React indexed a deterministic source, opened its persisted detail, updated its title/content through project-scoped PUT while retaining the same ID, confirmed the revised content through a fresh Python detail request, uploaded local text and RTF fixtures with extracted text verified in actual responses, then deleted the edited source through the Live UI and confirmed it disappeared from the project index.' });

  // Communications: seed synthetic inbox rows, create a reply draft, archive
  // one message, and move another to trash. No mail is sent or provider contacted.
  const importedMessageResponse = await requestJson(fixture.baseUrl, '/api/communication/messages/import', fixture.tokens.communication, {
    method: 'POST', body: {
      platform: 'email', sender: 'batch2a@example.test', recipients: ['local-user@example.test'],
      subject: 'Batch 2A synthetic inbox message', body: 'Local-only message fixture for draft and archive checks.', project: 'default',
    },
  });
  assert(importedMessageResponse.response.status === 201, `Synthetic inbox fixture import returned ${importedMessageResponse.response.status}`);
  const importedMessage = importedMessageResponse.payload.message;
  const trashMessageResponse = await requestJson(fixture.baseUrl, '/api/communication/messages/import', fixture.tokens.communication, {
    method: 'POST', body: {
      platform: 'email', sender: 'batch2a@example.test', recipients: ['local-user@example.test'],
      subject: 'Batch 2A synthetic trash message', body: 'Local-only message fixture for reversible trash checks.', project: 'default',
    },
  });
  assert(trashMessageResponse.response.status === 201, `Synthetic trash fixture import returned ${trashMessageResponse.response.status}`);
  const trashMessage = trashMessageResponse.payload.message;
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
  const copiedDraftPromise = page.waitForResponse(response => response.url().endsWith('/api/communication/drafts') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Copy to drafts', exact: true }).click();
  const copiedDraftResponse = await copiedDraftPromise;
  assert(copiedDraftResponse.status() === 201, `React inbox draft copy returned ${copiedDraftResponse.status()}: ${await copiedDraftResponse.text()}`);
  const copiedDraft = (await copiedDraftResponse.json()).draft;
  assert(copiedDraft?.id && String(copiedDraft.message_id) === String(importedMessage.id), 'Inbox copy draft did not retain its source-message reference');
  assert(copiedDraft.subject === importedMessage.subject && copiedDraft.body === importedMessage.body, 'Inbox copy draft did not preserve the message subject and body');
  const draftsAfterCopy = await requestJson(fixture.baseUrl, '/api/communication/drafts?project=default&limit=100', fixture.tokens.communication);
  assert(draftsAfterCopy.response.ok && draftsAfterCopy.payload.drafts?.some(draft => String(draft.id) === String(copiedDraft.id)), 'Copied inbox draft did not persist');
  await page.getByRole('button', { name: 'Edit message', exact: true }).click();
  const editMessageDialog = page.getByRole('dialog', { name: 'Edit inbox message' });
  await editMessageDialog.getByLabel('Subject').fill('Batch 2A synthetic inbox message · revised');
  await editMessageDialog.getByLabel('Sender').fill('batch2a-updated@example.test');
  await editMessageDialog.getByRole('textbox', { name: 'Message' }).fill('Local-only revised message fixture for draft and archive checks.');
  const editMessagePromise = page.waitForResponse(response => response.url().endsWith('/api/communication/messages/update') && response.request().method() === 'POST', { timeout: 15_000 });
  await editMessageDialog.getByRole('button', { name: 'Save changes' }).click();
  const editMessageResponse = await editMessagePromise;
  assert(editMessageResponse.status() === 200, `React inbox edit returned ${editMessageResponse.status()}: ${await editMessageResponse.text()}`);
  const editedInboxMessage = (await editMessageResponse.json()).message;
  assert(editedInboxMessage?.subject === 'Batch 2A synthetic inbox message · revised' && editedInboxMessage.body === 'Local-only revised message fixture for draft and archive checks.', 'React inbox edit response did not return the updated record');
  const inboxAfterEdit = await requestJson(fixture.baseUrl, '/api/communication/inbox?project=default', fixture.tokens.communication);
  assert(inboxAfterEdit.response.ok && inboxAfterEdit.payload.messages?.some(message => String(message.id) === String(importedMessage.id) && message.subject === editedInboxMessage.subject && message.sender === editedInboxMessage.sender && message.body === editedInboxMessage.body), 'Communication message edit did not persist');
  const archiveMessagePromise = page.waitForResponse(response => response.url().endsWith('/api/communication/messages/state') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Archive', exact: true }).click();
  const archiveMessage = await archiveMessagePromise;
  assert(archiveMessage.ok(), `React message archive returned ${archiveMessage.status()}`);
  const inboxAfterArchive = await requestJson(fixture.baseUrl, '/api/communication/inbox?project=default', fixture.tokens.communication);
  assert(inboxAfterArchive.response.ok && inboxAfterArchive.payload.messages?.some(message => String(message.id) === String(importedMessage.id) && (message.archived === true || Number(message.archived) === 1 || message.folder === 'archive')), 'Communication archive state was not persisted');
  await page.getByRole('button', { name: /Batch 2A synthetic trash message/ }).click();
  const trashMessagePromise = page.waitForResponse(response => response.url().endsWith('/api/communication/messages/state') && response.request().method() === 'POST', { timeout: 15_000 });
  page.once('dialog', dialog => dialog.accept());
  await page.getByRole('button', { name: 'Move to trash', exact: true }).click();
  const trashResponse = await trashMessagePromise;
  assert(trashResponse.ok(), `React message move-to-trash returned ${trashResponse.status()}`);
  const inboxAfterTrash = await requestJson(fixture.baseUrl, '/api/communication/inbox?project=default', fixture.tokens.communication);
  assert(inboxAfterTrash.response.ok && inboxAfterTrash.payload.messages?.some(message => String(message.id) === String(trashMessage.id) && message.folder === 'trash'), 'Communication trash state was not persisted');
  const restoreMessagePromise = page.waitForResponse(response => response.url().endsWith('/api/communication/messages/state') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Undo', exact: true }).click();
  const restoreMessage = await restoreMessagePromise;
  assert(restoreMessage.ok(), `React message restore returned ${restoreMessage.status()}`);
  const inboxAfterRestore = await requestJson(fixture.baseUrl, '/api/communication/inbox?project=default', fixture.tokens.communication);
  assert(inboxAfterRestore.response.ok && inboxAfterRestore.payload.messages?.some(message => String(message.id) === String(trashMessage.id) && message.folder === 'inbox'), 'Restored message was not persisted in the inbox');
  assert(!apiRequests.some(request => request.url().includes('/api/communication/send')), 'The local communications smoke unexpectedly invoked a send operation');
  results.push({ workflow: 'Communications', result: 'PASS', evidence: 'The actual React Inbox created reply and source-linked copy drafts, edited the synthetic message and refreshed its contextual text, archived it, moved a second message to trash, restored it through the owner-scoped mailbox-state API, verified both folder transitions persisted, and issued no send request.' });

  // Contacts CRM uses the standalone /api/contacts contract. Exercise the UI
  // field adapters against the backend's handles object and verify each write
  // through a fresh Python list/export response.
  await open('communications', fixture.tokens.communication);
  await page.getByRole('tab', { name: 'Contacts' }).click();
  await page.getByText('Add CRM contact', { exact: true }).click();
  await page.getByLabel('Name').fill('Batch 2 CRM smoke contact');
  await page.getByLabel('Email').fill('crm-smoke@example.test');
  await page.getByLabel('Notes').fill('Disposable local CRM lifecycle record');
  const crmCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/contacts') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Save contact' }).click();
  const crmCreateResponse = await crmCreatePromise;
  assert(crmCreateResponse.status() === 201, `React CRM contact creation returned HTTP ${crmCreateResponse.status()}`);
  const createdCrmContact = (await crmCreateResponse.json()).contact;
  assert(createdCrmContact?.id && createdCrmContact.handles?.email === 'crm-smoke@example.test', 'CRM create did not persist email under handles.email');
  let crmRead = await requestJson(fixture.baseUrl, '/api/contacts?project=default', fixture.tokens.communication);
  assert(crmRead.response.ok && crmRead.payload.contacts?.some(contact => String(contact.id) === String(createdCrmContact.id)), 'Fresh CRM read did not return the new contact');
  const crmRow = page.locator('.p-record-section').filter({ hasText: 'Batch 2 CRM smoke contact' });
  await crmRow.getByRole('button', { name: 'Edit' }).click();
  await page.getByPlaceholder('Email').fill('crm-smoke-updated@example.test');
  // Editing replaces the contact heading with inputs, so the original
  // name-based row locator no longer matches after the edit transition.
  const editingCrmRow = page.locator('.p-record-section').filter({ has: page.getByPlaceholder('Email') });
  const crmPatchPromise = page.waitForResponse(response => response.url().includes(`/api/contacts/${encodeURIComponent(createdCrmContact.id)}`) && response.request().method() === 'PATCH', { timeout: 15_000 });
  await editingCrmRow.getByRole('button', { name: 'Save', exact: true }).click();
  const crmPatchResponse = await crmPatchPromise;
  assert(crmPatchResponse.ok(), `React CRM contact update returned HTTP ${crmPatchResponse.status()}`);
  crmRead = await requestJson(fixture.baseUrl, '/api/contacts?project=default', fixture.tokens.communication);
  assert(crmRead.response.ok && crmRead.payload.contacts?.some(contact => String(contact.id) === String(createdCrmContact.id) && contact.handles?.email === 'crm-smoke-updated@example.test'), 'Fresh CRM read did not return the updated handle');
  const crmExportPromise = page.waitForResponse(response => response.url().includes('/api/contacts/export') && response.request().method() === 'GET', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Export CRM' }).click();
  const crmExportResponse = await crmExportPromise;
  const crmExport = await crmExportResponse.json();
  assert(crmExportResponse.ok() && crmExport.contacts?.some(contact => String(contact.id) === String(createdCrmContact.id)), 'CRM export omitted the persisted contact');
  page.once('dialog', dialog => dialog.accept());
  const crmDeletePromise = page.waitForResponse(response => response.url().includes(`/api/contacts/${encodeURIComponent(createdCrmContact.id)}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await crmRow.getByRole('button', { name: 'Delete' }).click();
  assert((await crmDeletePromise).ok(), 'React CRM contact deletion failed');
  crmRead = await requestJson(fixture.baseUrl, '/api/contacts?project=default', fixture.tokens.communication);
  assert(crmRead.response.ok && !crmRead.payload.contacts?.some(contact => String(contact.id) === String(createdCrmContact.id)), 'CRM deletion did not persist');
  results.push({ workflow: 'Contacts CRM', result: 'PASS', evidence: 'React created a disposable contact with handles.email, edited its email while preserving the backend handles shape, exported the persisted contact, deleted it after confirmation and verified each state through fresh Python reads.' });

  // Voice: create a real local transcription session from React, then upload a
  // short synthetic WAV through the existing STT contract. The fixture is
  // silent and does not claim that speech recognition produced a transcript.
  await open('voice', fixture.tokens.voice);
  const voiceSessionPromise = page.waitForResponse(response => response.url().includes('/api/voice/sessions') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'New session', exact: true }).click();
  const voiceSessionResponse = await voiceSessionPromise;
  assert(voiceSessionResponse.status() === 201, `React Voice session creation returned ${voiceSessionResponse.status()}`);
  const voiceSession = (await voiceSessionResponse.json()).session;
  assert(voiceSession?.id && voiceSession.status === 'active', 'Voice session was not returned as an active persisted session');
  await page.getByRole('tab', { name: 'Transcripts' }).click();
  await page.locator('input[aria-label="Choose audio file"]').setInputFiles({
    name: 'openagent-smoke-silence.wav', mimeType: 'audio/wav', buffer: syntheticSilenceWav(),
  });
  const voiceSttPromise = page.waitForResponse(response => response.url().endsWith('/api/voice/stt') && response.request().method() === 'POST', { timeout: 30_000 });
  await page.getByRole('button', { name: 'Upload and transcribe' }).click();
  const voiceSttResponse = await voiceSttPromise;
  assert(voiceSttResponse.status() === 201, `React Voice upload returned ${voiceSttResponse.status()}: ${await voiceSttResponse.text()}`);
  const voiceStt = (await voiceSttResponse.json()).transcription;
  assert(voiceStt?.session_id === voiceSession.id && voiceStt.turn?.id && voiceStt.audio_asset?.id, 'Voice STT response omitted its session, persisted turn or local audio asset');
  assert(typeof voiceStt.display_text === 'string' && voiceStt.display_text.length > 0, 'Voice STT did not provide its truthful status or transcript text');
  const persistedVoiceTurns = await requestJson(fixture.baseUrl, `/api/voice/turns?project=default&session_id=${encodeURIComponent(voiceSession.id)}`, fixture.tokens.voice);
  assert(persistedVoiceTurns.response.ok && persistedVoiceTurns.payload.turns?.some(turn => String(turn.id) === String(voiceStt.turn.id) && String(turn.audio_asset_id) === String(voiceStt.audio_asset.id)), 'Voice audio/transcript record did not persist in the session history');
  results.push({ workflow: 'Voice local session and buffered audio', result: 'PASS', evidence: `React created session ${voiceSession.id}, uploaded a generated 0.1-second silent WAV to the existing local STT route, and verified persisted turn ${voiceStt.turn.id} and audio asset ${voiceStt.audio_asset.id}. This validates transport and persistence only; it does not claim speech recognition.` });
  await page.getByRole('tab', { name: 'Voice settings' }).click();
  await page.getByRole('textbox', { name: 'Text to synthesize' }).fill('Local playback integration fixture.');
  const voicePlaybackPath = `/api/voice/assets/`;
  const audioPlaybackPromise = page.waitForResponse(response => new URL(response.url()).pathname.startsWith(voicePlaybackPath) && new URL(response.url()).pathname.endsWith('/content') && response.request().method() === 'GET', { timeout: 15_000 });
  const ttsCreatePromise = page.waitForResponse(response => response.url().endsWith('/api/voice/tts') && response.request().method() === 'POST', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Generate local speech artifact' }).click();
  const ttsResponse = await ttsCreatePromise;
  assert(ttsResponse.status() === 201, `React local TTS returned ${ttsResponse.status()}: ${await ttsResponse.text()}`);
  const ttsArtifact = (await ttsResponse.json()).script;
  assert(ttsArtifact?.audio_asset_id, 'Local TTS response did not identify its stored audio asset');
  const voiceAudioUrl = `/api/voice/assets/${encodeURIComponent(ttsArtifact.audio_asset_id)}/content?project=default`;
  assert((await requestJson(fixture.baseUrl, voiceAudioUrl, '')).response.status === 401, 'Voice audio content was readable without authentication');
  assert((await requestJson(fixture.baseUrl, voiceAudioUrl, fixture.tokens.neitherAgentScope)).response.status === 403, 'Voice audio content was readable without the voice read scope');
  assert((await requestJson(fixture.baseUrl, `/api/voice/assets/${encodeURIComponent(ttsArtifact.audio_asset_id)}/content?project=other`, fixture.tokens.voice)).response.status === 404, 'Voice audio content crossed the requested project scope');
  const directAudio = await requestBytes(fixture.baseUrl, voiceAudioUrl, fixture.tokens.voice);
  assert(directAudio.status === 200 && directAudio.body.subarray(0, 4).toString('ascii') === 'RIFF', `The local HTTP route did not return WAV bytes (status=${directAudio.status}, bytes=${directAudio.body.length}, contentLength=${directAudio.headers['content-length']})`);
  const audioPlayback = await audioPlaybackPromise;
  assert(audioPlayback.status() === 200 && audioPlayback.headers()['content-type']?.startsWith('audio/wav'), `Authenticated Voice asset read returned ${audioPlayback.status()} ${audioPlayback.headers()['content-type']}`);
  assert(String(audioPlayback.request().headers()['authorization'] ?? '').startsWith('Bearer '), 'Voice asset playback request omitted its authenticated token');
  await expect(page.getByLabel('Generated local voice audio')).toHaveAttribute('data-audio-bytes', /^(?:[5-9]\d|\d{3,})$/, { timeout: 15_000 });
  assert(Number(await page.getByLabel('Generated local voice audio').getAttribute('data-audio-bytes')) === directAudio.body.length, 'The React playback blob size did not match the bytes returned by the Python route');
  await expect(page.getByLabel('Generated local voice audio')).toHaveAttribute('src', /^blob:/, { timeout: 15_000 });
  results.push({ workflow: 'Voice TTS playback', result: 'PASS', evidence: `React requested the existing local TTS service, then fetched ${directAudio.body.length} stored WAV bytes (RIFF header verified) through authenticated project-scoped GET /api/voice/assets/{id}/content. The React playback blob matched the real HTTP response and exposed a native audio player and download link; no external provider was called.` });

  }

  // Calendar: create a local event in the selected React calendar and verify
  // persistence through the read contract; no sync provider is involved.
  await open('calendar', fixture.tokens.calendar);
  const scheduleAnalysisResponse = page.waitForResponse(response => new URL(response.url()).pathname === '/api/calendar/free-slots', { timeout: 15_000 });
  await page.getByRole('button', { name: 'Find time for a team sync' }).click();
  const scheduleAnalysis = await scheduleAnalysisResponse;
  assert(scheduleAnalysis.ok(), `Calendar assistant availability request returned ${scheduleAnalysis.status()}`);
  await expect(page.locator('#toast-region')).toContainText('Calendar backend found', { timeout: 10_000 });
  const scheduleSummaryResponse = page.waitForResponse(response => {
    const url = new URL(response.url());
    return url.pathname === '/api/calendar/optimize' && url.searchParams.has('from');
  }, { timeout: 15_000 });
  await page.getByRole('button', { name: 'Summarise my week' }).click();
  const scheduleSummary = await scheduleSummaryResponse;
  const scheduleSummaryData = await scheduleSummary.json();
  assert(scheduleSummary.ok() && Number.isFinite(Number(scheduleSummaryData.total_events)), `Calendar summary request did not return backend counts: ${JSON.stringify(scheduleSummaryData)}`);
  await expect(page.locator('#toast-region')).toContainText('Calendar backend reviewed', { timeout: 10_000 });
  await page.getByRole('button', { name: 'New', exact: true }).click();
  const calendarTitle = 'Batch 2A local calendar fixture';
  await page.locator('[aria-labelledby="calendar-create-title"] input').first().fill(calendarTitle);
  const calendarStart = page.getByLabel('Starts', { exact: true });
  const calendarEnd = page.getByLabel('Ends', { exact: true });
  // Keep the event near enough to appear in the React Upcoming list while
  // staying outside the repeated 01:00 hour on DST transition days.
  const futureStart = new Date(Date.now() + 4 * 60 * 60_000);
  futureStart.setMinutes(0, 0, 0);
  const futureEnd = new Date(futureStart.getTime() + 60 * 60_000);
  const toLocalInput = date => { const shifted = new Date(date.getTime() - date.getTimezoneOffset() * 60_000); return shifted.toISOString().slice(0, 16); };
  await calendarStart.fill(toLocalInput(futureStart));
  await calendarEnd.fill(toLocalInput(futureEnd));
  await page.locator('[aria-labelledby="calendar-create-title"] select').selectOption('FREQ=DAILY');
  assert(await calendarStart.inputValue() && await calendarEnd.inputValue(), 'Calendar event form did not retain its start and end fields');
  const calendarCreateButton = page.getByRole('button', { name: 'Create event', exact: true });
  await expect(calendarCreateButton).toBeEnabled();
  const calendarFormState = await page.locator('[aria-labelledby="calendar-create-title"] form').evaluate(form => ({
    valid: form.checkValidity(),
    fields: [...form.querySelectorAll('input')].map(input => ({ name: input.getAttribute('aria-label') || input.type, value: input.value, valid: input.checkValidity() })),
  }));
  assert(calendarFormState.valid, `Calendar create form is invalid before submit: ${JSON.stringify(calendarFormState)}`);
  const calendarCreatePromise = page.waitForResponse(response => new URL(response.url()).pathname === '/api/calendar/events' && response.request().method() === 'POST', { timeout: 15_000 }).catch(async error => {
    const recent = apiRequests.slice(-12).map(request => `${request.method()} ${new URL(request.url()).pathname}`).join(', ');
    throw new Error(`${error.message}; calendar form=${JSON.stringify(calendarFormState)}; recent API requests=${recent}; visible form errors=${await page.locator('[aria-labelledby="calendar-create-title"] [role="alert"]').allTextContents().catch(() => [])}`);
  });
  await calendarCreateButton.click();
  const calendarCreate = await calendarCreatePromise;
  assert(calendarCreate.status() === 201, `React calendar create returned ${calendarCreate.status()}`);
  const createdCalendarEvent = (await calendarCreate.json()).event;
  assert(createdCalendarEvent?.id, 'Calendar create response omitted its event ID');
  const calendarRead = await requestJson(fixture.baseUrl, '/api/calendar/events?project=default', fixture.tokens.calendar);
  assert(calendarRead.response.ok && calendarRead.payload.events?.some(event => String(event.id) === String(createdCalendarEvent.id)), 'Calendar event did not persist through its read endpoint');
  const nextOccurrenceStart = new Date(createdCalendarEvent.starts_at);
  nextOccurrenceStart.setUTCDate(nextOccurrenceStart.getUTCDate() + 1);
  const nextOccurrenceEnd = new Date(nextOccurrenceStart.getTime() + 60 * 60_000);
  const recurrenceRead = await requestJson(fixture.baseUrl, `/api/calendar/events?${new URLSearchParams({ project: 'default', from: nextOccurrenceStart.toISOString(), to: nextOccurrenceEnd.toISOString() })}`, fixture.tokens.calendar);
  const recurringRead = recurrenceRead.payload.events?.find(event => String(event.id) === String(createdCalendarEvent.id));
  assert(recurrenceRead.response.ok && recurringRead?.occurrences?.some(occurrence => new Date(String(occurrence.starts_at)).getTime() === nextOccurrenceStart.getTime()), `Recurring event occurrence was not returned in its future date range: ${JSON.stringify({ status: recurrenceRead.response.status, expected: nextOccurrenceStart.toISOString(), events: recurrenceRead.payload.events })}`);
  await page.getByRole('button', { name: 'Next period' }).click();
  await page.locator('.calendar-event').filter({ has: page.getByText(calendarTitle) }).first().waitFor({ state: 'visible', timeout: 15_000 });
  const calendarList = await requestJson(fixture.baseUrl, '/api/calendar/calendars?project=default', fixture.tokens.calendar);
  const otherCalendar = calendarList.payload.calendars?.find(calendar => String(calendar.id) !== String(createdCalendarEvent.calendar_id));
  assert(calendarList.response.ok && otherCalendar?.id, 'Calendar fixture needs at least two calendars to verify availability filtering');
  const busyStarts = new Date(createdCalendarEvent.ends_at);
  const busyEnds = new Date(busyStarts.getTime() + 60 * 60_000);
  const otherBusy = await requestJson(fixture.baseUrl, '/api/calendar/events', fixture.tokens.calendar, {
    method: 'POST', body: {
      project: 'default', calendar_id: String(otherCalendar.id), title: 'Batch 2B second-calendar busy block',
      starts_at: busyStarts.toISOString(), ends_at: busyEnds.toISOString(), reminders: [],
    },
  });
  assert(otherBusy.response.status === 201, `Could not seed second-calendar availability fixture: ${otherBusy.response.status}`);
  const searchStart = new Date(createdCalendarEvent.starts_at);
  const searchEnd = new Date(searchStart.getTime() + 2 * 60 * 60_000);
  const availabilityQuery = (calendarId = '') => {
    const query = new URLSearchParams({ project: 'default', from: searchStart.toISOString(), to: searchEnd.toISOString(), duration: '30' });
    if (calendarId) query.set('calendar_id', calendarId);
    return `/api/calendar/free-slots?${query}`;
  };
  const allCalendarSlots = await requestJson(fixture.baseUrl, availabilityQuery(), fixture.tokens.calendar);
  const selectedCalendarSlots = await requestJson(fixture.baseUrl, availabilityQuery(String(createdCalendarEvent.calendar_id)), fixture.tokens.calendar);
  assert(allCalendarSlots.response.ok && allCalendarSlots.payload.slots?.length === 0, 'Unfiltered availability should account for both calendars busy blocks');
  assert(selectedCalendarSlots.response.ok && selectedCalendarSlots.payload.slots?.length > 0, 'Selected-calendar availability did not exclude the other calendar busy block');
  results.push({ workflow: 'Calendar availability scope', result: 'PASS', evidence: 'Actual authenticated Python free-slot responses account for events across all calendars by default and ignore another calendar’s busy time when a specific calendar_id is selected.' });
  const calendarRow = page.locator('.calendar-upcoming-row').filter({ hasText: calendarTitle }).first();
  await calendarRow.waitFor({ state: 'visible', timeout: 15_000 });
  await calendarRow.click();
  const calendarDetails = page.getByRole('dialog', { name: calendarTitle });
  await calendarDetails.waitFor({ state: 'visible', timeout: 10_000 });
  await expect(calendarDetails).toContainText('Recurring series occurrence');
  await calendarDetails.getByRole('button', { name: 'Edit series' }).click();
  const calendarEdit = page.getByRole('dialog', { name: 'Edit calendar event' });
  await calendarEdit.getByLabel('Title').fill(`${calendarTitle} updated`);
  const calendarPatchPromise = page.waitForResponse(response => response.url().includes(`/api/calendar/events/${createdCalendarEvent.id}`) && response.request().method() === 'PATCH', { timeout: 15_000 });
  await calendarEdit.getByRole('button', { name: 'Save changes' }).click();
  const calendarPatch = await calendarPatchPromise;
  assert(calendarPatch.status() === 200, `React calendar update returned ${calendarPatch.status()}: ${await calendarPatch.text()}`);
  const updatedCalendarEvent = (await calendarPatch.json()).event;
  assert(updatedCalendarEvent?.title === `${calendarTitle} updated`, 'Calendar update response did not contain the edited title');
  const calendarAfterPatch = await requestJson(fixture.baseUrl, '/api/calendar/events?project=default', fixture.tokens.calendar);
  assert(calendarAfterPatch.response.ok && calendarAfterPatch.payload.events?.some(event => String(event.id) === String(createdCalendarEvent.id) && event.title === `${calendarTitle} updated`), 'Calendar update did not persist through the authenticated read endpoint');
  await page.locator('.calendar-upcoming-row').filter({ hasText: `${calendarTitle} updated` }).first().click();
  page.once('dialog', dialog => dialog.accept());
  const calendarDeletePromise = page.waitForResponse(response => response.url().includes(`/api/calendar/events/${createdCalendarEvent.id}`) && response.request().method() === 'DELETE', { timeout: 15_000 });
  await page.getByRole('dialog', { name: `${calendarTitle} updated` }).getByRole('button', { name: 'Delete series' }).click();
  const calendarDelete = await calendarDeletePromise;
  assert(calendarDelete.status() === 200, `React calendar delete returned ${calendarDelete.status()}`);
  const calendarAfterDelete = await requestJson(fixture.baseUrl, '/api/calendar/events?project=default', fixture.tokens.calendar);
  assert(calendarAfterDelete.response.ok && !calendarAfterDelete.payload.events?.some(event => String(event.id) === String(createdCalendarEvent.id)), 'Calendar delete did not persist through the authenticated read endpoint');
  await page.getByText('Calendar tools', { exact: true }).click();
  await page.getByRole('tab', { name: 'Reminders' }).click();
  const reminderRow = page.locator('.timeline-row').filter({ hasText: 'Batch 2B due reminder fixture' });
  await reminderRow.waitFor({ state: 'visible', timeout: 15_000 });
  const dueBeforeAck = await requestJson(fixture.baseUrl, '/api/calendar/reminders?project=default', fixture.tokens.calendar);
  assert(dueBeforeAck.response.ok && dueBeforeAck.payload.reminders?.some(reminder => String(reminder.title ?? reminder.event_title) === 'Batch 2B due reminder fixture'), 'Synthetic due reminder was not available from the real Python API');
  const reminderId = String(dueBeforeAck.payload.reminders.find(reminder => String(reminder.title ?? reminder.event_title) === 'Batch 2B due reminder fixture').reminder_id);
  const acknowledgePromise = page.waitForResponse(response => response.url().includes(`/api/calendar/reminders/${reminderId}/acknowledge`) && response.request().method() === 'POST', { timeout: 15_000 });
  await reminderRow.getByRole('button', { name: 'Acknowledge' }).click();
  const acknowledgeResponse = await acknowledgePromise;
  assert(acknowledgeResponse.status() === 200, `React reminder acknowledgement returned ${acknowledgeResponse.status()}: ${await acknowledgeResponse.text()}`);
  const dueAfterAck = await requestJson(fixture.baseUrl, '/api/calendar/reminders?project=default', fixture.tokens.calendar);
  assert(dueAfterAck.response.ok && !dueAfterAck.payload.reminders?.some(reminder => String(reminder.reminder_id) === reminderId), 'Acknowledged reminder remained due in the persistent Python store');
  results.push({ workflow: 'Calendar', result: 'PASS', evidence: 'React used authenticated backend availability and schedule-summary suggestions without changing an event, then created a recurring series. Python returned its future occurrence despite DTSTART being outside the requested range; React rendered that occurrence, opened its series-level details, edited/deleted the series, and acknowledged a seeded due reminder. Mutations persisted, and availability honored the selected-calendar scope.' });
  if (onlyCalendar) {
    finishResults({ selectedWorkflows: ['Calendar'] });
    return;
  }

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
