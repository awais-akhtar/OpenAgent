import { spawn } from 'node:child_process';
import { appendFile, mkdir, writeFile } from 'node:fs/promises';
import net from 'node:net';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const configuredOrigin = process.env.OPENAGENT_UI_ORIGIN;
let origin = configuredOrigin ?? '';
let port = configuredOrigin ? Number(new URL(configuredOrigin).port || 80) : 0;
const logDir = path.join(root, '.qa');
const logFile = path.join(logDir, 'playwright-last.log');
await mkdir(logDir, { recursive: true });
await writeFile(logFile, `E2E run started ${new Date().toISOString()}\nOrigin: ${origin || 'reserved isolated loopback port'}\n`, 'utf8');

function log(line) {
  process.stdout.write(line);
  void appendFile(logFile, line);
}

function reserveLocalPort(host) {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.once('error', reject);
    server.listen(0, host, () => {
      const address = server.address();
      const selectedPort = address && typeof address === 'object' ? address.port : 0;
      server.close(error => error ? reject(error) : selectedPort ? resolve(selectedPort) : reject(new Error('Could not reserve an isolated E2E port.')));
    });
  });
}

function waitForExit(child, timeoutMs) {
  return Promise.race([
    new Promise(resolve => child.once('exit', (code, signal) => resolve({ code, signal }))),
    new Promise(resolve => setTimeout(() => resolve(null), timeoutMs)),
  ]);
}

function run(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, windowsHide: true, ...options });
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({ code: code ?? (signal ? 1 : 0), signal }));
    child.stdout?.on('data', chunk => log(chunk.toString()));
    child.stderr?.on('data', chunk => log(chunk.toString()));
  });
}

let vite;
let serverStarted = false;
let apiOrigin = origin;
let exitCode = 1;
try {
  if (!process.env.OPENAGENT_UI_ORIGIN) {
    port = await reserveLocalPort('127.0.0.1');
    origin = `http://127.0.0.1:${port}`;
    const parsed = new URL(origin);
    vite = spawn(process.execPath, [path.join(root, 'node_modules/vite/bin/vite.js'), '--host', parsed.hostname, '--port', String(port), '--strictPort'], {
      cwd: root,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: process.env,
    });
    vite.stdout.on('data', chunk => log(`[vite] ${chunk.toString()}`));
    vite.stderr.on('data', chunk => log(`[vite] ${chunk.toString()}`));
    vite.once('error', error => { throw error; });
    const readyDeadline = Date.now() + 30_000;
    while (Date.now() < readyDeadline) {
      if (vite.exitCode !== null) throw new Error(`Vite exited before readiness (code ${vite.exitCode}). See ${logFile}.`);
      try {
        const response = await fetch(origin, { signal: AbortSignal.timeout(1_000) });
        if (response.ok) { serverStarted = true; break; }
      } catch { /* Wait for the owned Vite child to bind. */ }
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    if (!serverStarted) throw new Error(`Timed out waiting for the isolated Vite server at ${origin}. See ${logFile}.`);

    // The page HTML can answer before Vite has transformed the lazy-loaded
    // workspace modules. Warm those route entry points before launching a
    // browser so the first route assertion measures the app, not a cold
    // transform on a busy Windows runner.
    for (const modulePath of [
      '/src/screens/HomeChat.tsx',
      '/src/screens/WorkspaceScreens.tsx',
      '/src/screens/OperationsScreens.tsx',
      '/src/screens/AdvancedScreen.tsx',
    ]) {
      const response = await fetch(new URL(modulePath, origin), { signal: AbortSignal.timeout(60_000) });
      if (!response.ok) throw new Error(`Vite failed to prepare ${modulePath} (${response.status}). See ${logFile}.`);
      const moduleSource = await response.text();
      if (!moduleSource.trim()) throw new Error(`Vite returned an empty module for ${modulePath}. See ${logFile}.`);
    }
    apiOrigin = origin;
  }

  const cli = path.join(root, 'node_modules/@playwright/test/cli.js');
  const result = await run(process.execPath, [cli, 'test', ...process.argv.slice(2)], {
    env: { ...process.env, OPENAGENT_UI_ORIGIN: apiOrigin },
    stdio: ['inherit', 'pipe', 'pipe'],
  });
  exitCode = result.code;
  log(`\nPlaywright process exited with code ${result.code}${result.signal ? ` (${result.signal})` : ''}.\n`);
} catch (error) {
  log(`\nE2E runner failure: ${error instanceof Error ? error.stack ?? error.message : String(error)}\n`);
  exitCode = 1;
} finally {
  if (vite && serverStarted && vite.exitCode === null) {
    vite.kill('SIGTERM');
    const exited = await waitForExit(vite, 5_000);
    if (exited === null && process.platform === 'win32' && vite.pid) {
      await run('taskkill.exe', ['/PID', String(vite.pid), '/T', '/F'], { stdio: ['ignore', 'pipe', 'pipe'] }).catch(error => log(`\nVite taskkill fallback failed: ${String(error)}\n`));
      await waitForExit(vite, 2_000);
    }
    log(`\nIsolated Vite shutdown requested; child exit code: ${vite.exitCode ?? 'terminated by fallback'}.\n`);
  }
}

process.exitCode = exitCode;
