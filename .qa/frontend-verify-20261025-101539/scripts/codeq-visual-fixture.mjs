import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { chromium } from '@playwright/test';

const FIXTURE_ENV = 'OPENAGENT_ENABLE_VISUAL_FIXTURES';
const PROJECT_ID = 'qa-reference-ecommerce';

function loadFixture() {
  if (process.env[FIXTURE_ENV] !== '1' || process.env.NODE_ENV === 'production') {
    throw new Error(`QA guard: this fixture is test-only. Set ${FIXTURE_ENV}=1 in a non-production environment.`);
  }
  const root = process.cwd();
  if (path.basename(root) !== 'frontend-second') throw new Error('Run the CodeQ visual fixture from frontend-second only.');
  const fixture = JSON.parse(readFileSync('.qa/visual-fixtures/reference-fixtures.json', 'utf8'));
  const source = readFileSync('reference/frontend-final-approved.html');
  const sourceHash = createHash('sha256').update(source).digest('hex').toUpperCase();
  if (sourceHash !== 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1' || fixture.sourceSha256 !== sourceHash) throw new Error('The visual fixture was extracted from a different reference HTML file.');
  if (fixture.purpose !== 'Visual acceptance only. Never use these values in production data or production workflows.') {
    throw new Error('Fixture purpose marker is missing or invalid.');
  }
  const files = fixture.data?.topLevel?.codeFiles;
  if (!Array.isArray(files) || files.length !== 4 || files.some(file => typeof file.content !== 'string' || !file.path)) {
    throw new Error('CodeQ fixture is incomplete. Run `node scripts/extract-reference-visual-fixtures.mjs` followed by `node scripts/enrich-codeq-visual-fixture.mjs` with the QA opt-in enabled.');
  }
  return files;
}

const json = (route, status, body) => route.fulfill({ status, contentType: 'application/json; charset=utf-8', body: JSON.stringify(body) });

/** Install deterministic, read-only CodeQ fixture responses in a QA Playwright page. */
export async function installCodeQVisualFixture(page) {
  const files = loadFixture();
  const byPath = new Map(files.map(file => [file.path, file]));
  const project = {
    id: PROJECT_ID,
    name: 'E-commerce Platform',
    source_path: '[visual-fixture-only]',
    project: 'default',
    backend: 'local',
    workspace: { workspace_path: '[visual-fixture-only]/ecommerce-platform' },
  };
  const directories = ['.next', 'public', 'src', 'src/app', 'src/components', 'src/hooks', 'src/styles', 'api'];
  // Keep the visual reference's explorer sequence. The four sample source files
  // are fixture-only and intentionally placed after the expanded src folders.
  const entries = [
    ...['.next', 'public', 'src', 'src/app', 'src/components'].map(filePath => ({ path: filePath, type: 'directory' })),
    ...files.map(file => ({ path: file.path, type: 'file' })),
    ...['src/hooks', 'src/styles', 'api'].map(filePath => ({ path: filePath, type: 'directory' })),
    { path: 'package.json', type: 'file' },
    { path: 'README.md', type: 'file' },
  ];

  console.warn('QA ONLY: CodeQ visual fixture routes are active. Mutating API requests are blocked; fixture data must never enter production.');
  await page.addInitScript(() => { window.__OPENAGENT_VISUAL_FIXTURE__ = true; });
  // The application requests Monaco's AMD loader from jsDelivr. In offline QA
  // runs, serve the exact installed package files locally so editor rendering
  // is deterministic; this is static test tooling, not an API fixture.
  const monacoRoot = path.resolve('node_modules/monaco-editor/min/vs');
  await page.route('https://cdn.jsdelivr.net/npm/monaco-editor@0.55.1/min/vs/**', async route => {
    const url = new URL(route.request().url());
    const prefix = '/npm/monaco-editor@0.55.1/min/vs/';
    const relative = decodeURIComponent(url.pathname.slice(prefix.length));
    const assetPath = path.resolve(monacoRoot, relative);
    const insideMonaco = path.relative(monacoRoot, assetPath);
    if (!insideMonaco || insideMonaco.startsWith('..') || path.isAbsolute(insideMonaco) || !existsSync(assetPath)) {
      await route.fulfill({ status: 404, body: 'Monaco QA asset not found.' });
      return;
    }
    const extension = path.extname(assetPath).toLowerCase();
    const contentType = extension === '.css' ? 'text/css' : extension === '.json' ? 'application/json' : extension === '.wasm' ? 'application/wasm' : extension === '.ttf' ? 'font/ttf' : 'application/javascript';
    await route.fulfill({ status: 200, contentType, body: readFileSync(assetPath) });
  });
  await page.route('**/api/**', async route => {
    const request = route.request();
    const url = new URL(request.url());
    const method = request.method().toUpperCase();
    const codebase = url.pathname.match(/^\/api\/codebase\/projects(?:\/([^/]+))?(?:\/(tree|file))?$/);

    // Mutations are blocked globally while a visual fixture is active, including
    // CodeQ runs, file writes, terminal commands, and unrelated API writes.
    if (!['GET', 'HEAD'].includes(method)) {
      await json(route, 403, { error: 'Blocked by the read-only visual acceptance fixture.' });
      return;
    }

    if (method === 'GET' && url.pathname === '/api/codebase/projects') {
      await json(route, 200, { projects: [project] });
      return;
    }
    if (method === 'GET' && codebase?.[1] === PROJECT_ID && codebase[2] === 'tree') {
      await json(route, 200, { entries });
      return;
    }
    if (method === 'GET' && codebase?.[1] === PROJECT_ID && codebase[2] === 'file') {
      const file = byPath.get(url.searchParams.get('path') ?? '');
      if (!file) {
        await json(route, 404, { error: 'This path is not present in the reference CodeQ visual fixture.' });
        return;
      }
      const content = file.content;
      await json(route, 200, { file: { path: file.path, content, sha256: createHash('sha256').update(content).digest('hex') } });
      return;
    }
    if (method === 'GET' && codebase?.[1] === PROJECT_ID && !codebase[2]) {
      await json(route, 200, { project });
      return;
    }
    if (url.pathname.startsWith('/api/codebase/')) {
      await json(route, 403, { error: 'This CodeQ read endpoint is outside the explicit visual fixture allowlist.' });
      return;
    }
    if (url.pathname === '/api/terminal/runs') {
      await json(route, 200, { runs: [] });
      return;
    }
    await route.continue();
  });
  return { projectId: PROJECT_ID, fileCount: files.length };
}

async function captureFixture() {
  const browser = await chromium.launch({ headless: true, executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe' });
  try {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
    const { fileCount } = await installCodeQVisualFixture(page);
    const origin = (process.env.OPENAGENT_UI_ORIGIN || 'http://127.0.0.1:5180').replace(/\/$/, '');
    const theme = process.env.OPENAGENT_VISUAL_THEME === 'dark' ? 'dark' : 'light';
    await page.goto(`${origin}/?view=codeq`, { waitUntil: 'domcontentloaded' });
    await page.evaluate(value => { document.documentElement.dataset.theme = value; }, theme);
    await page.getByRole('treeitem', { name: 'ProductCard.tsx' }).click();
    await page.locator('.codeq-editor .monaco-editor .view-lines .view-line').first().waitFor({ state: 'visible' });
    await page.waitForTimeout(350);
    const output = path.resolve(`.qa/visual-codeq-fixture/${theme}/desktop-1440x900/codeq.png`);
    mkdirSync(path.dirname(output), { recursive: true });
    await page.screenshot({ path: output, fullPage: false, animations: 'disabled' });
    console.log(`Captured CodeQ with ${fileCount} test-only reference file records: ${output}`);
    console.log('Not fixture-backed: preview lifecycle, run activity/terminal output, Git diff, and assistant proposal. Those remain honest unavailable/idle states.');
  } finally {
    await browser.close();
  }
}

if (process.argv.includes('--capture')) await captureFixture();
