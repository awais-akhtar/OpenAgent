import { demoProvider } from './provider';

type DemoRow = Record<string, unknown>;

function calendarOccurrences(event: DemoRow, fromMs: number, toMs: number): DemoRow[] {
  const startMs = new Date(String(event.starts_at ?? '')).getTime();
  const endMs = new Date(String(event.ends_at ?? '')).getTime();
  const rule = String(event.rrule ?? '');
  if (!rule || !Number.isFinite(startMs) || !Number.isFinite(endMs)) return startMs < toMs && endMs > fromMs ? [event] : [];
  const parts = Object.fromEntries(rule.split(';').map(pair => pair.split('=', 2)).filter(pair => pair.length === 2));
  const frequency = parts.FREQ;
  const interval = Math.max(1, Math.min(1000, Number(parts.INTERVAL ?? 1) || 1));
  const count = Math.max(1, Math.min(1000, Number(parts.COUNT ?? 1000) || 1));
  const untilValue = String(parts.UNTIL ?? '');
  const untilMatch = untilValue.match(/^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?$/);
  const until = untilMatch ? new Date(`${untilMatch[1]}-${untilMatch[2]}-${untilMatch[3]}T${untilMatch[4]}:${untilMatch[5]}:${untilMatch[6]}Z`).getTime() : untilValue ? new Date(untilValue).getTime() : Number.POSITIVE_INFINITY;
  if (!['DAILY', 'WEEKLY', 'MONTHLY', 'YEARLY'].includes(String(frequency))) return startMs >= fromMs && startMs < toMs ? [event] : [];
  const start = new Date(startMs);
  const duration = endMs - startMs;
  const results: DemoRow[] = [];
  for (let index = 0; index < count && index < 1000; index += 1) {
    const occurrence = new Date(startMs);
    if (frequency === 'DAILY') occurrence.setDate(start.getDate() + index * interval);
    else if (frequency === 'WEEKLY') occurrence.setDate(start.getDate() + index * interval * 7);
    else {
      const monthOffset = index * interval * (frequency === 'YEARLY' ? 12 : 1);
      const monthStart = new Date(start.getFullYear(), start.getMonth() + monthOffset, 1, start.getHours(), start.getMinutes(), start.getSeconds(), start.getMilliseconds());
      const day = Math.min(start.getDate(), new Date(monthStart.getFullYear(), monthStart.getMonth() + 1, 0).getDate());
      monthStart.setDate(day);
      occurrence.setTime(monthStart.getTime());
    }
    const occurrenceMs = occurrence.getTime();
    if (occurrenceMs > until) break;
    if (occurrenceMs < toMs && occurrenceMs + duration > fromMs) results.push({ starts_at: occurrence.toISOString(), ends_at: new Date(occurrenceMs + duration).toISOString() });
  }
  return results;
}
export const codeQDemoProductCardContent = `import { useState } from 'react';
import { Heart, ShoppingCart } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface ProductCardProps {
  product: {
    id: string;
    name: string;
    price: number;
    image: string;
    rating: number;
  };
}

export function ProductCard({ product }: ProductCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isInCart, setIsInCart] = useState(false);

  const addToCart = async () => {
    try {
      await cart.add(product.id);
      setIsInCart(true);
    } catch (error) {
      console.error('Unable to add item', error);
    }
  };

  return (
    <Card className="group overflow-hidden">
      <img src={product.image} alt={product.name} />
      <h3>{product.name}</h3>
      <p>{formatPrice(product.price)}</p>
      <Button onClick={addToCart}>Add to cart</Button>
    </Card>
  );
}`;

const codeQDemoFiles: Record<string, string> = {
  'src/components/ProductCard.tsx': codeQDemoProductCardContent,
  'src/hooks/useCart.ts': `import { useState } from 'react';

export function useCart() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);

  async function add(id: string) {
    setLoading(true);
    try {
      // Replace this example with the connected API.
      return { id };
    } finally {
      setLoading(false);
    }
  }

  return { items, loading, add };
}`,
  'src/app/page.tsx': `import { ProductCard } from '@/components/ProductCard';

export default function ShopPage() {
  return (
    <main className="container mx-auto">
      <h1>Products</h1>
      <ProductCard product={demoProduct} />
    </main>
  );
}`,
  'tailwind.config.js': `export default {
  content: ['./src/**/*.{js,ts,jsx,tsx}'],
  theme: { extend: {} },
  plugins: [],
};`,
  'package.json': '{\n  "name": "ecommerce-platform",\n  "private": true,\n  "scripts": { "dev": "next dev" }\n}',
  'README.md': '# E-commerce Platform\n\nExample workspace for the approved CodeQ interface.\n',
};
type WorkspaceState = {
  files: DemoRow[];
  assets: DemoRow[];
  codeProjects: DemoRow[];
  codeFiles: DemoRow[];
  codeRuns: DemoRow[];
  terminalRuns: DemoRow[];
  researchSessions: DemoRow[];
  knowledgeDocuments: DemoRow[];
  knowledgeCollections: DemoRow[];
  calendarCalendars: DemoRow[];
  calendarEvents: DemoRow[];
  providers: DemoRow[];
  conflicts: DemoRow[];
  workspaceItems: DemoRow[];
};

const now = () => new Date().toISOString();
let nextId = 1;
const uid = (prefix: string) => `demo-${prefix}-${nextId++}`;
const seededContent = codeQDemoFiles;

function knowledgeSeed(advanced: Record<string, unknown>): { documents: DemoRow[]; collections: DemoRow[] } {
  const value = advanced.knowledge;
  const knowledge = value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
  const documents = Array.isArray(knowledge.documents) ? knowledge.documents as DemoRow[] : [];
  const collections = Array.isArray(knowledge.collections) ? knowledge.collections as DemoRow[] : [];
  return {
    documents: documents.map((item, index) => ({ ...item, id: item.id ?? `demo-knowledge-${index + 1}`, title: item.title ?? item.name ?? 'Knowledge source', content: item.content ?? item.description ?? '', chunks: item.chunks ?? item.count ?? 1, tags: Array.isArray(item.tags) ? item.tags : [] })),
    collections: collections.map((item, index) => ({ ...item, id: item.id ?? `demo-collection-${index + 1}`, name: item.name ?? item.title ?? 'Knowledge collection', dimensions: item.dimensions ?? item.count ?? 0, distance: item.distance ?? 'cosine' })),
  };
}

async function initialize(): Promise<WorkspaceState> {
  const stored = await demoProvider.get('workspaceDemo');
  if (stored.version === 2 && stored.state && typeof stored.state === 'object') {
    const persisted = structuredClone(stored.state as WorkspaceState);
    persisted.workspaceItems ??= [];
    return persisted;
  }

  const [shell, advanced] = await Promise.all([demoProvider.get('shell'), demoProvider.collections()]);
  const sourceFiles = Array.isArray(shell.files) ? shell.files as DemoRow[] : [];
  const codeFiles = Array.isArray(shell.codeFiles) ? shell.codeFiles as DemoRow[] : [];
  const docs = sourceFiles.map((file, index) => {
    const sizeLabel = String(file.size ?? '');
    const sizeMatch = sizeLabel.match(/([\d.]+)\s*(MB|KB|B)/i);
    const size_bytes = sizeMatch ? Number(sizeMatch[1]) * (sizeMatch[2].toUpperCase() === 'MB' ? 1_000_000 : sizeMatch[2].toUpperCase() === 'KB' ? 1_000 : 1) : 0;
    const ageMatch = String(file.updated ?? '').match(/(\d+)\s+(day|week)s?\s+ago/i);
    const ageDays = ageMatch ? Number(ageMatch[1]) * (ageMatch[2].toLowerCase() === 'week' ? 7 : 1) : 0;
    const title = String(file.name ?? 'Project document');
    const description = String(file.description ?? '');
    const content = index === 0 ? `# ${title}\n\n## 1. Overview\n${description} A connected application will render the actual document here and will respect file permissions and available preview formats.\n\n## 2. Requirements\n- Keep user tasks easy to discover.\n- Use a consistent interface across every workspace.\n- Make live execution status understandable and trustworthy.\n\n## 3. Implementation notes\nShow real file content from the existing backend; never substitute these examples for production data.`
      : `# ${title}\n\n${description}\n\n## Notes\n- Example content is stored in isolated demo data.\n- Editing this document changes only this browser's demo workspace.`;
    return {
      id: `demo-doc-${index + 1}`, doc_id: `demo-doc-${index + 1}`, title,
      description, format: String(file.type ?? 'MD').toLowerCase(), status: 'active', content,
      size_label: sizeLabel, version: 1, size_bytes,
      updated_at: new Date(Date.now() - ageDays * 86_400_000 - 30_000).toISOString(), project: 'default',
    };
  });
  const research = (await demoProvider.get('coreFixtures.research')) as DemoRow;
  const researchSeed = Array.isArray(research.sessions) ? research.sessions as DemoRow[] : [];
  const knowledgeRows = knowledgeSeed(advanced);
  const knowledgeDocs = knowledgeRows.documents;
  const knowledgeCollections = knowledgeRows.collections;
  const today = new Date();
  const calendarMonday = new Date(today);
  calendarMonday.setDate(calendarMonday.getDate() - ((calendarMonday.getDay() + 6) % 7));
  calendarMonday.setHours(0, 0, 0, 0);
  const at = (dayOffset: number, hour: number) => { const date = new Date(calendarMonday); date.setDate(date.getDate() + dayOffset); date.setHours(hour, 0, 0, 0); return date.toISOString(); };
  const state: WorkspaceState = {
    files: docs,
    assets: [],
    codeProjects: [{ id: 'demo-code-project', name: 'E-commerce Platform', workspace: { workspace_path: 'demo://e-commerce-platform' }, source_path: 'demo://e-commerce-platform' }],
    codeFiles: [...codeFiles.map((file, index) => ({ id: `demo-code-file-${index + 1}`, path: String(file.path ?? file.name ?? `file-${index}.txt`), name: String(file.name ?? 'example.txt'), type: 'file', size: (seededContent[String(file.path)] ?? '// Approved CodeQ demo file\n').length, sha256: `demo-sha-${index + 1}`, content: seededContent[String(file.path)] ?? `// ${String(file.name ?? 'Example file')}\n// Deterministic local demo content.\n` })),
      ...['package.json', 'README.md'].filter(path => !codeFiles.some(file => String(file.path ?? '') === path)).map((path, index) => ({ id: `demo-code-extra-${index + 1}`, path, name: path, type: 'file', size: seededContent[path].length, sha256: `demo-extra-sha-${index + 1}`, content: seededContent[path] }))],
    codeRuns: [], terminalRuns: [],
    researchSessions: researchSeed.length ? researchSeed : [{
      id: 'demo-research-completed', query: 'Open-source AI agents', status: 'completed', project: 'default',
      provider: 'Local example', allow_external: false, created_at: now(), completed_at: now(),
      summary: 'Agent frameworks are increasingly designed around structured tool use, coordination, and external data access. A production system should make source quality, permissions, and run progress easy to understand. This sample demonstrates the content layout; it is not a verified research result.',
      sources: [
        { id: 'demo-research-source-1', title: 'AutoGen project documentation', site: 'Official documentation', date: 'Source 01', url: 'https://example.test/autogen', snippet: 'Example source fixture; no external source is contacted in demo mode.' },
        { id: 'demo-research-source-2', title: 'CrewAI framework guide', site: 'Framework documentation', date: 'Source 02', url: 'https://example.test/crewai', snippet: 'Example source fixture; no external source is contacted in demo mode.' },
        { id: 'demo-research-source-3', title: 'LangGraph overview', site: 'LangChain documentation', date: 'Source 03', url: 'https://example.test/langgraph', snippet: 'Example source fixture; no external source is contacted in demo mode.' },
        { id: 'demo-research-source-4', title: 'Model Context Protocol specification', site: 'Official specification', date: 'Source 04', url: 'https://example.test/mcp', snippet: 'Example source fixture; no external source is contacted in demo mode.' },
        { id: 'demo-research-source-5', title: 'llama.cpp repository', site: 'GitHub project', date: 'Source 05', url: 'https://example.test/llama-cpp', snippet: 'Example source fixture; no external source is contacted in demo mode.' },
      ],
      events: [{ sequence: 1, type: 'research.completed', message: 'Example report and source review are ready.' }],
    }],
    knowledgeDocuments: knowledgeDocs.length ? knowledgeDocs : [
      { id: 'demo-knowledge-1', title: 'Product requirements', source: 'Product requirements specification', content: 'The product workspace keeps project decisions and requirements searchable.', status: 'indexed', chunks: 8, tags: ['product', 'requirements'] },
      { id: 'demo-knowledge-2', title: 'Architecture overview', source: 'System architecture overview', content: 'The application uses a modular service architecture with scoped project data.', status: 'indexed', chunks: 12, tags: ['architecture', 'engineering'] },
    ],
    knowledgeCollections: knowledgeCollections.length ? knowledgeCollections : [{ id: 'demo-knowledge-collection', name: 'Project knowledge', dimensions: 1536, distance: 'cosine', updated_at: now() }],
    calendarCalendars: [{ id: 'demo-calendar-primary', name: 'Personal', description: 'Local demo calendar', is_default: true }],
    calendarEvents: [
      { id: 'demo-event-standup', title: 'Team stand-up', starts_at: at(0, 9), ends_at: at(0, 10), calendar_id: 'demo-calendar-primary', description: 'Share progress and coordinate the day.', location: 'Project room', attendees: ['Alex Morgan'], reminders: [15] },
      { id: 'demo-event-planning', title: 'Project planning', starts_at: at(1, 10), ends_at: at(1, 11), calendar_id: 'demo-calendar-primary', description: 'Plan the coming product work.', location: '', attendees: [], reminders: [15] },
      { id: 'demo-event-reviews', title: 'Review PRs', starts_at: at(0, 11), ends_at: at(0, 12), calendar_id: 'demo-calendar-primary', description: 'Review changes ready for the project.', location: 'Engineering', attendees: [], reminders: [15] },
      { id: 'demo-event-research', title: 'Research sync', starts_at: at(2, 12), ends_at: at(2, 13), calendar_id: 'demo-calendar-primary', description: 'Share research findings and next steps.', location: 'Online', attendees: [], reminders: [30] },
      { id: 'demo-event-design', title: 'Design review', starts_at: at(3, 9), ends_at: at(3, 10), calendar_id: 'demo-calendar-primary', description: 'Review the approved product experience.', location: 'Project room', attendees: ['Alex Morgan'], reminders: [15] },
      { id: 'demo-event-catch-up', title: 'Team catch-up', starts_at: at(4, 10), ends_at: at(4, 11), calendar_id: 'demo-calendar-primary', description: 'Catch up on the team’s current priorities.', location: 'Online', attendees: [], reminders: [15] },
      { id: 'demo-event-roadmap', title: 'Agent roadmap', starts_at: at(4, 13), ends_at: at(4, 14), calendar_id: 'demo-calendar-primary', description: 'Review the upcoming agent roadmap.', location: 'Project room', attendees: [], reminders: [15] },
      { id: 'demo-event-weekly-review', title: 'Weekly review', starts_at: at(5, 14), ends_at: at(5, 15), calendar_id: 'demo-calendar-primary', description: 'Review this week’s progress and outcomes.', location: 'Online', attendees: [], reminders: [15] },
    ],
    providers: [], conflicts: [],
    workspaceItems: [
      { id: 'demo-note-1', kind: 'note', title: 'Launch checklist', content: 'Confirm the product review, update release notes, and share the final checklist.', metadata: { workspace_type: 'note' }, updated_at: now() },
      { id: 'demo-link-1', kind: 'note', title: 'React documentation', content: 'Reference documentation for the frontend workspace.', metadata: { workspace_type: 'link', url: 'https://react.dev/' }, updated_at: now() },
    ],
  };
  await demoProvider.mutate('workspaceDemo', { version: 1, state });
  return structuredClone(state);
}

async function save(state: WorkspaceState) { await demoProvider.mutate('workspaceDemo', { version: 2, state }); }

/** Restore only Knowledge demo records from the approved fixtures, leaving other workspaces untouched. */
export async function resetKnowledgeDemo(): Promise<void> {
  const state = await initialize();
  const knowledgeRows = knowledgeSeed(await demoProvider.collections());
  state.knowledgeDocuments = knowledgeRows.documents;
  state.knowledgeCollections = knowledgeRows.collections;
  await save(state);
  await demoProvider.resetActivity('knowledge');
}

function ok<T extends DemoRow>(data: T): T { return data; }
function parse(path: string) { return new URL(path, 'http://demo.local'); }

/** Local-only implementation of the workspace API shapes used by these screens. */
export async function workspaceDemoRequest<T extends DemoRow>(method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE', path: string, input?: unknown, signal?: AbortSignal): Promise<T> {
  if (signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError');
  const body: DemoRow = input instanceof FormData
    ? { filename: (input.get('file') as File | null)?.name ?? 'Uploaded file', title: (input.get('file') as File | null)?.name ?? 'Uploaded reference' }
    : input && typeof input === 'object' && !Array.isArray(input) ? input as DemoRow : {};
  const url = parse(path); const pathname = url.pathname; const parts = pathname.split('/').filter(Boolean).map(decodeURIComponent);
  const state = await initialize();
  const result = (value: DemoRow) => ok(value) as T;
  const filePath = url.searchParams.get('path') ?? String(body?.path ?? '');
  const docId = parts[parts.indexOf('document') + 1];

  if (pathname === '/api/codebase/projects' && method === 'GET') return result({ projects: state.codeProjects });
  if (pathname === '/api/codebase/projects' && method === 'POST') {
    const project = { id: uid('project'), name: String(body?.name ?? 'Demo project'), workspace: { workspace_path: 'demo://new-project' } };
    state.codeProjects.unshift(project); await save(state); return result({ project });
  }
  if (pathname === '/api/codebase/folder-picker') return result({ path: 'demo://selected-folder' });
  if (parts[0] === 'api' && parts[1] === 'codebase' && parts[2] === 'projects') {
    const projectId = parts[3];
    if (parts.length === 4 && method === 'GET') return result({ project: state.codeProjects.find(item => item.id === projectId) ?? state.codeProjects[0] ?? {} });
    if (parts[4] === 'tree') {
      const directories = new Set(['.next', 'public', 'src', 'src/app', 'src/components', 'src/hooks', 'src/styles', 'api']);
      for (const file of state.codeFiles) {
        const path = String(file.path ?? '');
        const segments = path.split('/');
        for (let index = 1; index < segments.length; index++) directories.add(segments.slice(0, index).join('/'));
      }
      const entries = [
        ...[...directories].map(path => ({ path, type: 'directory', size: 0 })),
        ...state.codeFiles.filter(file => file.type !== 'directory').map(({ path, type, size }) => ({ path, type, size })),
      ];
      return result({ entries });
    }
    if (parts[4] === 'file' && method === 'GET') {
      const file = state.codeFiles.find(item => item.path === filePath);
      return result({ file: file ?? null });
    }
    if (parts[4] === 'file' && method === 'POST') {
      let file = state.codeFiles.find(item => item.path === filePath);
      if (!file) { file = { id: uid('file'), path: filePath, name: filePath.split('/').pop(), type: 'file', content: '', size: 0, sha256: uid('hash') }; state.codeFiles.push(file); }
      file.content = String(body?.content ?? ''); file.size = String(file.content).length; file.sha256 = uid('hash'); await save(state);
      return result({ file, change: { new_sha256: file.sha256 } });
    }
    if (parts[4] === 'scan') return result({ project_map: { file_count: state.codeFiles.length } });
    if (parts[4] === 'paths') {
      const action = String(body?.action ?? ''); const target = String(body?.path ?? ''); const dest = String(body?.destination ?? '');
      const existing = state.codeFiles.findIndex(item => item.path === target);
      if (action === 'create_file' && existing < 0) state.codeFiles.push({ id: uid('file'), path: target, name: target.split('/').pop(), type: 'file', content: '', size: 0, sha256: uid('hash') });
      else if (action === 'create_directory' && existing < 0) state.codeFiles.push({ id: uid('folder'), path: target, name: target.split('/').pop(), type: 'directory', size: 0 });
      else if (action === 'delete' && existing >= 0) state.codeFiles.splice(existing, 1);
      else if ((action === 'rename' || action === 'move') && existing >= 0) state.codeFiles[existing].path = dest;
      await save(state); return result({ ok: true });
    }
    if (parts[4] === 'tools') {
      const tool = String(body?.tool ?? 'search'); const args = body?.arguments && typeof body.arguments === 'object' ? body.arguments as DemoRow : {};
      if (tool === 'grep') {
        const query = String(args.query ?? '').toLowerCase();
        const matches = state.codeFiles.flatMap(file => String(file.content ?? '').split('\n').flatMap((line, lineIndex) => query && line.toLowerCase().includes(query) ? [{ path: file.path, line: lineIndex + 1, preview: line.trim() }] : []));
        return result({ result: { matches, stdout: matches.length ? `${matches.length} local match(es) found.` : 'No local matches found.', exit_code: 0 } });
      }
      const output = tool === 'git_diff' ? 'Demo mode: no changes in the example workspace.' : 'a1b2c3d  Update product card\n8f7e6d5  Add cart hook';
      return result({ result: { stdout: output, exit_code: 0 } });
    }
  }
  if (pathname === '/api/codebase/preflight') return result({ preflight: { framework: { label: 'Next.js · React · TypeScript' }, missing_toolchains: [], ready: true } });
  if (pathname === '/api/terminal/runs' && method === 'GET') return result({ runs: state.terminalRuns });
  if (parts[0] === 'api' && parts[1] === 'terminal' && parts[2] === 'runs' && parts[3]) return result({ run: state.terminalRuns.find(item => item.id === parts[3]) ?? {} });
  if (pathname === '/api/terminal/stream') {
    const run = { id: uid('terminal'), status: 'completed', command: Array.isArray(body?.command) ? body.command.join(' ') : 'demo command', cwd: String(body?.cwd ?? 'demo://e-commerce-platform'), stdout: 'Demo mode: command preview completed locally.', stderr: '', exit_code: 0, events: [{ stream: 'stdout', chunk: 'Demo mode: command preview completed locally.' }], created_at: now() };
    state.terminalRuns.unshift(run); await save(state); return result({ run });
  }
  if (pathname === '/api/codebase/runs' && method === 'POST') {
    const run = { id: uid('code-run'), status: 'completed', prompt: String(body?.prompt ?? ''), summary: 'Demo run completed in the isolated CodeQ workspace.', created_at: now(), approvals: [], snapshots: [], events: [{ sequence: 1, type: 'run.completed', message: 'Demo changes are ready to review.' }] };
    state.codeRuns.unshift(run); await save(state); return result({ run });
  }
  if (parts[0] === 'api' && parts[1] === 'codebase' && parts[2] === 'runs' && parts[3]) {
    const run = state.codeRuns.find(item => item.id === parts[3]) ?? { id: parts[3], status: 'completed', approvals: [], snapshots: [] };
    if (parts[4] === 'cancel') run.status = 'cancelled'; if (parts[4] === 'resume') run.status = 'completed'; await save(state); return result({ run, restored: true });
  }
  if (pathname === '/api/codebase/approvals/demo/resolve') return result({ ok: true });
  if (pathname === '/api/research/sessions' && method === 'GET') return result({ sessions: state.researchSessions });
  if (pathname === '/api/research/sessions' && method === 'POST') {
    const id = uid('research'); const query = String(body?.query ?? '');
    const session = { id, query, status: 'completed', project: 'default', created_at: now(), completed_at: now(), summary: `A concise synthesis for “${query}” is ready.`, sources: [
      { id: `${id}-source-1`, title: 'Product research overview', url: 'https://example.test/research-overview', snippet: 'A deterministic reference source for the local demo.' },
      { id: `${id}-source-2`, title: 'Technical documentation', url: 'https://example.test/documentation', snippet: 'Supporting context for the research summary.' },
    ], events: [{ sequence: 1, type: 'research.completed', message: 'Report generated in demo mode.' }] };
    state.researchSessions.unshift(session); await save(state); return result({ session });
  }
  if (parts[0] === 'api' && parts[1] === 'research' && parts[2] === 'sessions' && parts[3]) {
    const session = state.researchSessions.find(item => item.id === parts[3]) ?? {};
    if (parts[4] === 'report') return result({ report: String(session.summary ?? session.report ?? 'Research report example'), title: String(session.query ?? 'Research report'), status: session.status ?? 'completed', sources: session.sources ?? [], session });
    if (parts[4] === 'checkpoints') return result({ checkpoints: [{ id: `${parts[3]}-checkpoint`, status: 'completed', message: 'Source review complete', created_at: session.created_at ?? now() }] });
    if (parts[4] === 'source-graph') return result({ nodes: [{ id: 'demo-entity-1', label: 'Product research', kind: 'topic' }, { id: 'demo-entity-2', label: 'Roadmap', kind: 'finding' }], edges: [{ from: 'Product research', to: 'Roadmap', relation: 'informs' }] });
    if (parts[4] === 'cancel') { session.status = 'cancelled'; await save(state); return result({ session }); }
    if (parts[4] === 'archive') session.status = 'archived';
    if (parts[4] === 'retry') session.status = 'completed';
    if (parts[4] === 'delete') { state.researchSessions = state.researchSessions.filter(item => item.id !== parts[3]); await save(state); return result({ deleted: true }); }
    if (parts[4] === 'branch') { const branch = { ...session, id: uid('research-branch'), status: 'completed', query: String(body?.query ?? session.query ?? '') }; state.researchSessions.unshift(branch); await save(state); return result({ session: branch }); }
    await save(state); return result({ session });
  }
  if (pathname === '/api/chat/research-record') return result({ conversation: { id: uid('conversation'), title: String(body?.query ?? 'Research notes') } });

  if (pathname === '/api/documents/library') return result({ items: state.files.filter(file => (url.searchParams.get('status') === 'archived') === (file.status === 'archived')).filter(file => !url.searchParams.get('q') || String(file.title).toLowerCase().includes(String(url.searchParams.get('q')).toLowerCase())), documents: state.files });
  if (pathname === '/api/document' && method === 'POST') {
    const doc = { id: uid('doc'), doc_id: uid('doc'), title: String(body?.title ?? 'Untitled'), content: String(body?.content ?? ''), format: body?.format ?? 'markdown', status: 'active', version: 1, updated_at: now(), project: 'default' };
    state.files.unshift(doc); await save(state); return result({ document: doc });
  }
  if (pathname === '/api/documents/import' || pathname === '/api/documents/import-pdf') {
    const doc = { id: uid('doc'), doc_id: uid('doc'), title: String(body?.title ?? body?.filename ?? 'Imported document'), content: 'Imported example content stored in the isolated demo workspace.', format: 'markdown', status: 'active', version: 1, updated_at: now(), project: 'default' };
    state.files.unshift(doc); await save(state); return result({ document: doc });
  }
  if (parts[0] === 'api' && parts[1] === 'document' && docId) {
    const index = state.files.findIndex(item => item.doc_id === docId || item.id === docId); const doc = state.files[index];
    if (parts[2] === 'versions') return result({ versions: doc ? [{ version: Number(doc.version ?? 1), title: doc.title, content: doc.content, created_at: doc.updated_at }] : [] });
    if (parts[2] === 'archive' && doc) { doc.status = body?.archived ? 'archived' : 'active'; await save(state); return result({ document: doc }); }
    if (parts[2] === 'restore' && doc) { doc.version = Number(parts[3] ?? doc.version); await save(state); return result({ document: doc }); }
    if (method === 'DELETE') { if (index >= 0) state.files.splice(index, 1); await save(state); return result({ deleted: true }); }
    if (method === 'PUT' && doc) { doc.title = String(body?.title ?? doc.title); doc.content = String(body?.content ?? doc.content); doc.version = Number(doc.version ?? 1) + 1; doc.updated_at = now(); await save(state); return result({ document: doc }); }
    return result({ document: doc ?? null });
  }
  if (pathname === '/api/media/assets') return result({ assets: state.assets });
  if (pathname === '/api/media/ingest') { const asset = { id: uid('asset'), title: String(body?.title ?? body?.filename ?? 'Uploaded file'), mime_type: 'application/octet-stream', size_bytes: 0, created_at: now(), media_type: 'file' }; state.assets.unshift(asset); await save(state); return result({ asset }); }
  if (pathname === '/api/workspace/items' && method === 'GET') return result({ items: state.workspaceItems });
  if (pathname === '/api/workspace/items' && method === 'POST') {
    const item = { ...body, id: uid('workspace-item'), updated_at: now() };
    state.workspaceItems.unshift(item); await save(state); return result({ item });
  }
  if (pathname === '/api/workspace/items/update') {
    const project = String(body?.project ?? 'default');
    const item = state.workspaceItems.find(candidate => candidate.id === body?.item_id && String(candidate.project ?? 'default') === project);
    if (item) { item.title = body?.title; item.content = body?.content; item.metadata = body?.metadata; item.updated_at = now(); }
    await save(state); return result({ item: item ?? null });
  }
  if (pathname.startsWith('/api/workspace/items/') && method === 'DELETE') {
    const itemId = parts[parts.indexOf('items') + 1];
    const itemIndex = state.workspaceItems.findIndex(candidate => String(candidate.id) === itemId);
    if (itemIndex < 0) return result({ error: 'Workspace item not found.' });
    if (state.workspaceItems.some(candidate => String(candidate.parent_id ?? '') === itemId)) return result({ error: 'Move or delete this item’s children before deleting it.' });
    state.workspaceItems.splice(itemIndex, 1); await save(state); return result({ deleted: true, item_id: itemId });
  }

  if (parts[0] === 'api' && parts[1] === 'knowledge' && parts[2] === 'documents' && parts[3] && method === 'PUT') {
    const document = state.knowledgeDocuments.find(item => String(item.id) === parts[3] && String(item.project ?? 'default') === String(body?.project ?? 'default'));
    if (!document) return result({ error: 'Knowledge source not found.' });
    const nextTitle = String(body?.title ?? '').trim();
    const nextContent = String(body?.content ?? '').trim();
    if (!nextTitle || !nextContent) return result({ error: 'A title and source text are required.' });
    document.title = nextTitle;
    document.content = nextContent;
    document.chunks = Math.max(1, Math.ceil(nextContent.length / 800));
    document.updated_at = now();
    await save(state);
    return result({ document });
  }
  if (pathname === '/api/knowledge/status') return result({ status: 'ready', documents: state.knowledgeDocuments.length, chunks: state.knowledgeDocuments.reduce((count, item) => count + Number(item.chunks ?? 1), 0) });
  if (pathname === '/api/knowledge/documents') return result({ documents: state.knowledgeDocuments });
  if (pathname === '/api/knowledge/tags') return result({ tags: [...new Set(state.knowledgeDocuments.flatMap(item => Array.isArray(item.tags) ? item.tags as string[] : []))].map((tag, index) => ({ id: `tag-${index}`, tag })) });
  if (pathname === '/api/knowledge/timeline') return result({ timeline: state.knowledgeDocuments.slice(0, 8).map(item => ({ source_id: item.id, title: item.title, type: 'indexed', timestamp: now() })) });
  if (pathname === '/api/knowledge/graph') return result({ entities: [{ id: 'entity-product', name: 'Product requirements', entity_type: 'document', source_ids: ['demo-knowledge-1'] }], relationships: [{ id: 'relation-product-roadmap', subject: 'Product requirements', predicate: 'informs', object: 'Roadmap', source_id: 'demo-knowledge-1', weight: 1 }] });
  if (pathname === '/api/rag/status') return result({ status: 'ready', documents: state.knowledgeDocuments.length });
  if (pathname === '/api/semantic/collections' && method === 'GET') return result({ collections: state.knowledgeCollections });
  if (pathname === '/api/knowledge/search' || pathname === '/api/knowledge/semantic-search' || pathname === '/api/knowledge/hybrid-search') {
    const query = String(body?.query ?? url.searchParams.get('q') ?? '').toLowerCase();
    const matches = state.knowledgeDocuments.filter(item => !query || `${item.title} ${item.content}`.toLowerCase().includes(query) || String(item.title).toLowerCase().includes(query.split(' ')[0] ?? ''));
    return result({ results: matches.map((item, index) => ({ ...item, score: Math.max(0.62, 0.96 - index * 0.08), snippet: item.content })) });
  }
  if (pathname === '/api/knowledge/synthesize') return result({ query: String(body?.query ?? ''), summary: `Demo synthesis: ${String(body?.query ?? 'your query')} is supported by the indexed project notes.`, sources: state.knowledgeDocuments.slice(0, 2).map(item => ({ id: item.id, title: item.title, content: item.content, score: 0.9 })) });
  if (pathname === '/api/knowledge/upload') {
    const doc = { id: uid('knowledge'), title: String(body.title ?? 'Uploaded reference'), content: 'Uploaded reference stored in the isolated demo workspace.', status: 'indexed', chunks: 1, tags: ['upload'], created_at: now() };
    state.knowledgeDocuments.unshift(doc); await save(state); return result({ document: doc });
  }
  if (pathname === '/api/knowledge/ingest') {
    const doc = { id: uid('knowledge'), title: String(body?.title ?? 'Untitled source'), content: String(body?.content ?? ''), status: 'indexed', chunks: 1, tags: ['demo'], created_at: now() };
    state.knowledgeDocuments.unshift(doc); await save(state); return result({ document: doc });
  }
  if (pathname === '/api/rag/sync') return result({ status: 'completed', state: 'completed', indexed: state.knowledgeDocuments.length });
  if (pathname === '/api/semantic/collections' && method === 'POST') { const collection = { id: uid('collection'), name: String(body?.name ?? 'Collection'), dimensions: 1536, distance: 'cosine', updated_at: now() }; state.knowledgeCollections.unshift(collection); await save(state); return result({ collection }); }

  if (pathname === '/api/calendar/calendars' && method === 'GET') return result({ calendars: state.calendarCalendars });
  if (pathname === '/api/calendar/events' && method === 'GET') {
    const from = new Date(url.searchParams.get('from') ?? 0).getTime();
    const to = new Date(url.searchParams.get('to') ?? 8640000000000000).getTime();
    const events = state.calendarEvents.flatMap(event => {
      if (url.searchParams.get('calendar_id') && event.calendar_id !== url.searchParams.get('calendar_id')) return [];
      const occurrences = calendarOccurrences(event, from, to);
      return occurrences.length
        ? [{ ...event, occurrences: String(event.rrule ?? '') ? occurrences : undefined }]
        : [];
    });
    return result({ events });
  }
  if (pathname === '/api/calendar/free-slots') return result({ slots: [{ starts_at: new Date(Date.now() + 86_400_000).toISOString(), duration_minutes: Number(url.searchParams.get('duration') ?? 60) }] });
  if (pathname === '/api/calendar/reminders') return result({ reminders: state.calendarEvents.slice(0, 2).map(event => ({ id: event.id, title: event.title, trigger_at: event.starts_at })) });
  if (pathname === '/api/calendar/providers') return result({ providers: state.providers });
  if (pathname === '/api/calendar/conflicts') return result({ conflicts: state.conflicts });
  if (pathname === '/api/calendar/optimize') return result({ total_events: state.calendarEvents.length, total_scheduled_minutes: state.calendarEvents.length * 60, suggestions: ['Your week has room for focused project work.'], conflicts: [] });
  if (pathname === '/api/calendar/events' && method === 'POST') { const event = { ...body, id: uid('event'), created_at: now() }; state.calendarEvents.push(event); await save(state); return result({ event }); }
  if (parts[0] === 'api' && parts[1] === 'calendar' && parts[2] === 'events' && parts[3] && (method === 'PATCH' || method === 'DELETE')) {
    const index = state.calendarEvents.findIndex(event => String(event.id) === parts[3] && (!url.searchParams.get('project') || String(event.project ?? 'default') === url.searchParams.get('project')));
    if (index < 0) throw new Error('Event not found.');
    if (method === 'DELETE') { const [deleted] = state.calendarEvents.splice(index, 1); await save(state); return result({ deleted: true, id: deleted.id }); }
    const current = state.calendarEvents[index];
    const updated: DemoRow = { ...current, ...body, id: current.id, updated_at: now() };
    if (!String(updated.title ?? '').trim()) throw new Error('Event title is required.');
    if (new Date(String(updated.ends_at)).getTime() <= new Date(String(updated.starts_at)).getTime()) throw new Error('Event end must be later than start.');
    state.calendarEvents[index] = updated; await save(state); return result({ event: updated });
  }
  if (pathname === '/api/calendar/calendars' && method === 'POST') { const calendar = { ...body, id: uid('calendar'), is_default: false }; state.calendarCalendars.push(calendar); await save(state); return result({ calendar }); }
  if (pathname === '/api/calendar/providers/register') {
    const supplied = body?.config && typeof body.config === 'object' ? body.config as DemoRow : {};
    const safeConfig = Object.fromEntries(Object.entries(supplied).filter(([key]) => !/password|token|secret|credential|ics_seed/i.test(key)));
    const provider = { ...body, config: safeConfig, provider_id: uid('provider'), status: 'ready' };
    state.providers.push(provider); await save(state); return result({ provider });
  }
  if (pathname === '/api/calendar/sync' && method === 'POST') return result({ sync_job_id: uid('sync'), state: 'completed', pulled: 0, pushed: 0, conflicts: 0, completed_at: now() });
  if (parts[0] === 'api' && parts[1] === 'calendar' && parts[2] === 'sync' && parts[3]) return result({ id: parts[3], state: 'completed', direction: 'demo', pulled: 0, pushed: 0, conflicts: 0, completed_at: now() });
  if (pathname === '/api/calendar/ical-export') return result({ ical: 'BEGIN:VCALENDAR\r\nVERSION:2.0\r\nEND:VCALENDAR\r\n' });
  if (pathname === '/api/calendar/ical-import') return result({ imported: 0 });
  if (parts[0] === 'api' && parts[1] === 'calendar' && parts[2] === 'conflicts' && parts[3]) return result({ resolved: true });

  return result({ status: 'demo', message: `No local demo handler is defined for ${method} ${pathname}.` });
}
