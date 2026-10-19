import { afterEach, describe, expect, it, vi } from 'vitest';
import { demoProvider } from '../src/demo/provider';
import { workspaceDemoRequest } from '../src/demo/workspace';

const hash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const fixtureFile = { sourceSha256: hash, data: { topLevel: {}, advanced: { knowledge: { documents: [], collections: [] } }, coreFixtures: {} } };
const shellFile = { sourceSha256: hash, data: { files: [], codeFiles: [{ path: 'src/app.tsx', name: 'app.tsx' }, { path: 'src/components/ProductCard.tsx', name: 'ProductCard.tsx' }] } };

afterEach(() => {
  localStorage.removeItem('oa-second-isolated-demo-v1');
  vi.restoreAllMocks();
});

describe('workspace demo adapter', () => {
  it('keeps workspace reads and writes local while persisting supported demo operations', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = String(input);
      if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify(fixtureFile), { status: 200 });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify(shellFile), { status: 200 });
      throw new Error(`Unexpected network request in workspace demo: ${url}`);
    });
    await demoProvider.reset();

    const code = await workspaceDemoRequest('GET', '/api/codebase/projects?project=default');
    expect(code.projects).toEqual([expect.objectContaining({ id: 'demo-code-project' })]);
    const tree = await workspaceDemoRequest('GET', '/api/codebase/projects/demo-code-project/tree');
    expect((tree.entries as Array<Record<string, unknown>>).map(entry => entry.path)).toContain('src/components/ProductCard.tsx');
    const productCard = await workspaceDemoRequest('GET', '/api/codebase/projects/demo-code-project/file?path=src/components/ProductCard.tsx');
    expect((productCard.file as Record<string, unknown>).content).toContain('ProductCardProps');
    const saved = await workspaceDemoRequest('POST', '/api/codebase/projects/demo-code-project/file', { path: 'src/app.tsx', content: 'export const answer = 42;' });
    expect((saved.file as Record<string, unknown>).content).toBe('export const answer = 42;');

    const created = await workspaceDemoRequest('POST', '/api/document', { title: 'Local design notes', content: '# Notes' });
    const doc = created.document as Record<string, unknown>;
    const loaded = await workspaceDemoRequest('GET', `/api/document/${String(doc.doc_id ?? doc.id)}`);
    expect((loaded.document as Record<string, unknown>).title).toBe('Local design notes');

    const research = await workspaceDemoRequest('POST', '/api/research/sessions', { query: 'Local example question' });
    expect((research.session as Record<string, unknown>).status).toBe('completed');
    expect(fetchSpy).toHaveBeenCalledTimes(2);
    const persisted = localStorage.getItem('oa-second-isolated-demo-v1') ?? '';
    expect(persisted).toContain('Local design notes');
    expect(persisted).toContain('Local example question');
  });

  it('does not persist calendar provider credentials from demo forms', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => String(input).endsWith('/demo/reference-fixtures.json')
      ? new Response(JSON.stringify(fixtureFile), { status: 200 })
      : new Response(JSON.stringify(shellFile), { status: 200 }));
    await demoProvider.reset();

    await workspaceDemoRequest('POST', '/api/calendar/providers/register', {
      provider: 'caldav_http', account_ref: 'example',
      config: { username: 'demo-user', password: 'private-canary', auth_token: 'token-canary', base_url: 'https://calendar.example.test' },
    });

    const persisted = localStorage.getItem('oa-second-isolated-demo-v1') ?? '';
    expect(persisted).not.toContain('private-canary');
    expect(persisted).not.toContain('token-canary');
  });

  it('provides the approved deterministic Calendar week in local demo mode', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => String(input).endsWith('/demo/reference-fixtures.json')
      ? new Response(JSON.stringify(fixtureFile), { status: 200 })
      : new Response(JSON.stringify(shellFile), { status: 200 }));
    await demoProvider.reset();

    const response = await workspaceDemoRequest('GET', '/api/calendar/events');
    const events = response.events as Array<Record<string, unknown>>;
    expect(events.map(event => event.title)).toEqual([
      'Team stand-up', 'Project planning', 'Review PRs', 'Research sync',
      'Design review', 'Team catch-up', 'Agent roadmap', 'Weekly review',
    ]);
    expect(events.every(event => String(event.id).startsWith('demo-event-'))).toBe(true);
  });
});
