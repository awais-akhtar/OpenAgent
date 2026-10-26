import { afterEach, describe, expect, it, vi } from 'vitest';
import { advancedDemoAction, advancedDemoScreen } from '../src/demo/advanced';
import { demoProvider } from '../src/demo/provider';

const hash = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const fixtures = { sourceSha256: hash, data: { topLevel: {}, advanced: { voice: { recordings: [{ id: 'voice-1', name: 'Planning sync', status: 'Transcribed' }] }, vault: { secrets: [{ id: 'vault-1', name: 'Example credential', status: 'Active' }], requests: [], audit: [] } } } };
const shell = { sourceSha256: hash, data: {} };
const communicationFixtures = { sourceSha256: hash, data: { topLevel: {}, advanced: { communications: { messages: [
  { id: 'cm1', name: 'Updated release checklist', description: 'Maya Chen · Please review the new acceptance items before Friday.', status: 'Inbox', owner: 'Maya Chen', time: '10:42' },
  { id: 'cm2', name: 'Code review follow-up', description: 'Daniel Park · The file permissions need one more pass.', status: 'Inbox', owner: 'Daniel Park', time: 'Yesterday' },
  { id: 'cm3', name: 'Weekly progress summary', description: 'To team@demo.example · Ready for review.', status: 'Draft', owner: 'Team', time: '09:15' },
  { id: 'cm4', name: 'Design handoff notes', description: 'To product@demo.example · Latest interaction specification attached.', status: 'Sent', owner: 'Product', time: 'Monday' },
] } } } };

afterEach(() => { localStorage.removeItem('oa-second-isolated-demo-v1'); vi.restoreAllMocks(); });

describe('advanced offline demo adapter', () => {
  it('maps voice fixtures to the workspace view model without requesting live services', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = String(input);
      if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify(fixtures), { status: 200 });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify(shell), { status: 200 });
      throw new Error(`Unexpected demo network request: ${url}`);
    });
    await demoProvider.reset();
    const view = await advancedDemoScreen('voice', 'Recordings');
    expect((view.sessions.sessions as Array<Record<string, unknown>>)[0]).toMatchObject({ id: 'voice-1', mode: 'transcription', status: 'completed' });
    expect(view.status.status).toBe('demo');
  });

  it('does not persist entered vault secrets or passphrases in demo storage', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => String(input).endsWith('/demo/reference-fixtures.json')
      ? new Response(JSON.stringify(fixtures), { status: 200 })
      : new Response(JSON.stringify(shell), { status: 200 }));
    await demoProvider.reset();
    const result = await advancedDemoAction('/api/vault/store', { label: 'Temporary test', category: 'secret', secret: 'private-canary-value', passphrase: 'passphrase-canary-value' });
    expect(result.demo).toBe(true);
    expect(localStorage.getItem('oa-second-isolated-demo-v1')).not.toContain('private-canary-value');
    expect(localStorage.getItem('oa-second-isolated-demo-v1')).not.toContain('passphrase-canary-value');
    expect((await advancedDemoScreen('vault', 'Secrets')).entries.entries).toEqual(expect.arrayContaining([expect.objectContaining({ label: 'Temporary test' })]));
  });

  it('maps the approved Communications fixture records deterministically', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => String(input).endsWith('/demo/reference-fixtures.json')
      ? new Response(JSON.stringify(communicationFixtures), { status: 200 })
      : new Response(JSON.stringify(shell), { status: 200 }));
    await demoProvider.reset();
    const view = await advancedDemoScreen('communications', 'Inbox');
    const messages = view.messages.messages as Array<Record<string, unknown>>;
    expect(messages).toHaveLength(4);
    expect(messages.map(message => message.subject)).toEqual([
      'Updated release checklist', 'Code review follow-up', 'Weekly progress summary', 'Design handoff notes',
    ]);
    expect(messages[0]).toMatchObject({ sender: 'Maya Chen', body: 'Maya Chen · Please review the new acceptance items before Friday.', created_at: '10:42', status: 'Inbox' });
  });
});
