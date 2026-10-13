import { describe, expect, it } from 'vitest';
import { resourcePageConfigs } from './ResourcePages';

const resourceScreens = ['assistant', 'agents', 'automation', 'calendar', 'communications', 'developer', 'engine', 'finetune', 'science', 'phd', 'scraper', 'twin'];

describe('API-backed resource screens', () => {
  it('provides named reads for all twelve resource screens', () => {
    for (const screen of resourceScreens) {
      const page = resourcePageConfigs[screen];
      expect(page, screen).toBeDefined();
      expect(page.title.trim(), screen).not.toBe('');
      expect(page.reads.length, screen).toBeGreaterThan(0);
      for (const read of page.reads) expect(read.path, `${screen}: ${read.label}`).toMatch(/^\/api\//);
    }
  });

  it('keeps every form action on same-origin API paths with a server payload', () => {
    for (const screen of resourceScreens) {
      for (const action of resourcePageConfigs[screen].actions ?? []) {
        const values = Object.fromEntries(action.fields.map(field => [field.name, field.name === 'steps' ? '[]' : 'test-value']));
        expect(action.path(values), `${screen}: ${action.label}`).toMatch(/^\/api\//);
        expect(action.payload(values), `${screen}: ${action.label}`).toBeDefined();
        expect(['GET', 'POST', 'PUT', 'PATCH', 'DELETE']).toContain(action.method ?? 'POST');
      }
    }
  });

  it('keeps outbound delivery approval separate and requires explicit confirmation', () => {
    const send = resourcePageConfigs.communications.actions?.find(action => action.label === 'Send an approved draft');
    expect(send?.path({})).toBe('/api/communication/send');
    expect(send?.confirmText).toBeTruthy();
    expect(send?.payload({ draft_id: 'draft-1', approval_id: 'approval-1' })).toMatchObject({
      draft_id: 'draft-1', approval_id: 'approval-1', allow_external: false,
    });
  });

  it('keeps live workflow execution distinct from dry-run and confirmation-gated', () => {
    const dry = resourcePageConfigs.automation.actions?.find(action => action.label === 'Run workflow in dry-run mode');
    const live = resourcePageConfigs.automation.actions?.find(action => action.label === 'Run a workflow live');
    expect(dry?.payload({ workflow_id: 'wf' })).toMatchObject({ workflow_id: 'wf', dry_run: true, allow_network: false });
    expect(live?.payload({ workflow_id: 'wf' })).toMatchObject({ workflow_id: 'wf', dry_run: false, allow_network: false });
    expect(live?.confirmText).toBeTruthy();
  });
});
