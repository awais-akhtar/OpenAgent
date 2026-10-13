import { describe, expect, it } from 'vitest';
import { resourcePageConfigs } from '../../pages/ResourcePages';
import { specialistFacts } from './SpecialistInspector';

const specialistViews = [
  'agents', 'assistant', 'automation', 'calendar', 'communications', 'developer',
  'engine', 'finetune', 'phd', 'science', 'scraper', 'twin',
] as const;

describe('specialist workspace API parity', () => {
  it('keeps each dossier-backed specialist view reachable with its configured reads', () => {
    for (const id of specialistViews) {
      const page = resourcePageConfigs[id];
      expect(page, id).toBeDefined();
      expect(page.reads.length, id).toBeGreaterThan(0);
      expect(page.reads.every(read => read.path.startsWith('/api/')), id).toBe(true);
    }
  });

  it('uses the declared Python methods and paths for configured actions', () => {
    const expected: Record<string, Record<string, [string, string]>> = {
      agents: {
        'Run a checked agent': ['POST', '/api/agents/run-checked'],
        'Request an approval': ['POST', '/api/agents/approvals/request'],
        'Resolve an approval': ['POST', '/api/agents/approvals/resolve'],
        'Delegate a task': ['POST', '/api/agents/delegate'],
        'Grant scoped permission': ['POST', '/api/agents/permissions/grant'],
      },
      assistant: {
        'Add a task': ['POST', '/api/assistant/tasks'],
        'Complete a task': ['POST', '/api/assistant/tasks/complete'],
        'Add a goal': ['POST', '/api/assistant/goals'],
        'Create meeting brief': ['POST', '/api/assistant/meeting/brief'],
        'Add CRM relationship': ['POST', '/api/assistant/crm'],
      },
      automation: {
        'Create workflow': ['POST', '/api/workflows'],
        'Update workflow details': ['PUT', '/api/workflows/test-id'],
        'Run workflow in dry-run mode': ['POST', '/api/workflows/run'],
        'Run a workflow live': ['POST', '/api/workflows/run'],
      },
      calendar: {
        'Create a calendar event': ['POST', '/api/calendar/events'],
        'Create a calendar': ['POST', '/api/calendar/calendars'],
        'Find available time': ['GET', '/api/calendar/free-slots?from=from&to=to&duration=60&project=default'],
        'Resolve a calendar conflict': ['POST', '/api/calendar/conflicts/conflict-id/resolve'],
        'Sync a registered calendar': ['POST', '/api/calendar/sync'],
      },
      communications: {
        'Save a local draft': ['POST', '/api/communication/drafts'],
        'Request send approval': ['POST', '/api/communication/send/request-approval'],
        'Resolve send approval': ['POST', '/api/communication/send/resolve'],
        'Send an approved draft': ['POST', '/api/communication/send'],
      },
      developer: {
        'Register a webhook': ['POST', '/api/developer/webhooks'],
        'Create scoped API key': ['POST', '/api/developer/api-keys'],
        'Generate SDK artifact': ['POST', '/api/developer/sdk'],
      },
      engine: {
        'Inspect a local model file': ['POST', '/api/adaptive-engine/inspect'],
        'Benchmark a device': ['POST', '/api/adaptive-engine/benchmark'],
        'Compile a local model': ['POST', '/api/adaptive-engine/compile'],
      },
      finetune: {
        'Create dataset record': ['POST', '/api/finetune/datasets'],
        'Add training example': ['POST', '/api/finetune/examples'],
        'Create a training job': ['POST', '/api/finetune/jobs'],
        'Register adapter': ['POST', '/api/finetune/adapters'],
        'Activate a registered adapter': ['POST', '/api/finetune/adapters/activate'],
      },
      phd: {
        'Create an application record': ['POST', '/api/phd/applications'],
        'Start private draft generation': ['POST', '/api/phd/applications/application-id/generate/start'],
      },
      science: {
        'Add a scientific note': ['POST', '/api/science/memory'],
        'Analyze a catalog dataset': ['POST', '/api/science/materials/catalog/analyse'],
        'Plan a bounded science run': ['POST', '/api/science/run'],
      },
      scraper: {
        'Schedule a scrape': ['POST', '/api/scraper/schedule'],
        'Scrape this URL now': ['POST', '/api/scraper/scrape'],
        'Run due scheduled scrapes': ['POST', '/api/scraper/run-due'],
      },
      twin: {
        'Add a personal goal': ['POST', '/api/twin/goals'],
        'Predict a likely next intent': ['POST', '/api/twin/predict'],
        'Record an interaction': ['POST', '/api/twin/interaction'],
        'Consolidate context memory': ['POST', '/api/twin/consolidate'],
      },
    };

    for (const [screen, actions] of Object.entries(expected)) {
      for (const [label, [method, path]] of Object.entries(actions)) {
        const action = resourcePageConfigs[screen].actions?.find(item => item.label === label);
        expect(action, `${screen}: ${label}`).toBeDefined();
        const values = Object.fromEntries((action?.fields ?? []).map(field => [field.name, ({ workflow_id: 'test-id', application_id: 'application-id', conflict_id: 'conflict-id', from: 'from', to: 'to', duration: '60', steps: '[]' } as Record<string, string>)[field.name] ?? 'test-value']));
        expect(action?.method ?? 'POST', `${screen}: ${label} method`).toBe(method);
        expect(action?.path(values), `${screen}: ${label} path`).toBe(path);
        expect(action?.payload(values), `${screen}: ${label} payload`).toBeDefined();
      }
    }
  });

  it('retains project and server-side safety flags in sensitive action payloads', () => {
    const values = { draft_id: 'draft-1', approval_id: 'approval-1' };
    expect(resourcePageConfigs.communications.actions?.find(item => item.label === 'Send an approved draft')?.payload(values)).toMatchObject({
      project: 'default', draft_id: 'draft-1', approval_id: 'approval-1', allow_external: false,
    });
    expect(resourcePageConfigs.automation.actions?.find(item => item.label === 'Run a workflow live')?.payload({ workflow_id: 'wf' })).toMatchObject({
      workflow_id: 'wf', dry_run: false, allow_network: false,
    });
    expect(resourcePageConfigs.scraper.actions?.find(item => item.label === 'Scrape this URL now')?.payload({ url: 'https://example.test', allow_external: 'false' })).toMatchObject({
      allow_external: false, respect_robots: true, ingest: true,
    });
    expect(resourcePageConfigs.agents.actions?.find(item => item.label === 'Grant scoped permission')?.payload({ agent: 'a', scope: 'read' })).toMatchObject({ project: 'default', granted_by: 'user' });
  });
});

describe('specialist detail projection', () => {
  it('shows known semantic fields and omits raw nested payloads', () => {
    expect(specialistFacts({ name: 'Local worker', status: 'ready', private_config: { token: 'do-not-render' }, events: [{ id: 1 }] })).toEqual([
      ['name', 'Local worker'], ['status', 'ready'],
    ]);
  });

  it('keeps an honest empty response when no supported summary field exists', () => {
    expect(specialistFacts({ opaque_payload: { records: [1, 2] } })).toEqual([]);
  });
});
