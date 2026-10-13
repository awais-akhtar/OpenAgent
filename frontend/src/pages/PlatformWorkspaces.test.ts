import { describe, expect, it } from 'vitest';
import { resourcePageConfigs } from './ResourcePages';
import { specialistFacts } from '../features/specialists/SpecialistInspector';

const ids = ['developer', 'engine', 'finetune', 'phd', 'science', 'scraper'] as const;
describe('dedicated platform workspaces', () => {
  it('covers the six inventory-driven surfaces and keeps their API contracts', () => {
    expect(ids).toHaveLength(6);
    for (const id of ids) {
      expect(resourcePageConfigs[id].reads.length).toBeGreaterThan(0);
      expect(resourcePageConfigs[id].reads.every(read => read.path.startsWith('/api/'))).toBe(true);
    }
    expect(resourcePageConfigs.developer.reads.map(x => x.path)).toContain('/api/developer/api-keys');
    expect(resourcePageConfigs.engine.reads.map(x => x.path)).toContain('/api/adaptive-engine/hardware?project={project}');
    expect(resourcePageConfigs.finetune.reads.map(x => x.path)).toContain('/api/finetune/jobs?project={project}');
    expect(resourcePageConfigs.phd.actions?.[1].payload({ command: 'draft' })).toMatchObject({ allow_external_model: false, allow_external_research: false });
    expect(resourcePageConfigs.science.actions?.[1].confirmText).toBeTruthy();
    expect(resourcePageConfigs.scraper.actions?.[1].payload({ url: 'https://example.org', allow_external: 'false' })).toMatchObject({ respect_robots: true, ingest: true, allow_external: false });
  });

  it('uses the allowlisted scalar summary rather than exposing arbitrary nested payloads', () => {
    expect(specialistFacts({ name: 'Local model', status: 'ready', settings: { password: 'secret' }, raw: { x: 1 } })).toEqual([['name', 'Local model'], ['status', 'ready']]);
  });
});
