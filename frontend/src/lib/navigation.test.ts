import { describe, expect, it } from 'vitest';
import { groups, isViewId, titles, views } from './navigation';

describe('navigation registry', () => {
  it('exposes every original route exactly once', () => {
    const expected = ['dashboard','chat','codebase','voice','research','phd','agents','knowledge','workspace','automation','assistant','communications','models','security','developer','settings','engine','science','calendar','twin','vault','scraper','finetune'];
    expect([...views.map(view => view.id)].sort()).toEqual([...expected].sort());
    expect(new Set(views.map(view => view.id)).size).toBe(23);
    expect(groups.flatMap(group => group.items)).toHaveLength(23);
  });

  it('validates deep-link IDs and provides a display title for each one', () => {
    for (const { id } of views) { expect(isViewId(id)).toBe(true); expect(titles[id]).toBeTruthy(); }
    expect(isViewId(null)).toBe(false);
    expect(isViewId('missing')).toBe(false);
  });
});
