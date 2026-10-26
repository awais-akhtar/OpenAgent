import { describe, expect, it } from 'vitest';
import { mergeAgentEventPage } from '../src/screens/OperationsScreens';

describe('recorded Agent event polling', () => {
  it('advances the cursor for a partial page and never duplicates repeated events', () => {
    const first = mergeAgentEventPage([], [{ id: 'one', event_type: 'started' }], '');
    expect(first.cursor).toBe('one');
    const repeated = mergeAgentEventPage(first.events, [{ id: 'one', event_type: 'started' }, { id: 'two', event_type: 'completed' }], first.cursor);
    expect(repeated.events.map(event => event.id)).toEqual(['one', 'two']);
    expect(repeated.cursor).toBe('two');
    expect(mergeAgentEventPage(repeated.events, [], repeated.cursor)).toEqual(repeated);
  });
  it('does not invent identifiers for malformed backend events', () => {
    expect(mergeAgentEventPage([], [{ event_type: 'unknown' }], '')).toEqual({ events: [], cursor: '' });
  });
});
