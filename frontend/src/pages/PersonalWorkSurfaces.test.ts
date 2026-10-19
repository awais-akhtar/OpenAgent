import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { actionBody, calendarEventStyle, calendarEventsPath, calendarWeekDays, calendarWeekLabel, defaultSectionIndex, factsOf, messageBodyOf, PersonalWorkSurface, projectScope, scopeReadPath, sortAgenda } from './PersonalWorkSurfaces';
import { resourcePageConfigs } from './ResourcePages';

const apiMocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn(), put: vi.fn(), patch: vi.fn(), delete: vi.fn() }));
vi.mock('../lib/react/api', () => ({
  api: apiMocks,
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
  records: (value: unknown, key?: string) => { const rows = Array.isArray(value) ? value : key && value && typeof value === 'object' ? (value as Record<string, unknown>)[key] : []; return Array.isArray(rows) ? rows.filter(row => row && typeof row === 'object') : []; },
}));

describe('personal AionUI work surfaces', () => {
  it('opens Calendar on Agenda and omits transport status flags from human-facing facts', () => {
    const calendarReads = resourcePageConfigs.calendar.reads;
    expect(calendarReads[defaultSectionIndex('calendar', calendarReads)].label).toBe('Events');
    expect(factsOf({ status: 'Success', success: true, ok: true, enabled: false, calendars: 2 })).toEqual([['calendars', 2]]);
  });
  it('places real timed events on the reference 11-hour calendar grid and keeps all-day events in its header lane', () => {
    expect(calendarEventStyle({ starts_at: '2026-10-14T10:30:00', ends_at: '2026-10-14T11:30:00' })).toEqual({ top: '165px', height: '64px' });
    expect(calendarEventStyle({ starts_at: '2026-10-14T10:30:00', all_day: true })).toBeUndefined();
  });
  it('opens Communications on the inbox and resolves full message content separately from its preview', () => {
    const reads = resourcePageConfigs.communications.reads;
    expect(reads[defaultSectionIndex('communications', reads)].label).toBe('Inbox');
    expect(messageBodyOf({ preview: 'Short preview', body: 'Full message\nwith a second paragraph.' })).toBe('Full message\nwith a second paragraph.');
    expect(messageBodyOf({ preview: 'Short preview' })).toBe('Short preview');
  });
  it('omits credentials and raw payloads from personal record summaries regardless of key casing', () => {
    expect(factsOf({ display_name: 'Avery', API_Key: 'one-time-value', refreshToken: 'refresh-value', Raw: 'private payload', city: 'York' })).toEqual([['display_name', 'Avery'], ['city', 'York']]);
  });
  it('sorts calendar items chronologically without mutating the API collection', () => {
    const later = { title: 'Later', starts_at: '2026-10-21T13:00:00Z' };
    const undated = { title: 'Unscheduled' };
    const earlier = { title: 'Earlier', starts_at: '2026-10-20T09:00:00Z' };
    const input = [later, undated, earlier];
    expect(sortAgenda(input)).toEqual([earlier, later, undated]);
    expect(input).toEqual([later, undated, earlier]);
  });
  it('builds a Monday-to-Sunday week range for the existing events API', () => {
    const anchor = new Date(2026, 9, 14, 12);
    const days = calendarWeekDays(anchor);
    expect(days).toHaveLength(7);
    expect(days[0].getDay()).toBe(1);
    expect(days[0].getDate()).toBe(12);
    expect(days[6].getDay()).toBe(0);
    expect(calendarWeekLabel(anchor)).toContain('12');
    expect(calendarWeekLabel(anchor)).toContain('18');
    expect(calendarWeekLabel(anchor)).toContain('2026');
    const request = new URL(calendarEventsPath('/api/calendar/events?project={project}&limit=100', 'team one', anchor), 'http://localhost');
    expect(request.pathname).toBe('/api/calendar/events');
    expect(request.searchParams.get('project')).toBe('team one');
    expect(request.searchParams.get('limit')).toBe('100');
    expect(request.searchParams.get('from')).toBe(days[0].toISOString());
    const end = new Date(days[6]); end.setHours(23, 59, 59, 999);
    expect(request.searchParams.get('to')).toBe(end.toISOString());
  });
  it('uses URL project scope before device preference and safely encodes it in resource reads', () => {
    expect(projectScope('http://localhost/?project=team%2Fone', 'other')).toBe('team/one');
    expect(projectScope('http://localhost/', 'other')).toBe('other');
    expect(scopeReadPath('/api/assistant/tasks?project={project}&limit=100', 'team/one')).toBe('/api/assistant/tasks?project=team%2Fone&limit=100');
  });
  it('builds only configured assistant, calendar and twin mutations', () => {
    const assistant = resourcePageConfigs.assistant.actions!;
    const task = assistant.find(a => a.label === 'Add a task')!;
    expect(task.path({})).toBe('/api/assistant/tasks');
    expect(actionBody(task, { title: 'Prepare', priority: '' })).toMatchObject({ title: 'Prepare', project: 'default' });
    const complete = assistant.find(a => a.label === 'Complete a task')!;
    expect(actionBody(complete, { task_id: 'task-alpha' })).toMatchObject({ task_id: 'task-alpha', project: 'default' });
    const calendar = resourcePageConfigs.calendar.actions!.find(a => a.label === 'Create a calendar event')!;
    expect(actionBody(calendar, { title: 'Review', starts_at: '2026-10-20T10:00', ends_at: '2026-10-20T11:00' })).toMatchObject({ title: 'Review', project: 'default' });
    const twin = resourcePageConfigs.twin.actions!.find(a => a.label === 'Add a personal goal')!;
    expect(twin.path({})).toBe('/api/twin/goals');
  });
  it('keeps communication delivery approval gated and external sending disabled', () => {
    const actions = resourcePageConfigs.communications.actions!;
    expect(actions.find(a => a.label === 'Request send approval')?.path({})).toBe('/api/communication/send/request-approval');
    const send = actions.find(a => a.label === 'Send an approved draft')!;
    expect(send.confirmText).toBeTruthy();
    expect(actionBody(send, { draft_id: 'd1', approval_id: 'a1' })).toMatchObject({ draft_id: 'd1', approval_id: 'a1', allow_external: false });
    expect(actionBody(actions.find(a => a.label === 'Save a local draft')!, { recipients: 'a@example.test', body: 'Draft' })).toMatchObject({ project: 'default' });
  });

  describe('communications inbox reading', () => {
    let host: HTMLDivElement;
    let root: Root;
    beforeEach(() => {
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
      localStorage.clear();
      history.replaceState({}, '', '/');
      apiMocks.get.mockReset().mockImplementation(async (path: string) => path.includes('/communication/inbox') ? ({ messages: [
        { id: 'm1', sender: 'Avery', subject: 'First update', preview: 'A short preview', body: 'Complete message content with\nline breaks.' },
        { id: 'm2', sender: 'Morgan', subject: 'Second update', preview: 'Another preview', body: 'The second full message.' },
      ] }) : {});
      host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    });
    afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

    it('shows the complete selected message and changes it when an inbox row is selected', async () => {
      await act(async () => root.render(createElement(PersonalWorkSurface, { id: 'communications' })));
      expect(host.querySelector('.pw-inbox')).toBeTruthy();
      expect(host.querySelector('.pw-message-detail')?.textContent).toContain('Complete message content with\nline breaks.');
      expect(host.querySelector('.pw-inbox')?.textContent).toContain('A short preview');
      const secondMessage = [...host.querySelectorAll<HTMLButtonElement>('.pw-inbox button')].find(button => button.textContent?.includes('Second update'));
      await act(async () => secondMessage?.click());
      expect(host.querySelector('.pw-message-detail')?.textContent).toContain('The second full message.');
    });

    it('shows real free-slot results returned by the Calendar API instead of a saved acknowledgement', async () => {
      apiMocks.get.mockImplementation(async (path: string) => path.includes('/api/calendar/free-slots')
        ? { slots: [{ start: '2026-10-20T09:00:00Z', end: '2026-10-20T10:00:00Z' }] }
        : {});
      await act(async () => root.render(createElement(PersonalWorkSurface, { id: 'calendar' })));
      const more = host.querySelector<HTMLDetailsElement>('.pw-more');
      await act(async () => more?.querySelector('summary')?.click());
      const find = [...host.querySelectorAll<HTMLButtonElement>('.pw-more button')].find(button => button.textContent === 'Find available time');
      await act(async () => find?.click());
      const inputs = [...document.querySelectorAll<HTMLInputElement>('.pw-dialog input')];
      const setterFor = (input: HTMLInputElement) => Object.getOwnPropertyDescriptor(input.constructor.prototype, 'value')?.set;
      await act(async () => {
        setterFor(inputs[0])?.call(inputs[0], '2026-10-20T08:00:00Z'); inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
        setterFor(inputs[1])?.call(inputs[1], '2026-10-20T18:00:00Z'); inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
      });
      const submit = [...document.querySelectorAll<HTMLButtonElement>('.pw-dialog button[type="submit"]')].at(-1);
      await act(async () => submit?.click());
      expect(apiMocks.get).toHaveBeenCalledWith(expect.stringContaining('/api/calendar/free-slots?from=2026-10-20T08%3A00%3A00Z&to=2026-10-20T18%3A00%3A00Z'));
      expect(host.querySelector('.pw-availability')?.textContent).toContain('Available times');
      expect(host.querySelector('.pw-availability')?.textContent).toContain('20 Oct 2026');
      expect(host.querySelector('.pw-feedback')?.textContent).toContain('Find available time loaded.');
      expect(host.querySelector('.pw-feedback')?.textContent).not.toContain('saved.');
    });
  });

  describe('calendar week navigation', () => {
    let host: HTMLDivElement;
    let root: Root;
    beforeEach(() => {
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
      localStorage.clear();
      history.replaceState({}, '', '/?project=calendar-project');
      vi.useFakeTimers();
      vi.setSystemTime(new Date(2026, 9, 14, 12));
      apiMocks.get.mockReset().mockImplementation(async (path: string) => {
        if (path.startsWith('/api/calendar/events')) {
          const request = new URL(path, location.origin);
          const start = new Date(request.searchParams.get('from') || '2026-10-12T00:00:00.000Z');
          const eventDate = new Date(start); eventDate.setUTCDate(eventDate.getUTCDate() + 2); eventDate.setUTCHours(13, 0, 0, 0);
          return { events: [{ id: 'event-week', title: 'Planning review', starts_at: eventDate.toISOString(), ends_at: new Date(eventDate.getTime() + 60 * 60 * 1000).toISOString(), location: 'Studio' }] };
        }
        if (path.startsWith('/api/calendar/calendars')) return { calendars: [] };
        return {};
      });
      host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    });
    afterEach(async () => { await act(async () => root.unmount()); host.remove(); vi.useRealTimers(); });

    it('shows seven real event days and reloads the backend for previous, next, and current weeks', async () => {
      await act(async () => root.render(createElement(PersonalWorkSurface, { id: 'calendar' })));
      expect(host.querySelectorAll('.pw-calendar-day')).toHaveLength(7);
      expect(host.textContent).toContain(calendarWeekLabel(new Date(2026, 9, 14, 12)));
      expect(host.textContent).toContain('Planning review');
      expect(host.textContent).toContain('Studio');
      const eventsCalls = () => apiMocks.get.mock.calls.map(([path]) => String(path)).filter(path => path.startsWith('/api/calendar/events'));
      expect(eventsCalls()).toHaveLength(1);
      const currentRange = new URL(eventsCalls()[0], location.origin).searchParams.get('from');
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Previous week"]')?.click());
      expect(eventsCalls()).toHaveLength(2);
      expect(new URL(eventsCalls()[1], location.origin).searchParams.get('from')).not.toBe(currentRange);
      expect(host.textContent).toContain(calendarWeekLabel(new Date(2026, 9, 7, 12)));
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Next week"]')?.click());
      expect(host.textContent).toContain(calendarWeekLabel(new Date(2026, 9, 14, 12)));
      await act(async () => host.querySelector<HTMLButtonElement>('[aria-label="Previous week"]')?.click());
      await act(async () => host.querySelector<HTMLButtonElement>('.pw-calendar-navigation button:not([aria-label])')?.click());
      expect(host.textContent).toContain(calendarWeekLabel(new Date(2026, 9, 14, 12)));
      const dayHeadings = [...host.querySelectorAll('.pw-calendar-day-heading')].map(day => day.textContent);
      expect(dayHeadings).toHaveLength(7);
      expect(host.querySelector('.pw-calendar-navigation')?.getAttribute('aria-label')).toBe('Calendar date navigation');
      expect(host.querySelector('.pw-calendar-toolbar small')?.textContent).toContain(Intl.DateTimeFormat().resolvedOptions().timeZone);
    });

    it('shows an empty week without fabricating events or changing the supported event action', async () => {
      apiMocks.get.mockImplementation(async (path: string) => path.startsWith('/api/calendar/events') ? { events: [] } : {});
      await act(async () => root.render(createElement(PersonalWorkSurface, { id: 'calendar' })));
      expect(host.textContent).toContain('No events this week.');
      expect(host.querySelectorAll('.pw-calendar-day')).toHaveLength(7);
      expect(host.querySelector('.pw-head .pw-primary')?.textContent).toBe('Schedule event');
      expect(apiMocks.post).not.toHaveBeenCalled();
    });
  });
});
