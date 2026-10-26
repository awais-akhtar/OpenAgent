import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { fireEvent, render, screen, waitFor, cleanup, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { HomeScreen, ChatScreen } from '../src/screens/HomeChat';
import { AgentsScreen, AutomationsScreen, ModelsScreen } from '../src/screens/OperationsScreens';
import { CalendarScreen, KnowledgeScreen } from '../src/screens/WorkspaceScreens';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

function renderWithQuery(ui: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

function json(data: unknown, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { 'Content-Type': 'application/json' } });
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); sessionStorage.clear(); window.history.replaceState({}, '', '/'); useWorkspaceState.getState().setDataMode('live'); });

describe('backend connected screens', () => {
  it('loads scoped audit activity on the Live Personal Assistant Today tab', async () => {
    const requests: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requests.push(url);
      if (url.includes('/api/assistant/tasks')) return json({ tasks: [] });
      if (url.includes('/api/assistant/daily-brief')) return json({ project: 'default', priority_count: 0, recommended_focus: [], open_tasks: [], active_goals: [] });
      if (url === '/api/audit?project=default&scope=assistant&limit=20') return json({ events: [{ id: 'assistant-audit-1', action: 'assistant.task.created', scope: 'assistant', created_at: '2026-10-25T09:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="assistant" />);

    expect(await screen.findByText('assistant.task.created')).toBeVisible();
    expect(requests).toContain('/api/audit?project=default&scope=assistant&limit=20');
    expect(requests.some(url => url === '/api/audit' || url.startsWith('/api/audit?project=default&limit='))).toBe(false);
  });

  it('uses checked agent execution, scoped access records, approvals and delegation contracts', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    let approvals: Array<Record<string, unknown>> = [];
    let agentJobs: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET has no body. */ }
      requests.push({ url, method, body });
      if (url.includes('/api/agents/permissions') && method === 'GET') return json({ permissions: [{ id: 'perm-1', project: 'default', agent_id: 'executive', scope: 'agent.run', granted_by: 'system' }] });
      if (url.includes('/api/agents/approvals') && method === 'GET') return json({ approvals });
      if (url.includes('/api/agents/jobs') && method === 'GET') return json({ jobs: agentJobs });
      if (url.includes('/api/agents') && method === 'GET') return json({ agents: [{ id: 'executive', name: 'Executive' }, { id: 'research', name: 'Research' }] });
      if (url.includes('/api/agent-runs')) return json({ runs: [] });
      if (url.endsWith('/api/agents/approvals/request')) {
        const approval = { id: 'approval-1', project: 'default', requested_by: 'user', action: body.action, reason: body.reason, status: 'pending', payload: body.payload };
        approvals = [approval, ...approvals];
        return json({ approval }, 201);
      }
      if (url.endsWith('/api/agents/approvals/resolve')) {
        approvals = approvals.map(item => item.id === body.approval_id ? { ...item, status: body.approved ? 'approved' : 'rejected' } : item);
        return json({ approval: approvals[0] });
      }
      if (url.endsWith('/api/agents/permissions/grant')) return json({ permission: { agent_id: body.agent, scope: body.scope, project: body.project, granted_by: body.granted_by } }, 201);
      if (url.endsWith('/api/agents/jobs') && method === 'POST') { agentJobs = [{ id: 'job-1', agent: body.agent, goal: body.goal, status: 'completed', progress: 100, current_step: 'Completed', result: { run: { id: 'run-1', agent: body.agent, goal: body.goal, status: 'completed', response: 'Backend result' }, verification: { status: 'passed', findings: [] } } }]; return json({ job: agentJobs[0] }, 202); }
      if (url.endsWith('/api/agents/delegate')) return json({ id: 'delegation-1', result: { run: { response: 'Delegated backend result' } } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AgentsScreen />);
    fireEvent.click(screen.getByText('Run or delegate'));
    fireEvent.change(screen.getByRole('textbox', { name: 'Task goal' }), { target: { value: 'Review the release plan' } });
    fireEvent.click(screen.getByRole('button', { name: 'Request approval' }));
    await screen.findByText(/Approval approval-1: pending/);
    fireEvent.click(screen.getByRole('button', { name: 'Access & approvals' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Load task' }));
    fireEvent.click(screen.getByText('Run or delegate'));
    expect(screen.getByRole('textbox', { name: 'Task goal' })).toHaveValue('Review the release plan');
    fireEvent.click(screen.getByRole('button', { name: 'Access & approvals' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Approve' }));
    await screen.findAllByText('Backend approval status: approved.');
    fireEvent.click(await screen.findByRole('button', { name: 'Run approved task' }));
    fireEvent.click(screen.getByRole('button', { name: 'Run with approval' }));
    await screen.findByText('Backend result');
    const checkedRun = requests.find(request => request.url.endsWith('/api/agents/jobs') && request.method === 'POST');
    expect(checkedRun?.body).toMatchObject({ agent: 'executive', goal: 'Review the release plan', project: 'default', require_approval: true, approval_id: 'approval-1' });
    expect(requests.some(request => request.url.endsWith('/api/agents/run'))).toBe(false);

    fireEvent.click(screen.getByRole('button', { name: 'Access & approvals' }));
    fireEvent.change(await screen.findByRole('combobox', { name: 'Permission scope' }), { target: { value: 'memory.read' } });
    fireEvent.click(screen.getByRole('button', { name: 'Grant permission' }));
    await screen.findByText(/Backend permission recorded: executive · memory.read/);
    const grant = requests.find(request => request.url.endsWith('/api/agents/permissions/grant'));
    expect(grant?.body).toMatchObject({ agent: 'executive', scope: 'memory.read', project: 'default', granted_by: 'user' });

    fireEvent.click(screen.getByRole('button', { name: 'Workspace' }));
    fireEvent.click(screen.getByText('Run or delegate'));
    fireEvent.click(screen.getByRole('button', { name: 'Delegate' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Task goal' }), { target: { value: 'Research the release risks' } });
    fireEvent.change(screen.getByRole('combobox', { name: 'Target agent' }), { target: { value: 'research' } });
    fireEvent.click(screen.getByRole('button', { name: 'Delegate task' }));
    await screen.findByText(/delegation-1/);
    expect(requests.find(request => request.url.endsWith('/api/agents/delegate'))?.body).toMatchObject({ from_agent: 'executive', to_agent: 'research', goal: 'Research the release risks', project: 'default' });
  });

  it('loads project-scoped Home data without requesting agent history', async () => {
    const requests: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requests.push(url);
      if (url.includes('/api/codebase/projects')) return json({ projects: [{ id: 'project-1', name: 'Atlas workspace', path: 'D:/atlas' }] });
      if (url.includes('/api/conversations')) return json({ conversations: [{ id: 'conversation-1', title: 'Planning notes', updated_at: '2026-10-14T09:00:00Z' }] });
      if (url.includes('/api/agents')) return json({ agents: [{ id: 'executive', name: 'Executive agent', description: 'Plans work' }] });
      if (url.includes('/api/agent-runs')) return json({ runs: [{ id: 'run-home-1', agent: 'executive', goal: 'Review launch plan', created_at: '2026-10-14T10:00:00Z' }] });
      if (url.includes('/api/events')) return json({ events: [{ id: 'event-home-1', type: 'note', title: 'Reviewed milestones', created_at: '2026-10-14T08:00:00Z' }] });
      if (url.includes('/api/assistant/tasks')) return json({ tasks: [{ id: 'task-home-1', title: 'Prepare agenda', status: 'open', due_at: '2026-10-14T13:00:00Z' }] });
      if (url.includes('/api/calendar/events')) return json({ events: [{ id: 'calendar-home-1', title: 'Design review', starts_at: '2026-10-14T15:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<HomeScreen />);
    expect(await screen.findByText('Atlas workspace')).toBeVisible();
    expect(await screen.findByText('Executive agent')).toBeVisible();
    expect(await screen.findByText('Prepare agenda')).toBeVisible();
    expect(await screen.findByText('Design review')).toBeVisible();
    expect(await screen.findByText('Planning notes')).toBeVisible();
    expect(await screen.findByText('Reviewed milestones')).toBeVisible();
    expect(screen.getByText(/Agent run history is shown in Agents when this account has the explicit agents:read scope for the active project/i)).toBeVisible();
    expect(screen.queryByText('Review launch plan')).not.toBeInTheDocument();
    expect(requests.some(url => url.includes('/api/agent-runs'))).toBe(false);
  });

  it('posts a selected Image Studio upscale to the backend global upscale route', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* multipart upload is outside this test. */ }
      requests.push({ url, method, body });
      if (url.includes('/api/gallery/library')) return json({ assets: [{ id: 'image-1', filename: 'Source image' }] });
      if (url.includes('/api/gallery/albums')) return json({ albums: [] });
      if (url.endsWith('/api/gallery/upscale') && method === 'POST') return json({ asset: { id: 'image-1-upscaled' } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="gallery" />);
    fireEvent.click(await screen.findByRole('button', { name: /Source image/ }));
    fireEvent.click(screen.getByRole('tab', { name: 'Operations' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Upscale' }));

    await waitFor(() => expect(requests.some(request => request.url === '/api/gallery/upscale' && request.method === 'POST')).toBe(true));
    expect(requests.find(request => request.url === '/api/gallery/upscale')?.body).toMatchObject({ asset_id: 'image-1', project: 'default', scale: 2 });
    expect(requests.some(request => request.url.includes('/api/gallery/image-1/upscale'))).toBe(false);
  });

  it('uploads Image Studio files using the backend JSON/base64 contract', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    let uploaded = false;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* no request body */ }
      requests.push({ url, method, body });
      if (url.includes('/api/gallery/library')) return json({ assets: uploaded ? [{ id: 'uploaded-1', filename: 'sample.png' }] : [] });
      if (url.includes('/api/gallery/albums')) return json({ albums: [] });
      if (url.endsWith('/api/gallery/upload') && method === 'POST') { uploaded = true; return json({ asset: { id: 'uploaded-1', filename: body.filename } }, 201); }
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="gallery" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Upload' }));
    const image = new File([new Uint8Array([137, 80, 78, 71])], 'sample.png', { type: 'image/png' });
    fireEvent.change(screen.getByLabelText('Select image file'), { target: { files: [image] } });
    fireEvent.click(screen.getByRole('button', { name: 'Upload' }));

    await waitFor(() => expect(requests.some(request => request.url === '/api/gallery/upload' && request.method === 'POST')).toBe(true));
    expect(requests.find(request => request.url === '/api/gallery/upload')?.body).toMatchObject({
      filename: 'sample.png', content_type: 'image/png', content_base64: 'iVBORw==', project: 'default',
    });
    fireEvent.click(screen.getByRole('tab', { name: 'Library' }));
    expect(await screen.findByRole('button', { name: /sample\.png/ })).toBeVisible();
  });

  it('persists Personal OS notes using the backend body field and renders returned notes', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    let notes: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* no request body */ }
      requests.push({ url, method, body });
      if (url.includes('/api/personal-os/status')) return json({ status: 'ready', note_count: notes.length });
      if (url.includes('/api/personal-os/presets')) return json({ presets: [] });
      if (url.includes('/api/personal-os/notes') && method === 'GET') return json({ notes });
      if (url.includes('/api/personal-os/notes') && method === 'POST') {
        notes = [{ id: 'note-1', title: String(body.title), body: String(body.body) }];
        return json({ note: notes[0] }, 201);
      }
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="personalos" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Notes' }));
    fireEvent.change(await screen.findByPlaceholderText('Write a note…'), { target: { value: 'Keep the weekly review concise.' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save note' }));

    expect(await screen.findByText('Keep the weekly review concise.')).toBeVisible();
    expect(requests.find(request => request.url === '/api/personal-os/notes' && request.method === 'POST')?.body).toMatchObject({ title: 'Personal note', body: 'Keep the weekly review concise.', project: 'default' });
  });

  it('creates and closes a Model Cookbook serve plan with persisted session logs', async () => {
    const requests: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    let hosts: Array<Record<string, unknown>> = [];
    let sessions: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* no request body */ }
      requests.push({ url, method, body });
      if (url.includes('/api/model-cookbook/hosts') && method === 'GET') return json({ hosts });
      if (url.includes('/api/model-cookbook/sessions')) return json({ sessions });
      if (url.includes('/api/model-cookbook/hardware-fit')) return json({ recommended_bits: 4 });
      if (url.includes('/api/model-cookbook/hosts') && method === 'POST') {
        hosts = [{ id: 'host-1', label: String(body.label), address: String(body.address), host_type: body.host_type }];
        return json({ host: hosts[0] }, 201);
      }
      if (url.endsWith('/api/model-cookbook/serve')) {
        sessions = [{ id: 'session-1', host_id: String(body.host_id), model_ref: String(body.model_ref), runner: String(body.runner), status: 'planned', mode: 'plan' }];
        return json({ session: sessions[0] }, 201);
      }
      if (url.endsWith('/api/model-cookbook/stop/session-1')) {
        sessions = sessions.map(session => ({ ...session, status: 'stopped' }));
        return json({ session: sessions[0] });
      }
      if (url.endsWith('/api/model-cookbook/logs/session-1')) return json({ logs: [{ id: 'log-1', message: 'Serve plan created' }] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="cookbook" />);
    fireEvent.change(screen.getByPlaceholderText('Host name'), { target: { value: 'Local host' } });
    fireEvent.change(screen.getByPlaceholderText('Host address'), { target: { value: '127.0.0.1' } });
    fireEvent.click(screen.getByRole('button', { name: 'Add' }));
    expect((await screen.findAllByText('Local host', { exact: true })).length).toBeGreaterThan(0);
    fireEvent.change(screen.getByRole('textbox', { name: 'Model reference to serve' }), { target: { value: 'small-model' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create serve plan' }));
    expect(await screen.findByText(/No process was launched/)).toBeVisible();
    expect(requests.find(request => request.url.endsWith('/api/model-cookbook/hosts') && request.method === 'POST')?.body).toMatchObject({ label: 'Local host', address: '127.0.0.1', project: 'default' });
    expect(requests.find(request => request.url.endsWith('/api/model-cookbook/serve'))?.body).toMatchObject({ host_id: 'host-1', model_ref: 'small-model', runner: 'ollama', allow_process: false });
    fireEvent.click(await screen.findByRole('button', { name: 'Close plan' }));
    await waitFor(() => expect(requests.some(request => request.url.endsWith('/api/model-cookbook/stop/session-1'))).toBe(true));
    fireEvent.click(screen.getByRole('tab', { name: 'Logs' }));
    expect(await screen.findByText('Serve plan created')).toBeVisible();
  });

  it('filters calendar events with the selected backend calendar', async () => {
    const eventUrls: string[] = [];
    const freeSlotUrls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [{ id: 'team-calendar', name: 'Team schedule' }, { id: 'personal-calendar', name: 'Personal' }] });
      if (url.includes('/api/calendar/free-slots')) { freeSlotUrls.push(url); return json({ slots: [{ starts_at: '2026-10-14T14:00:00Z', ends_at: '2026-10-14T15:00:00Z', duration_minutes: 60 }] }); }
      if (url.includes('/api/calendar/events')) { eventUrls.push(url); return json({ events: [{ id: 'event-team', title: 'Team planning', starts_at: '2026-10-14T15:00:00Z' }] }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<CalendarScreen />);
    expect(await screen.findByRole('option', { name: 'Team schedule' })).toBeInTheDocument();
    fireEvent.change(screen.getByRole('combobox', { name: 'Filter calendar' }), { target: { value: 'team-calendar' } });
    await waitFor(() => expect(eventUrls.some(url => url.includes('calendar_id=team-calendar'))).toBe(true));
    expect((await screen.findAllByText('Team planning')).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('button', { name: 'Find slots' }));
    expect(await screen.findByText(/backend suggested/)).toBeInTheDocument();
    expect(freeSlotUrls.some(url => url.includes('duration=60') && url.includes('calendar_id=team-calendar'))).toBe(true);
  });

  it('runs Calendar assistant suggestions against existing backend analysis endpoints without writing events', async () => {
    const requests: Array<{ url: string; method: string }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, method: String(init?.method ?? 'GET') });
      if (url.includes('/api/calendar/calendars')) return json({ calendars: [{ id: 'personal', name: 'Personal' }] });
      if (url.includes('/api/calendar/events')) return json({ events: [] });
      if (url.includes('/api/calendar/free-slots')) return json({ slots: [{ starts_at: '2026-10-26T09:00:00Z', ends_at: '2026-10-26T10:00:00Z', duration_minutes: 60 }] });
      if (url.includes('/api/calendar/optimize')) return json({ total_events: 3, total_scheduled_minutes: 180, conflicts: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<CalendarScreen />);
    fireEvent.click(await screen.findByRole('button', { name: 'Find time for a team sync' }));
    expect(await screen.findByText(/Calendar backend found 1 available time/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Move a meeting to next week' }));
    await waitFor(() => expect(requests.filter(request => request.url.includes('/api/calendar/free-slots'))).toHaveLength(2));
    fireEvent.click(screen.getByRole('button', { name: 'Summarise my week' }));
    expect(await screen.findByText(/Calendar backend reviewed 3 events and found 0 conflicts/)).toBeVisible();

    const availabilityCalls = requests.filter(request => request.url.includes('/api/calendar/free-slots'));
    expect(availabilityCalls).toHaveLength(2);
    const firstRange = new URL(availabilityCalls[0].url, 'http://localhost');
    const nextRange = new URL(availabilityCalls[1].url, 'http://localhost');
    expect(Date.parse(nextRange.searchParams.get('from') ?? '') - Date.parse(firstRange.searchParams.get('from') ?? '')).toBe(7 * 86_400_000);
    expect(requests.some(request => request.url.includes('/api/calendar/optimize'))).toBe(true);
    expect(requests.some(request => request.url.includes('/api/calendar/events') && request.method === 'POST')).toBe(false);
  });

  it('loads providers and streams chat, persisting the backend-created conversation ID', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [{ id: 'auto', label: 'Automatic routing', available: true }] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/messages')) return json({ messages: [] });
      if (url.includes('/api/chat/stream')) {
        const frames = [
          { type: 'conversation', conversation: { id: 'conversation-created', title: 'Hello' } },
          { type: 'status', status: 'started' },
          { type: 'chunk', content: 'A real ' },
          { type: 'chunk', content: 'response.' },
          { type: 'message', message: { content: 'A real response.' }, done: true },
        ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(frames)); controller.close(); } }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<ChatScreen />);
    fireEvent.click(screen.getByRole('button', { name: /Select model/ }));
    expect(await screen.findByRole('option', { name: 'Automatic routing' })).toBeVisible();
    fireEvent.change(screen.getByRole('textbox', { name: 'Message OpenAgent' }), { target: { value: 'Hello backend' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
    expect(await screen.findByText('A real response.')).toBeVisible();
    await waitFor(() => expect(sessionStorage.getItem('oa-second-conversation-id')).toBe('conversation-created'));
    const streamRequest = requests.find(request => request.url.includes('/api/chat/stream'));
    expect(streamRequest).toBeDefined();
    expect(JSON.parse(String(streamRequest?.init?.body))).toMatchObject({ message: 'Hello backend', provider: 'auto', allow_external: false });
  });

  it('ingests selected files and sends the returned asset metadata with the chat stream', async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); requests.push({ url, init });
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/media/ingest')) return json({ asset: { id: 'asset-123', title: 'notes.txt', media_type: 'document', mime_type: 'text/plain', size_bytes: 5 } }, 201);
      if (url.includes('/api/chat/stream')) {
        const frames = [
          { type: 'conversation', conversation: { id: 'conversation-attachment' } },
          { type: 'message', message: { content: 'Read the attached notes.' } },
        ].map(event => `data: ${JSON.stringify(event)}\n\n`).join('');
        return new Response(new ReadableStream({ start(controller) { controller.enqueue(new TextEncoder().encode(frames)); controller.close(); } }), { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
      }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<ChatScreen />);
    const file = new File(['hello'], 'notes.txt', { type: 'text/plain' });
    fireEvent.change(screen.getByLabelText('Choose files to attach'), { target: { files: [file] } });
    expect(await screen.findByTestId('chat-attachment')).toHaveTextContent('notes.txt');
    fireEvent.click(screen.getByRole('button', { name: 'Remove notes.txt' }));
    expect(screen.queryByTestId('chat-attachment')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Choose files to attach'), { target: { files: [file] } });
    expect(await screen.findByTestId('chat-attachment')).toHaveTextContent('notes.txt');
    const uploadRequest = requests.find(request => request.url.includes('/api/media/ingest'));
    const uploadBody = JSON.parse(String(uploadRequest?.init?.body));
    expect(uploadBody).toMatchObject({ filename: 'notes.txt', project: 'default', source: 'chat' });
    expect(uploadBody.data_base64).toBeTruthy();

    fireEvent.change(screen.getByRole('textbox', { name: 'Message OpenAgent' }), { target: { value: 'Read this file' } });
    fireEvent.click(screen.getByRole('button', { name: 'Send message' }));
    expect(await screen.findByText('Read the attached notes.')).toBeVisible();
    const streamRequest = requests.find(request => request.url.includes('/api/chat/stream'));
    expect(JSON.parse(String(streamRequest?.init?.body))).toMatchObject({ attachments: [{ id: 'asset-123', name: 'notes.txt', media_type: 'document', mime_type: 'text/plain' }] });
  });

  it('shows media ingest errors without adding a false attachment chip', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/models')) return json({ providers: [] });
      if (url.includes('/api/conversations')) return json({ conversations: [] });
      if (url.includes('/api/media/ingest')) return json({ error: 'Upload rejected by media storage.' }, 400);
      return json({ error: `Unexpected request ${url} ${init?.method ?? ''}` }, 404);
    }));
    renderWithQuery(<ChatScreen />);
    fireEvent.change(screen.getByLabelText('Choose files to attach'), { target: { files: [new File(['no'], 'blocked.txt', { type: 'text/plain' })] } });
    expect(await screen.findByRole('alert')).toHaveTextContent('Upload rejected by media storage.');
    expect(screen.queryByTestId('chat-attachment')).not.toBeInTheDocument();
  });

  it('loads agent profiles and submits an actual goal to the run endpoint', async () => {
    let posted: unknown;
    let jobs: Record<string, unknown>[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/agents/jobs') && init?.method === 'POST') { posted = JSON.parse(String(init?.body)); jobs = [{ id: 'job-1', agent: 'executive', goal: 'Review this task', status: 'completed', progress: 100, current_step: 'Completed', result: { run: { id: 'run-1', agent: 'executive', goal: 'Review this task', response: 'Backend agent result' } } }]; return json({ job: jobs[0] }, 202); }
      if (url.includes('/api/agents/jobs')) return json({ jobs });
      if (url.includes('/api/agents')) return json({ agents: [{ id: 'executive', name: 'Executive', description: 'General agent' }] });
      if (url.includes('/api/agent-runs')) return json({ runs: [{ id: 'older-run', agent: 'executive', goal: 'Earlier request', response: 'Older result', status: 'completed' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AgentsScreen />);
    fireEvent.change(await screen.findByRole('textbox', { name: 'Task goal' }), { target: { value: 'Review this task' } });
    fireEvent.click(screen.getByRole('button', { name: 'Run agent' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Tasks' }));
    expect(await screen.findByText('Backend agent result')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Review this task' })).toBeInTheDocument();
    expect(screen.queryByText('Older result')).not.toBeInTheDocument();
    expect(posted).toMatchObject({ agent: 'executive', goal: 'Review this task', project: 'default' });
  });

  it('loads and creates a valid automation then runs it through backend routes', async () => {
    const calls: Array<{ url: string; body?: unknown }> = [];
    let workflows: unknown[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) : undefined; calls.push({ url, body });
      if (url.endsWith('/api/workflows') && init?.method === 'POST') { workflows = [{ id: 'wf-1', name: (body as Record<string, unknown>).name, description: (body as Record<string, unknown>).description, steps: (body as Record<string, unknown>).steps, trigger_type: 'manual' }]; return json({ workflow: workflows[0] }, 201); }
      if (url.endsWith('/api/workflows/builder')) return json({ actions: { add_event: { label: 'Add event', category: 'local' } } });
      if (url.includes('/api/workflows/wf-1/run')) return json({ run: { id: 'wf-run-1', workflow_id: 'wf-1', status: 'completed', result: { dry_run: (body as Record<string, unknown>).dry_run === true } } }, 201);
      if (url.includes('/api/workflows?')) return json({ workflows });
      if (url.includes('/api/workflow-runs')) return json({ runs: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AutomationsScreen />);
    fireEvent.click(await screen.findByRole('button', { name: 'New' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: 'Daily review' } });
    fireEvent.change(await screen.findByRole('combobox', { name: 'New step 1 action' }), { target: { value: 'add_event' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Title' }), { target: { value: 'Daily review' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'New step 1 Content' }), { target: { value: 'Review the daily queue' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save automation' }));
    expect((await screen.findAllByText('Daily review')).length).toBeGreaterThan(0);
    fireEvent.click(await screen.findByRole('button', { name: 'Preview run' }));
    expect(await screen.findByText('Preview result')).toBeVisible();
    expect(calls.find(call => call.url.includes('/api/workflows/wf-1/run'))?.body).toMatchObject({ dry_run: true, allow_network: false });
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Confirm live run' }));
    await waitFor(() => expect(calls.filter(call => call.url.includes('/api/workflows/wf-1/run')).at(-1)?.body).toMatchObject({ dry_run: false, allow_network: false }));
    expect(calls.find(call => call.url.endsWith('/api/workflows') && call.body)?.body).toMatchObject({ trigger_type: 'manual', steps: [{ action: 'add_event', title: 'Daily review', content: 'Review the daily queue' }] });
    expect(calls.some(call => call.url.includes('/api/workflows/wf-1/run'))).toBe(true);
  });

  it('loads actual provider and local model records for the model filters', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'ollama', label: 'Ollama', local: true, available: true }] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [{ id: 'registered-1', name: 'Local test model', provider: 'ollama', status: 'ready' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<ModelsScreen />);
    expect(await screen.findByText('Local test model')).toBeVisible();
    expect(screen.getAllByText('Ollama').some(element => element.checkVisibility?.() ?? true)).toBe(true);
    fireEvent.click(screen.getByRole('tab', { name: 'Local' }));
    expect(screen.getByText('Local test model')).toBeVisible();
  });

  it('loads saved backend benchmark history without exposing prompts or generated text', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const calls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); calls.push(url);
      if (url.endsWith('/api/models')) return json({ providers: [] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [] });
      if (url.endsWith('/api/local-ai/hardware')) return json({ cpu_count: 8, ram_gb: 16 });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first' });
      if (url.endsWith('/api/local-ai/setup/status')) return json({});
      if (url.endsWith('/api/local-ai/marketplace')) return json({ models: [] });
      if (url.endsWith('/api/local-ai/adapters')) return json({ adapters: [] });
      if (url.includes('/api/local-ai/jobs?')) return json({ jobs: [] });
      if (url.includes('/api/local-ai/benchmarks?')) return json({ benchmarks: [{ id: 'benchmark-1', model_id: 'model-1', model: 'Local model', provider: 'local', prompt_tokens: 3, output_tokens: 9, duration_ms: 250, tokens_per_second: 36, created_at: '2026-10-24T10:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<ModelsScreen />);
    fireEvent.click(screen.getByText('Provider setup'));
    fireEvent.click(screen.getByRole('tab', { name: 'Setup activity' }));
    expect(await screen.findByText('Local model')).toBeInTheDocument();
    expect(screen.getByText('250 ms')).toBeInTheDocument();
    expect(screen.getByText('36 tokens/s')).toBeInTheDocument();
    expect(screen.getByText('3 / 9')).toBeInTheDocument();
    expect(screen.queryByText(/Hello OpenAgent|Synthetic private prompt|Synthetic generated output/i)).not.toBeInTheDocument();
    expect(calls.some(url => url.includes('/api/local-ai/benchmarks?limit=50'))).toBe(true);
  });

  it('keeps benchmark history local and clearly unmeasured in Demo mode', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    const calls: string[] = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input); calls.push(url);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<ModelsScreen />);
    fireEvent.click(screen.getByText('Add provider'));
    fireEvent.click(screen.getByRole('tab', { name: 'Setup activity' }));
    expect(await screen.findByRole('heading', { name: 'No measured benchmarks in Demo' })).toBeInTheDocument();
    expect(calls.filter(url => url.includes('/api/'))).toEqual([]);
    useWorkspaceState.getState().setDataMode('live');
  });

  it('uses backend setup, registration, and confirmed removal contracts for local models', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.endsWith('/api/models')) return json({ providers: [{ id: 'ollama', label: 'Ollama', local: true, available: true }] });
      if (url.endsWith('/api/local-ai/models')) return json({ models: [{ id: 'local-1', name: 'Local model', provider: 'local', status: 'ready', local_path: 'D:/models/local.gguf' }] });
      if (url.endsWith('/api/local-ai/hardware')) return json({ system: 'Windows', machine: 'x64' });
      if (url.endsWith('/api/model-policy')) return json({ routing_policy: 'local-first' });
      if (url.endsWith('/api/local-ai/setup/status')) return json({ ollama: { available: true, server_available: true }, recommended: { chat: 'local-1' } });
      if (url.endsWith('/api/local-ai/marketplace')) return json({ models: [{ id: 'catalog-1', name: 'Catalog model', metadata: { model: 'provider/model:latest' } }] });
      if (url.endsWith('/api/local-ai/adapters')) return json({ adapters: [] });
      if (url.includes('/api/local-ai/jobs?')) return json({ jobs: [] });
      if (url.endsWith('/api/local-ai/models/setup/job') && method === 'POST') return json({ job: { id: 'job-1', status: 'queued' } }, 202);
      if (url.endsWith('/api/local-ai/models/register') && method === 'POST') return json({ model: { id: 'registered-1', name: body?.name } }, 201);
      if (url.endsWith('/api/local-ai/models/delete') && method === 'POST') return json({ removed: true });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);

    renderWithQuery(<ModelsScreen />);
    await screen.findByText('Local model');
    fireEvent.click(screen.getByText('Provider setup'));
    fireEvent.click(screen.getByRole('tab', { name: 'Marketplace & runtime' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Queue setup' }));
    expect(calls.some(call => call.url.endsWith('/api/local-ai/models/setup/job'))).toBe(false);
    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole('button', { name: 'Queue setup' }));
    await waitFor(() => expect(calls.find(call => call.url.endsWith('/api/local-ai/models/setup/job'))?.body).toMatchObject({ task: 'chat', model: 'provider/model:latest', allow_network: false, project: 'default' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Model name' }), { target: { value: 'Registered local' } });
    fireEvent.click(screen.getByRole('button', { name: 'Register model' }));
    await waitFor(() => expect(calls.find(call => call.url.endsWith('/api/local-ai/models/register'))?.body).toMatchObject({ name: 'Registered local', provider: 'local' }));
    fireEvent.click(screen.getByRole('tab', { name: 'Inventory' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Remove' }));
    await waitFor(() => expect(calls.find(call => call.url.endsWith('/api/local-ai/models/delete'))?.body).toMatchObject({ model_id: 'local-1' }));
  });

  it('loads and creates named semantic collections through the supported backend routes', async () => {
    const calls: Array<{ url: string; method?: string; body?: unknown }> = [];
    let collections: Array<Record<string, unknown>> = [{ id: 'collection-1', name: 'Research', dimensions: 64, distance: 'cosine', updated_at: '2026-10-14T08:00:00Z' }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; const body = init?.body ? JSON.parse(String(init.body)) : undefined; calls.push({ url, method, body });
      if (url.includes('/api/knowledge/status')) return json({ documents: 0, chunks: 0 });
      if (url.includes('/api/knowledge/documents')) return json({ documents: [] });
      if (url.includes('/api/semantic/collections') && method === 'POST') { collections = [...collections, { id: 'collection-2', name: (body as Record<string, unknown>).name, dimensions: 64, distance: 'cosine' }]; return json({ collection: collections[1] }, 201); }
      if (url.includes('/api/semantic/collections')) return json({ collections });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<KnowledgeScreen />);
    fireEvent.click(screen.getByRole('tab', { name: 'Collections' }));
    expect(await screen.findByText('Research')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Open collection Research' }));
    const details = await screen.findByRole('dialog', { name: 'Research' });
    expect(within(details).getByText('Vector collection details · backend data')).toBeVisible();
    expect(within(details).getByText('64')).toBeVisible();
    expect(calls.some(call => call.url.includes('/api/knowledge/documents/collection-1'))).toBe(false);
    fireEvent.click(within(details).getByRole('button', { name: 'Close collection details' }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Collection name' }), { target: { value: 'Project memory' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create collection' }));
    expect(await screen.findByText('Project memory')).toBeVisible();
    expect(calls.find(call => call.url.includes('/api/semantic/collections') && call.method === 'POST')?.body).toMatchObject({ name: 'Project memory', project: 'default' });
  });

  it('loads stored communication drafts and sent records in their own tabs', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/communication/sent')) return json({ sent: [{ id: 'sent-1', subject: 'Approved update', status: 'simulated_local', created_at: '2026-10-14T08:00:00Z' }] });
      if (url.includes('/api/communication/drafts')) return json({ drafts: [{ id: 'draft-1', subject: 'Follow up', status: 'draft', updated_at: '2026-10-14T09:00:00Z' }] });
      if (url.includes('/api/communication/approvals')) return json({ approvals: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Sent' }));
    expect(await screen.findByText('Approved update')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Drafts' }));
    expect((await screen.findAllByText('Follow up')).length).toBeGreaterThan(0);
  });

  it('searches communication contacts through the project-scoped communications API', async () => {
    let contacts = [{ id: 'contact-1', name: 'Ada Lovelace', handles: { email: 'ada@example.org' } }];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      if (url.includes('/api/communication/contacts')) return json({ contacts });
      if (url.includes('/api/contacts')) return json({ contacts: [] });
      if (url.includes('/api/communication/inbox')) return json({ messages: [] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Contacts' }));
    expect(await screen.findByText('Ada Lovelace')).toBeVisible();
    fireEvent.change(screen.getByRole('textbox', { name: 'Search communication contacts' }), { target: { value: 'Ada' } });
    expect(screen.getByText('Ada Lovelace')).toBeVisible();
    fireEvent.change(screen.getByRole('textbox', { name: 'Search communication contacts' }), { target: { value: 'not found' } });
    expect(screen.queryByText('Ada Lovelace')).not.toBeInTheDocument();
  });

  it('performs CRM create, edit, delete and export using the backend handles contract', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let crm = [{ id: 'crm-1', project: 'default', name: 'Ada Lovelace', handles: { email: 'ada@example.org' }, notes: 'First programmer' }];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('URL', { ...URL, createObjectURL: vi.fn(() => 'blob:crm-export'), revokeObjectURL: vi.fn() });
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      const body = typeof init?.body === 'string' ? JSON.parse(init.body) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/communication/contacts')) return json({ contacts: [] });
      if (url.endsWith('/api/contacts/export')) return json({ project: 'default', count: crm.length, contacts: crm });
      if (url.endsWith('/api/contacts') && method === 'GET') return json({ contacts: crm });
      if (url.endsWith('/api/contacts') && method === 'POST') { const contact = { id: 'crm-2', project: 'default', ...body }; crm = [...crm, contact]; return json({ contact }, 201); }
      if (url.endsWith('/api/contacts/crm-1') && method === 'PATCH') { crm = crm.map(contact => contact.id === 'crm-1' ? { ...contact, ...body } : contact); return json({ contact: crm[0] }); }
      if (url.endsWith('/api/contacts/crm-1') && method === 'DELETE') { crm = crm.filter(contact => contact.id !== 'crm-1'); return json({ deleted: true, contact_id: 'crm-1' }); }
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Contacts' }));
    expect(await screen.findByText('Ada Lovelace')).toBeVisible();
    expect(screen.getByText('ada@example.org')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    fireEvent.change(screen.getByPlaceholderText('Email'), { target: { value: 'ada.new@example.org' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(calls.find(call => call.method === 'PATCH')?.body).toMatchObject({ handles: { email: 'ada.new@example.org' } }));

    fireEvent.click(screen.getByText('Add CRM contact'));
    const addForm = screen.getByText('Add CRM contact').closest('details') as HTMLElement;
    fireEvent.change(within(addForm).getByLabelText('Name'), { target: { value: 'Grace Hopper' } });
    fireEvent.change(within(addForm).getByLabelText('Email'), { target: { value: 'grace@example.org' } });
    fireEvent.change(within(addForm).getByLabelText('Notes'), { target: { value: 'Compiler pioneer' } });
    fireEvent.click(within(addForm).getByRole('button', { name: 'Save contact' }));
    await waitFor(() => expect(calls.find(call => call.method === 'POST' && call.url.endsWith('/api/contacts'))?.body).toMatchObject({ name: 'Grace Hopper', handles: { email: 'grace@example.org' }, notes: 'Compiler pioneer', project: 'default' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export CRM' }));
    await waitFor(() => expect(calls.some(call => call.method === 'GET' && call.url.endsWith('/api/contacts/export'))).toBe(true));
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete' })[0]);
    await waitFor(() => expect(calls.some(call => call.method === 'DELETE' && call.url.endsWith('/api/contacts/crm-1'))).toBe(true));
    await waitFor(() => expect(screen.queryByText('Ada Lovelace')).not.toBeInTheDocument());
  });

  it('creates local communication calendar events using validated event times', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined; calls.push({ url, method, body });
      if (url.includes('/api/communication/calendar') && method === 'POST') return json({ event: { id: 'event-new', title: body?.title } }, 201);
      if (url.includes('/api/communication/calendar')) return json({ events: [{ id: 'event-1', title: 'Planning', starts_at: '2026-10-15T09:00:00Z', ends_at: '2026-10-15T10:00:00Z' }] });
      if (url.includes('/api/communication/inbox')) return json({ messages: [] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Calendar' }));
    expect(await screen.findByText('Planning')).toBeVisible();
    fireEvent.click(screen.getByText('Add local calendar event'));
    fireEvent.change(screen.getByRole('textbox', { name: 'Title' }), { target: { value: 'Review' } });
    fireEvent.change(screen.getByLabelText('Starts'), { target: { value: '2026-10-16T09:00' } });
    fireEvent.change(screen.getByLabelText('Ends'), { target: { value: '2026-10-16T10:00' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save event' }));
    await waitFor(() => expect(calls.some(call => call.url.includes('/api/communication/calendar') && call.method === 'POST')).toBe(true));
    expect(calls.find(call => call.url.includes('/api/communication/calendar') && call.method === 'POST')?.body).toMatchObject({ title: 'Review', project: 'default', calendar_id: 'local', source: 'local' });
  });

  it('requires deliberate consent and enabled backend policy before polling email externally', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined; calls.push({ url, method, body });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [{ id: 'email', label: 'Email', enabled: true, capabilities: ['inbox'] }] });
      if (url.includes('/api/communication/oauth/accounts')) return json({ accounts: [{ id: 'account-1', provider: 'generic', platform: 'email', label: 'Work email', status: 'connected', token_present: true }] });
      if (url.includes('/api/communication/oauth/providers')) return json({ providers: [{ id: 'generic' }] });
      if (url.includes('/api/communication/sync-runs')) return json({ runs: [] });
      if (url.includes('/api/communication/sync') && method === 'POST') return json({ run: { id: 'run-1', status: 'completed' } }, 201);
      if (url.includes('/api/communication/inbox')) return json({ messages: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Connected accounts' }));
    expect(await screen.findByText('Work email')).toBeVisible();
    const syncButton = screen.getByRole('button', { name: 'Sync email inbox' });
    expect(syncButton).toBeDisabled();
    expect(calls.some(call => call.url.includes('/api/communication/sync') && call.method === 'POST')).toBe(false);
    fireEvent.click(screen.getByRole('checkbox', { name: 'I explicitly authorize one external inbox poll now.' }));
    fireEvent.click(syncButton);
    await waitFor(() => expect(calls.some(call => call.url.includes('/api/communication/sync') && call.method === 'POST')).toBe(true));
    expect(calls.find(call => call.url.includes('/api/communication/sync') && call.method === 'POST')?.body).toMatchObject({ platform: 'email', project: 'default', allow_external: true });
  });

  it('processes only backend-due scheduled sends after the user explicitly authorizes external delivery', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET'; const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined; calls.push({ url, method, body });
      if (url.includes('/api/communication/scheduled/process') && method === 'POST') return json({ count: 1, processed: [{ id: 'scheduled-1', status: 'sent' }] });
      if (url.includes('/api/communication/scheduled')) return json({ scheduled: [{ id: 'scheduled-1', draft_id: 'draft-approved', approval_id: 'approval-approved', status: 'scheduled', send_at: '2020-01-01T00:00:00Z' }] });
      if (url.includes('/api/communication/inbox')) return json({ messages: [] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Scheduled' }));
    expect(await screen.findByText('draft-approved')).toBeVisible();
    const processButton = screen.getByRole('button', { name: 'Process due sends' });
    expect(processButton).toBeDisabled();
    expect(calls.some(call => call.url.includes('/api/communication/scheduled/process') && call.method === 'POST')).toBe(false);
    fireEvent.click(screen.getByRole('checkbox', { name: /explicitly authorize processing due sends/ }));
    fireEvent.click(processButton);
    expect(await screen.findByText('Backend processed 1 due scheduled send(s).')).toBeVisible();
    expect(calls.find(call => call.url.includes('/api/communication/scheduled/process') && call.method === 'POST')?.body).toMatchObject({ project: 'default', allow_external: true });
  });

  it('requires backend approval resolution and explicit confirmation before a safe communication send', async () => {
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let approvalStatus = '';
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/communication/drafts')) return json({ drafts: [{ id: 'draft-1', subject: 'Follow up', recipients: ['person@example.org'], body: 'Hello', status: 'draft' }] });
      if (url.includes('/api/communication/approvals')) return json({ approvals: approvalStatus ? [{ id: 'approval-1', draft_id: 'draft-1', status: approvalStatus }] : [] });
      if (url.includes('/api/communication/send/request-approval')) { approvalStatus = 'pending'; return json({ approval: { id: 'approval-1', draft_id: 'draft-1', status: 'pending' } }, 201); }
      if (url.includes('/api/communication/send/resolve')) { approvalStatus = body?.approved ? 'approved' : 'rejected'; return json({ approval: { id: 'approval-1', draft_id: 'draft-1', status: approvalStatus } }); }
      if (url.endsWith('/api/communication/send')) return json({ sent: { id: 'sent-1', status: 'sent' } }, 201);
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Drafts' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Request approval' }));
    expect(await screen.findByText(/backend status:/)).toHaveTextContent('pending');
    expect(calls.find(call => call.url.includes('/send/request-approval'))?.body).toMatchObject({ draft_id: 'draft-1' });

    fireEvent.click(screen.getByRole('button', { name: 'Approve request' }));
    expect(await screen.findByRole('checkbox', { name: /explicitly confirm/ })).not.toBeChecked();
    expect(screen.getByRole('button', { name: 'Send approved draft' })).toBeDisabled();
    fireEvent.click(screen.getByRole('checkbox', { name: /explicitly confirm/ }));
    fireEvent.click(screen.getByRole('button', { name: 'Send approved draft' }));
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/communication/send'))).toBe(true));
    expect(calls.find(call => call.url.endsWith('/api/communication/send'))?.body).toMatchObject({ draft_id: 'draft-1', approval_id: 'approval-1', allow_external: false });
    expect(calls.findIndex(call => call.url.includes('/send/request-approval'))).toBeLessThan(calls.findIndex(call => call.url.includes('/send/resolve')));
    expect(calls.findIndex(call => call.url.includes('/send/resolve'))).toBeLessThan(calls.findIndex(call => call.url.endsWith('/api/communication/send')));
  });

  it('loads a selectable inbox message and marks it read through the communication state API', async () => {
    const calls: Array<{ url: string; method: string; body?: unknown }> = [];
    let read = false;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/communication/inbox')) return json({ messages: [{ id: 'message-1', subject: 'Deployment notes', sender: 'ops@example.org', summary: 'Release window', body: 'The release starts at 14:00.', read, created_at: '2026-10-14T08:00:00Z' }] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      if (url.includes('/api/communication/sync-runs')) return json({ runs: [{ id: 'sync-1', platform: 'email', status: 'completed', started_at: '2026-10-14T07:30:00Z' }] });
      if (url.includes('/api/communication/messages/state')) { read = Boolean((body as Record<string, unknown>).read); return json({ message: { id: 'message-1', read } }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    expect((await screen.findAllByText('Deployment notes')).length).toBe(2);
    expect(await screen.findByText('email · completed')).toBeInTheDocument();
    // Motion's page entrance keeps initial opacity in JSDOM, so assert the
    // backend-provided message is rendered; browser E2E covers actual visibility.
    expect(screen.getByText('The release starts at 14:00.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Mark read' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Read' })).toBeDisabled());
    expect(calls.find(call => call.url.includes('/api/communication/messages/state'))).toMatchObject({ method: 'POST', body: { message_id: 'message-1', read: true, project: 'default' } });
    expect(calls.some(call => call.url.includes('/api/communication/inbox'))).toBe(true);
  });

  it('filters real inbox messages and opens the draft workspace from Compose', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/communication/inbox')) return json({ messages: [{ id: 'message-2', subject: 'Release plan', sender: 'team@example.org', summary: 'Friday deployment', body: 'Review the runbook before Friday.', read: true }] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      if (url.includes('/api/communication/sync-runs')) return json({ runs: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    expect((await screen.findAllByText('Release plan')).length).toBeGreaterThan(0);
    fireEvent.change(screen.getByRole('textbox', { name: 'Search inbox messages' }), { target: { value: 'runbook' } });
    expect(screen.getAllByText('Release plan').length).toBeGreaterThan(0);
    fireEvent.change(screen.getByRole('textbox', { name: 'Search inbox messages' }), { target: { value: 'missing' } });
    expect(screen.getByText('No messages match “missing”.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Compose message' }));
    expect(await screen.findByRole('tab', { name: 'Drafts' })).toHaveAttribute('aria-selected', 'true');
    expect(await screen.findByRole('heading', { name: 'Create a draft' })).toBeInTheDocument();
  });

  it('creates a Live communication draft through the existing backend contract', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const calls: Array<{ url: string; method: string; body?: Record<string, unknown> }> = [];
    let drafts: Array<Record<string, unknown>> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = init?.method ?? 'GET';
      const body = init?.body ? JSON.parse(String(init.body)) as Record<string, unknown> : undefined;
      calls.push({ url, method, body });
      if (url.includes('/api/communication/drafts') && method === 'POST') {
        const created = { id: 'draft-created', subject: body?.subject, recipients: body?.recipients, body: body?.body, status: 'draft' };
        drafts = [created, ...drafts]; return json({ draft: created }, 201);
      }
      if (url.includes('/api/communication/drafts')) return json({ drafts });
      if (url.includes('/api/communication/approvals')) return json({ approvals: [] });
      if (url.includes('/api/communication/inbox')) return json({ messages: [] });
      if (url.includes('/api/communication/connectors')) return json({ connectors: [] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="communications" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Drafts' }));
    expect(await screen.findByRole('form', { name: 'Create communication draft' })).toBeInTheDocument();
    expect(screen.getAllByRole('form')).toHaveLength(1);
    expect(screen.getAllByRole('button', { name: 'Save draft' })).toHaveLength(1);
    fireEvent.change(await screen.findByRole('textbox', { name: 'To' }), { target: { value: 'owner@example.test, reviewer@example.test' } });
    fireEvent.change(screen.getAllByRole('textbox', { name: 'Subject' })[0], { target: { value: 'Local review' } });
    fireEvent.change(screen.getAllByRole('textbox', { name: 'Message' })[0], { target: { value: 'Please review this draft.' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'To' }), { target: { value: 'invalid-address' } });
    fireEvent.submit(screen.getByRole('form', { name: 'Create communication draft' }));
    expect(await screen.findByText('Enter one or more valid email addresses, separated by commas.')).toBeInTheDocument();
    expect(calls.some(call => call.url.includes('/api/communication/drafts') && call.method === 'POST')).toBe(false);
    fireEvent.change(screen.getByRole('textbox', { name: 'To' }), { target: { value: 'owner@example.test, reviewer@example.test' } });
    fireEvent.submit(screen.getByRole('form', { name: 'Create communication draft' }));
    await waitFor(() => expect(calls.some(call => call.url.includes('/api/communication/drafts') && call.method === 'POST')).toBe(true));
    expect(await screen.findByText('Draft saved by the backend. It has not been sent.')).toBeInTheDocument();
    expect(await screen.findByText('Local review')).toBeInTheDocument();
    expect(calls.find(call => call.url.includes('/api/communication/drafts') && call.method === 'POST')?.body).toMatchObject({
      platform: 'email', recipients: ['owner@example.test', 'reviewer@example.test'], subject: 'Local review', body: 'Please review this draft.', project: 'default',
    });
    expect(calls.some(call => call.url.endsWith('/api/communication/send'))).toBe(false);
  });

  it('loads the backend scraper result archive in the Results tab', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/scraper/history')) return json({ results: [{ id: 'scrape-result-1', url: 'https://example.org/report', title: 'Research report', status_code: 200, created_at: '2026-10-14T08:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="scraper" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Results' }));
    expect(await screen.findByText('Research report')).toBeVisible();
    expect(screen.getByText(/Status Code: 200/)).toBeVisible();
  });

  it('loads vault access records without exposing secret values', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/vault/access-log')) return json({ access_log: [{ id: 'access-1', entry_id: 'entry-1', action: 'retrieve', success: false, created_at: '2026-10-14T08:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="vault" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Access' }));
    const events = await screen.findAllByText('retrieve');
    expect(events.length).toBeGreaterThan(0);
    await waitFor(() => expect(screen.getByText(/Entry entry-1.*failed/)).toBeVisible());
  });

  it('reveals a vault value only after a passphrase-backed request and conceals it again on demand', async () => {
    let revealBody: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/vault/status')) return json({ project: 'default', total_entries: 1 });
      if (url.includes('/api/vault/entries')) return json({ entries: [{ id: 'entry-1', label: 'Service token', category: 'api_key' }] });
      if (url.includes('/api/vault/retrieve')) { revealBody = JSON.parse(String(init?.body)) as Record<string, unknown>; return json({ secret: { secret: 'secret-value-for-test' } }); }
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="vault" />);
    fireEvent.click(await screen.findByRole('button', { name: 'Reveal' }));
    fireEvent.change((await screen.findAllByLabelText('Vault passphrase'))[0], { target: { value: 'correct horse battery staple' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reveal for 15 seconds' }));
    expect(await screen.findByText('secret-value-for-test')).toBeVisible();
    expect(revealBody).toMatchObject({ entry_id: 'entry-1', passphrase: 'correct horse battery staple', project: 'default' });
    fireEvent.click(screen.getByRole('button', { name: 'Conceal now' }));
    expect(screen.queryByText('secret-value-for-test')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Secret value masked')).toBeVisible();
  });

  it('creates fine-tuning dataset metadata through the backend-backed workspace', async () => {
    let datasetBody: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/finetune/datasets') && init?.method === 'POST') { datasetBody = JSON.parse(String(init.body)) as Record<string, unknown>; return json({ dataset: { id: 'dataset-1', name: 'Support examples', format: 'jsonl' } }, 201); }
      if (url.includes('/api/finetune/datasets')) return json({ datasets: [] });
      if (url.includes('/api/finetune/status')) return json({ datasets: 0, jobs: 0, adapters: 0 });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="finetune" />);
    fireEvent.change(await screen.findByLabelText('Dataset name'), { target: { value: 'Support examples' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create dataset record' }));
    await waitFor(() => expect(datasetBody).toMatchObject({ name: 'Support examples', format: 'jsonl', project: 'default' }));
    expect(await screen.findByText('The backend accepted the request. Check its returned state before assuming work has started.')).toBeVisible();
  });

  it('retains Fine-tuning form values after a Live write fails and clears them only after success', async () => {
    useWorkspaceState.getState().setDataMode('live');
    let rejectWrites = true;
    const writes: Array<{ path: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/finetune/datasets') && init?.method === 'POST') {
        const body = JSON.parse(String(init.body)) as Record<string, unknown>;
        writes.push({ path: url, body });
        return rejectWrites ? json({ error: 'Dataset permission denied.' }, 403) : json({ dataset: { id: 'dataset-1', name: body.name, format: 'jsonl' } }, 201);
      }
      if (url.includes('/api/finetune/datasets')) return json({ datasets: [] });
      if (url.includes('/api/finetune/status')) return json({ datasets: 0, jobs: 0, adapters: 0 });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="finetune" />);
    fireEvent.click(await screen.findByText('Create dataset'));
    const nameInput = await screen.findByLabelText('Dataset name');
    fireEvent.change(nameInput, { target: { value: 'Retryable dataset' } });
    fireEvent.click(screen.getByRole('button', { name: 'Create dataset record' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/does not have permission/i);
    expect(nameInput).toHaveValue('Retryable dataset');

    rejectWrites = false;
    fireEvent.click(screen.getByRole('button', { name: 'Create dataset record' }));
    await waitFor(() => expect(nameInput).toHaveValue(''));
    expect(await screen.findByText('The backend accepted the request. Check its returned state before assuming work has started.')).toBeVisible();
    expect(writes).toHaveLength(2);
    expect(writes.every(write => write.body.name === 'Retryable dataset')).toBe(true);
  });

  it('cancels only a backend-reported queued Fine-tuning job', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const writes: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes('/api/finetune/jobs/') && url.endsWith('/cancel') && init?.method === 'POST') {
        writes.push({ url, body: JSON.parse(String(init.body)) as Record<string, unknown> });
        return json({ job: { id: 'queued-1', project: 'default', status: 'cancelled' } });
      }
      if (url.includes('/api/finetune/jobs')) return json({ jobs: [{ id: 'queued-1', project: 'default', status: 'queued', dataset_id: 'dataset-1', method: 'lora', base_model: 'qwen' }] });
      if (url.includes('/api/finetune/datasets')) return json({ datasets: [{ id: 'dataset-1', name: 'Support', format: 'jsonl' }] });
      if (url.includes('/api/finetune/status')) return json({ datasets: 1, jobs: 1, adapters: 0 });
      if (url.includes('/api/finetune/adapters')) return json({ adapters: [] });
      if (url.includes('/api/finetune/templates')) return json({ templates: [] });
      if (url.includes('/api/local-ai/models')) return json({ models: [] });
      if (url.includes('/api/local-ai/jobs')) return json({ jobs: [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="finetune" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel queued job' }));
    await waitFor(() => expect(writes).toHaveLength(1));
    expect(writes[0]).toMatchObject({ url: expect.stringContaining('/api/finetune/jobs/queued-1/cancel'), body: { project: 'default' } });
    expect(await screen.findByText('The backend accepted the request. Check its returned state before assuming work has started.')).toBeVisible();
  });

  it('runs approved Scientific AI code and displays persisted real output in Reports', async () => {
    useWorkspaceState.getState().setDataMode('live');
    let submitted: Record<string, unknown> | undefined;
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith('/api/science/execute') && init?.method === 'POST') {
        submitted = JSON.parse(String(init.body)) as Record<string, unknown>;
        return json({ execution: { id: 'science-run-1', status: 'completed', returncode: 0, elapsed_ms: 12, attempts: 1, peak_memory_bytes: 1024, stdout: 'local science result\\n', stderr: '', code_sha256: 'abc123' } }, 201);
      }
      if (url.includes('/api/science/executions')) return json({ executions: submitted ? [{ id: 'science-run-1', status: 'completed', returncode: 0, elapsed_ms: 12, attempts: 1, stdout: 'local science result\\n', stderr: '', code_sha256: 'abc123' }] : [] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="science" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Reports' }));
    const source = await screen.findByLabelText('Python source');
    const runButton = await screen.findByRole('button', { name: 'Run approved code' });
    expect(runButton).toBeDisabled();
    fireEvent.change(source, { target: { value: "print('local science result')" } });
    fireEvent.click(screen.getByLabelText(/I reviewed this Python source/));
    fireEvent.click(runButton);
    await waitFor(() => expect(submitted).toMatchObject({ project: 'default', approved: true, timeout_seconds: 10, memory_limit_mb: 512 }));
    expect(submitted?.code).toBe("print('local science result')");
    const reportSection = (await screen.findByText('Recorded scientific executions')).closest('section');
    expect(reportSection).toBeVisible();
    expect(reportSection?.querySelector('pre')).toHaveTextContent('local science result');
  });

  it('loads persisted science proposals with their honest execution state', async () => {
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/science/experiments')) return json({ experiments: [{ id: 'proposal-1', title: 'Experiment proposal: Compare materials', goal: 'Compare materials', status: 'proposed', execution_status: 'not_started', created_at: '2026-10-14T08:00:00Z', proposals: [{ title: 'Candidate A', rationale: 'Test candidate A under the stated conditions.', variables: { material: 'A', temperature: 20 } }] }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="science" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Experiments' }));
    await waitFor(() => expect(screen.getByText('Experiment proposal: Compare materials')).toBeVisible());
    await waitFor(() => expect(screen.getByText(/proposed · not_started/)).toBeVisible());
    await waitFor(() => expect(screen.getByText('Candidate A')).toBeVisible());
  });

  it('loads Twin source metadata while omitting original interaction text', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes('/api/twin/sources')) return json({ sources: [{ id: 'source-1', kind: 'research', topics: ['materials', 'planning'], preference_signals: [{ category: 'topic', key: 'materials', value: 'materials', relationship: 'topic signal supplied to the preference learner' }], content: 'private original prompt', context: { local_path: 'private/path' }, created_at: '2026-10-14T08:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderWithQuery(<AdvancedScreen route="twin" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Sources' }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/twin/sources'), expect.anything()));
    await waitFor(() => expect(screen.getByText('research')).toBeVisible());
    await waitFor(() => expect(screen.getByText(/Topics: materials, planning/)).toBeVisible());
    await waitFor(() => expect(screen.getByText('Inferred preference signals:')).toBeVisible());
    expect(screen.getByText('Inferred preference signals:').closest('p')).toHaveTextContent('materials');
    expect(screen.queryByText(/private original prompt|private\/path/)).not.toBeInTheDocument();
  });

  it('requires dataset license consent and confirms catalog analysis against the science API', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.includes('/api/science/materials/datasets')) return json({ datasets: [] });
      if (url.includes('/api/science/memory')) return json({ memory: [] });
      if (url.includes('/api/science/materials/catalog') && method === 'GET') return json({ datasets: [{ dataset_id: 'mat-7', name: 'Alloy properties', license: 'CC-BY-4.0' }] });
      if (url.endsWith('/api/science/materials/catalog/analyse')) return json({ analysis: { dataset_id: 'mat-7', records_analysed: 9 } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="science" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Data' }));
    await screen.findByRole('option', { name: 'Alloy properties · CC-BY-4.0' });
    fireEvent.change(screen.getByLabelText('Dataset'), { target: { value: 'mat-7' } });
    fireEvent.click(screen.getByLabelText(/accept the selected dataset license/i));
    expect(screen.getByRole('button', { name: 'Analyse selected data' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: 'Analyse selected data' }));
    expect(await screen.findByText('The backend returned a dataset analysis.')).toBeVisible();
    expect(vi.mocked(window.confirm)).toHaveBeenCalled();
    expect(calls.find(call => call.url.endsWith('/api/science/materials/catalog/analyse'))?.body).toMatchObject({ dataset_id: 'mat-7', allow_external: false, license_accepted: true, project: 'default' });
  });

  it('labels Twin predictions as suggestions and requires confirmation before consolidation', async () => {
    const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      if (init?.method === 'POST') calls.push({ url, body });
      if (url.includes('/api/twin/patterns')) return json({ patterns: [] });
      if (url.includes('/api/twin/suggestions')) return json({ suggestions: [] });
      if (url.includes('/api/twin/status')) return json({ status: 'ready' });
      if (url.includes('/api/twin/profile')) return json({ profile: {} });
      if (url.includes('/api/twin/goals')) return json({ goals: [] });
      if (url.endsWith('/api/twin/predict')) return json({ prediction: { intent: 'plan a meeting', confidence: 0.62 } });
      if (url.endsWith('/api/twin/consolidate')) return json({ consolidated: true, updated_records: 2 });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="twin" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Preferences' }));
    fireEvent.change(await screen.findByLabelText('Context to consider'), { target: { value: 'I need to schedule a meeting' } });
    fireEvent.click(screen.getByRole('button', { name: 'Get suggestion' }));
    expect(await screen.findByText(/suggestion, not a fact/)).toBeVisible();
    expect(calls.find(call => call.url.endsWith('/api/twin/predict'))?.body).toMatchObject({ context: 'I need to schedule a meeting', project: 'default' });

    fireEvent.click(screen.getByRole('tab', { name: 'Data controls' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Consolidate context memory' }));
    expect(vi.mocked(window.confirm)).toHaveBeenCalledWith(expect.stringContaining('Consolidate this project'));
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/twin/consolidate'))).toBe(true));
  });

  it('requires network consent and confirmation before processing due scraper jobs', async () => {
    const calls: Array<{ url: string; body: Record<string, unknown> }> = [];
    const confirmAction = vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      if (init?.method === 'POST') calls.push({ url, body });
      if (url.includes('/api/scraper/jobs')) return json({ jobs: [{ id: 'job-1', label: 'Docs refresh', url: 'https://docs.example.test', allow_external: true, status: 'scheduled' }] });
      if (url.includes('/api/scraper/status')) return json({ status: 'ready' });
      if (url.endsWith('/api/scraper/run-due')) return json({ processed: 1, results: [{ status: 'completed' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="scraper" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }));
    const run = await screen.findByRole('button', { name: 'Process due jobs' });
    expect(run).toBeDisabled();
    await screen.findByText('1 backend records');
    fireEvent.click(screen.getByLabelText(/authorize processing due jobs/i));
    await waitFor(() => expect(run).toBeEnabled());
    fireEvent.click(run);
    await waitFor(() => expect(calls.some(call => call.url.endsWith('/api/scraper/run-due'))).toBe(true));
    expect(await screen.findByText('The backend processed its currently due scheduled jobs.')).toBeVisible();
    expect(confirmAction).toHaveBeenCalled();
    expect(calls.find(call => call.url.endsWith('/api/scraper/run-due'))?.body).toEqual({ project: 'default' });
  });

  it('cancels a pending Live scraper schedule and retains its history row', async () => {
    let cancelled = false;
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    const confirmAction = vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* response is not JSON */ }
      calls.push({ url, method, body });
      if (url.includes('/api/scraper/jobs/') && url.endsWith('/cancel') && method === 'POST') {
        cancelled = true;
        return json({ cancelled: true, job: { id: 'job-1', status: 'cancelled' } });
      }
      if (url.includes('/api/scraper/jobs')) return json({ jobs: [{ id: 'job-1', url: 'https://docs.example.test', interval_hours: 24, status: cancelled ? 'cancelled' : 'scheduled', metadata: { label: 'Docs refresh', external_approved: false } }] });
      if (url.includes('/api/scraper/status')) return json({ status: 'ready' });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="scraper" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }));
    const cancel = await screen.findByRole('button', { name: 'Cancel scheduled extraction for Docs refresh' });
    fireEvent.click(cancel);

    await waitFor(() => expect(calls.some(call => call.method === 'POST' && call.url.endsWith('/api/scraper/jobs/job-1/cancel'))).toBe(true));
    expect(confirmAction).toHaveBeenCalledWith(expect.stringContaining('Cancel the recurring extraction'));
    expect(calls.find(call => call.method === 'POST' && call.url.endsWith('/api/scraper/jobs/job-1/cancel'))?.body).toEqual({ project: 'default' });
    expect(await screen.findByText('Schedule cancelled. The saved job history has been retained.')).toBeVisible();
    expect(await screen.findByText('cancelled')).toBeVisible();
    expect(screen.queryByRole('button', { name: 'Cancel scheduled extraction for Docs refresh' })).not.toBeInTheDocument();
  });

  it('reports a due-job claim race as running instead of claiming cancellation succeeded', async () => {
    let running = false;
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      if (url.includes('/api/scraper/jobs/') && url.endsWith('/cancel') && method === 'POST') {
        running = true;
        return json({ cancelled: false, job: { id: 'job-race', status: 'running' } });
      }
      if (url.includes('/api/scraper/jobs')) return json({ jobs: [{ id: 'job-race', url: 'https://docs.example.test', interval_hours: 24, status: running ? 'running' : 'scheduled', metadata: { label: 'Claimed Docs Refresh', external_approved: true } }] });
      if (url.includes('/api/scraper/status')) return json({ status: 'ready' });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="scraper" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Cancel scheduled extraction for Claimed Docs Refresh' }));

    expect(await screen.findByText('This job is already running. Its current request was not interrupted.')).toBeVisible();
    expect(screen.getByText('running')).toBeVisible();
    expect(screen.queryByText(/Schedule cancelled/)).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Cancel scheduled extraction for Claimed Docs Refresh' })).not.toBeInTheDocument();
  });

  it('creates external access review requests and confirms pending decisions through the security service', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {};
      try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* multipart or empty request */ }
      calls.push({ url, method, body });
      if (url.includes('/api/security/status')) return json({ status: 'ok' });
      if (url.includes('/api/security/shares')) return json({ grants: [] });
      if (url.includes('/api/security/egress') && method === 'GET') return json({ requests: [{ id: 'egress-1', destination: 'files.example.test', purpose: 'Upload a user-selected report', resource_type: 'document', status: 'pending' }] });
      if (url.includes('/api/audit') && method === 'GET') return json({ events: [{ id: 'audit-default-1', actor: 'owner', action: 'secret.reveal', resource: 'vault-entry', project: 'default', status: 'success', created_at: '2026-10-24T20:00:00Z', metadata: { secret_value: 'must-not-render' } }] });
      if (url.endsWith('/api/security/egress/request') && method === 'POST') return json({ request: { id: 'egress-2', status: 'pending' } }, 201);
      if (url.endsWith('/api/security/egress/resolve') && method === 'POST') return json({ request: { id: 'egress-1', status: body.approved ? 'approved' : 'denied' } });
      if (url.endsWith('/api/security/audit-chain/verify') && method === 'POST') return json({ ok: true, verified_events: 12 });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));

    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Access' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/egress') && call.method === 'GET')).toBe(true));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Approve request' })).toBeVisible());
    fireEvent.change(screen.getByRole('textbox', { name: 'Purpose' }), { target: { value: 'Upload a user-selected report' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Destination' }), { target: { value: 'files.example.test' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'External-access resource type' }), { target: { value: 'document' } });
    fireEvent.click(screen.getByRole('button', { name: 'Submit for review' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/egress/request') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/egress/request'))?.body).toMatchObject({
      requester: 'user', purpose: 'Upload a user-selected report', destination: 'files.example.test', resource_type: 'document', risk_level: 'medium',
    });

    fireEvent.click(screen.getByRole('button', { name: 'Approve request' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/egress/resolve') && call.method === 'POST')).toBe(true));
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Approve external access to files.example.test'));
    expect(calls.find((call) => call.url.endsWith('/api/security/egress/resolve'))?.body).toMatchObject({ request_id: 'egress-1', approved: true, approved_by: 'user' });

    fireEvent.click(screen.getByRole('tab', { name: 'Audit' }));
    expect(await screen.findByText('secret.reveal')).toBeVisible();
    expect(screen.getByText("security:admin")).toBeVisible();
    expect(screen.getByText(/read-only events for project/i)).toBeVisible();
    expect(screen.queryByText('must-not-render')).toBeNull();
    expect(calls.some((call) => call.url === '/api/audit?project=default&limit=100' && call.method === 'GET')).toBe(true);
    fireEvent.click(await screen.findByRole('button', { name: 'Verify chain' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/audit-chain/verify') && call.method === 'POST')).toBe(true));
    expect(await screen.findByText('Audit-chain verification completed.')).toBeVisible();
  });

  it('creates and revokes read-only share grants through Security Access', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.includes('/api/security/status')) return json({});
      if (url.includes('/api/security/egress')) return json({ requests: [] });
      if (url.includes('/api/security/shares') && method === 'GET') return json({ grants: [{ id: 'share-1', resource_type: 'document', resource_id: 'doc-7', grantee: 'reviewer@example.test', scope: 'read', status: 'active' }] });
      if (url.endsWith('/api/security/shares') && method === 'POST') return json({ grant: { id: 'share-2', status: 'active' } }, 201);
      if (url.endsWith('/api/security/shares/resolve') && method === 'POST') return json({ grant: { id: 'share-1', status: 'revoked' } });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    const expiresAt = new Date(Date.now() + 86_400_000);
    const expiresAtLocal = new Date(expiresAt.getTime() - expiresAt.getTimezoneOffset() * 60_000).toISOString().slice(0, 16);
    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Access' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Revoke read grant' })).toBeVisible());
    fireEvent.change(screen.getByRole('textbox', { name: 'Share resource type' }), { target: { value: 'document' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Resource ID' }), { target: { value: 'doc-8' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Grantee' }), { target: { value: 'reader@example.test' } });
    fireEvent.change(screen.getByLabelText('Share expiry'), { target: { value: expiresAtLocal } });
    fireEvent.click(screen.getByRole('button', { name: 'Create read grant' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/shares') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/shares') && call.method === 'POST')?.body).toMatchObject({ resource_type: 'document', resource_id: 'doc-8', grantee: 'reader@example.test', scope: 'read', expires_at: new Date(expiresAtLocal).toISOString() });
    fireEvent.click(screen.getByRole('button', { name: 'Revoke read grant' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/shares/resolve') && call.method === 'POST')).toBe(true));
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Revoke the read grant'));
    expect(calls.find((call) => call.url.endsWith('/api/security/shares/resolve'))?.body).toEqual({ grant_id: 'share-1', status: 'revoked' });
  });

  it('generates stored privacy reports using count-only summaries', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.endsWith('/api/security/privacy-reports')) return json({ reports: [{ id: 'report-1', created_at: '2026-10-14T09:00:00Z', report: { secrets: { count: 1, names: ['private-secret-name'] }, data_inventory: { documents: 3 }, network_default: false } }] });
      if (url.endsWith('/api/security/privacy-report') && method === 'POST') return json({ report: { generated_at: '2026-10-14T10:00:00Z', secrets: { count: 2, names: ['another-private-name'] }, data_inventory: { documents: 4 }, network_default: false } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Privacy' }));
    fireEvent.click(await screen.findByRole('button', { name: 'Generate report' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/privacy-report') && call.method === 'POST')).toBe(true));
    expect(screen.getByText('Secret records: 2')).toBeVisible();
    expect(screen.queryByText(/private-secret-name|another-private-name/)).not.toBeInTheDocument();
  });

  it('previews retention changes before requiring confirmation for deletion', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.endsWith('/api/security/retention') && method === 'GET') return json({ policies: [{ id: 'retention-1', resource_type: 'agent_runs', retention_days: 30, action: 'delete', status: 'active' }] });
      if (url.endsWith('/api/security/retention/apply') && method === 'POST') return json({ policy_id: 'retention-1', dry_run: body.dry_run, planned: 3, deleted: body.dry_run === true ? 0 : 3, cutoff: '2026-09-01T00:00:00Z' });
      if (url.endsWith('/api/security/retention') && method === 'POST') return json({ policy: { id: 'retention-2', resource_type: body.resource_type } }, 201);
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Retention' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Preview deletions' })).toBeVisible());
    fireEvent.click(screen.getByRole('button', { name: 'Preview deletions' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/retention/apply') && call.body.dry_run === true)).toBe(true));
    expect(screen.getByText('Preview completed. No records were deleted.')).toBeVisible();
    expect(screen.getByText(/3 records matched · 0 deleted/)).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Apply deletion' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/retention/apply') && call.body.dry_run === false)).toBe(true));
    expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('Permanently delete records'));
    expect(screen.getByText(/3 records matched · 3 deleted/)).toBeVisible();
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Keep for days' }), { target: { value: '90' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save policy' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/retention') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/retention') && call.method === 'POST')?.body).toMatchObject({ resource_type: 'context_events', retention_days: 90, action: 'delete', status: 'active' });
  });

  it('creates a policy-only sandbox profile and evaluates access against returned backend profiles', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.endsWith('/api/security/sandbox-profiles') && method === 'GET') return json({ profiles: [{ id: 'profile-1', name: 'default-deny', permissions: [], network_allowed: false }] });
      if (url.endsWith('/api/security/sandbox-profiles') && method === 'POST') return json({ profile: { id: 'profile-2', name: body.name } }, 201);
      if (url.endsWith('/api/security/sandbox/evaluate') && method === 'POST') return json({ profile: { name: 'default-deny' }, allowed: false, requested_network: body.network, requested_permissions: body.permissions, denied: [{ type: 'permissions' }] });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Sandbox' }));
    await waitFor(() => expect(screen.getByRole('radio', { name: /default-deny/ })).toBeVisible());
    fireEvent.change(screen.getByRole('textbox', { name: 'Profile name' }), { target: { value: 'limited-review' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Allowed permissions, separated by commas' }), { target: { value: 'filesystem.read' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save profile' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/sandbox-profiles') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/sandbox-profiles') && call.method === 'POST')?.body).toMatchObject({ name: 'limited-review', permissions: ['filesystem.read'], network_allowed: false });
    fireEvent.change(screen.getByRole('combobox', { name: 'Profile' }), { target: { value: 'profile-1' } });
    fireEvent.change(screen.getByRole('textbox', { name: 'Requested permissions' }), { target: { value: 'process.spawn' } });
    fireEvent.click(screen.getByRole('button', { name: 'Evaluate only' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/sandbox/evaluate') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/sandbox/evaluate'))?.body).toMatchObject({ profile_id: 'profile-1', permissions: ['process.spawn'], network: false });
    expect(await screen.findByText('Denied by this profile')).toBeVisible();
  });

  it('stores, temporarily reveals, conceals, and rotates secrets without exposing stored values in metadata', async () => {
    const calls: Array<{ url: string; method: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal('confirm', vi.fn(() => true));
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input); const method = String(init?.method ?? 'GET');
      let body: Record<string, unknown> = {}; try { body = JSON.parse(String(init?.body ?? '{}')); } catch { /* GET */ }
      calls.push({ url, method, body });
      if (url.endsWith('/api/security/secrets') && method === 'GET') return json({ secrets: [{ id: 'secret-1', name: 'Build credential', description: 'Local CI', encrypted: true, status: 'active', updated_at: '2026-10-14T09:00:00Z' }] });
      if (url.endsWith('/api/security/secrets') && method === 'POST') return json({ secret: { id: 'secret-2', name: body.name, encrypted: true } }, 201);
      if (url.endsWith('/api/security/secrets/reveal') && method === 'POST') return json({ secret: { id: 'secret-1', name: 'Build credential', value: 'returned-secret-value' } });
      if (url.endsWith('/api/security/secrets/rotate') && method === 'POST') return json({ secret: { id: 'secret-1', name: 'Build credential', encrypted: true } });
      return json({ error: `Unexpected request ${method} ${url}` }, 404);
    }));
    renderWithQuery(<AdvancedScreen route="security" />);
    fireEvent.click(screen.getByRole('tab', { name: 'Secrets' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Reveal for 15 seconds' })).toBeVisible());
    expect(screen.queryByText('returned-secret-value')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Passphrase'), { target: { value: 'unlock phrase' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reveal for 15 seconds' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/secrets/reveal') && call.method === 'POST')).toBe(true));
    expect(await screen.findByText('returned-secret-value')).toBeVisible();
    fireEvent.click(screen.getByRole('button', { name: 'Conceal now' }));
    expect(screen.queryByText('returned-secret-value')).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Passphrase'), { target: { value: 'unlock phrase' } });
    fireEvent.change(screen.getByLabelText('Replacement value'), { target: { value: 'replacement-secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Replace secret value' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/secrets/rotate') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/secrets/rotate'))?.body).toMatchObject({ secret_id: 'secret-1', value: 'replacement-secret', passphrase: 'unlock phrase' });
    fireEvent.change(screen.getByRole('textbox', { name: 'Name' }), { target: { value: 'New credential' } });
    fireEvent.change(screen.getByLabelText('Secret value'), { target: { value: 'never-render-this' } });
    fireEvent.change(screen.getByLabelText('Encryption passphrase'), { target: { value: 'store phrase' } });
    fireEvent.click(screen.getByRole('button', { name: 'Encrypt and store' }));
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/api/security/secrets') && call.method === 'POST')).toBe(true));
    expect(calls.find((call) => call.url.endsWith('/api/security/secrets') && call.method === 'POST')?.body).toMatchObject({ name: 'New credential', value: 'never-render-this', passphrase: 'store phrase' });
    expect(screen.queryByText('never-render-this')).not.toBeInTheDocument();
  });

  it('renders verified Adaptive Engine backend data in each live workspace tab', async () => {
    useWorkspaceState.getState().setDataMode('live');
    const requests: string[] = [];
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith('/api/adaptive-engine/status')) return json({
        status: 'ready', project: 'default',
        hardware_profile: {
          platform: 'Test workstation', total_ram_gb: 32, available_ram_gb: 18, cpu_threads: 16,
          devices: [{ id: 'gpu-test-1', name: 'Test GPU', kind: 'gpu', vendor: 'Example', memory_gb: 12, available: true, estimate_source: 'reported' }],
        },
        backends: [{ id: 'backend-test', name: 'Verified CPU backend', status: 'ready', available: true }],
        inventory: { models: 2, plans: 1, orchestration_runs: 1 },
      });
      if (url.endsWith('/api/compute/status')) return json({ status: 'ready' });
      if (url.endsWith('/api/adaptive-engine/direct/status')) return json({ status: 'idle' });
      if (url.endsWith('/api/adaptive-engine/hardware')) return json({
        profile: { id: 'profile-test', platform: 'Test workstation', total_ram_gb: 32, available_ram_gb: 18, cpu_threads: 16, devices: [{ id: 'gpu-test-1', name: 'Test GPU', kind: 'gpu', vendor: 'Example', memory_gb: 12, available: true }] },
        backends: [{ id: 'backend-test', name: 'Verified CPU backend', status: 'ready', available: true }],
      });
      if (url.endsWith('/api/adaptive-engine/orchestrations')) return json({ runs: [{ id: 'run-test-1', task: 'Index project documents', mode: 'sequential', rounds: 2, approved: true, status: 'completed', final_output: 'Index completed by backend', completed_at: '2026-10-18T09:30:00Z' }] });
      if (url.endsWith('/api/adaptive-engine/plans')) return json({ plans: [{ id: 'plan-test-1', task: 'Local inference', status: 'planned', model: 'local-model', device: 'CPU', updated_at: '2026-10-18T09:00:00Z' }] });
      return json({ error: `Unexpected request ${url}` }, 404);
    });
    vi.stubGlobal('fetch', fetchMock);
    renderWithQuery(<AdvancedScreen route="engine" />);

    expect(await screen.findByText(/Test workstation/)).toBeVisible();
    expect((await screen.findAllByText('Verified CPU backend')).length).toBeGreaterThan(0);
    expect(screen.getByText('18 GB')).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Hardware' }));
    expect((await screen.findAllByText('Test GPU')).length).toBeGreaterThan(0);
    expect(screen.getByText(/32 GB total memory · 16 CPU threads/)).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Jobs' }));
    expect(await screen.findByText('Index project documents')).toBeVisible();
    expect(screen.getByText('Index completed by backend')).toBeVisible();
    expect((await screen.findAllByText(/^completed$/i)).length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole('tab', { name: 'Plans' }));
    expect(await screen.findByText('Local inference')).toBeVisible();
    expect((await screen.findAllByText(/^planned$/i)).length).toBeGreaterThan(0);
    expect(requests).toEqual(expect.arrayContaining([
      expect.stringContaining('/api/adaptive-engine/status'),
      expect.stringContaining('/api/compute/status'),
      expect.stringContaining('/api/adaptive-engine/direct/status'),
      expect.stringContaining('/api/adaptive-engine/hardware'),
      expect.stringContaining('/api/adaptive-engine/orchestrations'),
      expect.stringContaining('/api/adaptive-engine/plans'),
    ]));
    expect(screen.queryByText('Awaiting service')).not.toBeInTheDocument();
  });
});
