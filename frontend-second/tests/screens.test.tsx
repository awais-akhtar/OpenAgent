import { render, screen, fireEvent, cleanup, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { HomeScreen, ChatScreen } from '../src/screens/HomeChat';
import { CodeQScreen, ResearchScreen, FilesScreen } from '../src/screens/WorkspaceScreens';
import { AgentsScreen, AutomationsScreen, ModelsScreen, SettingsScreen } from '../src/screens/OperationsScreens';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

afterEach(cleanup);

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
}

describe('reference screen set', () => {
  it('opens the approved advanced demo integration guide and supports keyboard dismissal', async () => {
    const originalMode = useWorkspaceState.getState().dataMode;
    useWorkspaceState.getState().setDataMode('demo');
    try {
      renderWithClient(<AdvancedScreen route="voice" />);
      fireEvent.click(screen.getByRole('button', { name: 'Integration guide' }));
      expect(await screen.findByRole('dialog', { name: 'Demo and integration guide' })).toBeInTheDocument();
      expect(screen.getByText('Current screen').parentElement).toHaveTextContent('voice');
      expect(screen.getByText('Record collection').parentElement).toHaveTextContent('recordings');
      expect(screen.getByRole('button', { name: 'Export sample data' })).toBeInTheDocument();
      fireEvent.keyDown(document, { key: 'Escape' });
      await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Demo and integration guide' })).not.toBeInTheDocument());
    } finally { useWorkspaceState.getState().setDataMode(originalMode); }
  });

  it('opens a truthful Live integration guide without Demo export controls', async () => {
    const originalMode = useWorkspaceState.getState().dataMode;
    useWorkspaceState.getState().setDataMode('live');
    try {
      renderWithClient(<AdvancedScreen route="voice" />);
      fireEvent.click(screen.getByRole('button', { name: 'Integration guide' }));
      const guide = await screen.findByRole('dialog', { name: 'Demo and integration guide' });
      expect(within(guide).getByText('Backend permissions apply')).toBeInTheDocument();
      expect(within(guide).getByText('Live export is unavailable here. No backend export route is configured for this guide.')).toBeInTheDocument();
      expect(within(guide).queryByRole('button', { name: 'Export sample data' })).not.toBeInTheDocument();
      expect(within(guide).queryByText('Only example data')).not.toBeInTheDocument();
      fireEvent.keyDown(document, { key: 'Escape' });
      await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Demo and integration guide' })).not.toBeInTheDocument());
    } finally { useWorkspaceState.getState().setDataMode(originalMode); }
  });

  it('renders the technical Home dashboard without duplicate workspace shortcuts', async () => {
    renderWithClient(<HomeScreen />);
    expect(await screen.findByRole('meter', { name: 'CPU utilization' })).toBeVisible();
    expect(screen.getByRole('heading', { name: 'Local runtime' })).toBeVisible();
    expect(screen.queryByText('Recent activity')).not.toBeInTheDocument();
    expect(screen.queryByText('Projects')).not.toBeInTheDocument();
  });

  it('shows an honest unavailable state when telemetry requests fail', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(JSON.stringify({ detail: 'temporarily unavailable' }), { status: 503 }));
    try {
      renderWithClient(<HomeScreen />);
      await waitFor(() => expect(screen.getByRole('meter', { name: 'CPU utilization' })).not.toHaveAttribute('aria-valuenow'));
      expect(screen.getByRole('meter', { name: 'CPU utilization' })).not.toHaveAttribute('aria-valuenow');
    } finally { fetchMock.mockRestore(); }
  });
  it('loads the connected chat composer without claiming a local draft is an assistant reply', () => {
    renderWithClient(<ChatScreen />);
    expect(screen.getByRole('textbox', { name: 'Message OpenAgent' })).toBeVisible();
    expect(screen.getByRole('button', { name: 'Send message' })).toBeDisabled();
  });

  it('provides truthful CodeQ and file empty states', () => {
    const { unmount } = renderWithClient(<CodeQScreen />);
    expect(document.querySelector('.codeq-root')).toBeVisible();
    expect(screen.getByRole('button', { name: 'Open project folder' })).toBeVisible();
    unmount();
    renderWithClient(<FilesScreen />);
    expect(screen.getByRole('heading', { name: 'Files & knowledge' })).toBeVisible();
  });

  it('does not fabricate research reports or agent activity', () => {
    const { unmount } = renderWithClient(<ResearchScreen />);
    expect(screen.getByRole('heading', { name: 'No report selected' })).toBeVisible();
    fireEvent.click(screen.getByRole('tab', { name: 'Findings' }));
    expect(screen.getByRole('tab', { name: 'Findings' })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText(/Findings are shown with their source references/i)).toBeVisible();
    unmount();
    renderWithClient(<AgentsScreen />);
    expect(screen.getByRole('button', { name: /Run agent/i })).toBeInTheDocument();
  });

  it('exposes all main operation screens and the advanced workspace template', () => {
    const routes = [
      [<AgentsScreen />, /Agent workspace|No agents available/],
      [<AutomationsScreen />, /Automations|No automations/i],
      [<ModelsScreen />, /Models|Model registry unavailable/i],
      [<SettingsScreen />, /Settings/i],
      [<AdvancedScreen route="vault" />, /Vault/i],
      [<AdvancedScreen route="science" />, /Scientific AI/i],
    ] as const;
    for (const [component, text] of routes) {
      const { unmount } = renderWithClient(component);
      expect(screen.getAllByText(text).length).toBeGreaterThan(0);
      unmount();
    }
  });

  it('creates developer API keys only with scopes from the backend catalog and shows the returned token once', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes('/api/developer/api-keys') && init?.method === 'POST') return new Response(JSON.stringify({ api_key: { id: 'k1', label: 'CI', scopes: ['developer:read'], key: 'oa_dev_once' } }), { status: 201 });
      if (url.includes('/api/developer/api-keys')) return new Response(JSON.stringify({ api_keys: [] }), { status: 200 });
      if (url.includes('/api/auth/scopes')) return new Response(JSON.stringify({ scopes: [{ scope: 'developer:read', description: 'Read developer resources' }, { scope: 'developer:admin', description: 'Manage developer resources' }] }), { status: 200 });
      if (url.includes('/api/developer/status')) return new Response(JSON.stringify({ status: 'ready' }), { status: 200 });
      if (init?.method === 'POST') return new Response(JSON.stringify({ api_key: { id: 'w1', enabled: false } }), { status: 201 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    try {
      renderWithClient(<AdvancedScreen route="developer" />);
      fireEvent.click(await screen.findByRole('tab', { name: 'API keys' }));
      fireEvent.change(await screen.findByLabelText('Key label'), { target: { value: 'CI' } });
      fireEvent.click(await screen.findByLabelText(/developer:read/));
      fireEvent.click(screen.getByRole('button', { name: 'Create API key' }));
      await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/developer/api-keys'),
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ label: 'CI', scopes: ['developer:read'] }) }),
      ));
      expect(await screen.findByLabelText('New API token (shown once)')).toHaveValue('oa_dev_once');
    } finally { fetchMock.mockRestore(); }
  });

  it('registers webhook endpoints disabled by default and requires explicit activation confirmation', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      if (String(input).includes('/api/developer/webhooks')) return new Response(JSON.stringify({ webhooks: [], deliveries: [] }), { status: 200 });
      if (String(input).includes('/api/developer/status')) return new Response(JSON.stringify({ status: 'ready' }), { status: 200 });
      if (init?.method === 'POST') return new Response(JSON.stringify({ webhook: { id: 'w1', enabled: false } }), { status: 201 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    try {
      renderWithClient(<AdvancedScreen route="developer" />);
      fireEvent.click(await screen.findByRole('tab', { name: 'Webhooks' }));
      fireEvent.change(await screen.findByLabelText('Event pattern'), { target: { value: 'openagent.task.completed' } });
      fireEvent.change(await screen.findByLabelText('Target URL'), { target: { value: 'https://example.org/hook' } });
      fireEvent.click(screen.getByRole('button', { name: 'Register webhook' }));
      await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/developer/webhooks'),
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ event: 'openagent.task.completed', target_url: 'https://example.org/hook', enabled: false, secret_hint: '', metadata: {} }) }),
      ));
      fireEvent.click(screen.getByLabelText('Enable immediately for matching future events'));
      const register = screen.getByRole('button', { name: 'Register webhook' });
      expect(register).toBeDisabled();
      fireEvent.click(screen.getByLabelText('I understand this can cause outbound requests to the target URL.'));
      fireEvent.click(register);
      await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/developer/webhooks'),
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ event: 'openagent.task.completed', target_url: 'https://example.org/hook', enabled: true, secret_hint: '', metadata: {} }) }),
      ));
    } finally { fetchMock.mockRestore(); }
  });

  it('registers MCP definitions as disabled and excludes environment secrets', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes('/api/developer/mcp/servers') && init?.method === 'POST') return new Response(JSON.stringify({ server: { id: 'm1', name: 'Local tool', enabled: false } }), { status: 201 });
      if (url.includes('/api/developer/mcp/servers')) return new Response(JSON.stringify({ servers: [] }), { status: 200 });
      if (url.includes('/api/developer/mcp')) return new Response(JSON.stringify({ servers: [] }), { status: 200 });
      if (url.includes('/api/developer/status')) return new Response(JSON.stringify({ status: 'ready' }), { status: 200 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    try {
      renderWithClient(<AdvancedScreen route="developer" />);
      fireEvent.click(await screen.findByRole('tab', { name: 'MCP' }));
      fireEvent.change(await screen.findByLabelText('Server name'), { target: { value: 'Local tool' } });
      fireEvent.change(await screen.findByLabelText('Executable command'), { target: { value: 'local-tool' } });
      fireEvent.change(await screen.findByLabelText('Arguments (JSON string array)'), { target: { value: '["--stdio"]' } });
      fireEvent.click(screen.getByRole('button', { name: 'Register disabled server' }));
      await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining('/api/developer/mcp/servers'),
        expect.objectContaining({ method: 'POST', body: JSON.stringify({ name: 'Local tool', command: 'local-tool', args: ['--stdio'], env: {}, enabled: false, metadata: {} }) }),
      ));
    } finally { fetchMock.mockRestore(); }
  });

  it('keeps Settings theme and reduced-motion preferences shared and persistent', () => {
    useWorkspaceState.getState().setTheme('light');
    useWorkspaceState.getState().setReduceMotion(false);
    renderWithClient(<SettingsScreen />);
    fireEvent.click(screen.getByRole('button', { name: /Dark/i }));
    expect(useWorkspaceState.getState().theme).toBe('dark');
    expect(document.documentElement).toHaveAttribute('data-theme', 'dark');
    const reduceMotion = screen.getByRole('checkbox', { name: 'Reduce interface animation' });
    fireEvent.click(reduceMotion);
    expect(useWorkspaceState.getState().reduceMotion).toBe(true);
    expect(localStorage.getItem('oa-second-reduce-motion')).toBe('true');
    expect(localStorage.getItem('oa-second-theme')).toBe('dark');
  });

  it('shows the local demo marker between the Settings heading and tabs', () => {
    const originalMode = useWorkspaceState.getState().dataMode;
    useWorkspaceState.getState().setDataMode('demo');
    const rendered = renderWithClient(<SettingsScreen />);
    try {
      const badge = screen.getByText('Live demo');
      expect(badge).toBeInTheDocument();
      expect(badge.closest('.settings-demo-indicator')).toBeInTheDocument();
      expect(badge.compareDocumentPosition(screen.getByRole('tablist')) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    } finally {
      rendered.unmount();
      useWorkspaceState.getState().setDataMode(originalMode);
    }
  });

  it('loads and saves runtime settings through the backend and surfaces privilege errors', async () => {
    const snapshot = { chat_model: 'local-chat', code_model: 'local-code', privacy_boundary: 'Current process only.' };
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.includes('/api/settings/runtime') && init?.method === 'POST') {
        expect(JSON.parse(String(init.body))).toEqual({ settings: { chat_model: 'custom-chat' } });
        return new Response(JSON.stringify({ settings: { ...snapshot, chat_model: 'custom-chat' } }), { status: 200 });
      }
      if (url.includes('/api/settings/runtime')) return new Response(JSON.stringify({ settings: snapshot }), { status: 200 });
      if (url.includes('/api/health')) return new Response(JSON.stringify({ status: 'ok' }), { status: 200 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    try {
      const first = renderWithClient(<SettingsScreen />);
      fireEvent.click(screen.getByRole('tab', { name: 'Advanced' }));
      const field = await screen.findByLabelText('Chat model');
      expect(field).toHaveValue('local-chat');
      fireEvent.change(field, { target: { value: 'custom-chat' } });
      fireEvent.click(screen.getByRole('button', { name: 'Save runtime settings' }));
      expect(await screen.findByText(/accepted by the backend/i)).toBeVisible();
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/settings/runtime'), expect.objectContaining({ method: 'POST' }));
      first.unmount();
    } finally { fetchMock.mockRestore(); }

    const forbiddenFetch = vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      if (String(input).includes('/api/settings/runtime')) return new Response(JSON.stringify({ error: 'settings:read permission required' }), { status: 403 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    try {
      renderWithClient(<SettingsScreen />);
      fireEvent.click(screen.getByRole('tab', { name: 'Advanced' }));
      expect(await screen.findByRole('alert')).toHaveTextContent(/does not have permission/i);
    } finally { forbiddenFetch.mockRestore(); }
  });

  it('creates backups and requires a successful dry run plus explicit confirmation before restore', async () => {
    const archive = { name: 'openagent-backup.zip', path: '/server/data/backups/openagent-backup.zip', size_bytes: 512, manifest: { project: 'default' } };
    let restored = false;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith('/api/backups/create')) return new Response(JSON.stringify({ backup: archive }), { status: 201 });
      if (url.endsWith('/api/backups/restore-dry-run')) return new Response(JSON.stringify({ status: 'ready', will_write: false, database_entries: ['database/openagent.sqlite3'], media_entries: 0, conflicts: ['current_database_would_be_replaced'] }), { status: 200 });
      if (url.endsWith('/api/backups/restore')) {
        expect(JSON.parse(String(init?.body))).toEqual({ path: archive.path, confirm: true });
        restored = true;
        return new Response(JSON.stringify({ restored: true, safety_copy: '/server/data/backups/pre-restore.sqlite3' }), { status: 200 });
      }
      if (url.endsWith('/api/backups')) return new Response(JSON.stringify({ backups: [archive] }), { status: 200 });
      if (url.includes('/api/')) return new Response(JSON.stringify({}), { status: 200 });
      return new Response(JSON.stringify({}), { status: 200 });
    });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    try {
      renderWithClient(<SettingsScreen />);
      fireEvent.click(screen.getByRole('tab', { name: 'Data & backups' }));
      fireEvent.click(await screen.findByRole('button', { name: 'Create backup' }));
      expect(await screen.findByRole('status')).toHaveTextContent(/Backup created: openagent-backup.zip/);
      fireEvent.change(await screen.findByLabelText('Backup archive'), { target: { value: archive.path } });
      fireEvent.click(screen.getByRole('button', { name: 'Run restore preflight' }));
      const restore = await screen.findByRole('button', { name: /Restore this backup/ });
      expect(restore).toBeEnabled();
      expect(screen.getByRole('alert')).toHaveTextContent(/current database may be replaced/i);
      fireEvent.click(restore);
      expect(confirm).toHaveBeenCalled();
      expect(restored).toBe(false);
      confirm.mockReturnValue(true);
      fireEvent.click(restore);
      expect(await screen.findByText(/Restore completed according to the backend/i)).toBeVisible();
      expect(restored).toBe(true);
      expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining('/api/backups/restore'), expect.objectContaining({ method: 'POST' }));
    } finally { confirm.mockRestore(); fetchMock.mockRestore(); }
  });

  it('supports keyboard arrow navigation for workspace tabs', () => {
    renderWithClient(<CodeQScreen />);
    const code = screen.getByRole('tab', { name: 'Code' });
    const preview = screen.getByRole('tab', { name: 'Preview' });
    code.focus();
    fireEvent.keyDown(code, { key: 'ArrowRight' });
    expect(preview).toHaveFocus();
    expect(preview).toHaveAttribute('aria-selected', 'true');
  });
});
