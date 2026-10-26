import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { demoProvider } from '../src/demo/provider';
import { useWorkspaceState } from '../src/state';

const sourceSha256 = 'BDCF0F5E24121A94C562B524EF49AB2F9B8582ED5F083B79045AE3C01AAC6CE1';
const recordings = [
  { id: 'vo1', name: 'Product planning sync', description: 'Review onboarding feedback and release blockers', status: 'Transcribed', duration: '08:42', owner: 'Workspace' },
  { id: 'vo2', name: 'Quick idea — agent UI', description: 'Notes about task execution feedback', status: 'Transcribed', duration: '02:16', owner: 'Workspace' },
];

afterEach(() => { cleanup(); vi.restoreAllMocks(); });

describe('Voice demo recordings', () => {
  it('searches, creates, edits, opens details, duplicates and deletes local fixture records', async () => {
    const previousMode = useWorkspaceState.getState().dataMode;
    useWorkspaceState.getState().setDataMode('demo');
    vi.spyOn(globalThis, 'fetch').mockImplementation(async input => {
      const url = String(input);
      if (url.endsWith('/demo/reference-fixtures.json')) return new Response(JSON.stringify({ sourceSha256, data: { advanced: { voice: { recordings, settings: [] } } } }), { status: 200 });
      if (url.endsWith('/demo/approved-shell-fixtures.json')) return new Response(JSON.stringify({ sourceSha256, data: {} }), { status: 200 });
      throw new Error(`Unexpected request in local Voice demo: ${url}`);
    });
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    try {
      await demoProvider.reset();
      render(<QueryClientProvider client={client}><AdvancedScreen route="voice"/></QueryClientProvider>);
      const planningButton = await screen.findByRole('button', { name: 'Open Product planning sync details' });
      await waitFor(() => expect(planningButton).toBeVisible());
      expect(planningButton.querySelector('.lucide-mic')).toBeInTheDocument();
      expect(screen.getByText('2 items')).toBeVisible();

      fireEvent.change(screen.getByRole('textbox', { name: 'Search recordings' }), { target: { value: 'agent UI' } });
      expect(screen.getByRole('button', { name: 'Open Quick idea — agent UI details' })).toBeVisible();
      expect(screen.queryByRole('button', { name: 'Open Product planning sync details' })).not.toBeInTheDocument();
      expect(screen.getByText('1 item')).toBeVisible();
      fireEvent.change(screen.getByRole('textbox', { name: 'Search recordings' }), { target: { value: '' } });

      fireEvent.click(screen.getByRole('button', { name: 'Add' }));
      const createDialog = await screen.findByRole('dialog', { name: 'New recording' });
      fireEvent.change(within(createDialog).getByLabelText('Name'), { target: { value: 'Workshop recap' } });
      fireEvent.change(within(createDialog).getByLabelText('Description'), { target: { value: 'Decisions and next steps from the workshop.' } });
      fireEvent.change(within(createDialog).getByLabelText('Duration'), { target: { value: '03:20' } });
      fireEvent.click(within(createDialog).getByRole('button', { name: 'Create item' }));
      const createdButton = await screen.findByRole('button', { name: 'Open Workshop recap details' });
      expect(createdButton).toBeVisible();
      expect(within(createdButton.closest('.demo-record') as HTMLElement).getByText('Ready')).toBeVisible();

      fireEvent.click(screen.getByRole('button', { name: 'Edit Product planning sync' }));
      const editDialog = await screen.findByRole('dialog', { name: 'Edit recording' });
      fireEvent.change(within(editDialog).getByLabelText('Name'), { target: { value: 'Product planning sync · edited' } });
      fireEvent.click(within(editDialog).getByRole('button', { name: 'Save changes' }));
      expect(await screen.findByRole('button', { name: 'Open Product planning sync · edited details' })).toBeVisible();

      fireEvent.click(screen.getByRole('button', { name: 'Open Product planning sync · edited details' }));
      expect(await screen.findByRole('dialog', { name: 'Product planning sync · edited' })).toHaveTextContent('08:42');
      fireEvent.keyDown(document, { key: 'Escape' });
      await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument());

      fireEvent.click(screen.getByRole('button', { name: 'More actions for Quick idea — agent UI' }));
      const menu = await screen.findByRole('dialog', { name: 'Quick idea — agent UI' });
      fireEvent.click(within(menu).getByRole('button', { name: 'Run / test' }));
      expect(await screen.findByRole('status', { name: 'Local workflow running' })).toBeVisible();
      expect(screen.getByRole('progressbar', { name: 'Simulated recording workflow progress' })).toHaveAttribute('aria-valuenow', '0');
      expect(screen.queryByRole('progressbar', { name: 'Simulated activity progress' })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: 'Cancel local run' }));
      await waitFor(() => expect(screen.queryByRole('status', { name: 'Local workflow running' })).not.toBeInTheDocument());
      expect(await within(screen.getByRole('button', { name: 'Open Quick idea — agent UI details' }).closest('.demo-record') as HTMLElement).findByText('Cancelled')).toBeVisible();

      fireEvent.click(screen.getByRole('button', { name: 'More actions for Quick idea — agent UI' }));
      const menuActions = await screen.findByRole('dialog', { name: 'Quick idea — agent UI' });
      fireEvent.click(within(menuActions).getByRole('button', { name: 'Duplicate' }));
      expect(await screen.findByRole('button', { name: 'Open Quick idea — agent UI (copy) details' })).toBeVisible();

      fireEvent.click(screen.getByRole('button', { name: 'More actions for Workshop recap' }));
      const workshopActions = await screen.findByRole('dialog', { name: 'Workshop recap' });
      fireEvent.click(within(workshopActions).getByRole('button', { name: 'Delete' }));
      const confirmation = await screen.findByRole('dialog', { name: 'Delete “Workshop recap”?'});
      fireEvent.click(within(confirmation).getByRole('button', { name: 'Delete record' }));
      await waitFor(() => expect(screen.queryByRole('button', { name: 'Open Workshop recap details' })).not.toBeInTheDocument());
      expect(screen.getByText('Demo recording deleted locally.')).toBeVisible();
    } finally {
      client.clear();
      await demoProvider.reset();
      useWorkspaceState.getState().setDataMode(previousMode);
    }
  }, 15_000);
});
