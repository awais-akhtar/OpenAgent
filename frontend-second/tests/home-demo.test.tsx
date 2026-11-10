import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { HomeScreen } from '../src/screens/HomeChat';
import { useWorkspaceState } from '../src/state';

afterEach(cleanup);

describe('Home technical dashboard Demo boundary', () => {
  it('labels its fixture meters and never requests the backend', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('Demo must remain offline'));
    try {
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      render(<QueryClientProvider client={client}><HomeScreen /></QueryClientProvider>);
      expect(await screen.findByRole('meter', { name: 'CPU utilization' })).toBeVisible();
      expect(screen.getByText('Simulated demo')).toBeVisible();
      expect(await screen.findByRole('meter', { name: 'CPU utilization' })).toHaveAttribute('aria-valuenow', '24.6');
      expect(screen.queryByRole('button', { name: 'Pause' })).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: '5 min' }));
      expect(screen.getByRole('button', { name: '5 min' })).toHaveAttribute('aria-pressed', 'true');
      expect(fetchMock).not.toHaveBeenCalled();
      expect(screen.queryByText('Projects')).not.toBeInTheDocument();
      expect(screen.queryByText('Recent activity')).not.toBeInTheDocument();
    } finally {
      useWorkspaceState.getState().setDataMode('live');
      fetchMock.mockRestore();
    }
  });
});
