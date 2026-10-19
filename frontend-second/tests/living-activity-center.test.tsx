import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { LivingActivityCenter } from '../src/components/LivingActivityCenter';
import { useWorkspaceState } from '../src/state';

afterEach(() => { cleanup(); localStorage.clear(); useWorkspaceState.getState().setDataMode('demo'); vi.restoreAllMocks(); });

describe('LivingActivityCenter', () => {
  it('hides the generic activity walkthrough on Chat where no tool execution is active', () => {
    useWorkspaceState.getState().setDataMode('demo');
    render(<LivingActivityCenter route="chat"/>);
    expect(screen.queryByRole('button', { name: 'Open Design Preview sandbox' })).not.toBeInTheDocument();
  });

  it('keeps the design tutorial out of workspace routes and exposes it from Developer only', () => {
    useWorkspaceState.getState().setDataMode('demo');
    const workspaceRoutes = ['agents', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'assistant'];
    for (const route of workspaceRoutes) {
      const { unmount } = render(<LivingActivityCenter route={route}/>);
      expect(screen.queryByRole('button', { name: 'Open Design Preview sandbox' }), `${route} must not advertise the tutorial as workspace activity`).not.toBeInTheDocument();
      unmount();
    }
    render(<LivingActivityCenter route="developer"/>);
    expect(screen.getByRole('button', { name: 'Open Design Preview sandbox' })).toHaveTextContent('Design Preview');
  });

  it('requires explicit entry into the separate demo sandbox before local tutorial playback', () => {
    useWorkspaceState.getState().setDataMode('demo');
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    render(<LivingActivityCenter route="developer"/>);
    fireEvent.click(screen.getByRole('button', { name: 'Open Design Preview sandbox' }));
    expect(screen.getByRole('region', { name: 'Demo tutorial sandbox' })).toBeInTheDocument();
    expect(screen.getByText('This separate walkthrough demonstrates the Developer activity visuals. It does not report or change an active workspace run.')).toBeInTheDocument();
    expect(screen.queryByText('Ready when you are')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Start sandbox walkthrough' }));
    expect(screen.getByText('Developer walkthrough')).toBeInTheDocument();
    expect(screen.getAllByText('Inspecting example connections')).toHaveLength(2);
    expect(screen.getByText('Simulation only.')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Visual options' }));
    expect(screen.getByText('Interactive light orb')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('tab', { name: 'Tutorial history' }));
    expect(screen.getByText('Recent tutorial playback')).toBeInTheDocument();
    expect(screen.getByText('No tutorial history')).toBeInTheDocument();
  });

  it('keeps a local run stoppable and opens its simulation details', () => {
    useWorkspaceState.getState().setDataMode('demo');
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue(null);
    render(<LivingActivityCenter route="developer"/>);
    fireEvent.click(screen.getByRole('button', { name: 'Open Design Preview sandbox' }));
    fireEvent.click(screen.getByRole('button', { name: 'Start sandbox walkthrough' }));
    fireEvent.click(screen.getByRole('button', { name: 'Stop sandbox' }));
    expect(screen.getByText('Demo · Cancelled')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Details/ }));
    expect(screen.getByText('Local UI simulation')).toBeInTheDocument();
    expect(screen.getByText('Real backend calls')).toBeInTheDocument();
    expect(screen.getByText('None')).toBeInTheDocument();
  });

  it('hides local walkthrough controls in Live mode', () => {
    useWorkspaceState.getState().setDataMode('live');
    render(<LivingActivityCenter route="developer"/>);
    expect(screen.queryByRole('button', { name: 'Open Design Preview sandbox' })).not.toBeInTheDocument();
  });
});
