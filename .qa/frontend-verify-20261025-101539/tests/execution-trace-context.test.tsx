import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { ExecutionTrace } from '../src/components/ExecutionTrace';
import { useWorkspaceState } from '../src/state';

afterEach(() => {
  cleanup();
  useWorkspaceState.getState().setDataMode('live');
});

describe('ExecutionTrace contextual and mode boundaries', () => {
  const executionRoutes = ['agents', 'chat', 'research', 'codeq', 'models', 'automations', 'science', 'finetune', 'scraper', 'voice'] as const;

  it('does not show a sample trace on a routine Demo workspace visit', () => {
    useWorkspaceState.getState().setDataMode('demo');
    const { container } = render(<ExecutionTrace route="agents" />);
    expect(container.querySelector('.trace-mount')).not.toBeInTheDocument();
  });

  it('shows the labelled simulated trace only after the parent starts a Demo operation', async () => {
    useWorkspaceState.getState().setDataMode('demo');
    render(<ExecutionTrace route="agents" startSignal={1} />);
    expect(await screen.findByLabelText('Multi-agent orchestration')).toBeVisible();
    expect(screen.getByText(/DEMO DATA/)).toBeVisible();
    expect(screen.getByText(/PLAYING LOCAL EXAMPLE/)).toBeVisible();
  });

  it.each(['codeq', 'research'] as const)('%s stays compact after an explicit start until expanded deliberately', async route => {
    useWorkspaceState.getState().setDataMode('demo');
    render(<ExecutionTrace route={route} startSignal={1} />);
    const trace = await screen.findByLabelText(route === 'codeq' ? 'CodeQ execution trace' : 'Research execution');
    expect(trace.querySelector('.trace-collapsed')).toBeInTheDocument();
    expect(trace.querySelector('.trace-layout')).not.toBeInTheDocument();
    const expand = screen.getByRole('button', { name: 'Expand' });
    expand.click();
    expect(await screen.findByRole('progressbar', { name: 'Simulated activity progress' })).toBeVisible();
  });

  it.each(executionRoutes)('%s renders no synthetic execution content in Live mode, even when explicitly started', route => {
    useWorkspaceState.getState().setDataMode('live');
    const { container } = render(<ExecutionTrace route={route} startSignal={1} />);
    expect(container.querySelector('.trace-mount')).not.toBeInTheDocument();
    expect(container.querySelector('.trace-step')).not.toBeInTheDocument();
    expect(container.querySelector('.trace-metric-card')).not.toBeInTheDocument();
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /Play example|Export example/ })).not.toBeInTheDocument();
    expect(screen.queryByText(/LOCAL EXAMPLE|PLAYING LOCAL EXAMPLE|SAMPLE METRICS/i)).not.toBeInTheDocument();
  });
});
