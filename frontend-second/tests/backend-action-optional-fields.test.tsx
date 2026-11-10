import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

/**
 * The shared "Backend action" form marked every input `required`, including
 * fields the backend treats as optional and one labelled "(optional)". A
 * required empty input fails the browser's own constraint validation, which
 * fires no submit event at all -- so the button did nothing, no request was
 * made, and no message appeared. Four forms could not be submitted the way
 * their own submit guard said they should be: Assistant Goals (Description),
 * Developer Webhooks (Event pattern), Scraper (Job label, Repeat interval),
 * Fine-tuning datasets (Description) and Adaptive engine Plans (three of its
 * four inputs).
 *
 * Found by driving the real built app against a real server; every automated
 * suite had missed it. These tests hold each form to the contract its own
 * submit guard states: filling only the fields that guard demands must leave
 * the form valid and must reach the backend.
 */

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useWorkspaceState.getState().setDataMode('live');
});

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
}

/** Renders a screen and records the POSTs it makes. */
function renderScreen(route: string, reads: (url: string) => unknown = () => ({})) {
  const posts: Array<{ url: string; body: string }> = [];
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (input, init) => {
    const url = String(input);
    if (String(init?.method ?? 'GET') === 'POST') {
      posts.push({ url, body: String(init?.body ?? '') });
      return json({ ok: true }, 201);
    }
    return json(reads(url) ?? {});
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><AdvancedScreen route={route} /></QueryClientProvider>);
  return posts;
}

/** The form holding the given submit control. */
function formOf(button: HTMLElement): HTMLFormElement {
  const form = button.closest('form');
  if (!form) throw new Error('the submit control is not inside a form');
  return form as HTMLFormElement;
}

/**
 * The shared action form, scoped. Several screens render a second control with
 * the same accessible name as the form's own submit button -- the Assistant
 * record list has its own "Add goal" -- so these queries have to be scoped to
 * the action section rather than the whole page.
 */
async function actionForm(): Promise<HTMLFormElement> {
  const section = await waitFor(() => {
    const found = document.getElementById('advanced-backend-action');
    if (!found) throw new Error('the backend action section has not rendered');
    return found;
  });
  return waitFor(() => {
    const form = section.querySelector('form');
    if (!form) throw new Error('the backend action section has no form');
    return form as HTMLFormElement;
  });
}

/** The action form's own submit control. */
function submitOf(form: HTMLFormElement): HTMLButtonElement {
  const button = form.querySelector('button:not([type="button"])');
  if (!button) throw new Error('the action form has no submit control');
  return button as HTMLButtonElement;
}

describe('Backend action forms submit with only their required fields filled', () => {
  it('adds an assistant goal without a description', async () => {
    const posts = renderScreen('assistant');
    fireEvent.click(await screen.findByRole('tab', { name: 'Goals' }));
    const form = await actionForm();
    fireEvent.change(form.querySelector('input')!, { target: { value: 'Ship the release' } });

    // Description is left blank, which is what the submit guard permits.
    expect(form.checkValidity()).toBe(true);
    fireEvent.click(submitOf(form));
    await waitFor(() => expect(posts.some(p => p.url.includes('/api/assistant/goals'))).toBe(true));
    expect(JSON.parse(posts[0].body).title).toBe('Ship the release');
  });

  it('marks the assistant description optional rather than required', async () => {
    renderScreen('assistant');
    fireEvent.click(await screen.findByRole('tab', { name: 'Goals' }));
    const form = await actionForm();
    const inputs = [...form.querySelectorAll('input')];
    // Exactly one input is required: the goal itself.
    expect(inputs.filter(input => input.required)).toHaveLength(1);
  });

  it('asks for a compute plan with only the task filled', async () => {
    const posts = renderScreen('engine');
    fireEvent.click(await screen.findByRole('tab', { name: 'Plans' }));
    const task = await screen.findByPlaceholderText('chat, code, embeddings…');
    fireEvent.change(task, { target: { value: 'chat' } });
    const button = screen.getByRole('button', { name: 'Ask backend for a compute plan' });
    const form = formOf(button);

    // Preferred model, target latency and memory budget are all left blank.
    expect(form.checkValidity()).toBe(true);
    fireEvent.click(button);
    await waitFor(() => expect(posts.some(p => p.url.includes('/api/compute/plan'))).toBe(true));
  });

  it('leaves the field labelled optional optional', async () => {
    renderScreen('engine');
    fireEvent.click(await screen.findByRole('tab', { name: 'Plans' }));
    const model = await screen.findByPlaceholderText('Model identifier');
    // The label says "(optional)", so the input must not be required.
    expect(model).not.toBeRequired();
  });

  it('registers a webhook without an event pattern', async () => {
    const posts = renderScreen('developer');
    fireEvent.click(await screen.findByRole('tab', { name: 'Webhooks' }));
    const target = await screen.findByPlaceholderText('https://example.org/webhook');
    fireEvent.change(target, { target: { value: 'https://example.org/hook' } });
    const button = screen.getByRole('button', { name: 'Register webhook' });

    // The event pattern defaults to "*" in the request body, so the form must
    // not insist on one.
    expect(formOf(button).checkValidity()).toBe(true);
    fireEvent.click(button);
    await waitFor(() => expect(posts.some(p => p.url.includes('/api/developer/webhooks'))).toBe(true));
  });

  it('schedules an extraction without a label or an interval', async () => {
    // Scraper confirms before submitting, and the dialog is not what is under
    // test here.
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const posts = renderScreen('scraper');
    const form = await actionForm();
    const url = form.querySelector('input[type="url"]') as HTMLInputElement;
    fireEvent.change(url, { target: { value: 'https://example.org/page' } });

    // Job label's own placeholder says "Optional label", and the repeat
    // interval has a default, so neither may block the form.
    expect(form.checkValidity()).toBe(true);
    fireEvent.click(submitOf(form));
    await waitFor(() => expect(posts.some(p => p.url.includes('/api/scraper/'))).toBe(true));
  });

  it('leaves the fields a screen excludes from this component alone', async () => {
    // Vault and Fine-tuning render their own workspaces and never mount the
    // shared action form, so their field() calls are unreachable. Asserting
    // that here keeps a later reader from treating those calls as live
    // contracts, and fails if either screen starts rendering the form without
    // its required flags being reviewed.
    renderScreen('finetune');
    await waitFor(() => expect(document.querySelector('.p-card')).toBeTruthy());
    expect(document.getElementById('advanced-backend-action')).toBeNull();
  });

  it('still blocks a form whose own required field is blank', async () => {
    // The fix is per field, not a blanket removal of required. The task is the
    // one input the compute-plan guard demands, so a blank task must still
    // leave the form invalid.
    renderScreen('engine');
    fireEvent.click(await screen.findByRole('tab', { name: 'Plans' }));
    const task = await screen.findByPlaceholderText('chat, code, embeddings…');
    const button = screen.getByRole('button', { name: 'Ask backend for a compute plan' });

    expect(formOf(button).checkValidity()).toBe(false);
    expect(task).toBeRequired();
    fireEvent.change(task, { target: { value: 'chat' } });
    expect(formOf(button).checkValidity()).toBe(true);
  });
});
