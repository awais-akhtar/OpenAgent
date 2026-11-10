import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, describe, expect, it, vi } from 'vitest';
import AdvancedScreen from '../src/screens/AdvancedScreen';
import { useWorkspaceState } from '../src/state';

/**
 * No form may invite a click it cannot honour.
 *
 * A form whose submit control is enabled while the form itself cannot validate
 * is a dead control: the browser fires no submit event at all, so the click
 * does nothing, no request is made and nothing is shown to the person who
 * clicked. That is how `field()` marking every input `required` made four
 * screens unusable while every suite stayed green -- Playwright and the
 * React-to-Python smoke fill every visible input before submitting, so a
 * wrongly-required field is always satisfied there, and the route probes never
 * touch a form at all.
 *
 * This sweeps every advanced screen and every tab, fills each form only until
 * its submit control enables -- the point at which the UI says the request is
 * ready -- and then asserts the form can actually be submitted. The detector
 * was first written and validated against the real built application, where it
 * flagged a reproduction of the original defect and left a healthy form alone.
 *
 * Two limits are worth knowing, because a green run here is not a proof that
 * every form is correct:
 *
 * - Fields are filled in DOM order, so a wrongly-required field that sits
 *   *before* the one that enables the control gets filled on the way past and
 *   the form looks valid. Reverting the original fix makes this file fail on
 *   assistant, engine and scraper, where the optional fields come after the
 *   enabling one, but not on the Developer webhook form, where the optional
 *   Event pattern comes first. `backend-action-optional-fields` covers that
 *   form, and the others, by asserting each form's validity directly.
 * - Forms behind a dialog or a collapsed `details` are not reached by
 *   rendering alone and are not covered at all.
 */

const ROUTES = [
  'voice', 'assistant', 'communications', 'security', 'developer', 'engine', 'science',
  'twin', 'vault', 'scraper', 'finetune', 'gallery', 'cookbook', 'ultramodel',
  'personalos', 'g25advanced', 'voiceconv', 'modelserve', 'factprovenance',
];

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  useWorkspaceState.getState().setDataMode('live');
});

function emptyBackend() {
  vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) =>
    new Response(JSON.stringify(String(init?.method ?? 'GET') === 'GET' ? {} : { ok: true }), {
      status: String(init?.method ?? 'GET') === 'GET' ? 200 : 201,
      headers: { 'content-type': 'application/json' },
    }),
  );
}

/** Sets a controlled input's value the way a person typing would. */
function typeInto(element: HTMLInputElement | HTMLTextAreaElement, value: string) {
  fireEvent.change(element, { target: { value } });
}

function sampleFor(element: HTMLInputElement | HTMLTextAreaElement): string {
  const type = (element as HTMLInputElement).type;
  if (type === 'number') return '1';
  if (type === 'url') return 'https://example.org/probe';
  if (type === 'email') return 'probe@example.org';
  if (type === 'date') return '2026-11-03';
  return 'probe';
}

/** Every form on the page that invites a click it cannot honour. */
function deadControls(): Array<{ submit: string; blockedBy: string[] }> {
  const dead: Array<{ submit: string; blockedBy: string[] }> = [];
  for (const form of Array.from(document.querySelectorAll('form'))) {
    const submit = form.querySelector('button:not([type="button"])') as HTMLButtonElement | null;
    if (!submit) continue;
    const fields = Array.from(
      form.querySelectorAll<HTMLInputElement | HTMLTextAreaElement>(
        'input:not([type="checkbox"]):not([type="radio"]):not([type="file"]),textarea',
      ),
    );
    // Fill in order and stop the moment the control enables.
    for (const field of fields) {
      if (!submit.disabled) break;
      if (field.value) continue;
      typeInto(field, sampleFor(field));
    }
    if (!submit.disabled && !form.checkValidity()) {
      dead.push({
        submit: submit.textContent?.trim().slice(0, 48) ?? '(unlabelled)',
        blockedBy: fields
          .filter(field => !field.checkValidity())
          .map(field => field.getAttribute('placeholder') || (field.closest('label')?.textContent ?? '').trim() || '(unlabelled field)'),
      });
    }
  }
  return dead;
}

describe('every rendered form can actually be submitted', () => {
  for (const route of ROUTES) {
    it(`has no dead control on ${route}`, async () => {
      emptyBackend();
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
      render(<QueryClientProvider client={client}><AdvancedScreen route={route} /></QueryClientProvider>);
      await waitFor(() => expect(document.querySelector('.p-card, .p-empty, section')).toBeTruthy());

      const tabs = Array.from(document.querySelectorAll('[role="tab"]')).map(tab => tab.textContent?.trim() ?? '');
      const found: Array<{ tab: string; submit: string; blockedBy: string[] }> = [];

      const sweep = (tab: string) => {
        for (const entry of deadControls()) found.push({ tab, ...entry });
      };

      if (!tabs.length) sweep('(no tabs)');
      else {
        for (const name of tabs) {
          const tab = Array.from(document.querySelectorAll('[role="tab"]')).find(
            candidate => candidate.textContent?.trim() === name,
          );
          if (!tab) continue;
          fireEvent.click(tab);
          await waitFor(() => expect(document.querySelector('section')).toBeTruthy());
          sweep(name);
        }
      }

      expect(found, `${route}: these submit controls are enabled but their form cannot validate, so clicking them does nothing and says nothing: ${JSON.stringify(found, null, 2)}`).toEqual([]);
    });
  }
});
