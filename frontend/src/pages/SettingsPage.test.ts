import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, createElement } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { settingsFields } from './SettingsPage';
import { accessibility } from '../state/accessibility';
import { theme } from '../state/theme';

const apiMocks = vi.hoisted(() => ({ get: vi.fn(), post: vi.fn() }));
vi.mock('../lib/react/api', () => ({
  api: apiMocks,
  record: (value: unknown) => value && typeof value === 'object' && !Array.isArray(value) ? value : {},
}));

describe('focused Settings sections', () => {
  it('keeps common runtime endpoints in view and groups tuning under advanced settings', () => {
    const fields = settingsFields('runtime');
    expect(fields.primary.map(field => field.key)).toEqual(['ollama_url', 'ollama_model', 'llamacpp_url', 'llamacpp_model']);
    expect(fields.advanced.map(field => field.key)).toContain('olmoe_io_mode');
    expect(fields.primary.length + fields.advanced.length).toBeGreaterThan(4);
  });

  it('shows core role defaults first while retaining every advanced role field', () => {
    const fields = settingsFields('models');
    expect(fields.primary.map(field => field.key)).toEqual(['chat_model', 'code_model', 'research_model', 'voice_model']);
    expect(fields.advanced.map(field => field.key)).toEqual(['speed_model', 'low_resource_code_model', 'extraction_model', 'high_context_research_model']);
    expect(fields.advanced.every(field => field.writable === false)).toBe(true);
  });

  describe('runtime edit flow', () => {
    let host: HTMLDivElement;
    let root: Root;

    beforeEach(() => {
      (globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
      apiMocks.get.mockReset().mockImplementation(async (path: string) => {
        if (path === '/api/settings/runtime') return { settings: { ollama_url: 'http://localhost:11434', ollama_model: 'qwen2.5', llamacpp_url: '', llamacpp_model: '', chat_model: 'qwen2.5' } };
        if (path === '/api/model-policy') return { routing_policy: 'local-first' };
        if (path === '/api/security/status') return {};
        if (path === '/api/auth/me') return { user: { id: 'user-1' } };
        if (path === '/api/backups') return { backups: [] };
        return {};
      });
      apiMocks.post.mockReset().mockImplementation(async (path: string, body: { settings?: Record<string, unknown> }) => path === '/api/settings/runtime' ? { settings: body.settings } : {});
      host = document.createElement('div'); document.body.append(host); root = createRoot(host);
    });

    afterEach(async () => { await act(async () => root.unmount()); host.remove(); });

    it('keeps refresh from discarding edits and exposes save/revert while settings are dirty', async () => {
      const { SettingsPage } = await import('./SettingsPage');
      await act(async () => root.render(createElement(SettingsPage)));
      await act(async () => [...host.querySelectorAll<HTMLButtonElement>('.settings-sections button')].find(button => button.textContent === 'Runtime')?.click());
      const endpoint = host.querySelector<HTMLInputElement>('.settings-workspace input');
      expect(endpoint?.value).toBe('http://localhost:11434');
      const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set;
      await act(async () => {
        if (endpoint) { setter?.call(endpoint, 'http://localhost:11435'); endpoint.dispatchEvent(new Event('input', { bubbles: true })); }
      });
      expect(host.querySelector('.settings-savebar')?.textContent).toContain('Unsaved changes');
      expect(host.querySelector<HTMLButtonElement>('button[aria-label="Refresh settings"]')?.disabled).toBe(true);

      await act(async () => host.querySelector<HTMLButtonElement>('.settings-savebar .btn:not(.primary)')?.click());
      expect(host.querySelector<HTMLInputElement>('.settings-workspace input')?.value).toBe('http://localhost:11434');
      expect(host.querySelector('.settings-savebar')).toBeNull();

      await act(async () => {
        const input = host.querySelector<HTMLInputElement>('.settings-workspace input');
        if (input) { setter?.call(input, 'http://localhost:11436'); input.dispatchEvent(new Event('input', { bubbles: true })); }
      });
      await act(async () => { host.querySelector<HTMLButtonElement>('.settings-savebar .btn.primary')?.click(); await new Promise(resolve => setTimeout(resolve, 0)); });
      expect(apiMocks.post).toHaveBeenCalledWith('/api/settings/runtime', expect.objectContaining({ settings: expect.objectContaining({ ollama_url: 'http://localhost:11436' }) }));
      const saved = apiMocks.post.mock.calls[0][1].settings as Record<string, unknown>;
      expect(saved).not.toHaveProperty('speed_model');
      expect(saved).not.toHaveProperty('olmoe_kernel');
      expect(host.querySelector('.settings-savebar')).toBeNull();
      expect(host.textContent).toContain('Settings saved.');
    });

    it('marks runtime-managed model defaults as read-only instead of presenting false editors', async () => {
      const { SettingsPage } = await import('./SettingsPage');
      await act(async () => root.render(createElement(SettingsPage)));
      await act(async () => [...host.querySelectorAll<HTMLButtonElement>('.settings-sections button')].find(button => button.textContent === 'Model defaults')?.click());
      await act(async () => host.querySelector<HTMLElement>('.settings-advanced summary')?.click());
      const speedField = [...host.querySelectorAll('label')].find(label => label.querySelector('strong')?.textContent === 'Speed model');
      expect(speedField?.querySelector('output')?.textContent).toBe('Not reported');
      expect(speedField?.querySelector('input,select')).toBeNull();
      expect(speedField?.textContent).toContain('not writable through the current settings service');
    });

    it('does not render blank runtime editors when the configuration request fails', async () => {
      apiMocks.get.mockImplementation(async (path: string) => {
        if (path === '/api/settings/runtime') throw new Error('offline');
        if (path === '/api/backups') return { backups: [] };
        return {};
      });
      const { SettingsPage } = await import('./SettingsPage');
      await act(async () => root.render(createElement(SettingsPage)));
      await act(async () => [...host.querySelectorAll<HTMLButtonElement>('.settings-sections button')].find(button => button.textContent === 'Runtime')?.click());
      expect(host.querySelector('[role="alert"]')?.textContent).toContain('runtime configuration');
      expect(host.querySelector('.settings-workspace .empty')?.textContent).toBe('Runtime settings could not be loaded.');
      expect(host.querySelector('.settings-workspace input')).toBeNull();
    });

    it('persists Appearance and reduced-motion choices locally and stays synchronized with shell theme changes', async () => {
      const { SettingsPage } = await import('./SettingsPage');
      await act(async () => root.render(createElement(SettingsPage)));
      const motionSelect = host.querySelector<HTMLSelectElement>('select[aria-label="Reduced motion"]');
      const themeOption = (value: string) => host.querySelector<HTMLButtonElement>(`.settings-theme-option[data-theme="${value}"]`);
      expect(host.querySelector<HTMLButtonElement>('.settings-theme-option[aria-pressed="true"]')?.textContent).toContain('System');
      expect(motionSelect?.value).toBe('system');
      await act(async () => {
        themeOption('dark')?.click();
        if (motionSelect) { motionSelect.value = 'reduce'; motionSelect.dispatchEvent(new Event('change', { bubbles: true })); }
      });
      expect(theme.read()).toBe('dark');
      expect(document.documentElement.dataset.theme).toBe('dark');
      expect(accessibility.read()).toBe('reduce');
      expect(document.documentElement.dataset.reducedMotion).toBe('reduce');
      expect(apiMocks.post).not.toHaveBeenCalled();

      await act(async () => theme.set('light'));
      expect(host.querySelector<HTMLButtonElement>('.settings-theme-option[aria-pressed="true"]')?.textContent).toContain('Light');
      await act(async () => accessibility.set('allow'));
      expect(host.querySelector<HTMLSelectElement>('select[aria-label="Reduced motion"]')?.value).toBe('allow');
    });
  });
});
