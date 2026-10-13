import { defineConfig, presetWind3 } from 'unocss';

export default defineConfig({
  presets: [presetWind3()],
  shortcuts: {
    'oa-focus-ring': 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--oa-focus)]',
  },
  theme: {
    colors: {
      oa: {
        bg: 'var(--oa-bg)',
        subtle: 'var(--oa-surface-muted)',
        panel: 'var(--oa-surface)',
        text: 'var(--oa-text)',
        muted: 'var(--oa-muted)',
        border: 'var(--oa-border)',
        focus: 'var(--oa-focus)',
        primary: 'var(--oa-primary)',
      },
    },
  },
});
