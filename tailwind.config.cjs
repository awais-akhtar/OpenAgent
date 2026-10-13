/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./frontend/index.html', './frontend/src/**/*.{js,ts,jsx,tsx}'],
  darkMode: ['selector', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        background: 'var(--oa-bg)',
        foreground: 'var(--oa-text)',
        surface: 'var(--oa-surface)',
        muted: 'var(--oa-surface-muted)',
        border: 'var(--oa-border)',
        primary: { DEFAULT: 'var(--oa-primary)', foreground: 'var(--oa-primary-text)' },
        destructive: { DEFAULT: 'var(--oa-danger)', foreground: '#fff' },
        ring: 'var(--oa-focus)',
      },
      borderRadius: { md: 'var(--oa-radius)', lg: 'calc(var(--oa-radius) + 2px)' },
    },
  },
  plugins: [],
};
