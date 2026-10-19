import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import { fileURLToPath, URL } from 'node:url';
import { readFileSync } from 'node:fs';

const demoFixtureRoot = fileURLToPath(new URL('./public/demo', import.meta.url));

function localDemoFixtures() {
  return {
    name: 'openagent-local-demo-fixtures',
    apply: 'serve' as const,
    configureServer(server: import('vite').ViteDevServer) {
      server.middlewares.use('/demo', (request, response, next) => {
        const fileName = request.url?.split('?')[0]?.replace(/^\//, '');
        if (!['reference-fixtures.json', 'approved-shell-fixtures.json'].includes(fileName ?? '')) return next();
        response.setHeader('Content-Type', 'application/json; charset=utf-8');
        response.setHeader('Cache-Control', 'no-store');
        response.end(readFileSync(`${demoFixtureRoot}/${fileName}`));
      });
    },
  };
}

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  return {
  plugins: [react(), tailwindcss(), localDemoFixtures()],
  // Demo examples are loaded by the dev-only middleware above. Never copy
  // public/demo into Vite build output.
  publicDir: false,
  server: {
    host: '127.0.0.1',
    port: 5180,
    strictPort: true,
    proxy: {
      '/api': {
        target: env.OPENAGENT_API_TARGET || 'http://127.0.0.1:8765',
        changeOrigin: true,
        // The isolated personal_remote smoke emulates the headers produced by
        // a private HTTPS reverse proxy. This is test-only and never enabled
        // for ordinary development or production builds.
        configure: (proxy) => {
          if (env.OPENAGENT_TEST_PERSONAL_REMOTE_PROXY !== '1') return;
          proxy.on('proxyReq', (proxyRequest) => {
            proxyRequest.setHeader('Host', 'remote.test');
            proxyRequest.setHeader('X-Forwarded-Proto', 'https');
            proxyRequest.setHeader('Origin', 'https://remote.test');
          });
        },
      },
    },
  },
  build: {
    // Keep routine builds isolated. Promotion into the Python-served web/ tree
    // requires a separately approved deployment step.
    outDir: fileURLToPath(new URL('./dist', import.meta.url)),
    emptyOutDir: true,
    rollupOptions: {
      output: {
        entryFileNames: 'app.js',
        chunkFileNames: 'assets/[name]-[hash].js',
        assetFileNames: (asset) => asset.name?.endsWith('.css') ? 'styles.css' : 'assets/[name]-[hash][extname]',
        manualChunks(id) {
          if (!id.includes('node_modules')) return;
          if (/node_modules\/(react|react-dom|scheduler)\//.test(id)) return 'react-vendor';
          if (/node_modules\/(motion|framer-motion)\//.test(id)) return 'motion-vendor';
          if (/node_modules\/react-router/.test(id)) return 'router-vendor';
        },
      },
    },
  },
  };
});
