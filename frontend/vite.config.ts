import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import UnoCSS from 'unocss/vite';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, URL } from 'node:url';

const webOutput = fileURLToPath(new URL('../web', import.meta.url));

function serviceWorkerCacheFingerprint() {
  return {
    name: 'openagent-service-worker-cache-fingerprint',
    apply: 'build' as const,
    closeBundle() {
      const hash = createHash('sha256');
      for (const filename of ['index.html', 'app.js', 'styles.css']) {
        hash.update(filename).update('\0').update(readFileSync(`${webOutput}/${filename}`));
      }
      const cacheVersion = `openagent-vite-${hash.digest('hex').slice(0, 16)}`;
      const workerPath = `${webOutput}/sw.js`;
      const worker = readFileSync(workerPath, 'utf8');
      const updated = worker.replace(
        /const CACHE_VERSION = "[^"]+";/,
        `const CACHE_VERSION = "${cacheVersion}";`,
      );
      if (updated === worker && !worker.includes(`const CACHE_VERSION = "${cacheVersion}";`)) {
        throw new Error('Could not update the service worker cache fingerprint.');
      }
      writeFileSync(workerPath, updated);
    },
  };
}

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  base: '/',
  plugins: [UnoCSS(fileURLToPath(new URL('./uno.config.ts', import.meta.url))), react(), serviceWorkerCacheFingerprint()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  esbuild: { loader: 'tsx', include: /frontend\/src\/.*\.tsx?$/ },
  server: { proxy: { '/api': 'http://127.0.0.1:8765' } },
  build: {
    outDir: webOutput, emptyOutDir: false, cssCodeSplit: false,
    rollupOptions: { input: fileURLToPath(new URL('./index.html', import.meta.url)), output: { entryFileNames: 'app.js', chunkFileNames: 'assets/[name]-[hash].js', assetFileNames: asset => asset.name?.endsWith('.css') ? 'styles.css' : 'assets/[name]-[hash][extname]' } },
  },
});
