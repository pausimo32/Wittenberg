import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The built output is served by FastAPI from `web/dist` (docs/05 §2, §3), so the app is
// always same-origin in production and needs no CORS. In development Vite proxies /api to
// the Python server on 8000 for the same reason: keeping the browser on one origin means
// the API never has to relax anything to be usable.
//
// `host` is left at Vite's default (localhost). Never bind the dev server to 0.0.0.0 —
// `docs/adr/0004` says the app is not reachable from another device, and that has to hold
// for the development server too or someone will demo from it.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // No CDN, no remote font, no analytics: everything the page needs is in the bundle,
    // because a field session runs with the machine offline (docs/05 §3).
    assetsInlineLimit: 4096,
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
