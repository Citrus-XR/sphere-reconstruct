import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// FastAPI backend は 127.0.0.1:8787 で動く前提. dev server から proxy する.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8787',
        changeOrigin: false,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})
