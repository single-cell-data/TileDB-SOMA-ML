import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const isCI = process.env['CI']

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  ...(isCI ? { base: '/TileDB-SOMA-ML/', } : {}),
  build: {
    outDir: isCI ? 'dist' : 'dist/TileDB-SOMA-ML',
  }
})
