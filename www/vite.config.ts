import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const isCI = process.env['CI']

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: '/TileDB-SOMA-ML/shuffle',
  build: {
    outDir: isCI ? 'dist/shuffle' : 'dist/TileDB-SOMA-ML/shuffle',
  }
})
