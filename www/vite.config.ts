import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const isCI = process.env['CI']

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // CI deploys to GitHub Pages, which has an extra `/TileDB-SOMA-ML` path prefix (taken from the repo name)
  base: isCI ? '/TileDB-SOMA-ML/shuffle' : '/shuffle',
  build: { outDir: 'dist/shuffle', }
})
