import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy API calls to Flask backend so CORS is never an issue
      '/health': 'http://localhost:5000',
      '/predict_landmarks': 'http://localhost:5000',
      '/caption': 'http://localhost:5000',
      '/train': 'http://localhost:5000',
      '/model': 'http://localhost:5000',
      '/socket.io': {
        target: 'http://localhost:5000',
        ws: true,           // ← critical: proxy WebSocket (Socket.IO) too
        changeOrigin: true,
      },
    },
  },
})