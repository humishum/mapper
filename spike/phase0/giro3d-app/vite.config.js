import { defineConfig } from 'vite';

// The COPC artifacts are served by spike/phase0/range_server.py (byte ranges + CORS), not
// by Vite: `python -m http.server` and Vite's static middleware differ enough on Range
// handling that measuring against the same server we intend to ship behaviour for matters.
export default defineConfig({
    server: {
        host: '127.0.0.1',
        port: 5180,
        strictPort: true,
    },
    build: {
        target: 'esnext',
    },
});
