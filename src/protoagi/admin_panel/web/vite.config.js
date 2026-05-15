import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
// The Python admin server runs on :8765 by default; Vite proxies
// /api/* there during dev so we don't need CORS to be open in prod.
const ADMIN_TARGET = process.env.PROTOAGI_ADMIN_URL ?? "http://127.0.0.1:8765";
export default defineConfig({
    plugins: [react(), tailwindcss()],
    server: {
        port: 5173,
        proxy: {
            "/api": { target: ADMIN_TARGET, changeOrigin: true },
        },
    },
    build: {
        outDir: "dist",
        emptyOutDir: true,
    },
});
