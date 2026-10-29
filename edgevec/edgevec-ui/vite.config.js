import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: "/edgevec-ui/", // Required for GitHub Pages — assets resolve relative to the repo name

  optimizeDeps: {
    // Do NOT pre-bundle Transformers.js v3 — it uses dynamic WASM imports internally
    exclude: ["@huggingface/transformers"],
  },

  server: {
    headers: {
      // Required for SharedArrayBuffer (enables WASM multi-threading)
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});

