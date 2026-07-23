import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/@spotify/basic-pitch/model/*",
          dest: "basic-pitch-model",
          rename: { stripBase: true },
        },
      ],
    }),
  ],
  worker: { format: "es" },
  server: { host: "127.0.0.1", port: 5173 },
  preview: { host: "127.0.0.1", port: 4173 },
  test: { environment: "node" },
});
