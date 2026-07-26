import { defineConfig, loadEnv } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  // Production/release builds hard-code this to false even if a caller sets the
  // development flag. Rollup can then remove the worker's dynamic model import.
  const experimentalHybrid = mode !== "production"
    && env.TABSMITH_EXPERIMENTAL_LEARNED_HARMONY === "1";
  return {
    define: {
      __TABSMITH_EXPERIMENTAL_HYBRID__: JSON.stringify(experimentalHybrid),
    },
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
  };
});
