import express from "express";
import { existsSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  downloadYoutubeAudio,
  getYoutubeInfo,
  YoutubeError,
  youtubeDependencyVersion,
} from "./server/youtube.mjs";
import {
  SeparatorError,
  separatorDependencyVersion,
} from "./server/separator.mjs";
import {
  cancelJob,
  createSeparationJob,
  getJob,
  getJobResult,
  listJobs,
} from "./server/jobs.mjs";
import {
  CACHE_VERSIONS,
  cacheUsage,
  clearProcessingCache,
  loadTranscriptionCache,
  openProcessingCacheLocation,
  storeTranscriptionCache,
} from "./server/cache.mjs";
import {
  compareStoredReference,
  evaluationStorageInfo,
  importChordReference,
  latestAggregateReport,
  listChordReferences,
  saveManualReferenceAlignment,
} from "./server/evaluation-store.mjs";

export async function startTabsmithServer({
  production = false,
  host = "127.0.0.1",
  port = Number(process.env.PORT || 5173),
  staticRoot = process.env.TABSMITH_DIST_DIR || path.resolve("dist"),
} = {}) {
  const app = express();
  let activeDownload = false;
  let vite = null;

  app.disable("x-powered-by");
  // Cached transcription diagnostics can include thousands of retained raw
  // events. The server is loopback-only; keep enough room for that JSON while
  // retaining the existing audio-specific upload limit on separation.
  app.use(express.json({ limit: "50mb" }));

  app.get("/api/health", async (_request, response) => {
  const [youtube, separator] = await Promise.allSettled([
    youtubeDependencyVersion(),
    separatorDependencyVersion(),
  ]);
  response.status(youtube.status === "fulfilled" && separator.status === "fulfilled" ? 200 : 503).json({
    ok: true,
    youtube: youtube.status === "fulfilled",
    ytDlpVersion: youtube.status === "fulfilled" ? youtube.value : null,
    separator: separator.status === "fulfilled",
    separatorVersion: separator.status === "fulfilled" ? separator.value : null,
  });
  });

  app.get("/api/jobs", (_request, response) => response.json({ jobs: listJobs() }));

  app.get("/api/cache/config", (_request, response) => {
    response.json({ versions: CACHE_VERSIONS });
  });

  app.get("/api/cache/status", async (_request, response) => {
    try {
      response.json(await cacheUsage());
    } catch (error) {
      response.status(500).json({ error: error.message || "Could not inspect processing cache." });
    }
  });

  app.post("/api/cache/transcriptions/lookup", async (request, response) => {
    try {
      const cached = await loadTranscriptionCache(request.body?.key);
      response.status(cached.status === "hit" ? 200 : 404).json(cached);
    } catch (error) {
      response.status(400).json({ error: error.message || "Invalid transcription cache lookup." });
    }
  });

  app.put("/api/cache/transcriptions/:key", async (request, response) => {
    try {
      const stored = await storeTranscriptionCache(request.params.key, request.body);
      response.status(201).json({ stored: true, ...stored });
    } catch (error) {
      response.status(400).json({ error: error.message || "Could not store transcription cache." });
    }
  });

  app.delete("/api/cache/:kind", async (request, response) => {
    try {
      response.json(await clearProcessingCache(request.params.kind));
    } catch (error) {
      response.status(400).json({ error: error.message || "Could not clear processing cache." });
    }
  });

  app.post("/api/cache/open", async (_request, response) => {
    try {
      response.json(await openProcessingCacheLocation());
    } catch (error) {
      response.status(500).json({ error: error.message || "Could not open processing cache." });
    }
  });

  app.get("/api/evaluation/config", (_request, response) => {
    response.json(evaluationStorageInfo());
  });

  app.get("/api/evaluation/references", async (_request, response) => {
    try {
      response.json({ references: await listChordReferences() });
    } catch (error) {
      response.status(500).json({ error: error.message || "Could not list chord references." });
    }
  });

  app.post("/api/evaluation/references/import", async (request, response) => {
    try {
      response.status(201).json({ reference: await importChordReference(request.body || {}) });
    } catch (error) {
      response.status(400).json({ error: error.message || "Could not import the chord-only reference." });
    }
  });

  app.put("/api/evaluation/references/:id/alignment", async (request, response) => {
    try {
      response.json({
        reference: await saveManualReferenceAlignment(
          request.params.id,
          request.body?.marks,
          request.body?.duration,
        ),
      });
    } catch (error) {
      response.status(400).json({ error: error.message || "Could not save the manual alignment." });
    }
  });

  app.post("/api/evaluation/references/:id/compare", async (request, response) => {
    try {
      if (!request.body?.prediction?.chords) throw new Error("A timed Tabsmith chord prediction is required.");
      response.json(await compareStoredReference(
        request.params.id,
        request.body.prediction,
        request.body.context,
      ));
    } catch (error) {
      response.status(400).json({ error: error.message || "Could not evaluate the chord prediction." });
    }
  });

  app.get("/api/evaluation/report", async (_request, response) => {
    try {
      response.json(await latestAggregateReport());
    } catch (error) {
      response.status(500).json({ error: error.message || "Could not load the chord evaluation report." });
    }
  });

  app.post(
  "/api/jobs/separation",
  express.raw({ type: () => true, limit: "150mb" }),
  (request, response) => {
    try {
      let name = request.get("X-Input-Name") || "Audio";
      try { name = decodeURIComponent(name); } catch { /* Keep the safe server fallback. */ }
      const job = createSeparationJob(request.body, request.get("X-Input-Extension"), name);
      response.status(202).json(job);
    } catch (error) {
      const status = error instanceof SeparatorError ? error.status : 500;
      response.status(status).json({ error: error.message || "Guitar separation failed." });
    }
  },
  );

  app.get("/api/jobs/:id/result", (request, response) => {
    const audio = getJobResult(request.params.id, request.query.stem);
    if (!audio) return response.status(404).json({ error: "This job has no completed result." });
    response.set({
      "Content-Type": "audio/wav",
      "Content-Length": String(audio.size),
      "Cache-Control": "private, max-age=31536000, immutable",
      "X-Separation-Model": audio.model,
      "X-Separation-Stem": audio.stem,
    });
    response.sendFile(audio.path, { dotfiles: "allow" });
  });

  app.get("/api/jobs/:id", (request, response) => {
    const job = getJob(request.params.id);
    if (!job) return response.status(404).json({ error: "Processing job not found." });
    response.json(job);
  });

  app.delete("/api/jobs/:id", (request, response) => {
    const job = cancelJob(request.params.id);
    if (!job) return response.status(404).json({ error: "Processing job not found." });
    response.json(job);
  });

  app.post("/api/youtube/info", async (request, response) => {
  try {
    response.json(await getYoutubeInfo(request.body?.url));
  } catch (error) {
    const status = error instanceof YoutubeError ? error.status : 500;
    response.status(status).json({ error: error.message || "YouTube metadata failed." });
  }
  });

  app.post("/api/youtube/audio", async (request, response) => {
  if (activeDownload) return response.status(429).json({ error: "Another YouTube audio import is already running." });
  activeDownload = true;
  try {
    const audio = await downloadYoutubeAudio(request.body?.url);
    const mimeTypes = { m4a: "audio/mp4", mp4: "audio/mp4", webm: "audio/webm", opus: "audio/ogg", ogg: "audio/ogg", mp3: "audio/mpeg" };
    response.set({
      "Content-Type": mimeTypes[audio.extension] || "application/octet-stream",
      "Content-Length": String(audio.size),
      "Cache-Control": "no-store",
      "X-Audio-Extension": audio.extension,
    });
    response.sendFile(audio.path, async (error) => {
      await audio.cleanup().catch(() => {});
      activeDownload = false;
      if (error && !response.headersSent) response.status(500).json({ error: "Audio transfer failed." });
    });
  } catch (error) {
    activeDownload = false;
    const status = error instanceof YoutubeError ? error.status : 500;
    response.status(status).json({ error: error.message || "YouTube audio import failed." });
  }
  });

  if (production) {
    if (!existsSync(staticRoot)) throw new Error("Build the app with `npm run build` before starting production mode.");
    app.use(express.static(staticRoot));
    app.use((_request, response) => response.sendFile(path.join(staticRoot, "index.html")));
  } else {
    const { createServer: createViteServer } = await import("vite");
    vite = await createViteServer({ server: { middlewareMode: true }, appType: "spa" });
    app.use(vite.middlewares);
  }

  const server = await new Promise((resolve, reject) => {
    const listener = app.listen(port, host, () => resolve(listener));
    listener.once("error", reject);
  });
  const address = server.address();
  const actualPort = typeof address === "object" && address ? address.port : port;
  const url = `http://${host}:${actualPort}`;
  console.log(`Tabsmith running at ${url}`);
  return {
    app,
    server,
    url,
    async close() {
      await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
      await vite?.close();
    },
  };
}

const directEntry = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : null;
if (directEntry === import.meta.url) {
  await startTabsmithServer({ production: process.argv.includes("--production") });
}
