import { createHash, randomUUID } from "node:crypto";
import { copyFile, mkdir, rename, stat, unlink } from "node:fs/promises";
import path from "node:path";
import { normalizeAudioExtension, separateGuitar, validateAudioInput } from "./separator.mjs";
import {
  separationCacheRoot,
  SOURCE_SEPARATION_MODEL_VERSION,
} from "./cache.mjs";

const jobs = new Map();
let queue = Promise.resolve();

function publicJob(job) {
  return {
    id: job.id,
    kind: "guitar-separation",
    name: job.name,
    status: job.status,
    stage: job.stage,
    progress: job.progress,
    cached: job.cached,
    error: job.error,
    createdAt: job.createdAt,
    updatedAt: job.updatedAt,
  };
}

function update(job, changes) {
  Object.assign(job, changes, { updatedAt: new Date().toISOString() });
}

async function runJob(job) {
  if (job.status === "cancelled") return;
  update(job, { status: "running", stage: "Checking local cache", progress: 0.01 });
  const outputDirectory = separationCacheRoot();
  const resultPaths = Object.fromEntries(
    ["guitar", "bass", "harmony"].map(
      (stem) => [stem, path.join(outputDirectory, `${job.cacheKey}.${stem}.wav`)],
    ),
  );
  await mkdir(outputDirectory, { recursive: true });
  try {
    const cached = Object.fromEntries(await Promise.all(
      Object.entries(resultPaths).map(async ([stem, resultPath]) => [
        stem,
        await stat(resultPath).catch(() => null),
      ]),
    ));
    if (Object.values(cached).every((details) => details?.isFile() && details.size > 0)) {
      update(job, {
        status: "completed",
        stage: "Loaded cached harmonic stems",
        progress: 1,
        cached: true,
        resultPaths,
        sizes: Object.fromEntries(
          Object.entries(cached).map(([stem, details]) => [stem, details.size]),
        ),
      });
      return;
    }
    const audio = await separateGuitar(job.input, job.extension, {
      signal: job.controller.signal,
      onProgress: (progress, stage) => {
        if (job.status !== "cancelled") update(job, { progress, stage });
      },
    });
    const partialPaths = Object.fromEntries(
      Object.entries(resultPaths).map(
        ([stem, resultPath]) => [stem, `${resultPath}.${job.id}.partial`],
      ),
    );
    try {
      for (const stem of ["guitar", "bass", "harmony"]) {
        await copyFile(audio.paths[stem], partialPaths[stem]);
      }
      for (const stem of ["guitar", "bass", "harmony"]) {
        await rename(partialPaths[stem], resultPaths[stem]);
      }
    } finally {
      await Promise.all(
        Object.values(partialPaths).map((partialPath) => unlink(partialPath).catch(() => {})),
      );
      await audio.cleanup().catch(() => {});
    }
    const sizes = Object.fromEntries(await Promise.all(
      Object.entries(resultPaths).map(async ([stem, resultPath]) => [
        stem,
        (await stat(resultPath)).size,
      ]),
    ));
    update(job, {
      status: "completed",
      stage: "Harmonic stems ready",
      progress: 1,
      resultPaths,
      sizes,
    });
  } catch (error) {
    if (job.controller.signal.aborted || error?.status === 499) {
      update(job, { status: "cancelled", stage: "Cancelled", error: null });
    } else {
      update(job, {
        status: "failed",
        stage: "Separation failed",
        error: error instanceof Error ? error.message : "Guitar separation failed.",
      });
    }
  } finally {
    job.input = null;
    job.controller = null;
  }
}

export function createSeparationJob(input, extension, name = "Audio") {
  validateAudioInput(input, extension);
  const normalizedExtension = normalizeAudioExtension(extension);
  const cacheKey = createHash("sha256")
    .update(SOURCE_SEPARATION_MODEL_VERSION)
    .update("\0")
    .update(input)
    .digest("hex");
  const now = new Date().toISOString();
  const job = {
    id: randomUUID(),
    name: String(name || "Audio").replace(/[\u0000-\u001f]/g, "").slice(0, 160),
    extension: normalizedExtension,
    cacheKey,
    input,
    status: "queued",
    stage: "Queued",
    progress: 0,
    cached: false,
    error: null,
    resultPaths: null,
    sizes: null,
    controller: new AbortController(),
    createdAt: now,
    updatedAt: now,
  };
  jobs.set(job.id, job);
  queue = queue.then(() => runJob(job), () => runJob(job));
  return publicJob(job);
}

export function getJob(id) {
  const job = jobs.get(id);
  return job ? publicJob(job) : null;
}

export function listJobs() {
  return [...jobs.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, 30).map(publicJob);
}

export function cancelJob(id) {
  const job = jobs.get(id);
  if (!job) return null;
  if (!["completed", "failed", "cancelled"].includes(job.status)) {
    update(job, { status: "cancelled", stage: "Cancelling", progress: job.progress });
    job.controller?.abort();
  }
  return publicJob(job);
}

export function getJobResult(id, requestedStem = "guitar") {
  const job = jobs.get(id);
  const stem = ["guitar", "bass", "harmony"].includes(requestedStem)
    ? requestedStem
    : "guitar";
  if (!job || job.status !== "completed" || !job.resultPaths?.[stem]) return null;
  return {
    path: job.resultPaths[stem],
    size: job.sizes[stem],
    stem,
    model: SOURCE_SEPARATION_MODEL_VERSION,
  };
}
