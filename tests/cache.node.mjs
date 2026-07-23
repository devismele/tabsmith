import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import {
  mkdir,
  mkdtemp,
  rm,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test, { after, before } from "node:test";
import {
  CACHE_VERSIONS,
  loadTranscriptionCache,
  SOURCE_SEPARATION_MODEL_VERSION,
  storeTranscriptionCache,
  transcriptionCacheRoot,
} from "../server/cache.mjs";
import {
  createSeparationJob,
  getJob,
  getJobResult,
} from "../server/jobs.mjs";

let temporaryRoot;
let previousSeparation;
let previousTranscription;

before(async () => {
  temporaryRoot = await mkdtemp(path.join(os.tmpdir(), "tabsmith-cache-test-"));
  previousSeparation = process.env.TABSMITH_CACHE_DIR;
  previousTranscription = process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR;
  process.env.TABSMITH_CACHE_DIR = path.join(temporaryRoot, "separation");
  process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR = path.join(temporaryRoot, "transcription");
});

after(async () => {
  if (previousSeparation === undefined) delete process.env.TABSMITH_CACHE_DIR;
  else process.env.TABSMITH_CACHE_DIR = previousSeparation;
  if (previousTranscription === undefined) delete process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR;
  else process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR = previousTranscription;
  await rm(temporaryRoot, { recursive: true, force: true });
});

function metadata() {
  return {
    cacheSchemaVersion: CACHE_VERSIONS.transcriptionCacheSchemaVersion,
    transcriptionPipelineVersion: CACHE_VERSIONS.transcriptionPipelineVersion,
    noteCleanupVersion: CACHE_VERSIONS.noteCleanupVersion,
    arrangementVersion: CACHE_VERSIONS.arrangementVersion,
    fretboardMapperVersion: CACHE_VERSIONS.fretboardMapperVersion,
    chordAnalysisVersion: CACHE_VERSIONS.chordAnalysisVersion,
    createdAt: new Date().toISOString(),
  };
}

function result() {
  return {
    notes: [],
    chords: [],
    noteAnalysis: { rawNotes: [] },
    chordAnalysis: { rawFrames: [] },
  };
}

test("server rejects a legacy transcription file without metadata", async () => {
  const key = "a".repeat(64);
  await mkdir(transcriptionCacheRoot(), { recursive: true });
  await writeFile(
    path.join(transcriptionCacheRoot(), `${key}.json`),
    JSON.stringify({ cacheKey: key, settings: {}, result: result() }),
  );
  const cached = await loadTranscriptionCache(key);
  assert.equal(cached.status, "stale");
  assert.match(cached.reason, /no transcription cache metadata/);
});

test("server stores and reuses a current compatible transcription", async () => {
  const key = "b".repeat(64);
  await storeTranscriptionCache(key, {
    metadata: metadata(),
    cacheKey: key,
    settings: { arrangementMode: "lead" },
    result: result(),
  });
  const cached = await loadTranscriptionCache(key);
  assert.equal(cached.status, "hit");
  assert.equal(cached.entry.settings.arrangementMode, "lead");
});

test("a compatible Demucs stem is reused while transcription remains a miss", async () => {
  const wav = Buffer.alloc(44);
  wav.write("RIFF", 0, "ascii");
  wav.write("WAVE", 8, "ascii");
  const cacheKey = createHash("sha256")
    .update(SOURCE_SEPARATION_MODEL_VERSION)
    .update("\0")
    .update(wav)
    .digest("hex");
  await mkdir(process.env.TABSMITH_CACHE_DIR, { recursive: true });
  await Promise.all(
    ["guitar", "bass", "harmony"].map((stem) => writeFile(
      path.join(process.env.TABSMITH_CACHE_DIR, `${cacheKey}.${stem}.wav`),
      wav,
    )),
  );

  const job = createSeparationJob(wav, "wav", "Hotel regression.wav");
  let completed;
  for (let attempt = 0; attempt < 100; attempt += 1) {
    completed = getJob(job.id);
    if (completed.status === "completed") break;
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
  assert.equal(completed.status, "completed");
  assert.equal(completed.cached, true);
  assert.match(getJobResult(job.id, "guitar").path, /\.guitar\.wav$/);
  assert.match(getJobResult(job.id, "bass").path, /\.bass\.wav$/);
  assert.match(getJobResult(job.id, "harmony").path, /\.harmony\.wav$/);
  assert.equal((await loadTranscriptionCache("c".repeat(64))).status, "miss");
});
