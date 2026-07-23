import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";

const versions = JSON.parse(readFileSync(
  new URL("../shared/pipeline-versions.json", import.meta.url),
  "utf8",
));

export const CACHE_VERSIONS = Object.freeze(versions);
export const SOURCE_SEPARATION_MODEL_VERSION =
  versions.sourceSeparationModelVersion;

export function separationCacheRoot() {
  return process.env.TABSMITH_CACHE_DIR
    || path.resolve(".cache", "tabsmith", "separation");
}

export function transcriptionCacheRoot() {
  return process.env.TABSMITH_TRANSCRIPTION_CACHE_DIR
    || path.join(path.dirname(separationCacheRoot()), "transcription");
}

export function processingCacheRoot() {
  return path.dirname(separationCacheRoot());
}

function cachePathForKey(key) {
  if (!/^[a-f0-9]{64}$/.test(String(key))) {
    throw new Error("Invalid transcription cache key.");
  }
  return path.join(transcriptionCacheRoot(), `${key}.json`);
}

const METADATA_FIELDS = [
  ["cacheSchemaVersion", versions.transcriptionCacheSchemaVersion],
  ["transcriptionPipelineVersion", versions.transcriptionPipelineVersion],
  ["noteCleanupVersion", versions.noteCleanupVersion],
  ["arrangementVersion", versions.arrangementVersion],
  ["fretboardMapperVersion", versions.fretboardMapperVersion],
  ["chordAnalysisVersion", versions.chordAnalysisVersion],
];

export function validateTranscriptionMetadata(metadata) {
  if (!metadata || typeof metadata !== "object") {
    return { valid: false, reason: "entry has no transcription cache metadata" };
  }
  for (const [field, expected] of METADATA_FIELDS) {
    if (metadata[field] !== expected) {
      return {
        valid: false,
        reason: `${field} ${String(metadata[field] ?? "missing")} does not match ${String(expected)}`,
      };
    }
  }
  if (!metadata.createdAt || Number.isNaN(Date.parse(metadata.createdAt))) {
    return { valid: false, reason: "createdAt is missing or invalid" };
  }
  return { valid: true, reason: null };
}

export async function loadTranscriptionCache(key) {
  const file = cachePathForKey(key);
  const stored = await readFile(file, "utf8").catch((error) => {
    if (error?.code === "ENOENT") return null;
    throw error;
  });
  if (stored === null) return { status: "miss", entry: null, reason: null };
  let entry;
  try {
    entry = JSON.parse(stored);
  } catch {
    const reason = "entry is not valid JSON";
    console.warn(`Ignoring stale transcription cache: ${reason}`);
    return { status: "stale", entry: null, reason };
  }
  const validation = validateTranscriptionMetadata(entry?.metadata);
  if (!validation.valid) {
    console.warn(`Ignoring stale transcription cache: ${validation.reason}`);
    return { status: "stale", entry: null, reason: validation.reason };
  }
  if (entry.cacheKey !== key
    || !entry.result?.noteAnalysis?.rawNotes
    || !entry.result?.chordAnalysis?.rawFrames
    || !Array.isArray(entry.result?.notes)
    || !Array.isArray(entry.result?.chords)
    || !entry.settings) {
    const reason = "entry is incomplete or belongs to another key";
    console.warn(`Ignoring stale transcription cache: ${reason}`);
    return { status: "stale", entry: null, reason };
  }
  return { status: "hit", entry, reason: null };
}

export async function storeTranscriptionCache(key, entry) {
  const validation = validateTranscriptionMetadata(entry?.metadata);
  if (!validation.valid) {
    throw new Error(`Refusing invalid transcription cache entry: ${validation.reason}`);
  }
  if (entry.cacheKey !== key) throw new Error("Transcription cache key mismatch.");
  const directory = transcriptionCacheRoot();
  const destination = cachePathForKey(key);
  const partial = `${destination}.${process.pid}.${Date.now()}.partial`;
  await mkdir(directory, { recursive: true });
  try {
    await writeFile(partial, JSON.stringify(entry), { encoding: "utf8", flag: "wx" });
    // Reprocessing intentionally replaces the entry for the same canonical
    // settings only after the new JSON has been written successfully.
    await rm(destination, { force: true });
    await rename(partial, destination);
  } finally {
    await rm(partial, { force: true }).catch(() => {});
  }
  const details = await stat(destination);
  return { size: details.size, path: destination };
}

async function directoryUsage(directory) {
  const entries = await readdir(directory, { withFileTypes: true }).catch((error) => {
    if (error?.code === "ENOENT") return [];
    throw error;
  });
  let bytes = 0;
  let files = 0;
  for (const entry of entries) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      const nested = await directoryUsage(target);
      bytes += nested.bytes;
      files += nested.files;
    } else if (entry.isFile()) {
      const details = await stat(target);
      bytes += details.size;
      files += 1;
    }
  }
  return { bytes, files };
}

export async function cacheUsage() {
  const [separation, transcription] = await Promise.all([
    directoryUsage(separationCacheRoot()),
    directoryUsage(transcriptionCacheRoot()),
  ]);
  return {
    separation,
    transcription,
    total: {
      bytes: separation.bytes + transcription.bytes,
      files: separation.files + transcription.files,
    },
    location: processingCacheRoot(),
  };
}

async function clearExactCacheDirectory(directory) {
  const resolved = path.resolve(directory);
  const allowed = [
    path.resolve(separationCacheRoot()),
    path.resolve(transcriptionCacheRoot()),
  ];
  if (!allowed.includes(resolved)) throw new Error("Unsafe cache deletion target.");
  await rm(resolved, { recursive: true, force: true });
  await mkdir(resolved, { recursive: true });
}

export async function clearProcessingCache(kind) {
  if (kind === "separation") {
    await clearExactCacheDirectory(separationCacheRoot());
  } else if (kind === "transcription") {
    await clearExactCacheDirectory(transcriptionCacheRoot());
  } else if (kind === "all") {
    await clearExactCacheDirectory(transcriptionCacheRoot());
    await clearExactCacheDirectory(separationCacheRoot());
  } else {
    throw new Error("Unknown cache kind.");
  }
  return cacheUsage();
}

export async function openProcessingCacheLocation() {
  const location = processingCacheRoot();
  await mkdir(location, { recursive: true });
  const command = process.platform === "win32"
    ? { executable: "explorer.exe", args: [location] }
    : process.platform === "darwin"
      ? { executable: "open", args: [location] }
      : { executable: "xdg-open", args: [location] };
  const child = spawn(command.executable, command.args, {
    detached: true,
    stdio: "ignore",
    windowsHide: true,
  });
  child.unref();
  return { location };
}
