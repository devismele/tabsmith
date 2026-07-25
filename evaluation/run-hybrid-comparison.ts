import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  mkdir,
  readFile,
  readdir,
  stat,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  HYBRID_DECODER_VERSION,
} from "../src/learnedHarmony/hybridDecoder";
import {
  LEARNED_MODEL_CHECKSUM,
  LEARNED_MODEL_VERSION,
} from "../src/learnedHarmony/modelIdentity";
import {
  LearnedTcnProvider,
  type TcnWeights,
} from "../src/learnedHarmony/onnxProvider";
import { FEATURE_VERSION } from "../src/learnedHarmony/contract";
import type { HybridHarmonySettings } from "../src/learnedHarmony/types";
import bundledWeights from "../src/learnedHarmony/model/app-chord-model.weights.json";
import {
  compareTrackEngines,
  DEFAULT_ML_ONLY_TRANSITION_PENALTY,
  type EvaluationReferenceRegion,
  type PreparedEvaluationTrack,
  type SourceType,
} from "./hybridComparison";
import { scoreTrackComparison } from "./hybridMetrics";
import {
  assertNoAbsolutePaths,
  buildHybridAccuracyReport,
  hybridAccuracyPerTrackCsv,
  hybridAccuracyReportMarkdown,
  hybridDisagreementReport,
  type DatasetReportIdentity,
} from "./hybridReport";

interface DatasetConfig {
  datasetIdentifier: string;
  manifestPath: string;
  annotationDirectory: string;
  splitName: "development" | "validation" | "test";
  splitStrategy?: "manifest" | "artist-hash";
  splitSeed?: number;
  heldOutArtists?: string[];
  sourceType: SourceType;
  sourceMode: "full-mix" | "guitar-focused";
  audioRoot?: string;
  trackLimit?: number;
  runAblations?: boolean;
}

interface EvaluationConfig {
  schemaVersion: 1;
  sampleRate: number;
  outputDirectory: string;
  includeLegacyRegionHybrid: boolean;
  combineSourceTypes: boolean;
  timeoutMs?: number;
  mlOnlyTransitionPenalty?: number;
  bootstrap: { seed: number; iterations: number };
  shortRegionThresholdSeconds?: number;
  hybridSettings?: Partial<HybridHarmonySettings>;
  datasets: DatasetConfig[];
}

interface DatasetManifest {
  splitAssignment?: {
    trackSplit?: Record<string, string>;
    warnings?: string[];
  };
}

interface AnnotationTrack {
  track_id: string;
  artist: string;
  title: string;
  duration: number;
  source: string;
  audio_availability: string;
  split?: string;
  audio_path?: string;
  beats?: number[];
  chords: Array<{ start: number; end: number; label: string }>;
}

interface LoadedDataset {
  config: DatasetConfig;
  identity: DatasetReportIdentity;
  tracks: PreparedEvaluationTrack[];
}

interface DecodedWav {
  samples: Float32Array;
  sampleRate: number;
}

function sha256(value: Buffer | string): string {
  return createHash("sha256").update(value).digest("hex");
}

function artistKey(artist: string): string {
  return artist.trim().toLowerCase().replace(/\s+/g, " ");
}

function artistHashSplit(
  artist: string,
  seed: number,
): "training" | "development" | "validation" {
  const digest = sha256(`${seed}:${artistKey(artist)}`);
  const bucket = Number.parseInt(digest.slice(0, 8), 16) % 100;
  if (bucket < 70) return "training";
  if (bucket < 85) return "development";
  return "validation";
}

function parseArgs(argv: string[]): { configPath: string; outputDirectory?: string } {
  let configPath = "evaluation/hybrid-accuracy-config.json";
  let outputDirectory: string | undefined;
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] === "--config" && argv[index + 1]) {
      configPath = argv[++index];
    } else if (argv[index] === "--output" && argv[index + 1]) {
      outputDirectory = argv[++index];
    } else if (argv[index] === "--help" || argv[index] === "-h") {
      console.log(
        "Usage: npm run evaluate:hybrid -- [--config <json>] [--output <directory>]",
      );
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${argv[index]}`);
    }
  }
  return { configPath: path.resolve(configPath), outputDirectory };
}

function resolveFromConfig(configPath: string, target: string): string {
  return path.resolve(path.dirname(configPath), target);
}

function readUInt24LE(buffer: Buffer, offset: number): number {
  return buffer[offset]
    | buffer[offset + 1] << 8
    | buffer[offset + 2] << 16;
}

export function decodePcmWav(buffer: Buffer): DecodedWav {
  if (buffer.toString("ascii", 0, 4) !== "RIFF"
    || buffer.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Only RIFF/WAVE audio is supported by the parity harness.");
  }
  let offset = 12;
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;
  while (offset + 8 <= buffer.length) {
    const chunk = buffer.toString("ascii", offset, offset + 4);
    const length = buffer.readUInt32LE(offset + 4);
    const payload = offset + 8;
    if (chunk === "fmt ") {
      format = buffer.readUInt16LE(payload);
      channels = buffer.readUInt16LE(payload + 2);
      sampleRate = buffer.readUInt32LE(payload + 4);
      bitsPerSample = buffer.readUInt16LE(payload + 14);
    } else if (chunk === "data") {
      dataOffset = payload;
      dataLength = Math.min(length, buffer.length - payload);
      break;
    }
    offset = payload + length + (length % 2);
  }
  if (dataOffset < 0 || !channels || !sampleRate || !bitsPerSample) {
    throw new Error("WAV is missing a supported fmt/data chunk.");
  }
  if (format !== 1 && format !== 3) {
    throw new Error(`Unsupported WAV format ${format}; expected PCM or IEEE float.`);
  }
  const bytesPerSample = bitsPerSample / 8;
  const frameBytes = bytesPerSample * channels;
  const frameCount = Math.floor(dataLength / frameBytes);
  const samples = new Float32Array(frameCount);
  for (let frame = 0; frame < frameCount; frame += 1) {
    let mixed = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      const sampleOffset = dataOffset + frame * frameBytes + channel * bytesPerSample;
      if (format === 3 && bitsPerSample === 32) {
        mixed += buffer.readFloatLE(sampleOffset);
      } else if (format === 1 && bitsPerSample === 8) {
        mixed += (buffer[sampleOffset] - 128) / 128;
      } else if (format === 1 && bitsPerSample === 16) {
        mixed += buffer.readInt16LE(sampleOffset) / 32768;
      } else if (format === 1 && bitsPerSample === 24) {
        let value = readUInt24LE(buffer, sampleOffset);
        if (value & 0x800000) value |= 0xFF000000;
        mixed += value / 8388608;
      } else if (format === 1 && bitsPerSample === 32) {
        mixed += buffer.readInt32LE(sampleOffset) / 2147483648;
      } else {
        throw new Error(
          `Unsupported WAV encoding: format=${format}, bits=${bitsPerSample}`,
        );
      }
    }
    samples[frame] = mixed / channels;
  }
  return { samples, sampleRate };
}

export function resampleAudio(
  samples: Float32Array,
  sourceRate: number,
  targetRate: number,
): Float32Array {
  if (sourceRate === targetRate) return samples.slice();
  if (sourceRate <= 0 || targetRate <= 0 || !samples.length) return new Float32Array();
  const targetLength = Math.max(1, Math.round(samples.length * targetRate / sourceRate));
  const output = new Float32Array(targetLength);
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < targetLength; index += 1) {
    const position = index * ratio;
    const lower = Math.min(samples.length - 1, Math.floor(position));
    const upper = Math.min(samples.length - 1, lower + 1);
    const fraction = position - lower;
    output[index] = samples[lower] * (1 - fraction) + samples[upper] * fraction;
  }
  return output;
}

function bpmFromBeats(beats: number[]): number | null {
  const differences = beats.slice(1).flatMap((beat, index) =>
    beat > beats[index] ? [beat - beats[index]] : []);
  if (!differences.length) return null;
  const ordered = differences.sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  const median = ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
  return median > 0 ? 60 / median : null;
}

function resolveAudioPath(
  annotationPath: string,
  annotation: AnnotationTrack,
  dataset: DatasetConfig,
  configPath: string,
): string {
  if (!annotation.audio_path) {
    throw new Error(`${annotation.track_id}: annotation does not reference audio`);
  }
  if (dataset.audioRoot) {
    return path.join(
      resolveFromConfig(configPath, dataset.audioRoot),
      path.basename(annotation.audio_path),
    );
  }
  return path.isAbsolute(annotation.audio_path)
    ? annotation.audio_path
    : path.resolve(path.dirname(annotationPath), annotation.audio_path);
}

function validateReference(
  trackId: string,
  regions: EvaluationReferenceRegion[],
): void {
  if (!regions.length) throw new Error(`${trackId}: no timed chord annotations`);
  let end = -Infinity;
  for (const region of regions) {
    if (!Number.isFinite(region.start) || !Number.isFinite(region.end)
      || region.end <= region.start) {
      throw new Error(`${trackId}: invalid chord interval`);
    }
    if (region.start < end - 1e-6) {
      throw new Error(`${trackId}: overlapping chord annotations`);
    }
    end = region.end;
  }
}

async function exists(target: string): Promise<boolean> {
  try {
    await stat(target);
    return true;
  } catch {
    return false;
  }
}

async function loadDataset(
  dataset: DatasetConfig,
  configPath: string,
  sampleRate: number,
): Promise<LoadedDataset> {
  const manifestPath = resolveFromConfig(configPath, dataset.manifestPath);
  const annotationDirectory = resolveFromConfig(
    configPath,
    dataset.annotationDirectory,
  );
  if (!await exists(manifestPath)) {
    throw new Error(
      `${dataset.datasetIdentifier}: dataset manifest is missing at the configured relative path`,
    );
  }
  if (!await exists(annotationDirectory)) {
    throw new Error(
      `${dataset.datasetIdentifier}: annotation directory is missing.`
        + " Import the locally licensed raw-audio dataset before evaluation.",
    );
  }
  const manifestBuffer = await readFile(manifestPath);
  const manifest = JSON.parse(manifestBuffer.toString("utf8")) as DatasetManifest;
  const leakageWarnings = (manifest.splitAssignment?.warnings ?? [])
    .filter((warning) => /LEAKAGE/i.test(warning));
  if (leakageWarnings.length) {
    throw new Error(
      `${dataset.datasetIdentifier}: manifest contains leakage warnings:`
        + ` ${leakageWarnings.join("; ")}`,
    );
  }
  const files = (await readdir(annotationDirectory))
    .filter((file) => file.endsWith(".json"))
    .sort();
  const loadedAnnotations = await Promise.all(files.map(async (file) => {
    const annotationPath = path.join(annotationDirectory, file);
    const buffer = await readFile(annotationPath);
    return {
      annotationPath,
      buffer,
      annotation: JSON.parse(buffer.toString("utf8")) as AnnotationTrack,
    };
  }));
  const splitMap = manifest.splitAssignment?.trackSplit ?? {};
  const splitFor = (annotation: AnnotationTrack): string =>
    dataset.splitStrategy === "artist-hash"
      ? artistHashSplit(annotation.artist, dataset.splitSeed ?? 20260723)
      : (splitMap[annotation.track_id] ?? annotation.split ?? "unknown");
  const heldOutArtists = new Set(dataset.heldOutArtists ?? []);
  const selected = loadedAnnotations.filter(({ annotation }) => {
    const split = splitFor(annotation);
    return split === dataset.splitName
      && (!heldOutArtists.size || heldOutArtists.has(annotation.artist));
  }).slice(0, dataset.trackLimit ?? Number.POSITIVE_INFINITY);
  if (!selected.length) {
    throw new Error(
      `${dataset.datasetIdentifier}: no audio-backed tracks matched split`
        + ` "${dataset.splitName}"`
        + (heldOutArtists.size
          ? ` and held-out artist(s) ${[...heldOutArtists].join(", ")}`
          : "")
        + ".",
    );
  }
  const artistSplits = new Map<string, Set<string>>();
  for (const { annotation } of loadedAnnotations) {
    const split = splitFor(annotation);
    const splits = artistSplits.get(annotation.artist) ?? new Set<string>();
    splits.add(split);
    artistSplits.set(annotation.artist, splits);
  }
  for (const artist of heldOutArtists) {
    const splits = artistSplits.get(artist);
    if (splits && (splits.size !== 1 || !splits.has(dataset.splitName))) {
      throw new Error(
        `${dataset.datasetIdentifier}: held-out artist ${artist}`
          + ` crosses splits ${[...splits].join(", ")}`,
      );
    }
  }

  const tracks: PreparedEvaluationTrack[] = [];
  const annotationHashes: string[] = [];
  for (const [index, item] of selected.entries()) {
    const { annotation, annotationPath, buffer } = item;
    if (annotation.source === "synthetic") {
      throw new Error(
        `${annotation.track_id}: synthetic data is not accepted by the accuracy`
          + " report command; use unit tests for pipeline smoke coverage",
      );
    }
    if (annotation.audio_availability !== "audio") {
      throw new Error(
        `${annotation.track_id}: ${annotation.audio_availability} data cannot be`
          + " used for a raw-audio app comparison",
      );
    }
    const rawRegions = annotation.chords.map((region) => ({
      start: Number(region.start),
      end: Number(region.end),
      label: String(region.label),
    }));
    validateReference(annotation.track_id, rawRegions);
    const audioPath = resolveAudioPath(
      annotationPath,
      annotation,
      dataset,
      configPath,
    );
    if (!await exists(audioPath)) {
      throw new Error(
        `${annotation.track_id}: referenced raw audio is not locally available`,
      );
    }
    const audioBuffer = await readFile(audioPath);
    const decoded = decodePcmWav(audioBuffer);
    const resampled = resampleAudio(decoded.samples, decoded.sampleRate, sampleRate);
    const evaluationStart = Math.max(0, rawRegions[0].start);
    const evaluationEnd = Math.min(
      rawRegions.at(-1)!.end,
      resampled.length / sampleRate,
    );
    if (evaluationEnd <= evaluationStart) {
      throw new Error(`${annotation.track_id}: annotation/audio intervals do not overlap`);
    }
    const startSample = Math.round(evaluationStart * sampleRate);
    const endSample = Math.round(evaluationEnd * sampleRate);
    const samples = resampled.slice(startSample, endSample);
    const duration = samples.length / sampleRate;
    const referenceRegions = rawRegions.flatMap((region) => {
      const start = Math.max(evaluationStart, region.start);
      const end = Math.min(evaluationEnd, region.end);
      return end > start ? [{
        start: start - evaluationStart,
        end: Math.min(duration, end - evaluationStart),
        label: region.label,
      }] : [];
    });
    validateReference(annotation.track_id, referenceRegions);
    const referenceChecksum = sha256(buffer);
    annotationHashes.push(`${annotation.track_id}:${referenceChecksum}`);
    tracks.push({
      datasetId: dataset.datasetIdentifier,
      splitName: dataset.splitName,
      trackId: annotation.track_id,
      artist: annotation.artist,
      title: annotation.title,
      sourceType: dataset.sourceType,
      sourceMode: dataset.sourceMode,
      sampleRate,
      samples,
      duration,
      originalOffsetSeconds: evaluationStart,
      bpm: bpmFromBeats(annotation.beats ?? []),
      referenceRegions,
      audioHash: sha256(audioBuffer),
      referenceChecksum,
    });
    console.log(
      `[${dataset.datasetIdentifier}] loaded ${index + 1}/${selected.length}:`
        + ` ${annotation.track_id}`,
    );
  }
  const totalDuration = tracks.reduce((sum, track) => sum + track.duration, 0);
  return {
    config: dataset,
    tracks,
    identity: {
      datasetIdentifier: dataset.datasetIdentifier,
      manifestChecksum: sha256(manifestBuffer),
      annotationChecksum: sha256(annotationHashes.sort().join("\n")),
      splitName: dataset.splitName,
      splitDefinition: dataset.splitStrategy === "artist-hash"
        ? `artist-hash seed ${dataset.splitSeed ?? 20260723}`
        : "dataset manifest splitAssignment",
      sourceType: dataset.sourceType,
      trackCount: tracks.length,
      totalEvaluatedDurationSeconds: totalDuration,
      trackIds: tracks.map((track) => track.trackId),
      splitLeakageWarnings: leakageWarnings,
    },
  };
}

function validateConfig(config: EvaluationConfig): void {
  if (config.schemaVersion !== 1) throw new Error("Unsupported evaluation config schema");
  if (!Number.isFinite(config.sampleRate) || config.sampleRate <= 0) {
    throw new Error("sampleRate must be positive");
  }
  if (!config.datasets?.length) throw new Error("At least one dataset is required");
  if (config.bootstrap.iterations < 100) {
    throw new Error("bootstrap.iterations must be at least 100");
  }
  for (const dataset of config.datasets) {
    if (dataset.runAblations && dataset.splitName === "test") {
      throw new Error(
        `${dataset.datasetIdentifier}: ablations cannot run on the untouched test split`,
      );
    }
  }
}

function resolvedHybridSettings(
  partial: Partial<HybridHarmonySettings> | undefined,
): HybridHarmonySettings {
  return { ...CONSERVATIVE_HYBRID_SETTINGS, ...partial };
}

function currentCommit(): string {
  return execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: process.cwd(),
    encoding: "utf8",
  }).trim();
}

function validateBundledModel(weights: TcnWeights): void {
  if (weights.modelVersion !== LEARNED_MODEL_VERSION
    || weights.modelChecksum !== LEARNED_MODEL_CHECKSUM
    || weights.featureVersion !== FEATURE_VERSION) {
    throw new Error(
      "Bundled model version/checksum/feature identity is inconsistent;"
        + " evaluation aborted before scoring.",
    );
  }
}

export async function runHybridComparison(
  configPath: string,
  outputOverride?: string,
): Promise<ReturnType<typeof buildHybridAccuracyReport>> {
  const config = JSON.parse(await readFile(configPath, "utf8")) as EvaluationConfig;
  validateConfig(config);
  const weights = bundledWeights as unknown as TcnWeights;
  validateBundledModel(weights);
  const loadedDatasets: LoadedDataset[] = [];
  for (const dataset of config.datasets) {
    loadedDatasets.push(await loadDataset(dataset, configPath, config.sampleRate));
  }
  const provider = new LearnedTcnProvider(weights);
  const modelIdentity = {
    modelVersion: LEARNED_MODEL_VERSION,
    modelChecksum: LEARNED_MODEL_CHECKSUM,
    featureVersion: FEATURE_VERSION,
  };
  const settings = resolvedHybridSettings(config.hybridSettings);
  const scoredTracks = [];
  const allTracks = loadedDatasets.flatMap((dataset) =>
    dataset.tracks.map((track) => ({
      track,
      runAblations: Boolean(dataset.config.runAblations),
    })));
  for (const [index, item] of allTracks.entries()) {
    console.log(
      `Evaluating ${index + 1}/${allTracks.length}: ${item.track.trackId}`,
    );
    const comparison = await compareTrackEngines(
      item.track,
      { provider },
      {
        modelIdentity,
        hybridSettings: settings,
        mlOnlyTransitionPenalty:
          config.mlOnlyTransitionPenalty ?? DEFAULT_ML_ONLY_TRANSITION_PENALTY,
        timeoutMs: config.timeoutMs,
        includeLegacyRegionHybrid: config.includeLegacyRegionHybrid,
        runAblations: item.runAblations,
      },
    );
    scoredTracks.push(scoreTrackComparison(
      comparison,
      config.shortRegionThresholdSeconds ?? 0.5,
    ));
  }
  const successful = scoredTracks.filter((track) =>
    track.comparison.predictions["observation-hybrid"]?.usedLearned
      && track.comparison.predictions["ml-only"]);
  if (!successful.length) {
    const reasons = scoredTracks.map((track) =>
      `${track.comparison.track.trackId}:`
        + ` ${track.comparison.predictions["observation-hybrid"]?.fallbackReason ?? "unknown"}`);
    throw new Error(
      "No track completed learned inference through the TypeScript hybrid decoder;"
        + ` no accuracy reports were written. ${reasons.join("; ")}`,
    );
  }
  const hasFullBand = loadedDatasets.some(
    (dataset) => dataset.identity.sourceType === "full-band",
  );
  const usesP00 = loadedDatasets.some((dataset) =>
    dataset.config.heldOutArtists?.includes("guitarset-p00"));
  const limitations = [
    "Peak memory is a coarse Node heap high-water sample, not process-wide RSS profiling.",
    "ML-only uses a uniform transition penalty and intentionally omits the learned boundary head so it remains evidence-independent from Tabsmith's rule decoder.",
    ...(usesP00 ? [
      "GuitarSet performer p00 was previously used to choose the ML-only Viterbi transition penalty in repository experiments; this is held-out validation evidence, not a pristine final test.",
    ] : []),
    ...(!hasFullBand ? [
      "No legally available representative full-band raw-audio dataset was configured, so the production promotion gate remains open.",
      "Billboard precomputed-feature results are intentionally excluded from this raw-audio app comparison.",
    ] : []),
  ];
  const ablationDatasets = loadedDatasets.filter(
    (dataset) => dataset.config.runAblations,
  );
  const report = buildHybridAccuracyReport({
    commitSha: currentCommit(),
    createdAt: new Date().toISOString(),
    datasets: loadedDatasets.map((dataset) => dataset.identity),
    scoredTracks,
    modelIdentity,
    hybridDecoderVersion: HYBRID_DECODER_VERSION,
    hybridSettings: settings,
    bootstrapSeed: config.bootstrap.seed,
    bootstrapIterations: config.bootstrap.iterations,
    mlOnlyTransitionPenalty:
      config.mlOnlyTransitionPenalty ?? DEFAULT_ML_ONLY_TRANSITION_PENALTY,
    includeLegacyRegionHybrid: config.includeLegacyRegionHybrid,
    combineSourceTypes: config.combineSourceTypes,
    ablationSplitDescription: ablationDatasets.length
      ? `Frozen settings ablation on ${ablationDatasets.map((dataset) =>
        `${dataset.identity.datasetIdentifier}/${dataset.identity.splitName}`).join(", ")}.`
      : null,
    limitations,
  });
  assertNoAbsolutePaths(report);
  const outputDirectory = outputOverride
    ? path.resolve(outputOverride)
    : resolveFromConfig(configPath, config.outputDirectory);
  await mkdir(outputDirectory, { recursive: true });
  const jsonPath = path.join(
    outputDirectory,
    "hybrid-accuracy-comparison.json",
  );
  const markdownPath = path.join(
    outputDirectory,
    "hybrid-accuracy-comparison.md",
  );
  const csvPath = path.join(
    outputDirectory,
    "hybrid-accuracy-per-track.csv",
  );
  const disagreementPath = path.join(
    outputDirectory,
    "hybrid-disagreements.json",
  );
  const disagreement = hybridDisagreementReport(report, scoredTracks);
  assertNoAbsolutePaths(disagreement);
  await Promise.all([
    writeFile(jsonPath, `${JSON.stringify(report, null, 2)}\n`, "utf8"),
    writeFile(markdownPath, hybridAccuracyReportMarkdown(report), "utf8"),
    writeFile(csvPath, hybridAccuracyPerTrackCsv(report), "utf8"),
    writeFile(
      disagreementPath,
      `${JSON.stringify(disagreement, null, 2)}\n`,
      "utf8",
    ),
  ]);
  console.log(`JSON: ${jsonPath}`);
  console.log(`Markdown: ${markdownPath}`);
  console.log(`Per-track CSV: ${csvPath}`);
  console.log(`Diagnostics: ${disagreementPath}`);
  return report;
}

const args = parseArgs(process.argv.slice(2));
await runHybridComparison(args.configPath, args.outputDirectory).catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
});
