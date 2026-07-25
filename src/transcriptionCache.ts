import versions from "../shared/pipeline-versions.json";
import type { HybridHarmonySettings } from "./learnedHarmony/types";
import type {
  AnalysisResult,
  ArrangementMode,
  CachedTranscriptionMetadata,
  ChordDisplayMode,
  QualityPreset,
} from "./types";

export const TRANSCRIPTION_CACHE_SCHEMA_VERSION =
  versions.transcriptionCacheSchemaVersion;
export const TRANSCRIPTION_PIPELINE_VERSION =
  versions.transcriptionPipelineVersion;
export const NOTE_CLEANUP_VERSION = versions.noteCleanupVersion;
export const ARRANGEMENT_VERSION = versions.arrangementVersion;
export const FRETBOARD_MAPPER_VERSION = versions.fretboardMapperVersion;
export const CHORD_ANALYSIS_VERSION = versions.chordAnalysisVersion;
export const TEMPO_QUANTIZATION_VERSION = versions.tempoQuantizationVersion;
export const BASIC_PITCH_MODEL_VERSION = versions.basicPitchModelVersion;
export const SOURCE_SEPARATION_MODEL_VERSION =
  versions.sourceSeparationModelVersion;

export type ProcessingRange = {
  startSeconds: number;
  endSeconds: number | null;
};

export type TranscriptionCacheKeyInput = {
  audioContentHash: string;
  analysisAudioHash: string;
  harmonyAudioHash: string;
  bassAudioHash: string | null;
  harmonyEvidenceSource: "separated-harmonic-mix" | "full-mix" | "guitar-only";
  engine: "basic-pitch" | "dsp";
  basicPitchModelVersion: string;
  basicPitchThresholds: Record<string, number>;
  transcriptionPipelineVersion: string;
  noteCleanupVersion: number;
  noteCleanupSettings: Record<string, unknown>;
  arrangementVersion: number;
  arrangementMode: ArrangementMode;
  qualityPreset: QualityPreset;
  tuningMode: "automatic" | "selected";
  selectedTuning: string[];
  capoMode: "automatic" | "selected";
  capoPosition: number;
  fretboardMapperVersion: number;
  chordAnalysisVersion: number;
  chordAnalysisSettings: Record<string, unknown>;
  chordDisplayMode: ChordDisplayMode;
  tempoQuantizationVersion: number;
  processingRange: ProcessingRange;
  guitarIsolationEnabled: boolean;
  sourceSeparationModelVersion: string;
  chordEngine: "rule" | "hybrid";
  learnedModelVersion: string | null;
  learnedModelChecksum: string | null;
  hybridDecoderVersion: number | null;
  hybridSettings: HybridHarmonySettings | null;
};

export type CachedTranscriptionEntry = {
  metadata: CachedTranscriptionMetadata;
  cacheKey: string;
  settings: TranscriptionCacheKeyInput;
  result: AnalysisResult;
};

export type ProcessingCacheOutcome = {
  transcription:
    | "loaded-current"
    | "legacy-invalidated"
    | "reprocessed"
    | "fresh";
  separatedStemReused: boolean;
  detail?: string;
};

export type CacheValidation =
  | { valid: true; reason: null }
  | { valid: false; reason: string };

const METADATA_FIELDS: Array<{
  field: keyof CachedTranscriptionMetadata;
  expected: number | string;
}> = [
  { field: "cacheSchemaVersion", expected: TRANSCRIPTION_CACHE_SCHEMA_VERSION },
  { field: "transcriptionPipelineVersion", expected: TRANSCRIPTION_PIPELINE_VERSION },
  { field: "noteCleanupVersion", expected: NOTE_CLEANUP_VERSION },
  { field: "arrangementVersion", expected: ARRANGEMENT_VERSION },
  { field: "fretboardMapperVersion", expected: FRETBOARD_MAPPER_VERSION },
  { field: "chordAnalysisVersion", expected: CHORD_ANALYSIS_VERSION },
];

export function createCacheMetadata(
  createdAt = new Date().toISOString(),
): CachedTranscriptionMetadata {
  return {
    cacheSchemaVersion: TRANSCRIPTION_CACHE_SCHEMA_VERSION,
    transcriptionPipelineVersion: TRANSCRIPTION_PIPELINE_VERSION,
    noteCleanupVersion: NOTE_CLEANUP_VERSION,
    arrangementVersion: ARRANGEMENT_VERSION,
    fretboardMapperVersion: FRETBOARD_MAPPER_VERSION,
    chordAnalysisVersion: CHORD_ANALYSIS_VERSION,
    createdAt,
  };
}

export function validateCacheMetadata(
  metadata: CachedTranscriptionMetadata | null | undefined,
): CacheValidation {
  if (!metadata || typeof metadata !== "object") {
    return { valid: false, reason: "entry has no transcription cache metadata" };
  }
  for (const { field, expected } of METADATA_FIELDS) {
    const actual = metadata[field];
    if (actual !== expected) {
      return {
        valid: false,
        reason: `${String(field)} ${String(actual ?? "missing")} does not match ${String(expected)}`,
      };
    }
  }
  if (!metadata.createdAt || Number.isNaN(Date.parse(metadata.createdAt))) {
    return { valid: false, reason: "createdAt is missing or invalid" };
  }
  return { valid: true, reason: null };
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>)
        .filter(([, item]) => item !== undefined)
        .sort(([first], [second]) => first.localeCompare(second))
        .map(([key, item]) => [key, canonicalValue(item)]),
    );
  }
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new Error("Cache settings must use finite numbers.");
    return Object.is(value, -0) ? 0 : Number(value.toFixed(9));
  }
  return value;
}

export function canonicalizeCacheSettings(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

export async function sha256Hex(
  value: ArrayBuffer | Uint8Array | string,
): Promise<string> {
  const bytes = typeof value === "string"
    ? new TextEncoder().encode(value)
    : value instanceof Uint8Array
      ? value
      : new Uint8Array(value);
  const stableBytes = new Uint8Array(bytes.byteLength);
  stableBytes.set(bytes);
  const digest = await globalThis.crypto.subtle.digest(
    "SHA-256",
    stableBytes.buffer,
  );
  return [...new Uint8Array(digest)]
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function buildTranscriptionCacheKey(
  input: TranscriptionCacheKeyInput,
): Promise<string> {
  if (input.chordEngine === "rule") {
    // Preserve compatibility with safe existing rule-cache entries. The legacy
    // identity already meant rule-only, so the new hybrid-only fields are
    // intentionally omitted for that mode.
    const {
      chordEngine: _chordEngine,
      learnedModelVersion: _learnedModelVersion,
      learnedModelChecksum: _learnedModelChecksum,
      hybridDecoderVersion: _hybridDecoderVersion,
      hybridSettings: _hybridSettings,
      ...legacyRuleIdentity
    } = input;
    return sha256Hex(canonicalizeCacheSettings(legacyRuleIdentity));
  }
  return sha256Hex(canonicalizeCacheSettings(input));
}

/**
 * A hybrid fallback is deliberately not cached: persisting it under a hybrid key
 * would hide provider recovery on the next request. The exact rule result still
 * completes the current job.
 */
export function shouldStoreTranscriptionResult(
  input: TranscriptionCacheKeyInput,
  result: AnalysisResult,
): boolean {
  return input.chordEngine === "rule" || result.hybridEngine?.usedLearned === true;
}

export function cacheReadDecision(
  entry: CachedTranscriptionEntry | null | undefined,
  forceReprocess = false,
): { reuse: boolean; status: "miss" | "stale" | "bypassed" | "hit"; reason: string | null } {
  if (forceReprocess) {
    return { reuse: false, status: "bypassed", reason: "transcription cache bypass requested" };
  }
  if (!entry) return { reuse: false, status: "miss", reason: null };
  const validation = validateCacheMetadata(entry.metadata);
  if (!validation.valid) {
    return { reuse: false, status: "stale", reason: validation.reason };
  }
  if (!entry.result?.noteAnalysis?.rawNotes
    || !entry.result?.chordAnalysis?.rawFrames
    || !Array.isArray(entry.result.notes)
    || !Array.isArray(entry.result.chords)) {
    return {
      reuse: false,
      status: "stale",
      reason: "cached result lacks raw or cleaned pipeline diagnostics",
    };
  }
  return { reuse: true, status: "hit", reason: null };
}

export function pipelineLabel(): string {
  return TRANSCRIPTION_PIPELINE_VERSION.replace(/^\d{4}-\d{2}-/, "");
}

export function analysisCountLabels(result: AnalysisResult): {
  main: string;
  raw: string;
  cleaned: string;
  chords: string;
} {
  return {
    main: `Final arrangement: ${result.notes.length} notes`,
    raw: `Raw detections: ${result.noteAnalysis.diagnostics.rawNoteCount}`,
    cleaned: `After cleanup: ${result.noteAnalysis.diagnostics.cleanedNoteCount}`,
    chords: `Chord regions: ${result.chords.length}`,
  };
}

export function cacheOutcomeLabels(outcome: ProcessingCacheOutcome): string[] {
  const transcriptionLabels: Record<
    ProcessingCacheOutcome["transcription"],
    string
  > = {
    "loaded-current": "Loaded current transcription cache",
    "legacy-invalidated": "Legacy cache invalidated · Fresh transcription completed",
    reprocessed: "Reprocessing with updated pipeline completed",
    fresh: "Fresh transcription completed",
  };
  return [
    transcriptionLabels[outcome.transcription],
    ...(outcome.separatedStemReused ? ["Reused separated stems"] : []),
  ];
}

export async function reprocessPreservingPrevious<T>(
  previous: T,
  processor: () => Promise<T>,
): Promise<{ result: T; replaced: boolean; error: unknown | null }> {
  try {
    return { result: await processor(), replaced: true, error: null };
  } catch (error) {
    return { result: previous, replaced: false, error };
  }
}
