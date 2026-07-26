import { describe, expect, it } from "vitest";
import {
  analysisCountLabels,
  BASIC_PITCH_MODEL_VERSION,
  buildTranscriptionCacheKey,
  cacheOutcomeLabels,
  cacheReadDecision,
  CHORD_ANALYSIS_VERSION,
  createCacheMetadata,
  FRETBOARD_MAPPER_VERSION,
  NOTE_CLEANUP_VERSION,
  reprocessPreservingPrevious,
  shouldStoreTranscriptionResult,
  SOURCE_SEPARATION_MODEL_VERSION,
  TEMPO_QUANTIZATION_VERSION,
  TRANSCRIPTION_PIPELINE_VERSION,
  validateCacheMetadata,
  type CachedTranscriptionEntry,
  type TranscriptionCacheKeyInput,
} from "../src/transcriptionCache";
import type { AnalysisResult } from "../src/types";

function cacheInput(): TranscriptionCacheKeyInput {
  return {
    audioContentHash: "a".repeat(64),
    analysisAudioHash: "b".repeat(64),
    harmonyAudioHash: "a".repeat(64),
    bassAudioHash: null,
    harmonyEvidenceSource: "full-mix",
    engine: "basic-pitch",
    basicPitchModelVersion: BASIC_PITCH_MODEL_VERSION,
    basicPitchThresholds: { onset: 0.5, frame: 0.32 },
    transcriptionPipelineVersion: TRANSCRIPTION_PIPELINE_VERSION,
    noteCleanupVersion: NOTE_CLEANUP_VERSION,
    noteCleanupSettings: { minimumDuration: 0.08 },
    arrangementVersion: 1,
    arrangementMode: "lead",
    qualityPreset: "balanced",
    tuningMode: "automatic",
    selectedTuning: ["E2", "A2", "D3", "G3", "B3", "E4"],
    capoMode: "automatic",
    capoPosition: 0,
    fretboardMapperVersion: FRETBOARD_MAPPER_VERSION,
    chordAnalysisVersion: CHORD_ANALYSIS_VERSION,
    chordAnalysisSettings: { minimumBeats: 2 },
    chordDisplayMode: "simple",
    tempoQuantizationVersion: TEMPO_QUANTIZATION_VERSION,
    processingRange: { startSeconds: 0, endSeconds: null },
    guitarIsolationEnabled: true,
    sourceSeparationModelVersion: SOURCE_SEPARATION_MODEL_VERSION,
    chordEngine: "rule",
    learnedModelVersion: null,
    learnedModelChecksum: null,
    hybridDecoderVersion: null,
    hybridSettings: null,
  };
}

function analysis(): AnalysisResult {
  return {
    duration: 2,
    bpm: 120,
    tuning: ["E2", "A2", "D3", "G3", "B3", "E4"],
    capo: 0,
    chords: [{ start: 0, end: 2, name: "E", confidence: 1 }],
    notes: [{ start: 0, end: 0.5, midi: 64, name: "E4", confidence: 1, string: 1, fret: 0 }],
    noteAnalysis: {
      rawNotes: [
        { start: 0, end: 0.1, midi: 64, name: "E4", confidence: 0.7 },
        { start: 0, end: 0.5, midi: 64, name: "E4", confidence: 1 },
      ],
      cleanedNotes: [{ start: 0, end: 0.5, midi: 64, name: "E4", confidence: 1 }],
      arrangedNotes: [{ start: 0, end: 0.5, midi: 64, name: "E4", confidence: 1, string: 1, fret: 0 }],
      requestedMode: "lead",
      resolvedMode: "lead",
      preset: "balanced",
      settings: {} as AnalysisResult["noteAnalysis"]["settings"],
      diagnostics: {
        rawNoteCount: 2,
        cleanedNoteCount: 1,
        arrangedNoteCount: 1,
      } as AnalysisResult["noteAnalysis"]["diagnostics"],
      warnings: [],
    },
    chordAnalysis: {
      rawFrames: [{
        start: 0,
        end: 0.25,
        bestChord: "E",
        bestScore: 1,
        secondBestChord: "Em",
        secondBestScore: 0.5,
        confidence: 1,
        scoreMargin: 0.5,
        uncertain: false,
      }],
      windows: [],
      regions: [{ start: 0, end: 2, name: "E", confidence: 1 }],
      settings: {} as AnalysisResult["chordAnalysis"]["settings"],
      diagnostics: {} as AnalysisResult["chordAnalysis"]["diagnostics"],
    },
    engine: "basic-pitch",
    separation: "guitar",
  };
}

function entry(metadata = createCacheMetadata()): CachedTranscriptionEntry {
  return {
    metadata,
    cacheKey: "c".repeat(64),
    settings: cacheInput(),
    result: analysis(),
  };
}

describe("transcription cache versioning", () => {
  it("rejects a legacy cache entry without metadata", () => {
    expect(validateCacheMetadata(undefined)).toEqual({
      valid: false,
      reason: "entry has no transcription cache metadata",
    });
  });

  it("rejects an entry from an older pipeline version", () => {
    const metadata = {
      ...createCacheMetadata(),
      transcriptionPipelineVersion: "old-v1",
    };
    const validation = validateCacheMetadata(metadata);
    expect(validation.valid).toBe(false);
    expect(validation.reason).toContain("old-v1");
  });

  it("reuses a current compatible entry", () => {
    expect(cacheReadDecision(entry())).toEqual({
      reuse: true,
      status: "hit",
      reason: null,
    });
  });

  it("bypasses only the transcription entry when reprocessing", () => {
    expect(cacheReadDecision(entry(), true)).toMatchObject({
      reuse: false,
      status: "bypassed",
    });
  });
});

describe("canonical cache keys", () => {
  it("changes for arrangement mode and quality preset", async () => {
    const base = cacheInput();
    const lead = await buildTranscriptionCacheKey(base);
    const rhythm = await buildTranscriptionCacheKey({ ...base, arrangementMode: "rhythm" });
    const clean = await buildTranscriptionCacheKey({ ...base, qualityPreset: "clean" });
    expect(new Set([lead, rhythm, clean]).size).toBe(3);
  });

  it("changes for selected tuning or capo", async () => {
    const base = cacheInput();
    const standard = await buildTranscriptionCacheKey(base);
    const dropD = await buildTranscriptionCacheKey({
      ...base,
      tuningMode: "selected",
      selectedTuning: ["D2", "A2", "D3", "G3", "B3", "E4"],
    });
    const capoTwo = await buildTranscriptionCacheKey({
      ...base,
      capoMode: "selected",
      capoPosition: 2,
    });
    expect(new Set([standard, dropD, capoTwo]).size).toBe(3);
  });

  it("canonicalizes equivalent settings before hashing", async () => {
    const first = cacheInput();
    const second = {
      ...first,
      basicPitchThresholds: { frame: 0.32, onset: 0.5 },
    };
    expect(await buildTranscriptionCacheKey(first))
      .toBe(await buildTranscriptionCacheKey(second));
  });

  it("isolates rule and hybrid cache identities", async () => {
    const rule = cacheInput();
    const hybrid = {
      ...rule,
      chordEngine: "hybrid" as const,
      learnedModelVersion: "model-v1",
      learnedModelChecksum: "checksum-a",
      hybridDecoderVersion: 2,
      hybridSettings: {
        ruleObservationWeight: 1,
        learnedChordWeight: 0.35,
        learnedBoundaryWeight: 0.25,
        bassRootWeight: 0.25,
        allowLearnedBoundaryBackdating: false,
        minimumLearnedConfidence: 0.6,
        maximumLearnedEntropy: 0.75,
        protectRuleConfidenceAbove: 0.72,
        maximumLearnedWeightFullMix: 0.3,
        maximumLearnedWeightGuitarOnly: 0.55,
      },
    };
    const ruleKey = await buildTranscriptionCacheKey(rule);
    const hybridKey = await buildTranscriptionCacheKey(hybrid);
    expect(ruleKey).not.toBe(hybridKey);
    expect(await buildTranscriptionCacheKey({
      ...hybrid,
      learnedModelChecksum: "checksum-b",
    })).not.toBe(hybridKey);
    expect(await buildTranscriptionCacheKey({
      ...hybrid,
      hybridSettings: { ...hybrid.hybridSettings, learnedChordWeight: 0.2 },
    })).not.toBe(hybridKey);
  });
});

describe("cache-facing result behavior", () => {
  it("preserves the previous usable result when reprocessing fails", async () => {
    const previous = analysis();
    const outcome = await reprocessPreservingPrevious(previous, async () => {
      throw new Error("model failed");
    });
    expect(outcome.result).toBe(previous);
    expect(outcome.replaced).toBe(false);
  });

  it("does not cache a hybrid request that fell back to rules", () => {
    const input = {
      ...cacheInput(),
      chordEngine: "hybrid" as const,
    };
    const fallback = {
      ...analysis(),
      hybridEngine: { usedLearned: false, fallbackReason: "timeout" },
    };
    expect(shouldStoreTranscriptionResult(input, fallback)).toBe(false);
    expect(shouldStoreTranscriptionResult(
      input,
      { ...fallback, hybridEngine: { usedLearned: true } },
    )).toBe(true);
  });

  it("uses final arranged notes as the main count", () => {
    expect(analysisCountLabels(analysis())).toEqual({
      main: "Final arrangement: 1 notes",
      raw: "Raw detections: 2",
      cleaned: "After cleanup: 1",
      chords: "Chord regions: 1",
    });
  });

  it("distinguishes stem reuse from transcription reuse", () => {
    expect(cacheOutcomeLabels({
      transcription: "loaded-current",
      separatedStemReused: true,
    })).toEqual([
      "Loaded current transcription cache",
      "Reused separated stems",
    ]);
  });
});
