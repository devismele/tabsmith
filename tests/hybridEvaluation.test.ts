import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  HYBRID_DECODER_VERSION,
  runHybridHarmony,
} from "../src/learnedHarmony/hybridDecoder";
import type {
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";
import {
  compareTrackEngines,
  type EvaluationModelIdentity,
  type PreparedEvaluationTrack,
} from "../evaluation/hybridComparison";
import {
  aggregateComparison,
  bootstrapMeanConfidenceInterval,
  scoreTrackComparison,
} from "../evaluation/hybridMetrics";
import {
  assertNoAbsolutePaths,
  buildHybridAccuracyReport,
  hybridAccuracyPerTrackCsv,
} from "../evaluation/hybridReport";
import {
  audioBasenameForCapture,
  reportFileNames,
  resolveDatasetPaths,
  resolveOutputTag,
} from "../evaluation/hybridEvaluationConfig";
import { buildCrossCaptureSummary } from "../evaluation/hybridCrossCapture";

const MODEL: EvaluationModelIdentity = {
  modelVersion: "evaluation-test-model",
  modelChecksum: "evaluation-test-checksum",
  featureVersion: "harmony-features-v1",
};

function synthesizeTriad(duration: number, sampleRate: number): Float32Array {
  const frequencies = [261.6256, 329.6276, 391.9954];
  return Float32Array.from(
    { length: Math.round(duration * sampleRate) },
    (_unused, index) => {
      const time = index / sampleRate;
      const fade = Math.min(1, time * 8, (duration - time) * 8);
      return fade * frequencies.reduce(
        (sum, frequency) => sum + Math.sin(2 * Math.PI * frequency * time),
        0,
      ) / frequencies.length * 0.35;
    },
  );
}

function track(): PreparedEvaluationTrack {
  const sampleRate = 22050;
  const duration = 3;
  return {
    datasetId: "test-solo-guitar",
    splitName: "validation",
    trackId: "held-out-track",
    artist: "held-out-player",
    title: "Test progression",
    sourceType: "solo-guitar",
    sourceMode: "guitar-focused",
    sampleRate,
    samples: synthesizeTriad(duration, sampleRate),
    duration,
    originalOffsetSeconds: 0,
    bpm: 120,
    referenceRegions: [
      { start: 0, end: 1.5, label: "C:maj" },
      { start: 1.5, end: 3, label: "G:maj" },
    ],
    audioHash: "test-audio-hash",
    referenceChecksum: "test-reference-checksum",
  };
}

class CountingProvider implements LearnedHarmonyProvider {
  readonly id = "evaluation-test-provider";
  predictCalls = 0;
  constructor(private readonly available = true) {}

  async isAvailable(): Promise<boolean> {
    return this.available;
  }

  async getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return {
      modelVersion: MODEL.modelVersion,
      modelChecksum: MODEL.modelChecksum,
      contractVersion: 1,
      featureVersion: MODEL.featureVersion,
      format: "mock",
      vocabulary: {
        roots: 12,
        qualities: ["maj", "min", "7"],
        hasNoChordHead: true,
        hasBoundaryHead: true,
      },
      inputShapes: { features: ["batch", "frames", 25] },
      disclaimer: "Evaluation test provider.",
    };
  }

  async predict(request: LearnedHarmonyRequest): Promise<LearnedHarmonyResponse> {
    this.predictCalls += 1;
    const root = (rootIndex: number) => Array.from(
      { length: 12 },
      (_unused, index) => index === rootIndex ? 0.98 : 0.02 / 11,
    );
    const quality = [0.98, 0.01, 0.01];
    return {
      contractVersion: request.contractVersion,
      requestId: request.requestId,
      modelVersion: MODEL.modelVersion,
      modelChecksum: MODEL.modelChecksum,
      featureVersion: MODEL.featureVersion,
      frameTimes: [...request.frameTimes],
      rootProbabilities: request.frameTimes.map((time) =>
        root(time < 1.5 ? 0 : 7)),
      qualityProbabilities: request.frameTimes.map(() => [...quality]),
      noChordProbabilities: request.frameTimes.map(() => 0.01),
      boundaryProbabilities: request.frameTimes.map((time) =>
        Math.abs(time - 1.5) <= 0.26 ? 0.95 : 0.02),
      diagnostics: {
        inferenceMilliseconds: 2,
        backend: "mock",
        warnings: [],
      },
    };
  }
}

async function successfulComparison(runAblations = false) {
  const provider = new CountingProvider();
  let productionHybridCalls = 0;
  const comparison = await compareTrackEngines(
    track(),
    {
      provider,
      hybridRunner: async (params) => {
        productionHybridCalls += 1;
        return runHybridHarmony(params);
      },
      now: (() => {
        let value = 0;
        return () => ++value;
      })(),
      heapUsed: () => 1024 * 1024,
    },
    {
      modelIdentity: MODEL,
      hybridSettings: CONSERVATIVE_HYBRID_SETTINGS,
      includeLegacyRegionHybrid: true,
      runAblations,
      timeoutMs: 1000,
    },
  );
  return { comparison, provider, productionHybridCalls };
}

describe("TypeScript hybrid parity evaluation", () => {
  it("invokes the actual production hybrid and temporal decoder", async () => {
    const { comparison, provider, productionHybridCalls } =
      await successfulComparison(true);

    expect(productionHybridCalls).toBe(1);
    expect(provider.predictCalls).toBe(1);
    expect(comparison.execution.actualHybridDecoderInvoked).toBe(true);
    expect(comparison.execution.actualTemporalDecoderCompleted).toBe(true);
    expect(comparison.execution.hybridImplementation).toBe(
      "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony",
    );
    expect(comparison.execution.productionDecoder).toBe(
      "src/chordAnalysis.ts#decodeHarmonyObservations",
    );
    expect(comparison.predictions["observation-hybrid"]?.chordAnalysis)
      .toBeDefined();
    expect(comparison.ablations.map((ablation) => ablation.id)).toEqual([
      "rule-only",
      "hybrid-chord-probabilities-only",
      "hybrid-without-boundary",
      "hybrid-without-adaptive-weighting",
      "full-hybrid",
    ]);
  });

  it("uses one learned probability source for ML-only, hybrid, and legacy", async () => {
    const { comparison, provider } = await successfulComparison();
    const mlOnly = comparison.predictions["ml-only"];
    const hybrid = comparison.predictions["observation-hybrid"];
    const legacy = comparison.predictions["legacy-region-hybrid"];

    expect(provider.predictCalls).toBe(1);
    expect(comparison.execution.sharedLearnedResponseObject).toBe(true);
    expect(mlOnly?.probabilitySourceId).toBe(hybrid?.probabilitySourceId);
    expect(legacy?.probabilitySourceId).toBe(mlOnly?.probabilitySourceId);
    expect(mlOnly?.usedLearned).toBe(true);
    expect(hybrid?.usedLearned).toBe(true);
  });

  it("keeps track IDs, durations, and references identical across engines", async () => {
    const { comparison } = await successfulComparison();
    const scored = scoreTrackComparison(comparison);
    const metrics = Object.values(scored.engineMetrics);
    const durations = new Set(
      metrics.map((metric) => metric?.evaluatedDurationSeconds),
    );

    expect(durations.size).toBe(1);
    expect(comparison.track.trackId).toBe("held-out-track");
    expect(comparison.track.referenceRegions).toEqual(track().referenceRegions);
    expect(Object.keys(scored.engineMetrics).sort()).toEqual([
      "legacy-region-hybrid",
      "ml-only",
      "observation-hybrid",
      "rule-only",
    ]);
  });

  it("counts hybrid fallbacks without adding them to the successful common set", async () => {
    const successful = scoreTrackComparison(
      (await successfulComparison()).comparison,
    );
    const unavailable = await compareTrackEngines(
      { ...track(), trackId: "fallback-track" },
      { provider: new CountingProvider(false) },
      {
        modelIdentity: MODEL,
        hybridSettings: CONSERVATIVE_HYBRID_SETTINGS,
        includeLegacyRegionHybrid: true,
      },
    );
    const fallback = scoreTrackComparison(unavailable);
    const aggregate = aggregateComparison([successful, fallback], 1234, 200);

    expect(unavailable.predictions["observation-hybrid"]?.fallbackReason)
      .toBe("provider-unavailable");
    expect(unavailable.predictions["observation-hybrid"]?.regions).toBe(
      unavailable.predictions["rule-only"]?.regions,
    );
    expect(aggregate.allRequestedHybridTrackIds).toEqual([
      "held-out-track",
      "fallback-track",
    ]);
    expect(aggregate.commonSuccessfulTrackIds).toEqual(["held-out-track"]);
    expect(aggregate.fallbackCounts).toEqual({ "provider-unavailable": 1 });
    expect(aggregate.allRequestedHybrid.trackCount).toBe(2);
  });

  it("reconciles per-track counts, records identity, and excludes absolute paths", async () => {
    const scored = scoreTrackComparison(
      (await successfulComparison()).comparison,
    );
    const report = buildHybridAccuracyReport({
      commitSha: "0123456789abcdef",
      createdAt: "2026-07-25T00:00:00.000Z",
      datasets: [{
        datasetIdentifier: "test-solo-guitar",
        officialDistributionRecord: "zenodo-test",
        datasetVersion: "1.0",
        annotationArchiveMd5: "annotation-md5",
        audioArchiveMd5: "audio-md5",
        captureType: "audio_mono-mic",
        manifestChecksum: "manifest-checksum",
        annotationChecksum: "annotation-checksum",
        splitName: "validation",
        splitDefinition: "artist-hash seed 20260723",
        sourceType: "solo-guitar",
        trackCount: 1,
        totalEvaluatedDurationSeconds: 3,
        trackIds: ["held-out-track"],
        splitLeakageWarnings: [],
        leakageAudit: {
          heldOutPerformers: ["held-out-player"],
          heldOutPerformerSplits: {
            "held-out-player": "validation",
          },
          heldOutPerformersExcludedFromTraining: true,
          duplicateTrackIds: 0,
          notes: ["Held-out performer is excluded from training."],
        },
      }],
      scoredTracks: [scored],
      modelIdentity: MODEL,
      hybridDecoderVersion: HYBRID_DECODER_VERSION,
      hybridSettings: CONSERVATIVE_HYBRID_SETTINGS,
      bootstrapSeed: 42,
      bootstrapIterations: 200,
      includeLegacyRegionHybrid: true,
      combineSourceTypes: false,
      ablationSplitDescription: null,
      limitations: ["Synthetic unit fixture; no accuracy claim."],
    });

    expect(report.engineIdentities["ml-only"].modelChecksum)
      .toBe(MODEL.modelChecksum);
    expect(report.fairness.commonSuccessfulTrackIds).toEqual(["held-out-track"]);
    expect(report.perTrack).toHaveLength(4);
    expect(new Set(report.perTrack.map((row) => row.trackId)))
      .toEqual(new Set(["held-out-track"]));
    expect(new Set(report.perTrack.map((row) => row.durationSeconds)))
      .toEqual(new Set([3]));
    expect(hybridAccuracyPerTrackCsv(report).split(/\r?\n/)).toHaveLength(6);
    expect(() => assertNoAbsolutePaths(report)).not.toThrow();
    expect(() => assertNoAbsolutePaths({ path: "C:\\private\\audio.wav" }))
      .toThrow(/Absolute path/);

    const pickup = structuredClone(report);
    pickup.datasets[0].captureType = "audio_mono-pickup_mix";
    pickup.datasets[0].audioArchiveMd5 = "pickup-md5";
    const crossCapture = buildCrossCaptureSummary(
      report,
      pickup,
      "2026-07-25T01:00:00.000Z",
    );
    expect(crossCapture.dataset.trackCount).toBe(1);
    expect(crossCapture.methodology.alternateCapturesNotCombinedAsIndependentTracks)
      .toBe(true);
    expect(crossCapture.conclusions.evidenceLevel).toBe("held-out-validation");
    expect(() => assertNoAbsolutePaths(crossCapture)).not.toThrow();
  });

  it("produces deterministic paired bootstrap intervals", () => {
    const values = [0.1, -0.05, 0.2, 0, 0.15];
    const first = bootstrapMeanConfidenceInterval(values, 20260725, 500);
    const second = bootstrapMeanConfidenceInterval(values, 20260725, 500);
    const differentSeed = bootstrapMeanConfidenceInterval(values, 7, 500);

    expect(first).toEqual(second);
    expect(first).not.toEqual(differentSeed);
  });

  it("resolves local GuitarSet overrides without putting them in report names", () => {
    const fixtureRoot = path.resolve("hybrid-evaluation-fixture");
    const configPath = path.join(fixtureRoot, "config", "hybrid-config.json");
    const manifestOverride = path.join(
      fixtureRoot,
      "datasets",
      "prepared",
      "manifest.json",
    );
    const annotationsOverride = path.join(
      fixtureRoot,
      "datasets",
      "prepared",
      "annotations",
    );
    const audioOverride = path.join(
      fixtureRoot,
      "datasets",
      "audio_mono-pickup_mix",
    );
    const resolved = resolveDatasetPaths({
      datasetIdentifier: "guitarset-zenodo-1492449",
      manifestPath: "../fallback/manifest.json",
      annotationDirectory: "../fallback/annotations",
      captureType: "audio_mono-mic",
    }, configPath, {
      TABSMITH_GUITARSET_MANIFEST: manifestOverride,
      TABSMITH_GUITARSET_ANNOTATIONS: annotationsOverride,
      TABSMITH_GUITARSET_AUDIO: audioOverride,
      TABSMITH_HYBRID_EVAL_CAPTURE: "audio_mono-pickup_mix",
      TABSMITH_HYBRID_EVAL_OUTPUT_TAG: "pickup-mix",
    });

    expect(resolved.manifestPath).toBe(manifestOverride);
    expect(resolved.annotationDirectory).toBe(annotationsOverride);
    expect(resolved.audioRoot).toBe(audioOverride);
    expect(resolved.captureType).toBe("audio_mono-pickup_mix");
    expect(audioBasenameForCapture(
      "00_BN1-129-Eb_comp_mic.wav",
      resolved.captureType,
    )).toBe("00_BN1-129-Eb_comp_mix.wav");
    expect(reportFileNames(resolveOutputTag(undefined, {
      TABSMITH_HYBRID_EVAL_OUTPUT_TAG: "pickup-mix",
    }))).toEqual({
      json: "hybrid-accuracy-pickup-mix.json",
      markdown: "hybrid-accuracy-pickup-mix.md",
      perTrackCsv: "hybrid-accuracy-pickup-mix-per-track.csv",
      disagreements: "hybrid-accuracy-pickup-mix-disagreements.json",
    });
    expect(reportFileNames("calibrated-mono-mic")).toEqual({
      json: "hybrid-calibrated-mono-mic.json",
      markdown: "hybrid-calibrated-mono-mic.md",
      perTrackCsv: "hybrid-calibrated-mono-mic-per-track.csv",
      disagreements: "hybrid-calibrated-mono-mic-disagreements.json",
    });
  });

  it("rejects path-like output tags", () => {
    expect(() => resolveOutputTag(undefined, {
      TABSMITH_HYBRID_EVAL_OUTPUT_TAG: "C:\\private\\report",
    })).toThrow(/portable label/);
  });
});
