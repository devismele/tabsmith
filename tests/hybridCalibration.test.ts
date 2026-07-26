import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { describe, expect, it } from "vitest";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
} from "../src/learnedHarmony/hybridDecoder";
import { resolveChordEngineSelection } from "../src/learnedHarmony/buildGate";
import type {
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";
import {
  CachedLearnedHarmonyProvider,
  learnedInferenceCacheKey,
} from "../evaluation/hybridInferenceCache";
import {
  assertFinalValidationIsFrozen,
  assertNonEmptyCalibrationTracks,
  assignGroupedFolds,
  calibrateLearnedResponse,
  candidateSatisfiesConstraints,
  generateCalibrationCandidates,
  settingsIdentity,
  validateCalibrationSplit,
  type CandidateSelectionRow,
  type HybridCalibrationSplitManifest,
} from "../evaluation/hybridCalibration";

const SPLIT: HybridCalibrationSplitManifest = {
  schemaVersion: 1,
  seed: 20260726,
  strategy: "leave-one-performer-out",
  captures: ["audio_mono-mic", "audio_mono-pickup_mix"],
  finalValidationPerformer: "guitarset-p00",
  calibrationPerformers: [
    { performerId: "guitarset-p01", learnedModelRole: "training" },
    { performerId: "guitarset-p02", learnedModelRole: "development" },
  ],
  folds: [
    {
      foldId: "p01",
      validationPerformers: ["guitarset-p01"],
      calibrationTrainPerformers: ["guitarset-p02"],
    },
    {
      foldId: "p02",
      validationPerformers: ["guitarset-p02"],
      calibrationTrainPerformers: ["guitarset-p01"],
    },
  ],
  notes: [],
};

function request(checksum = "checksum", featureVersion = "harmony-features-v1"):
LearnedHarmonyRequest {
  return {
    contractVersion: 1,
    requestId: "calibration-cache-test",
    audioHash: "audio-hash",
    sectionStartSeconds: 0,
    sectionEndSeconds: 1,
    featureVersion,
    frameTimes: [0],
    harmonicChroma: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    bassChroma: [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    onsetStrength: [0.5],
    modelMetadata: {
      modelVersion: "model",
      modelChecksum: checksum,
      expectedFeatureVersion: featureVersion,
    },
  };
}

function response(input: LearnedHarmonyRequest): LearnedHarmonyResponse {
  return {
    contractVersion: 1,
    requestId: input.requestId,
    modelVersion: input.modelMetadata.modelVersion,
    modelChecksum: input.modelMetadata.modelChecksum,
    featureVersion: input.featureVersion,
    frameTimes: input.frameTimes.slice(),
    rootProbabilities: [[0.8, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
    qualityProbabilities: [[0.7, 0.2, 0.1]],
    noChordProbabilities: [0.05],
    boundaryProbabilities: [0.2],
    diagnostics: {
      inferenceMilliseconds: 1,
      backend: "mock",
      warnings: [],
    },
  };
}

class CountingProvider implements LearnedHarmonyProvider {
  readonly id = "counting";
  calls = 0;

  async isAvailable(): Promise<boolean> {
    return true;
  }

  async getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return {
      modelVersion: "model",
      modelChecksum: "checksum",
      contractVersion: 1,
      featureVersion: "harmony-features-v1",
      format: "mock",
      vocabulary: {
        roots: 12,
        qualities: ["maj", "min", "7"],
        hasNoChordHead: true,
        hasBoundaryHead: true,
      },
      inputShapes: { features: ["batch", "frames", 25] },
      disclaimer: "test",
    };
  }

  async predict(input: LearnedHarmonyRequest): Promise<LearnedHarmonyResponse> {
    this.calls += 1;
    return response(input);
  }
}

describe("hybrid calibration protocol", () => {
  it("keeps p00 sealed from calibration and requires a valid freeze", () => {
    validateCalibrationSplit(SPLIT);
    expect(() => validateCalibrationSplit({
      ...SPLIT,
      calibrationPerformers: [
        ...SPLIT.calibrationPerformers,
        { performerId: "guitarset-p00", learnedModelRole: "training" },
      ],
    })).toThrow(/p00/);
    expect(() => assertFinalValidationIsFrozen(null)).toThrow(/cannot execute/);
    expect(() => assertFinalValidationIsFrozen({
      status: "frozen-before-p00-validation",
      settingsIdentity: "a".repeat(64),
    })).not.toThrow();
  });

  it("groups alternate captures and produces deterministic fold assignments", () => {
    const tracks = [
      {
        trackId: "guitarset-01_track",
        performerId: "guitarset-p01",
        captureType: "audio_mono-mic",
      },
      {
        trackId: "guitarset-01_track",
        performerId: "guitarset-p01",
        captureType: "audio_mono-pickup_mix",
      },
      {
        trackId: "guitarset-02_track",
        performerId: "guitarset-p02",
        captureType: "audio_mono-mic",
      },
    ];
    const first = assignGroupedFolds(tracks, SPLIT);
    const second = assignGroupedFolds([...tracks].reverse(), SPLIT);
    expect(first).toEqual(second);
    expect(first["guitarset-01_track:audio_mono-mic"])
      .toBe(first["guitarset-01_track:audio_mono-pickup_mix"]);
  });

  it("fails a zero-track calibration", () => {
    expect(() => assertNonEmptyCalibrationTracks(0)).toThrow(/zero tracks/);
    expect(() => assertNonEmptyCalibrationTracks(1)).not.toThrow();
  });

  it("reuses identical learned responses and isolates cache identities", async () => {
    const directory = await mkdtemp(path.join(os.tmpdir(), "tabsmith-calibration-test-"));
    try {
      const inner = new CountingProvider();
      const cached = new CachedLearnedHarmonyProvider(
        inner,
        directory,
        "audio_mono-mic",
      );
      const input = request();
      const first = await cached.predict(input);
      const second = await cached.predict(input);
      expect(first).toBe(second);
      expect(inner.calls).toBe(1);
      expect(cached.stats.memoryHits).toBe(1);
      expect(JSON.parse(await readFile(
        path.join(directory, `${learnedInferenceCacheKey(input, "audio_mono-mic")}.json`),
        "utf8",
      )).response.rootProbabilities).toEqual(first.rootProbabilities);
      expect(learnedInferenceCacheKey(input, "audio_mono-mic"))
        .not.toBe(learnedInferenceCacheKey(input, "audio_mono-pickup_mix"));
      expect(learnedInferenceCacheKey(input, "audio_mono-mic"))
        .not.toBe(learnedInferenceCacheKey(request("different"), "audio_mono-mic"));
      expect(learnedInferenceCacheKey(input, "audio_mono-mic"))
        .not.toBe(learnedInferenceCacheKey(
          request("checksum", "different-feature"),
          "audio_mono-mic",
        ));
    } finally {
      await rm(directory, { recursive: true, force: true });
    }
  });

  it("makes probability calibration deterministic without mutating its source", () => {
    const source = response(request());
    const calibrated = calibrateLearnedResponse(source, {
      kind: "temperature",
      value: 0.75,
    });
    expect(calibrated).not.toBe(source);
    expect(source.rootProbabilities[0][0]).toBe(0.8);
    expect(calibrated.rootProbabilities[0][0]).toBeGreaterThan(0.8);
    expect(calibrateLearnedResponse(source, { kind: "none", value: 1 })).toBe(source);
  });

  it("generates deterministic searches and reproduces the old settings baseline", () => {
    const first = generateCalibrationCandidates(20260726, 8);
    const second = generateCalibrationCandidates(20260726, 8);
    expect(first).toEqual(second);
    expect(first[0]).toMatchObject({
      id: "current-adaptive",
      settings: CONSERVATIVE_HYBRID_SETTINGS,
      adaptiveWeighting: true,
    });
  });

  it("enforces fragmentation constraints before accuracy can win", () => {
    const candidate = generateCalibrationCandidates(1, 0)[0];
    const row: CandidateSelectionRow = {
      candidate,
      captures: {
        mic: {
          detailedAccuracy: 0.9,
          rootAccuracy: 0.9,
          fragmentationRate: 0.57,
          regionsPerMinute: 18,
          meanAbsoluteBoundaryErrorMs: 1000,
          incorrectOverrides: 0,
          fallbacks: 0,
          regionInvariantsValid: true,
        },
      },
      meanDetailedAccuracy: 0.9,
      meanRootAccuracy: 0.9,
      worstFoldDetailedAccuracy: 0.9,
    };
    const rule = {
      mic: {
        ...row.captures.mic,
        fragmentationRate: 0.54,
      },
    };
    expect(candidateSatisfiesConstraints(row, rule, {
      maximumFragmentationAboveRule: 0.01,
      maximumRegionsPerMinuteAboveRule: 0.5,
      maximumBoundaryMaeAboveRuleMs: 100,
    })).toBe(false);
  });

  it("changes settings identity when any fusion setting changes", () => {
    const base = settingsIdentity(CONSERVATIVE_HYBRID_SETTINGS);
    expect(settingsIdentity({
      ...CONSERVATIVE_HYBRID_SETTINGS,
      learnedChordWeight: 0.36,
    })).not.toBe(base);
    expect(settingsIdentity(
      CONSERVATIVE_HYBRID_SETTINGS,
      { kind: "temperature", value: 0.9 },
    )).not.toBe(base);
  });

  it("leaves the release build gate closed", () => {
    expect(resolveChordEngineSelection("hybrid", false)).toBe("rule");
  });
});
