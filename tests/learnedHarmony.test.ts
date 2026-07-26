import { describe, expect, it } from "vitest";
import {
  resolveSettings,
  type ChordObservation,
  type HarmonyObservationPackage,
} from "../src/chordAnalysis";
import {
  HYBRID_HARMONY_BUILD_ENABLED,
  resolveChordEngineSelection,
} from "../src/learnedHarmony/buildGate";
import type {
  LearnedHarmonyProvider,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  CONTRACT_VERSION,
  DisabledLearnedHarmonyProvider,
  MockLearnedHarmonyProvider,
  assertNotStoredAsProduction,
  buildExperimentalCacheKey,
  buildFeaturePackage,
  describeDevControl,
  isLearnedHarmonyEnabled,
  isMockResult,
  isProductionCacheable,
  resolveLearnedHarmonyProvider,
  runHybridHarmony,
  validateModelManifest,
  validateRequest,
  validateResponse,
  type FeatureSource,
  type ModelBinding,
  type RuleBasedHarmonyEvidence,
} from "../src/learnedHarmony";

const BINDING: ModelBinding = {
  modelVersion: "mock-temporal-baseline-v0",
  modelChecksum: "mock-0000000000000000",
  expectedFeatureVersion: "harmony-features-v1",
};

function featureSource(frames = 24): FeatureSource {
  const frameTimes = Array.from({ length: frames }, (_v, t) => Number((t * 0.1).toFixed(4)));
  const chroma = frameTimes.map((_v, t) => Array.from({ length: 12 }, (_c, c) => ((t + c) % 12) / 12));
  return {
    audioHash: "audio123",
    sectionStartSeconds: 0,
    sectionEndSeconds: frames * 0.1,
    frameTimes,
    harmonicChroma: chroma,
    bassChroma: chroma,
    onsetStrength: frameTimes.map((_v, t) => (t % 4 === 0 ? 0.9 : 0.1)),
  };
}

function makeRequest(frames = 24) {
  return buildFeaturePackage(featureSource(frames), BINDING, "req-1").request;
}

const RULE_BASED: RuleBasedHarmonyEvidence = {
  pipelineVersion: "2026-07-harmonic-context-v3-reduced-latency",
  regions: [
    { start: 0, end: 1, name: "C", confidence: 0.8 },
    { start: 1, end: 2, name: "Am", confidence: 0.7 },
  ],
};

function ruleObservation(
  start: number,
  bestChord: string,
  secondBestChord: string,
): ChordObservation {
  return {
    start,
    end: start + 1,
    bestChord,
    bestScore: 0.55,
    secondBestChord,
    secondBestScore: 0.5,
    confidence: 0.5,
    scoreMargin: 0.05,
    uncertain: false,
    candidateScores: { [bestChord]: 0.55, [secondBestChord]: 0.5 },
    noChordScore: -0.5,
    seventhEvidence: 0,
    boundaryStrength: start ? 0.86 : 0,
  };
}

const OBSERVATION_PACKAGE: HarmonyObservationPackage = {
  observations: [
    ruleObservation(0, "C", "Am"),
    ruleObservation(1, "Am", "C"),
  ],
  rawFrames: [],
  beatGrid: { bpm: 60, beatDuration: 1, phase: 0 },
  keyEstimate: null,
  duration: 2,
  settings: resolveSettings({
    minimumChordDurationSeconds: 1,
    minimumChordDurationBeats: 1,
    requiredConsecutiveWindows: 1,
  }),
  harmonyEvidenceSource: "full-mix",
  beatAlignedBoundaries: 1,
  learnedFeatures: featureSource(20),
};
RULE_BASED.observationPackage = OBSERVATION_PACKAGE;

describe("learned harmony: disabled by default", () => {
  it("resolves the disabled provider with no flag (app works with no ML deps/models)", async () => {
    const provider = resolveLearnedHarmonyProvider();
    expect(provider.id).toBe("disabled");
    expect(await provider.isAvailable()).toBe(false);
    await expect(new DisabledLearnedHarmonyProvider().predict()).rejects.toThrow();
  });

  it("is disabled unless the hidden flag is set", () => {
    expect(isLearnedHarmonyEnabled()).toBe(false);
    expect(isLearnedHarmonyEnabled({
      env: { TABSMITH_EXPERIMENTAL_LEARNED_HARMONY: "1" },
    })).toBe(false);
    expect(isLearnedHarmonyEnabled({
      env: { TABSMITH_EXPERIMENTAL_LEARNED_HARMONY: "1" },
      isReleaseBuild: false,
    })).toBe(true);
  });

  it("never exposes the dev control in a release build", () => {
    const env = { TABSMITH_EXPERIMENTAL_LEARNED_HARMONY: "1" };
    expect(isLearnedHarmonyEnabled({ env, isReleaseBuild: true })).toBe(false);
    expect(describeDevControl({ env, isReleaseBuild: true }).visible).toBe(false);
    expect(describeDevControl({ env, isReleaseBuild: false }).visible).toBe(true);
  });

  it("sanitizes direct hybrid selection when the compile-time gate is closed", () => {
    expect(HYBRID_HARMONY_BUILD_ENABLED).toBe(false);
    expect(resolveChordEngineSelection("hybrid", false)).toBe("rule");
    expect(resolveChordEngineSelection("onehotchord", false)).toBe("rule");
    expect(resolveChordEngineSelection("hybrid", true)).toBe("hybrid");
  });
});

describe("learned harmony: contract validation", () => {
  it("accepts a well-formed request and mock response", async () => {
    const request = makeRequest();
    expect(validateRequest(request).ok).toBe(true);
    const response = await new MockLearnedHarmonyProvider().predict(request);
    expect(validateResponse(response, request).ok).toBe(true);
  });

  it("rejects dimension mismatches", () => {
    const request = makeRequest();
    request.harmonicChroma = request.harmonicChroma.map((row) => row.slice(0, 11));
    expect(validateRequest(request).ok).toBe(false);
  });

  it("rejects non-finite probabilities/features", () => {
    const request = makeRequest();
    request.onsetStrength[3] = Number.NaN;
    expect(validateRequest(request).ok).toBe(false);
  });

  it("rejects incompatible feature and contract versions", () => {
    const a = makeRequest();
    a.featureVersion = "harmony-features-v999";
    expect(validateRequest(a).ok).toBe(false);
    const b = makeRequest();
    b.contractVersion = CONTRACT_VERSION + 1;
    expect(validateRequest(b).ok).toBe(false);
  });

  it("rejects non-monotonic timestamps", () => {
    const request = makeRequest();
    request.frameTimes[5] = request.frameTimes[4];
    expect(validateRequest(request).ok).toBe(false);
  });

  it("validates a model manifest against the contract", () => {
    const good = validateModelManifest({
      modelVersion: "temporal-baseline-v0", contractVersion: CONTRACT_VERSION,
      featureVersion: "harmony-features-v1", checksum: "abc", format: "onnx",
      vocabulary: { roots: 12, qualities: ["maj", "min", "7"], hasNoChordHead: true, hasBoundaryHead: true },
      inputShapes: { features: ["batch", "frames", 25] },
    });
    expect(good.ok).toBe(true);
  });
});

describe("learned harmony: mock provider", () => {
  it("is deterministic", async () => {
    const request = makeRequest();
    const a = await new MockLearnedHarmonyProvider().predict(request);
    const b = await new MockLearnedHarmonyProvider().predict(request);
    expect(a.rootProbabilities).toEqual(b.rootProbabilities);
    expect(a.boundaryProbabilities).toEqual(b.boundaryProbabilities);
    expect(a.diagnostics.warnings[0]).toContain("not musical inference");
  });

  it("supports cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    await expect(new MockLearnedHarmonyProvider({ latencyMs: 50 }).predict(makeRequest(), controller.signal))
      .rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("learned harmony: hybrid decoder + fallback", () => {
  const base = { ruleBased: RULE_BASED, settings: CONSERVATIVE_HYBRID_SETTINGS, enabled: true };

  it("reaches the decoder and combines on success", async () => {
    const result = await runHybridHarmony({ ...base, provider: new MockLearnedHarmonyProvider(), request: makeRequest() });
    expect(result.engine).toBe("hybrid-experimental");
    expect(result.usedLearned).toBe(true);
    expect(result.regions.length).toBeGreaterThan(0);
    expect(result.diagnostics.comparison.length)
      .toBe(OBSERVATION_PACKAGE.observations.length);
    expect(result.diagnostics.modelVersion).toBe("mock-temporal-baseline-v0");
  });

  it("falls back to rule-based when disabled", async () => {
    const result = await runHybridHarmony({ ...base, enabled: false, provider: new MockLearnedHarmonyProvider(), request: makeRequest() });
    expect(result.engine).toBe("rule-based");
    expect(result.usedLearned).toBe(false);
    expect(result.fallbackReason).toBe("learned-disabled");
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("falls back exactly when the provider is unavailable", async () => {
    const result = await runHybridHarmony({
      ...base,
      provider: new DisabledLearnedHarmonyProvider(),
      request: makeRequest(),
    });
    expect(result.fallbackReason).toBe("provider-unavailable");
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("does not run a provider without a feature package", async () => {
    const result = await runHybridHarmony({
      ...base,
      provider: new MockLearnedHarmonyProvider(),
      request: null,
    });
    expect(result.fallbackReason).toBe("no-feature-package");
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("falls back on timeout without losing the job", async () => {
    const result = await runHybridHarmony({
      ...base, provider: new MockLearnedHarmonyProvider({ latencyMs: 200 }), request: makeRequest(), timeoutMs: 5,
    });
    expect(result.engine).toBe("rule-based");
    expect(result.fallbackReason).toBe("timeout");
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("falls back on provider failure", async () => {
    const result = await runHybridHarmony({
      ...base, provider: new MockLearnedHarmonyProvider({ failWith: "boom" }), request: makeRequest(),
    });
    expect(result.fallbackReason).toBe("provider-error");
    expect(result.usedLearned).toBe(false);
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("falls back on version mismatch", async () => {
    const request = makeRequest();
    request.modelMetadata.modelVersion = "some-other-model";
    const result = await runHybridHarmony({ ...base, provider: new MockLearnedHarmonyProvider(), request });
    expect(result.fallbackReason).toBe("version-mismatch");
    expect(result.regions).toBe(RULE_BASED.regions);
  });

  it("falls back exactly on cancellation, invalid output, and response checksum mismatch", async () => {
    const cancelled = new AbortController();
    cancelled.abort();
    const cancellationResult = await runHybridHarmony({
      ...base,
      provider: new MockLearnedHarmonyProvider({ latencyMs: 20 }),
      request: makeRequest(),
      signal: cancelled.signal,
    });
    expect(cancellationResult.fallbackReason).toBe("cancelled");
    expect(cancellationResult.regions).toBe(RULE_BASED.regions);

    const delegate = new MockLearnedHarmonyProvider();
    const provider = (
      responseMutation: (response: LearnedHarmonyResponse) => void
    ): LearnedHarmonyProvider => ({
      id: "mutating-test-provider",
      isAvailable: () => delegate.isAvailable(),
      getMetadata: () => delegate.getMetadata(),
      predict: async (request, signal) => {
        const response = await delegate.predict(request, signal);
        responseMutation(response);
        return response;
      },
    });
    const invalidResult = await runHybridHarmony({
      ...base,
      provider: provider((response) => {
        response.rootProbabilities[0][0] = 2;
      }),
      request: makeRequest(),
    });
    expect(invalidResult.fallbackReason).toBe("invalid-response");
    expect(invalidResult.regions).toBe(RULE_BASED.regions);

    const checksumResult = await runHybridHarmony({
      ...base,
      provider: provider((response) => {
        response.modelChecksum = "wrong-response-checksum";
      }),
      request: makeRequest(),
    });
    expect(checksumResult.fallbackReason).toBe("version-mismatch");
    expect(checksumResult.regions).toBe(RULE_BASED.regions);
  });
});

describe("learned harmony: cache isolation", () => {
  it("never stores mock/experimental results as production", async () => {
    const response = await new MockLearnedHarmonyProvider().predict(makeRequest());
    expect(isMockResult(response)).toBe(true);
    expect(isProductionCacheable("hybrid-experimental")).toBe(false);
    expect(isProductionCacheable("rule-based")).toBe(true);
    expect(() => assertNotStoredAsProduction("hybrid-experimental", response)).toThrow();
  });

  it("embeds the model checksum in experimental cache keys", () => {
    const common = {
      engine: "hybrid-experimental" as const, audioHash: "a", modelVersion: "m",
      ruleBasedPipelineVersion: "p", hybridSettings: CONSERVATIVE_HYBRID_SETTINGS,
      sectionStartSeconds: 0, sectionEndSeconds: 2,
    };
    const keyA = buildExperimentalCacheKey({ ...common, modelChecksum: "checksum-A" });
    const keyB = buildExperimentalCacheKey({ ...common, modelChecksum: "checksum-B" });
    expect(keyA.startsWith("experimental:hybrid-experimental:")).toBe(true);
    expect(keyA).not.toBe(keyB); // checksum participates in the key
  });
});
