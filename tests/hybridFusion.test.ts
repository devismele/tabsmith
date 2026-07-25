import { describe, expect, it } from "vitest";
import {
  decodeChordSequence,
  decodeHarmonyObservations,
  resolveSettings,
  type ChordObservation,
  type HarmonyObservationPackage,
} from "../src/chordAnalysis";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  alignLearnedEvidence,
  fuseHybridObservations,
} from "../src/learnedHarmony";
import type {
  HybridHarmonySettings,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";

const ROOT_INDEX: Record<string, number> = {
  C: 0, G: 7, A: 9,
};

function observation(
  index: number,
  bestChord = "C",
  bestScore = 0.5,
  secondBestChord = "G",
  secondBestScore = 0.45,
  confidence = 0.45,
): ChordObservation {
  return {
    start: index,
    end: index + 1,
    bestChord,
    bestScore,
    secondBestChord,
    secondBestScore,
    confidence,
    scoreMargin: bestScore - secondBestScore,
    uncertain: confidence < 0.4,
    candidateScores: {
      C: bestChord === "C" ? bestScore : secondBestScore,
      G: bestChord === "G" ? bestScore : secondBestScore,
      Am: 0.2,
    },
    noChordScore: -0.45,
    seventhEvidence: 0,
    rootClass: bestChord === "N" ? null : bestChord,
    bassRootClass: bestChord === "N" ? "C" : bestChord,
    bassConfidence: 0.55,
    bassAttackRootClass: bestChord === "N" ? "C" : bestChord,
    bassAttackConfidence: 0.55,
    bassSustainRootClass: bestChord === "N" ? "C" : bestChord,
    bassSustainConfidence: 0.55,
    bassIsPassingTone: false,
    usedBassSupport: true,
    boundaryStrength: index === 0 ? 0 : 0.86,
  };
}

function peakedDistribution(size: number, index: number, peak = 0.96): number[] {
  const remainder = (1 - peak) / (size - 1);
  return Array.from({ length: size }, (_, candidate) =>
    candidate === index ? peak : remainder);
}

function learnedResponse(
  labels: string[],
  options: {
    boundary?: number[];
    uniform?: boolean;
    noChord?: number[];
  } = {},
): LearnedHarmonyResponse {
  return {
    contractVersion: 1,
    requestId: "hybrid-test",
    modelVersion: "test-model",
    modelChecksum: "test-checksum",
    featureVersion: "harmony-features-v1",
    frameTimes: labels.map((_label, index) => index + 0.5),
    rootProbabilities: labels.map((label) =>
      options.uniform
        ? Array(12).fill(1 / 12)
        : peakedDistribution(12, ROOT_INDEX[label.replace(/m$/, "")] ?? 0)),
    qualityProbabilities: labels.map((label) =>
      options.uniform
        ? Array(3).fill(1 / 3)
        : peakedDistribution(3, label.endsWith("m") ? 1 : 0, 0.98)),
    noChordProbabilities: options.noChord
      ?? labels.map((label) => label === "N" ? 0.98 : 0.01),
    boundaryProbabilities: options.boundary ?? labels.map(() => 0.05),
    diagnostics: {
      inferenceMilliseconds: 1,
      backend: "mock",
      warnings: [],
    },
  };
}

function observationPackage(
  observations: ChordObservation[],
): HarmonyObservationPackage {
  return {
    observations,
    rawFrames: [],
    beatGrid: { bpm: 60, beatDuration: 1, phase: 0 },
    keyEstimate: null,
    duration: observations.at(-1)?.end ?? 0,
    settings: resolveSettings({
      minimumChordDurationSeconds: 1,
      minimumChordDurationBeats: 1,
      requiredConsecutiveWindows: 2,
      changeEvidenceThreshold: 0.025,
      shortRegionPenalty: 0.08,
    }),
    harmonyEvidenceSource: "guitar-only",
    beatAlignedBoundaries: Math.max(0, observations.length - 1),
    learnedFeatures: {
      frameTimes: [],
      harmonicChroma: [],
      bassChroma: [],
      onsetStrength: [],
    },
  };
}

function fuse(
  observations: ChordObservation[],
  response: LearnedHarmonyResponse,
  settings: HybridHarmonySettings = CONSERVATIVE_HYBRID_SETTINGS,
) {
  return fuseHybridObservations(
    observations,
    response,
    settings,
    { sourceMode: "guitar-focused" },
  );
}

function assertRegionInvariants(
  regions: ReturnType<typeof decodeHarmonyObservations>["regions"],
  duration: number,
): void {
  expect(regions.length).toBeGreaterThan(0);
  expect(regions[0].start).toBe(0);
  expect(regions.at(-1)?.end).toBe(duration);
  regions.forEach((region, index) => {
    expect(region.end).toBeGreaterThan(region.start);
    if (index > 0) {
      expect(region.start).toBe(regions[index - 1].end);
      expect(region.start).toBeGreaterThanOrEqual(regions[index - 1].start);
    }
  });
}

describe("observation-level hybrid evidence", () => {
  it("retains the full aligned chord distribution including no-chord", () => {
    const observations = [observation(0)];
    const aligned = alignLearnedEvidence(observations, learnedResponse(["G"]));
    expect(aligned[0].chordProbabilities.G).toBeGreaterThan(
      aligned[0].chordProbabilities.C,
    );
    expect(aligned[0].chordProbabilities.C).toBeGreaterThan(0);
    expect(aligned[0].chordProbabilities.N).toBe(
      aligned[0].noChordProbability,
    );
    expect(aligned[0].noChordProbability).toBeGreaterThan(0);
    expect(aligned[0].contributingFrameCount).toBe(1);
  });

  it("reinforces rule/ML agreement without adding fragmentation", () => {
    const rules = Array.from({ length: 6 }, (_, index) => observation(index));
    const fused = fuse(rules, learnedResponse(rules.map(() => "C")));
    expect(fused.observations.every((window) => window.bestChord === "C")).toBe(true);
    expect(fused.diagnostics.ruleLearnedAgreementRate).toBe(1);
    const decoded = decodeHarmonyObservations({
      ...observationPackage(rules),
      observations: fused.observations,
    });
    expect(decoded.regions.map((region) => region.name)).toEqual(["C"]);
  });

  it("lets confident ML correct an uncertain rule observation before decoding", () => {
    const rules = Array.from(
      { length: 5 },
      (_, index) => observation(index, "C", 0.31, "G", 0.3, 0.28),
    );
    const fused = fuse(rules, learnedResponse(rules.map(() => "G")));
    expect(fused.diagnostics.changedTopCandidateWindows).toBe(5);
    expect(fused.observations.every((window) => window.bestChord === "G")).toBe(true);
    const decoded = decodeHarmonyObservations({
      ...observationPackage(rules),
      observations: fused.observations,
    });
    expect(decoded.regions.map((region) => region.name)).toEqual(["G"]);
  });

  it("protects a highly confident rule result from learned disagreement", () => {
    const rules = Array.from(
      { length: 4 },
      (_, index) => observation(index, "C", 0.9, "G", 0.25, 0.96),
    );
    const fused = fuse(rules, learnedResponse(rules.map(() => "G")));
    expect(fused.diagnostics.effectiveLearnedWeight.maximum).toBe(0);
    expect(fused.observations.every((window) => window.bestChord === "C")).toBe(true);
  });

  it("makes high-entropy learned output negligible", () => {
    const rules = Array.from(
      { length: 3 },
      (_, index) => observation(index, "C", 0.42, "G", 0.4, 0.35),
    );
    const fused = fuse(
      rules,
      learnedResponse(rules.map(() => "G"), {
        uniform: true,
        noChord: rules.map(() => 0.5),
      }),
    );
    expect(fused.diagnostics.effectiveLearnedWeight.maximum).toBe(0);
    fused.observations.forEach((window, index) => {
      expect(window.bestChord).toBe(rules[index].bestChord);
      expect(window.scoreMargin).toBeCloseTo(rules[index].scoreMargin, 5);
    });
  });

  it("turns a reliable learned boundary spike into capped transition evidence", () => {
    const rules = [
      observation(0, "C", 0.6, "G", 0.25, 0.6),
      observation(1, "C", 0.58, "G", 0.3, 0.55),
      observation(2, "G", 0.52, "C", 0.45, 0.4),
      observation(3, "G", 0.58, "C", 0.3, 0.55),
    ];
    const noBoundary = fuse(rules, learnedResponse(["C", "C", "G", "G"]));
    const withBoundary = fuse(
      rules,
      learnedResponse(["C", "C", "G", "G"], {
        boundary: [0.05, 0.05, 0.99, 0.05],
      }),
    );
    expect(withBoundary.observations[2].learnedBoundaryAdjustment).toBeGreaterThan(0);
    expect(withBoundary.observations[2].learnedBoundaryAdjustment).toBeLessThanOrEqual(0.06);
    const withoutLabels = decodeChordSequence(noBoundary.observations, {
      chordChangePenalty: 0.13,
    });
    const withLabels = decodeChordSequence(withBoundary.observations, {
      chordChangePenalty: 0.13,
    });
    const firstG = (labels: string[]) => {
      const index = labels.indexOf("G");
      return index < 0 ? Number.POSITIVE_INFINITY : index;
    };
    expect(firstG(withLabels)).toBeLessThanOrEqual(firstG(withoutLabels));
  });

  it("suppresses learned boundary flicker and rapidly alternating labels", () => {
    const rules = Array.from(
      { length: 8 },
      (_, index) => observation(index, "C", 0.55, "G", 0.42, 0.55),
    );
    const labels = rules.map((_window, index) => index % 2 ? "G" : "C");
    const response = learnedResponse(labels, {
      boundary: rules.map((_window, index) => index % 2 ? 0.95 : 0.8),
    });
    const fused = fuse(rules, response);
    expect(fused.diagnostics.effectiveLearnedWeight.maximum)
      .toBeLessThan(CONSERVATIVE_HYBRID_SETTINGS.learnedChordWeight);
    const decoded = decodeHarmonyObservations({
      ...observationPackage(rules),
      observations: fused.observations,
    });
    expect(decoded.regions).toHaveLength(1);
    expect(decoded.regions[0].name).toBe("C");
    assertRegionInvariants(decoded.regions, rules.length);
  });

  it("prevents an incorrect learned no-chord from defeating sustained rule/bass evidence", () => {
    const rules = Array.from(
      { length: 5 },
      (_, index) => observation(index, "C", 0.62, "G", 0.4, 0.68),
    );
    const response = learnedResponse(rules.map(() => "N"), {
      noChord: rules.map(() => 0.98),
    });
    const fused = fuse(rules, response);
    expect(fused.observations.every((window) => window.bestChord !== "N")).toBe(true);
    const decoded = decodeHarmonyObservations({
      ...observationPackage(rules),
      observations: fused.observations,
    });
    expect(decoded.regions.some((region) => region.name === "N")).toBe(false);
  });
});
