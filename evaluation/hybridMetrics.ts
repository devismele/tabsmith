// The repository's canonical chord evaluator is intentionally plain ESM.
// @ts-expect-error No declaration file is shipped for this internal Node module.
import { evaluateChordReference } from "../server/evaluation.mjs";
import type { ChordEvent } from "../src/types";
import type {
  AblationPrediction,
  DisagreementWindow,
  EngineId,
  EnginePrediction,
  EvaluationReferenceRegion,
  HybridAblationId,
  SourceType,
  TrackEngineComparison,
} from "./hybridComparison";

export const MAIN_METRICS = [
  "rootAccuracy",
  "majorMinorAccuracy",
  "detailedAccuracy",
  "noChordF1",
  "fragmentationRate",
  "meanAbsoluteBoundaryErrorMs",
] as const;

export type MainMetricName = typeof MAIN_METRICS[number];

export interface ErrorTaxonomyValue {
  occurrences: number;
  durationSeconds: number;
}

export type ErrorTaxonomy = Record<
  | "wrongRoot"
  | "majorMinorConfusion"
  | "seventhQualityConfusion"
  | "falseNoChord"
  | "missedNoChord"
  | "earlyBoundary"
  | "lateBoundary"
  | "excessiveFragmentation"
  | "missedShortChord"
  | "incorrectMlOverride"
  | "ruleProtectionBlockedCorrectMl",
  ErrorTaxonomyValue
>;

interface DurationTotals {
  evaluated: number;
  rootCorrect: number;
  majorMinorCorrect: number;
  detailedCorrect: number;
  referenceNoChord: number;
  predictedNoChord: number;
  truePositiveNoChord: number;
}

export interface EngineTrackMetrics {
  evaluatedDurationSeconds: number;
  rootAccuracy: number;
  majorMinorAccuracy: number;
  detailedAccuracy: number;
  chordSymbolRecall: number;
  weightedChordSymbolRecall: number;
  noChordPrecision: number;
  noChordRecall: number;
  noChordF1: number;
  fragmentationRate: number;
  referenceRegionCount: number;
  predictedRegionCount: number;
  extraChordRegions: number;
  missingChordRegions: number;
  meanAbsoluteBoundaryErrorMs: number | null;
  medianAbsoluteBoundaryErrorMs: number | null;
  meanSignedBoundaryErrorMs: number | null;
  boundariesWithin100ms: number | null;
  boundariesWithin250ms: number | null;
  boundariesWithin500ms: number | null;
  boundariesWithin1000ms: number | null;
  referenceBoundaryCount: number;
  averageChordRegionDurationSeconds: number;
  regionsPerMinute: number;
  oneWindowRegionCount: number;
  oneWindowRegionRate: number;
  veryShortRegionCount: number;
  veryShortRegionRate: number;
  flickerEventCount: number;
  mlOverrideCountVsRule: number | null;
  percentRuleWindowsChangedByMl: number | null;
  ruleMlAgreementRate: number | null;
  averageEffectiveLearnedWeight: number | null;
  averageLearnedEntropy: number | null;
  runtimePerAudioMinuteSeconds: number;
  peakMemoryMiB: number | null;
  errorTaxonomy: ErrorTaxonomy;
  /** Kept in-memory for exact aggregate medians; omitted from saved reports. */
  boundaryErrorsMs: number[];
  /** Kept in-memory for duration weighting; omitted from saved reports. */
  durationTotals: DurationTotals;
}

export interface ScoredTrackComparison {
  comparison: TrackEngineComparison;
  engineMetrics: Partial<Record<EngineId, EngineTrackMetrics>>;
  ablationMetrics: Partial<Record<HybridAblationId, EngineTrackMetrics>>;
}

export interface MetricSummary {
  mean: number | null;
  median: number | null;
  bootstrap95ConfidenceInterval: [number, number] | null;
  trackCount: number;
}

export interface EngineAggregate {
  trackCount: number;
  totalEvaluatedDurationSeconds: number;
  durationWeighted: {
    rootAccuracy: number;
    majorMinorAccuracy: number;
    detailedAccuracy: number;
    weightedChordSymbolRecall: number;
    noChordPrecision: number;
    noChordRecall: number;
    noChordF1: number;
    fragmentationRate: number;
    meanAbsoluteBoundaryErrorMs: number | null;
    medianAbsoluteBoundaryErrorMs: number | null;
    meanSignedBoundaryErrorMs: number | null;
    boundariesWithin100ms: number | null;
    boundariesWithin250ms: number | null;
    boundariesWithin500ms: number | null;
    boundariesWithin1000ms: number | null;
    predictedRegionCount: number;
    extraChordRegions: number;
    missingChordRegions: number;
    regionsPerMinute: number;
    oneWindowRegionRate: number;
    veryShortRegionRate: number;
    flickerEventCount: number;
    averageChordRegionDurationSeconds: number;
    runtimePerAudioMinuteSeconds: number;
  };
  trackLevel: Partial<Record<MainMetricName, MetricSummary>>;
  diagnostics: {
    mlOverrideCountVsRule: number;
    percentRuleWindowsChangedByMl: number | null;
    ruleMlAgreementRate: number | null;
    averageEffectiveLearnedWeight: number | null;
    averageLearnedEntropy: number | null;
    peakMemoryMiB: number | null;
  };
  errorTaxonomy: ErrorTaxonomy;
}

export interface PairedDifference {
  metric: MainMetricName;
  comparison:
    | "hybrid-minus-rule"
    | "hybrid-minus-ml-only"
    | "hybrid-minus-legacy-region-hybrid";
  meanDifference: number | null;
  medianDifference: number | null;
  bootstrap95ConfidenceInterval: [number, number] | null;
  improvedTracks: number;
  tiedTracks: number;
  worsenedTracks: number;
  trackCount: number;
}

interface ParsedChord {
  root: number | null;
  family: "maj" | "min" | "dim" | "N";
  detailed: string;
  plain: string;
  noChord: boolean;
}

const PITCH_NAMES = [
  "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];
const PITCH_INDEX: Record<string, number> = {
  C: 0, "B#": 0, "C#": 1, DB: 1, D: 2, "D#": 3, EB: 3,
  E: 4, FB: 4, "E#": 5, F: 5, "F#": 6, GB: 6, G: 7,
  "G#": 8, AB: 8, A: 9, "A#": 10, BB: 10, B: 11, CB: 11,
};

function round(value: number, places = 6): number {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function safeDivide(numerator: number, denominator: number): number {
  return denominator ? numerator / denominator : 0;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  const middle = Math.floor(ordered.length / 2);
  return ordered.length % 2
    ? ordered[middle]
    : (ordered[middle - 1] + ordered[middle]) / 2;
}

function parseChord(label: string): ParsedChord {
  const compact = String(label ?? "")
    .trim()
    .replaceAll("♯", "#")
    .replaceAll("♭", "b")
    .replace(/\s+/g, "");
  if (!compact || compact.toUpperCase() === "N" || compact === "N.C.") {
    return { root: null, family: "N", detailed: "N", plain: "N", noChord: true };
  }
  if (compact.toUpperCase() === "X" || compact === "?") {
    return { root: null, family: "N", detailed: "N", plain: "N", noChord: true };
  }
  const withoutBass = compact.split("/", 1)[0];
  const match = /^([A-Ga-g])([#b]?)(?::)?(.*)$/.exec(withoutBass);
  if (!match) {
    return { root: null, family: "N", detailed: "N", plain: "N", noChord: true };
  }
  const pitchToken = `${match[1].toUpperCase()}${match[2]}`;
  const root = PITCH_INDEX[pitchToken]
    ?? PITCH_INDEX[pitchToken.toUpperCase()]
    ?? null;
  if (root === null) {
    return { root: null, family: "N", detailed: "N", plain: "N", noChord: true };
  }
  const rawQuality = match[3]
    .replace(/\(.*/, "")
    .replace(/,.*/, "")
    .toLowerCase();
  let quality: string;
  if (!rawQuality || rawQuality === "maj" || rawQuality === "major") {
    quality = "maj";
  } else if (["m", "min", "minor"].includes(rawQuality)) {
    quality = "min";
  } else if (["m7", "min7"].includes(rawQuality)) {
    quality = "min7";
  } else if (rawQuality === "maj7") {
    quality = "maj7";
  } else if (["dim", "dim7", "hdim7", "°"].includes(rawQuality)) {
    quality = "dim";
  } else if (rawQuality === "7" || rawQuality === "dom7") {
    quality = "7";
  } else if (["sus2", "sus4", "add9", "aug", "+", "5"].includes(rawQuality)) {
    quality = rawQuality === "+" ? "aug" : rawQuality;
  } else {
    quality = rawQuality.startsWith("m") && !rawQuality.startsWith("maj")
      ? "min"
      : "maj";
  }
  const family = quality === "min" || quality === "min7"
    ? "min"
    : quality === "dim" ? "dim" : "maj";
  const suffix = {
    maj: "",
    min: "m",
    min7: "m7",
    maj7: "maj7",
    dim: "dim",
    "7": "7",
    sus2: "sus2",
    sus4: "sus4",
    add9: "add9",
    aug: "aug",
    "5": "5",
  }[quality] ?? "";
  const plain = `${PITCH_NAMES[root]}${suffix}`;
  return {
    root,
    family,
    detailed: plain,
    plain,
    noChord: false,
  };
}

export function canonicalChordLabel(label: string): string {
  return parseChord(label).detailed;
}

function chordAt<T extends { start: number; end: number }>(
  regions: T[],
  time: number,
): T | undefined {
  return regions.find((region, index) =>
    time >= region.start
      && (time < region.end
        || (index === regions.length - 1 && time <= region.end)));
}

function normalizeReference(
  regions: EvaluationReferenceRegion[],
): EvaluationReferenceRegion[] {
  return regions.map((region) => ({
    ...region,
    label: parseChord(region.label).plain,
  }));
}

function normalizePrediction(regions: ChordEvent[]): ChordEvent[] {
  return regions.map((region) => ({
    ...region,
    name: parseChord(region.name).plain,
  }));
}

function emptyTaxonomy(): ErrorTaxonomy {
  return {
    wrongRoot: { occurrences: 0, durationSeconds: 0 },
    majorMinorConfusion: { occurrences: 0, durationSeconds: 0 },
    seventhQualityConfusion: { occurrences: 0, durationSeconds: 0 },
    falseNoChord: { occurrences: 0, durationSeconds: 0 },
    missedNoChord: { occurrences: 0, durationSeconds: 0 },
    earlyBoundary: { occurrences: 0, durationSeconds: 0 },
    lateBoundary: { occurrences: 0, durationSeconds: 0 },
    excessiveFragmentation: { occurrences: 0, durationSeconds: 0 },
    missedShortChord: { occurrences: 0, durationSeconds: 0 },
    incorrectMlOverride: { occurrences: 0, durationSeconds: 0 },
    ruleProtectionBlockedCorrectMl: { occurrences: 0, durationSeconds: 0 },
  };
}

function addTaxonomy(
  taxonomy: ErrorTaxonomy,
  key: keyof ErrorTaxonomy,
  durationSeconds = 0,
): void {
  taxonomy[key].occurrences += 1;
  taxonomy[key].durationSeconds += durationSeconds;
}

function durationTotals(
  reference: EvaluationReferenceRegion[],
  predicted: ChordEvent[],
  taxonomy: ErrorTaxonomy,
): DurationTotals {
  const points = [...new Set(
    [...reference, ...predicted].flatMap((region) => [region.start, region.end]),
  )].sort((left, right) => left - right);
  const totals: DurationTotals = {
    evaluated: 0,
    rootCorrect: 0,
    majorMinorCorrect: 0,
    detailedCorrect: 0,
    referenceNoChord: 0,
    predictedNoChord: 0,
    truePositiveNoChord: 0,
  };
  let previousError: keyof ErrorTaxonomy | null = null;
  for (let index = 0; index + 1 < points.length; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (end <= start) continue;
    const midpoint = (start + end) / 2;
    const expectedRegion = chordAt(reference, midpoint);
    if (!expectedRegion) continue;
    const detectedRegion = chordAt(predicted, midpoint);
    const expected = parseChord(expectedRegion.label);
    const detected = parseChord(detectedRegion?.name ?? "N");
    const duration = end - start;
    totals.evaluated += duration;
    if (expected.root === detected.root) totals.rootCorrect += duration;
    if (expected.root === detected.root && expected.family === detected.family) {
      totals.majorMinorCorrect += duration;
    }
    if (expected.detailed === detected.detailed) totals.detailedCorrect += duration;
    if (expected.noChord) totals.referenceNoChord += duration;
    if (detected.noChord) totals.predictedNoChord += duration;
    if (expected.noChord && detected.noChord) totals.truePositiveNoChord += duration;

    let error: keyof ErrorTaxonomy | null = null;
    if (!expected.noChord && detected.noChord) error = "falseNoChord";
    else if (expected.noChord && !detected.noChord) error = "missedNoChord";
    else if (expected.root !== detected.root) error = "wrongRoot";
    else if (expected.family !== detected.family) error = "majorMinorConfusion";
    else if (expected.detailed !== detected.detailed) {
      error = "seventhQualityConfusion";
    }
    if (error) {
      taxonomy[error].durationSeconds += duration;
      if (error !== previousError) taxonomy[error].occurrences += 1;
    }
    previousError = error;
  }
  return totals;
}

function nearestBoundaryErrors(
  reference: EvaluationReferenceRegion[],
  predicted: ChordEvent[],
): number[] {
  const expected = reference.slice(1).map((region) => region.start);
  const actual = predicted.slice(1).map((region) => region.start);
  if (!actual.length) return expected.map(() => Number.POSITIVE_INFINITY);
  return expected.map((boundary) => {
    const nearest = actual.reduce((best, candidate) =>
      Math.abs(candidate - boundary) < Math.abs(best - boundary)
        ? candidate
        : best);
    return (nearest - boundary) * 1000;
  });
}

function percentageWithin(errors: number[], milliseconds: number): number | null {
  if (!errors.length) return null;
  return safeDivide(
    errors.filter((error) => Number.isFinite(error) && Math.abs(error) <= milliseconds).length,
    errors.length,
  );
}

function countFlickerEvents(regions: ChordEvent[], shortSeconds: number): number {
  let count = 0;
  for (let index = 1; index + 1 < regions.length; index += 1) {
    const previous = parseChord(regions[index - 1].name).detailed;
    const current = parseChord(regions[index].name).detailed;
    const next = parseChord(regions[index + 1].name).detailed;
    if (previous === next
      && previous !== current
      && regions[index].end - regions[index].start <= shortSeconds) count += 1;
  }
  return count;
}

function countFragmentation(
  reference: EvaluationReferenceRegion[],
  predicted: ChordEvent[],
  taxonomy: ErrorTaxonomy,
): number {
  let fragmented = 0;
  for (const expected of reference) {
    const overlaps = predicted.filter((actual) =>
      actual.start < expected.end && actual.end > expected.start);
    if (overlaps.length > 1) {
      fragmented += 1;
      addTaxonomy(taxonomy, "excessiveFragmentation", expected.end - expected.start);
    }
  }
  return fragmented;
}

function countMissedShortChords(
  reference: EvaluationReferenceRegion[],
  predicted: ChordEvent[],
  shortSeconds: number,
  taxonomy: ErrorTaxonomy,
): void {
  for (const expected of reference) {
    const duration = expected.end - expected.start;
    if (duration > shortSeconds) continue;
    const actual = chordAt(predicted, (expected.start + expected.end) / 2);
    if (parseChord(actual?.name ?? "N").detailed !== parseChord(expected.label).detailed) {
      addTaxonomy(taxonomy, "missedShortChord", duration);
    }
  }
}

function countOverrides(
  comparison: TrackEngineComparison,
  engineId: EngineId,
  taxonomy: ErrorTaxonomy,
): { count: number | null; percent: number | null } {
  if (engineId !== "observation-hybrid") return { count: null, percent: null };
  const rule = comparison.predictions["rule-only"];
  const hybrid = comparison.predictions["observation-hybrid"];
  if (!rule || !hybrid || !hybrid.usedLearned) return { count: 0, percent: 0 };
  const windows = rule.chordAnalysis?.windows ?? [];
  let changed = 0;
  for (const window of windows) {
    const midpoint = (window.start + window.end) / 2;
    const ruleLabel = parseChord(regionAtTime(rule.regions, midpoint)?.name ?? "N").detailed;
    const hybridLabel = parseChord(regionAtTime(hybrid.regions, midpoint)?.name ?? "N").detailed;
    if (ruleLabel === hybridLabel) continue;
    changed += 1;
    const expected = parseChord(
      chordAt(comparison.track.referenceRegions, midpoint)?.label ?? "N",
    ).detailed;
    if (ruleLabel === expected && hybridLabel !== expected) {
      addTaxonomy(
        taxonomy,
        "incorrectMlOverride",
        window.end - window.start,
      );
    }
  }
  for (const window of comparison.disagreementWindows) {
    const localTime = window.startSeconds - comparison.track.originalOffsetSeconds;
    const expected = parseChord(
      chordAt(comparison.track.referenceRegions, localTime)?.label ?? "N",
    ).detailed;
    const learned = parseChord(window.learnedTopCandidate ?? "N").detailed;
    const ruleLabel = parseChord(window.ruleTopCandidate).detailed;
    const hybridLabel = parseChord(window.hybridFinalLabel).detailed;
    if (learned === expected && ruleLabel !== expected && hybridLabel === ruleLabel) {
      addTaxonomy(
        taxonomy,
        "ruleProtectionBlockedCorrectMl",
        window.endSeconds - window.startSeconds,
      );
    }
  }
  return {
    count: changed,
    percent: safeDivide(changed, windows.length),
  };
}

function regionAtTime(regions: ChordEvent[], time: number): ChordEvent | undefined {
  return chordAt(regions, time);
}

function scorePrediction(
  comparison: TrackEngineComparison,
  prediction: EnginePrediction,
  shortRegionThresholdSeconds: number,
): EngineTrackMetrics {
  const reference = normalizeReference(comparison.track.referenceRegions);
  const predicted = normalizePrediction(prediction.regions);
  const base = evaluateChordReference(
    {
      id: `${comparison.track.datasetId}:${comparison.track.trackId}`,
      artist: comparison.track.artist,
      title: comparison.track.title,
      source: comparison.track.datasetId,
      version: comparison.track.splitName,
      referenceUrl: "",
      selectionReason: "Frozen timed dataset annotation.",
      alignment: {
        method: "dataset-timed",
        verified: true,
        confidence: 1,
        regions: reference.map((region) => ({
          start: region.start,
          end: region.end,
          chord: region.label,
        })),
      },
    },
    {
      duration: comparison.track.duration,
      bpm: comparison.track.bpm,
      chords: predicted,
      chordAnalysis: prediction.chordAnalysis,
    },
  );
  const taxonomy = emptyTaxonomy();
  const totals = durationTotals(reference, predicted, taxonomy);
  const errors = nearestBoundaryErrors(reference, predicted);
  errors.forEach((error) => {
    if (!Number.isFinite(error)) return;
    if (error < -80) addTaxonomy(taxonomy, "earlyBoundary");
    else if (error > 80) addTaxonomy(taxonomy, "lateBoundary");
  });
  const fragmented = countFragmentation(reference, predicted, taxonomy);
  countMissedShortChords(reference, predicted, shortRegionThresholdSeconds, taxonomy);
  const overrides = countOverrides(comparison, prediction.engineId, taxonomy);
  const finiteErrors = errors.filter(Number.isFinite);
  const absoluteErrors = finiteErrors.map(Math.abs);
  const noChordPrecision = safeDivide(
    totals.truePositiveNoChord,
    totals.predictedNoChord,
  );
  const noChordRecall = safeDivide(
    totals.truePositiveNoChord,
    totals.referenceNoChord,
  );
  const noChordF1 = noChordPrecision + noChordRecall
    ? 2 * noChordPrecision * noChordRecall / (noChordPrecision + noChordRecall)
    : 0;
  const totalRegionDuration = predicted.reduce(
    (sum, region) => sum + Math.max(0, region.end - region.start),
    0,
  );
  const veryShortRegionCount = predicted.filter(
    (region) => region.end - region.start <= shortRegionThresholdSeconds,
  ).length;
  const sharedWindows = comparison.predictions["rule-only"]?.chordAnalysis?.windows ?? [];
  const oneWindowRegionCount = predicted.filter((region) =>
    sharedWindows.filter((window) => {
      const midpoint = (window.start + window.end) / 2;
      return midpoint >= region.start && midpoint < region.end;
    }).length <= 1).length;
  const durationMinutes = totals.evaluated / 60;

  return {
    evaluatedDurationSeconds: round(totals.evaluated),
    rootAccuracy: round(safeDivide(totals.rootCorrect, totals.evaluated)),
    majorMinorAccuracy: round(
      safeDivide(totals.majorMinorCorrect, totals.evaluated),
    ),
    detailedAccuracy: round(safeDivide(totals.detailedCorrect, totals.evaluated)),
    chordSymbolRecall: base.metrics.chordSymbolRecall,
    weightedChordSymbolRecall: round(
      safeDivide(totals.detailedCorrect, totals.evaluated),
    ),
    noChordPrecision: round(noChordPrecision),
    noChordRecall: round(noChordRecall),
    noChordF1: round(noChordF1),
    fragmentationRate: round(safeDivide(fragmented, reference.length)),
    referenceRegionCount: reference.length,
    predictedRegionCount: predicted.length,
    extraChordRegions: base.sequence.extraDetectedChords,
    missingChordRegions: base.sequence.missingReferenceChords,
    meanAbsoluteBoundaryErrorMs: absoluteErrors.length
      ? round(mean(absoluteErrors), 3)
      : null,
    medianAbsoluteBoundaryErrorMs: absoluteErrors.length
      ? round(median(absoluteErrors)!, 3)
      : null,
    meanSignedBoundaryErrorMs: finiteErrors.length
      ? round(mean(finiteErrors), 3)
      : null,
    boundariesWithin100ms: percentageWithin(errors, 100),
    boundariesWithin250ms: percentageWithin(errors, 250),
    boundariesWithin500ms: percentageWithin(errors, 500),
    boundariesWithin1000ms: percentageWithin(errors, 1000),
    referenceBoundaryCount: errors.length,
    averageChordRegionDurationSeconds: round(
      safeDivide(totalRegionDuration, predicted.length),
    ),
    regionsPerMinute: round(safeDivide(predicted.length, durationMinutes)),
    oneWindowRegionCount,
    oneWindowRegionRate: round(safeDivide(oneWindowRegionCount, predicted.length)),
    veryShortRegionCount,
    veryShortRegionRate: round(safeDivide(veryShortRegionCount, predicted.length)),
    flickerEventCount: countFlickerEvents(predicted, shortRegionThresholdSeconds),
    mlOverrideCountVsRule: overrides.count,
    percentRuleWindowsChangedByMl: overrides.percent,
    ruleMlAgreementRate:
      prediction.diagnostics?.ruleLearnedAgreementRate ?? null,
    averageEffectiveLearnedWeight:
      prediction.diagnostics?.effectiveLearnedWeightAverage ?? null,
    averageLearnedEntropy:
      prediction.diagnostics?.averageLearnedEntropy ?? null,
    runtimePerAudioMinuteSeconds: round(
      safeDivide(prediction.timing.endToEndMilliseconds / 1000, durationMinutes),
    ),
    peakMemoryMiB: prediction.timing.peakHeapBytes === null
      ? null
      : round(prediction.timing.peakHeapBytes / (1024 ** 2), 3),
    errorTaxonomy: taxonomy,
    boundaryErrorsMs: errors,
    durationTotals: totals,
  };
}

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

export function scoreTrackComparison(
  comparison: TrackEngineComparison,
  shortRegionThresholdSeconds = 0.5,
): ScoredTrackComparison {
  const engineMetrics: ScoredTrackComparison["engineMetrics"] = {};
  for (const [engineId, prediction] of Object.entries(comparison.predictions)) {
    if (!prediction) continue;
    engineMetrics[engineId as EngineId] = scorePrediction(
      comparison,
      prediction,
      shortRegionThresholdSeconds,
    );
  }
  const ablationMetrics: ScoredTrackComparison["ablationMetrics"] = {};
  for (const ablation of comparison.ablations) {
    const prediction: EnginePrediction = {
      engineId: ablation.id === "rule-only"
        ? "rule-only"
        : "observation-hybrid",
      regions: ablation.regions,
      usedLearned: ablation.id !== "rule-only",
      timing: {
        observationMilliseconds: 0,
        inferenceMilliseconds: 0,
        decodeMilliseconds: 0,
        endToEndMilliseconds: 0,
        peakHeapBytes: null,
      },
      diagnostics: ablation.diagnostics ? {
        averageLearnedEntropy: ablation.diagnostics.averageLearnedEntropy,
        ruleLearnedAgreementRate: ablation.diagnostics.ruleLearnedAgreementRate,
        ruleLearnedDisagreements: ablation.diagnostics.ruleLearnedDisagreements,
        changedTopCandidateWindows: ablation.diagnostics.changedTopCandidateWindows,
        learnedBoundaryPeaksConsidered:
          ablation.diagnostics.learnedBoundaryPeaksConsidered,
        effectiveLearnedWeightAverage:
          ablation.diagnostics.effectiveLearnedWeight.average,
        effectiveLearnedWeightMinimum:
          ablation.diagnostics.effectiveLearnedWeight.minimum,
        effectiveLearnedWeightMaximum:
          ablation.diagnostics.effectiveLearnedWeight.maximum,
        alignedLearnedWindows: ablation.diagnostics.alignedWindows,
        missingLearnedWindows: ablation.diagnostics.missingLearnedWindows,
      } : undefined,
    };
    ablationMetrics[ablation.id] = scorePrediction(
      comparison,
      prediction,
      shortRegionThresholdSeconds,
    );
  }
  return { comparison, engineMetrics, ablationMetrics };
}

function mulberry32(seed: number): () => number {
  let value = seed >>> 0;
  return () => {
    value += 0x6D2B79F5;
    let result = value;
    result = Math.imul(result ^ result >>> 15, result | 1);
    result ^= result + Math.imul(result ^ result >>> 7, result | 61);
    return ((result ^ result >>> 14) >>> 0) / 4294967296;
  };
}

function quantile(values: number[], probability: number): number {
  if (!values.length) return Number.NaN;
  const ordered = [...values].sort((left, right) => left - right);
  const index = (ordered.length - 1) * probability;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return ordered[lower];
  return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower);
}

export function bootstrapMeanConfidenceInterval(
  values: number[],
  seed: number,
  iterations: number,
): [number, number] | null {
  if (!values.length || iterations <= 0) return null;
  const random = mulberry32(seed);
  const samples = new Array<number>(iterations);
  for (let iteration = 0; iteration < iterations; iteration += 1) {
    let total = 0;
    for (let draw = 0; draw < values.length; draw += 1) {
      total += values[Math.floor(random() * values.length)];
    }
    samples[iteration] = total / values.length;
  }
  return [
    round(quantile(samples, 0.025)),
    round(quantile(samples, 0.975)),
  ];
}

function metricSeed(seed: number, metric: string): number {
  let hash = seed >>> 0;
  for (const character of metric) {
    hash = Math.imul(hash ^ character.charCodeAt(0), 16777619);
  }
  return hash >>> 0;
}

function summarizeValues(
  values: number[],
  seed: number,
  iterations: number,
): MetricSummary {
  return {
    mean: values.length ? round(mean(values)) : null,
    median: values.length ? round(median(values)!) : null,
    bootstrap95ConfidenceInterval: bootstrapMeanConfidenceInterval(
      values,
      seed,
      iterations,
    ),
    trackCount: values.length,
  };
}

function mergeTaxonomies(metrics: EngineTrackMetrics[]): ErrorTaxonomy {
  const merged = emptyTaxonomy();
  for (const metric of metrics) {
    for (const key of Object.keys(merged) as Array<keyof ErrorTaxonomy>) {
      merged[key].occurrences += metric.errorTaxonomy[key].occurrences;
      merged[key].durationSeconds += metric.errorTaxonomy[key].durationSeconds;
    }
  }
  for (const value of Object.values(merged)) {
    value.durationSeconds = round(value.durationSeconds);
  }
  return merged;
}

export function aggregateEngineMetrics(
  metrics: EngineTrackMetrics[],
  seed: number,
  bootstrapIterations: number,
): EngineAggregate {
  const totals = metrics.reduce<DurationTotals>((sum, metric) => ({
    evaluated: sum.evaluated + metric.durationTotals.evaluated,
    rootCorrect: sum.rootCorrect + metric.durationTotals.rootCorrect,
    majorMinorCorrect:
      sum.majorMinorCorrect + metric.durationTotals.majorMinorCorrect,
    detailedCorrect: sum.detailedCorrect + metric.durationTotals.detailedCorrect,
    referenceNoChord:
      sum.referenceNoChord + metric.durationTotals.referenceNoChord,
    predictedNoChord:
      sum.predictedNoChord + metric.durationTotals.predictedNoChord,
    truePositiveNoChord:
      sum.truePositiveNoChord + metric.durationTotals.truePositiveNoChord,
  }), {
    evaluated: 0,
    rootCorrect: 0,
    majorMinorCorrect: 0,
    detailedCorrect: 0,
    referenceNoChord: 0,
    predictedNoChord: 0,
    truePositiveNoChord: 0,
  });
  const boundaryErrors = metrics.flatMap((metric) => metric.boundaryErrorsMs);
  const finiteBoundaryErrors = boundaryErrors.filter(Number.isFinite);
  const absoluteBoundaryErrors = finiteBoundaryErrors.map(Math.abs);
  const noChordPrecision = safeDivide(
    totals.truePositiveNoChord,
    totals.predictedNoChord,
  );
  const noChordRecall = safeDivide(
    totals.truePositiveNoChord,
    totals.referenceNoChord,
  );
  const noChordF1 = noChordPrecision + noChordRecall
    ? 2 * noChordPrecision * noChordRecall / (noChordPrecision + noChordRecall)
    : 0;
  const totalPredictedRegions = metrics.reduce(
    (sum, metric) => sum + metric.predictedRegionCount,
    0,
  );
  const totalVeryShortRegions = metrics.reduce(
    (sum, metric) => sum + metric.veryShortRegionCount,
    0,
  );
  const totalOneWindowRegions = metrics.reduce(
    (sum, metric) => sum + metric.oneWindowRegionCount,
    0,
  );
  const trackLevel: EngineAggregate["trackLevel"] = {};
  for (const metricName of MAIN_METRICS) {
    const values = metrics.flatMap((metric) => {
      const value = metric[metricName];
      return typeof value === "number" && Number.isFinite(value) ? [value] : [];
    });
    trackLevel[metricName] = summarizeValues(
      values,
      metricSeed(seed, metricName),
      bootstrapIterations,
    );
  }
  return {
    trackCount: metrics.length,
    totalEvaluatedDurationSeconds: round(totals.evaluated),
    durationWeighted: {
      rootAccuracy: round(safeDivide(totals.rootCorrect, totals.evaluated)),
      majorMinorAccuracy: round(
        safeDivide(totals.majorMinorCorrect, totals.evaluated),
      ),
      detailedAccuracy: round(
        safeDivide(totals.detailedCorrect, totals.evaluated),
      ),
      weightedChordSymbolRecall: round(
        safeDivide(totals.detailedCorrect, totals.evaluated),
      ),
      noChordPrecision: round(noChordPrecision),
      noChordRecall: round(noChordRecall),
      noChordF1: round(noChordF1),
      fragmentationRate: round(
        safeDivide(
          metrics.reduce(
            (sum, metric) => sum
              + metric.fragmentationRate * metric.referenceRegionCount,
            0,
          ),
          metrics.reduce(
            (sum, metric) => sum + metric.referenceRegionCount,
            0,
          ),
        ),
      ),
      meanAbsoluteBoundaryErrorMs: absoluteBoundaryErrors.length
        ? round(mean(absoluteBoundaryErrors), 3)
        : null,
      medianAbsoluteBoundaryErrorMs: absoluteBoundaryErrors.length
        ? round(median(absoluteBoundaryErrors)!, 3)
        : null,
      meanSignedBoundaryErrorMs: finiteBoundaryErrors.length
        ? round(mean(finiteBoundaryErrors), 3)
        : null,
      boundariesWithin100ms: percentageWithin(boundaryErrors, 100),
      boundariesWithin250ms: percentageWithin(boundaryErrors, 250),
      boundariesWithin500ms: percentageWithin(boundaryErrors, 500),
      boundariesWithin1000ms: percentageWithin(boundaryErrors, 1000),
      predictedRegionCount: totalPredictedRegions,
      extraChordRegions: metrics.reduce(
        (sum, metric) => sum + metric.extraChordRegions,
        0,
      ),
      missingChordRegions: metrics.reduce(
        (sum, metric) => sum + metric.missingChordRegions,
        0,
      ),
      regionsPerMinute: round(
        safeDivide(totalPredictedRegions, totals.evaluated / 60),
      ),
      oneWindowRegionRate: round(
        safeDivide(totalOneWindowRegions, totalPredictedRegions),
      ),
      veryShortRegionRate: round(
        safeDivide(totalVeryShortRegions, totalPredictedRegions),
      ),
      flickerEventCount: metrics.reduce(
        (sum, metric) => sum + metric.flickerEventCount,
        0,
      ),
      averageChordRegionDurationSeconds: round(
        safeDivide(totals.evaluated, totalPredictedRegions),
      ),
      runtimePerAudioMinuteSeconds: round(
        safeDivide(
          metrics.reduce(
            (sum, metric) => sum
              + metric.runtimePerAudioMinuteSeconds
                * (metric.evaluatedDurationSeconds / 60),
            0,
          ),
          totals.evaluated / 60,
        ),
      ),
    },
    trackLevel,
    diagnostics: {
      mlOverrideCountVsRule: metrics.reduce(
        (sum, metric) => sum + (metric.mlOverrideCountVsRule ?? 0),
        0,
      ),
      percentRuleWindowsChangedByMl: (() => {
        const values = metrics.flatMap((metric) =>
          metric.percentRuleWindowsChangedByMl === null
            ? []
            : [metric.percentRuleWindowsChangedByMl]);
        return values.length ? round(mean(values)) : null;
      })(),
      ruleMlAgreementRate: (() => {
        const values = metrics.flatMap((metric) =>
          metric.ruleMlAgreementRate === null ? [] : [metric.ruleMlAgreementRate]);
        return values.length ? round(mean(values)) : null;
      })(),
      averageEffectiveLearnedWeight: (() => {
        const values = metrics.flatMap((metric) =>
          metric.averageEffectiveLearnedWeight === null
            ? []
            : [metric.averageEffectiveLearnedWeight]);
        return values.length ? round(mean(values)) : null;
      })(),
      averageLearnedEntropy: (() => {
        const values = metrics.flatMap((metric) =>
          metric.averageLearnedEntropy === null ? [] : [metric.averageLearnedEntropy]);
        return values.length ? round(mean(values)) : null;
      })(),
      peakMemoryMiB: (() => {
        const values = metrics.flatMap((metric) =>
          metric.peakMemoryMiB === null ? [] : [metric.peakMemoryMiB]);
        return values.length ? round(Math.max(...values), 3) : null;
      })(),
    },
    errorTaxonomy: mergeTaxonomies(metrics),
  };
}

function improvementDirection(metric: MainMetricName): 1 | -1 {
  return metric === "fragmentationRate"
    || metric === "meanAbsoluteBoundaryErrorMs"
    ? -1
    : 1;
}

export function pairedMetricDifference(
  tracks: ScoredTrackComparison[],
  comparator: "rule-only" | "ml-only" | "legacy-region-hybrid",
  metric: MainMetricName,
  seed: number,
  bootstrapIterations: number,
): PairedDifference {
  const pairs = tracks.flatMap((track) => {
    const hybrid = track.engineMetrics["observation-hybrid"]?.[metric];
    const baseline = track.engineMetrics[comparator]?.[metric];
    return typeof hybrid === "number" && Number.isFinite(hybrid)
      && typeof baseline === "number" && Number.isFinite(baseline)
      ? [{ hybrid, baseline }]
      : [];
  });
  const differences = pairs.map(({ hybrid, baseline }) => hybrid - baseline);
  const direction = improvementDirection(metric);
  let improvedTracks = 0;
  let tiedTracks = 0;
  let worsenedTracks = 0;
  for (const difference of differences) {
    if (Math.abs(difference) <= 1e-9) tiedTracks += 1;
    else if (difference * direction > 0) improvedTracks += 1;
    else worsenedTracks += 1;
  }
  return {
    metric,
    comparison: comparator === "rule-only"
      ? "hybrid-minus-rule"
      : comparator === "ml-only"
        ? "hybrid-minus-ml-only"
        : "hybrid-minus-legacy-region-hybrid",
    meanDifference: differences.length ? round(mean(differences)) : null,
    medianDifference: differences.length ? round(median(differences)!) : null,
    bootstrap95ConfidenceInterval: bootstrapMeanConfidenceInterval(
      differences,
      metricSeed(seed, `${comparator}:${metric}`),
      bootstrapIterations,
    ),
    improvedTracks,
    tiedTracks,
    worsenedTracks,
    trackCount: differences.length,
  };
}

export interface AggregateComparison {
  commonSuccessfulTrackIds: string[];
  allRequestedHybridTrackIds: string[];
  fallbackCounts: Record<string, number>;
  engines: Partial<Record<EngineId, EngineAggregate>>;
  allRequestedHybrid: EngineAggregate;
  bySourceType: Partial<Record<
    SourceType,
    Partial<Record<EngineId, EngineAggregate>>
  >>;
  pairedDifferences: PairedDifference[];
  ablations: Partial<Record<HybridAblationId, EngineAggregate>>;
}

export function aggregateComparison(
  tracks: ScoredTrackComparison[],
  seed = 20260725,
  bootstrapIterations = 2000,
): AggregateComparison {
  const common = tracks.filter((track) =>
    Boolean(track.engineMetrics["rule-only"])
      && Boolean(track.engineMetrics["ml-only"])
      && track.comparison.predictions["observation-hybrid"]?.usedLearned === true);
  const engineIds: EngineId[] = [
    "rule-only",
    "ml-only",
    "observation-hybrid",
    "legacy-region-hybrid",
  ];
  const engines: AggregateComparison["engines"] = {};
  for (const engineId of engineIds) {
    const values = common.flatMap((track) => {
      const metric = track.engineMetrics[engineId];
      return metric ? [metric] : [];
    });
    if (values.length) {
      engines[engineId] = aggregateEngineMetrics(
        values,
        metricSeed(seed, engineId),
        bootstrapIterations,
      );
    }
  }
  const allRequestedHybridMetrics = tracks.flatMap((track) => {
    const metric = track.engineMetrics["observation-hybrid"];
    return metric ? [metric] : [];
  });
  const fallbackCounts: Record<string, number> = {};
  for (const track of tracks) {
    const reason = track.comparison.predictions["observation-hybrid"]?.fallbackReason;
    if (reason) fallbackCounts[reason] = (fallbackCounts[reason] ?? 0) + 1;
  }
  const bySourceType: AggregateComparison["bySourceType"] = {};
  for (const sourceType of ["solo-guitar", "full-band"] as const) {
    const sourceTracks = common.filter(
      (track) => track.comparison.track.sourceType === sourceType,
    );
    if (!sourceTracks.length) continue;
    bySourceType[sourceType] = {};
    for (const engineId of engineIds) {
      const values = sourceTracks.flatMap((track) => {
        const metric = track.engineMetrics[engineId];
        return metric ? [metric] : [];
      });
      if (values.length) {
        bySourceType[sourceType]![engineId] = aggregateEngineMetrics(
          values,
          metricSeed(seed, `${sourceType}:${engineId}`),
          bootstrapIterations,
        );
      }
    }
  }
  const comparators = (
    ["rule-only", "ml-only", "legacy-region-hybrid"] as const
  ).filter((comparator) => comparator !== "legacy-region-hybrid"
    || common.some((track) => Boolean(track.engineMetrics[comparator])));
  const pairedDifferences = comparators.flatMap((comparator) => MAIN_METRICS.map((metric) =>
    pairedMetricDifference(
      common,
      comparator,
      metric,
      seed,
      bootstrapIterations,
    )));
  const ablations: AggregateComparison["ablations"] = {};
  for (const id of [
    "rule-only",
    "hybrid-chord-probabilities-only",
    "hybrid-without-boundary",
    "hybrid-without-adaptive-weighting",
    "full-hybrid",
  ] as const) {
    const values = common.flatMap((track) => {
      const metric = track.ablationMetrics[id];
      return metric ? [metric] : [];
    });
    if (values.length) {
      ablations[id] = aggregateEngineMetrics(
        values,
        metricSeed(seed, `ablation:${id}`),
        bootstrapIterations,
      );
    }
  }
  return {
    commonSuccessfulTrackIds: common.map(
      (track) => track.comparison.track.trackId,
    ),
    allRequestedHybridTrackIds: tracks.map(
      (track) => track.comparison.track.trackId,
    ),
    fallbackCounts,
    engines,
    allRequestedHybrid: aggregateEngineMetrics(
      allRequestedHybridMetrics,
      metricSeed(seed, "all-requested-hybrid"),
      bootstrapIterations,
    ),
    bySourceType,
    pairedDifferences,
    ablations,
  };
}

export function publicTrackMetrics(
  metrics: EngineTrackMetrics,
): Omit<EngineTrackMetrics, "boundaryErrorsMs" | "durationTotals"> {
  const { boundaryErrorsMs: _errors, durationTotals: _totals, ...publicMetrics } = metrics;
  return publicMetrics;
}
