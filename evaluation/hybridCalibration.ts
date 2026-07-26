import { createHash } from "node:crypto";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
} from "../src/learnedHarmony/hybridDecoder";
import type {
  HybridHarmonySettings,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";

export const SEALED_FINAL_VALIDATION_PERFORMER = "guitarset-p00";

export interface CalibrationPerformer {
  performerId: string;
  learnedModelRole: "training" | "development";
}

export interface CalibrationFold {
  foldId: string;
  validationPerformers: string[];
  calibrationTrainPerformers: string[];
}

export interface HybridCalibrationSplitManifest {
  schemaVersion: 1;
  seed: number;
  strategy: "leave-one-performer-out";
  captures: string[];
  finalValidationPerformer: string;
  calibrationPerformers: CalibrationPerformer[];
  folds: CalibrationFold[];
  notes: string[];
}

export interface GroupedCaptureTrack {
  trackId: string;
  performerId: string;
  captureType: string;
}

export type ProbabilityCalibration =
  | { kind: "none"; value: 1 }
  | { kind: "temperature"; value: number }
  | { kind: "confidence-power"; value: number };

export interface CalibrationCandidate {
  id: string;
  settings: HybridHarmonySettings;
  adaptiveWeighting: boolean;
  probabilityCalibration: "none" | "temperature";
  description: string;
  complexity: number;
}

export interface CandidateCaptureMetrics {
  detailedAccuracy: number;
  rootAccuracy: number;
  fragmentationRate: number;
  regionsPerMinute: number;
  meanAbsoluteBoundaryErrorMs: number | null;
  incorrectOverrides: number;
  fallbacks: number;
  regionInvariantsValid: boolean;
}

export interface CandidateSelectionRow {
  candidate: CalibrationCandidate;
  captures: Record<string, CandidateCaptureMetrics>;
  meanDetailedAccuracy: number;
  meanRootAccuracy: number;
  worstFoldDetailedAccuracy: number;
}

export interface SelectionConstraints {
  maximumFragmentationAboveRule: number;
  maximumRegionsPerMinuteAboveRule: number;
  maximumBoundaryMaeAboveRuleMs: number;
}

export interface ReliabilityPoint {
  confidence: number;
  correct: boolean;
}

export interface ReliabilityBin {
  lower: number;
  upper: number;
  count: number;
  averageConfidence: number;
  accuracy: number;
}

export interface ReliabilityReport {
  sampleCount: number;
  accuracy: number;
  averageConfidence: number;
  expectedCalibrationError: number;
  bins: ReliabilityBin[];
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableValue);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, entry]) => [key, stableValue(entry)]));
  }
  return value;
}

export function settingsIdentity(
  settings: HybridHarmonySettings,
  probabilityCalibration: ProbabilityCalibration = { kind: "none", value: 1 },
): string {
  return createHash("sha256")
    .update(JSON.stringify(stableValue({ settings, probabilityCalibration })))
    .digest("hex");
}

export function validateCalibrationSplit(
  manifest: HybridCalibrationSplitManifest,
): void {
  if (manifest.schemaVersion !== 1
    || manifest.strategy !== "leave-one-performer-out") {
    throw new Error("Unsupported hybrid calibration split manifest");
  }
  if (manifest.finalValidationPerformer !== SEALED_FINAL_VALIDATION_PERFORMER) {
    throw new Error("The frozen final validation performer must remain guitarset-p00");
  }
  const performers = manifest.calibrationPerformers.map(({ performerId }) => performerId);
  if (!performers.length || new Set(performers).size !== performers.length) {
    throw new Error("Calibration performers must be non-empty and unique");
  }
  if (performers.includes(SEALED_FINAL_VALIDATION_PERFORMER)) {
    throw new Error("p00 cannot enter calibration search");
  }
  if (manifest.folds.length !== performers.length) {
    throw new Error("Leave-one-performer-out requires one fold per performer");
  }
  const validationCounts = new Map<string, number>();
  for (const fold of manifest.folds) {
    if (fold.validationPerformers.includes(SEALED_FINAL_VALIDATION_PERFORMER)
      || fold.calibrationTrainPerformers.includes(SEALED_FINAL_VALIDATION_PERFORMER)) {
      throw new Error("p00 cannot enter a calibration fold");
    }
    const validation = new Set(fold.validationPerformers);
    const training = new Set(fold.calibrationTrainPerformers);
    if (validation.size !== 1
      || [...validation].some((performer) => training.has(performer))) {
      throw new Error(`${fold.foldId}: invalid performer separation`);
    }
    const union = new Set([...validation, ...training]);
    if (union.size !== performers.length
      || performers.some((performer) => !union.has(performer))) {
      throw new Error(`${fold.foldId}: fold does not cover every calibration performer`);
    }
    for (const performer of validation) {
      validationCounts.set(performer, (validationCounts.get(performer) ?? 0) + 1);
    }
  }
  if (performers.some((performer) => validationCounts.get(performer) !== 1)) {
    throw new Error("Every calibration performer must be held out exactly once");
  }
}

export function assignGroupedFolds(
  tracks: GroupedCaptureTrack[],
  manifest: HybridCalibrationSplitManifest,
): Record<string, string> {
  validateCalibrationSplit(manifest);
  const foldForPerformer = new Map(manifest.folds.flatMap((fold) =>
    fold.validationPerformers.map((performer) => [performer, fold.foldId] as const)));
  const assignments: Record<string, string> = {};
  const performanceFolds = new Map<string, string>();
  for (const track of [...tracks].sort((left, right) =>
    `${left.trackId}:${left.captureType}`.localeCompare(`${right.trackId}:${right.captureType}`))) {
    if (track.performerId === SEALED_FINAL_VALIDATION_PERFORMER) {
      throw new Error("p00 cannot enter calibration search");
    }
    const fold = foldForPerformer.get(track.performerId);
    if (!fold) throw new Error(`${track.trackId}: performer is absent from calibration folds`);
    const key = `${track.trackId}:${track.captureType}`;
    assignments[key] = fold;
    const previous = performanceFolds.get(track.trackId);
    if (previous && previous !== fold) {
      throw new Error(`${track.trackId}: alternate captures crossed folds`);
    }
    performanceFolds.set(track.trackId, fold);
  }
  return assignments;
}

function normalizePower(probabilities: number[], power: number): number[] {
  if (!probabilities.length) return [];
  const adjusted = probabilities.map((value) =>
    Math.pow(Math.max(1e-9, Math.min(1, value)), power));
  const total = adjusted.reduce((sum, value) => sum + value, 0) || 1;
  return adjusted.map((value) => value / total);
}

function binaryPower(probability: number, power: number): number {
  const p = Math.max(1e-9, Math.min(1 - 1e-9, probability));
  const yes = Math.pow(p, power);
  const no = Math.pow(1 - p, power);
  return yes / (yes + no);
}

export function calibrateLearnedResponse(
  response: LearnedHarmonyResponse,
  calibration: ProbabilityCalibration,
): LearnedHarmonyResponse {
  if (calibration.kind === "none" || calibration.value === 1) return response;
  if (!Number.isFinite(calibration.value) || calibration.value <= 0) {
    throw new Error("Probability calibration value must be positive");
  }
  const power = calibration.kind === "temperature"
    ? 1 / calibration.value
    : calibration.value;
  return {
    ...response,
    rootProbabilities: response.rootProbabilities.map((row) =>
      normalizePower(row, power)),
    qualityProbabilities: response.qualityProbabilities.map((row) =>
      normalizePower(row, power)),
    noChordProbabilities: response.noChordProbabilities.map((value) =>
      binaryPower(value, power)),
    // The boundary head is a transition event probability, not a chord-class
    // confidence, so chord probability calibration deliberately leaves it intact.
    boundaryProbabilities: response.boundaryProbabilities.slice(),
    diagnostics: {
      ...response.diagnostics,
      warnings: [
        ...response.diagnostics.warnings,
        `Evaluation probability calibration: ${calibration.kind}=${calibration.value}.`,
      ],
    },
  };
}

export function reliabilityReport(
  points: ReliabilityPoint[],
  binCount = 10,
): ReliabilityReport {
  if (!Number.isInteger(binCount) || binCount <= 0) {
    throw new Error("Reliability bin count must be positive");
  }
  const bins: ReliabilityBin[] = [];
  let weightedError = 0;
  for (let index = 0; index < binCount; index += 1) {
    const lower = index / binCount;
    const upper = (index + 1) / binCount;
    const selected = points.filter(({ confidence }) =>
      confidence >= lower
      && (index === binCount - 1 ? confidence <= upper : confidence < upper));
    const averageConfidence = selected.length
      ? selected.reduce((sum, point) => sum + point.confidence, 0) / selected.length
      : 0;
    const accuracy = selected.length
      ? selected.filter(({ correct }) => correct).length / selected.length
      : 0;
    weightedError += selected.length * Math.abs(accuracy - averageConfidence);
    bins.push({
      lower,
      upper,
      count: selected.length,
      averageConfidence,
      accuracy,
    });
  }
  return {
    sampleCount: points.length,
    accuracy: points.length
      ? points.filter(({ correct }) => correct).length / points.length
      : 0,
    averageConfidence: points.length
      ? points.reduce((sum, point) => sum + point.confidence, 0) / points.length
      : 0,
    expectedCalibrationError: points.length ? weightedError / points.length : 0,
    bins,
  };
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

function round(value: number, digits = 3): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

export function generateCalibrationCandidates(
  seed: number,
  randomCandidateCount: number,
): CalibrationCandidate[] {
  const base = CONSERVATIVE_HYBRID_SETTINGS;
  const candidates: CalibrationCandidate[] = [
    {
      id: "current-adaptive",
      settings: { ...base },
      adaptiveWeighting: true,
      probabilityCalibration: "none",
      description: "Current conservative adaptive configuration.",
      complexity: 0,
    },
    {
      id: "fixed-current-weight",
      settings: { ...base },
      adaptiveWeighting: false,
      probabilityCalibration: "none",
      description: "Current learned weight without adaptive gates.",
      complexity: 1,
    },
    {
      id: "adaptive-without-rule-protection",
      settings: { ...base, protectRuleConfidenceAbove: 0.89 },
      adaptiveWeighting: true,
      probabilityCalibration: "none",
      description: "Adaptive weighting with high-confidence protection effectively relaxed.",
      complexity: 1,
    },
    {
      id: "adaptive-without-entropy-suppression",
      settings: { ...base, maximumLearnedEntropy: 1 },
      adaptiveWeighting: true,
      probabilityCalibration: "none",
      description: "Adaptive weighting without the entropy hard gate.",
      complexity: 1,
    },
    {
      id: "adaptive-temperature-calibrated",
      settings: { ...base },
      adaptiveWeighting: true,
      probabilityCalibration: "temperature",
      description: "Current adaptive weighting with fold-fitted temperature scaling.",
      complexity: 1,
    },
    {
      id: "chord-evidence-only",
      settings: { ...base, learnedBoundaryWeight: 0, bassRootWeight: 0 },
      adaptiveWeighting: true,
      probabilityCalibration: "none",
      description: "Learned chord evidence only; no learned boundary or fusion bass bonus.",
      complexity: 1,
    },
  ];
  const random = mulberry32(seed);
  for (let index = 0; index < randomCandidateCount; index += 1) {
    const learnedChordWeight = round(0.35 + random() * 0.5, 3);
    const maximumLearnedWeightGuitarOnly = round(
      Math.max(learnedChordWeight, 0.5 + random() * 0.4),
      3,
    );
    const boundaryOptions = [0, 0.1, 0.25];
    const bassOptions = [0, 0.15, 0.25];
    candidates.push({
      id: `random-${String(index + 1).padStart(2, "0")}`,
      settings: {
        ...base,
        learnedChordWeight,
        learnedBoundaryWeight:
          boundaryOptions[Math.floor(random() * boundaryOptions.length)],
        bassRootWeight: bassOptions[Math.floor(random() * bassOptions.length)],
        minimumLearnedConfidence: round(0.35 + random() * 0.27, 3),
        maximumLearnedEntropy: round(0.7 + random() * 0.29, 3),
        protectRuleConfidenceAbove: round(0.68 + random() * 0.2, 3),
        maximumLearnedWeightGuitarOnly,
      },
      adaptiveWeighting: true,
      probabilityCalibration: random() < 0.25 ? "temperature" : "none",
      description: "Deterministic bounded random-search candidate.",
      complexity: 2,
    });
  }
  return candidates;
}

export function candidateSatisfiesConstraints(
  row: CandidateSelectionRow,
  ruleByCapture: Record<string, CandidateCaptureMetrics>,
  constraints: SelectionConstraints,
): boolean {
  return Object.entries(row.captures).every(([capture, metrics]) => {
    const rule = ruleByCapture[capture];
    if (!rule || metrics.fallbacks || !metrics.regionInvariantsValid) return false;
    const boundaryAllowed = rule.meanAbsoluteBoundaryErrorMs === null
      || metrics.meanAbsoluteBoundaryErrorMs === null
      || metrics.meanAbsoluteBoundaryErrorMs
        <= rule.meanAbsoluteBoundaryErrorMs + constraints.maximumBoundaryMaeAboveRuleMs;
    return metrics.fragmentationRate
      <= rule.fragmentationRate + constraints.maximumFragmentationAboveRule
      && metrics.regionsPerMinute
        <= rule.regionsPerMinute + constraints.maximumRegionsPerMinuteAboveRule
      && boundaryAllowed;
  });
}

export function paretoFrontier(rows: CandidateSelectionRow[]): CandidateSelectionRow[] {
  return rows.filter((candidate) => !rows.some((other) => {
    if (other === candidate) return false;
    const otherFragmentation = Object.values(other.captures)
      .reduce((sum, value) => sum + value.fragmentationRate, 0)
      / Object.values(other.captures).length;
    const candidateFragmentation = Object.values(candidate.captures)
      .reduce((sum, value) => sum + value.fragmentationRate, 0)
      / Object.values(candidate.captures).length;
    const noWorse = other.meanDetailedAccuracy >= candidate.meanDetailedAccuracy
      && other.meanRootAccuracy >= candidate.meanRootAccuracy
      && otherFragmentation <= candidateFragmentation;
    const strictlyBetter = other.meanDetailedAccuracy > candidate.meanDetailedAccuracy
      || other.meanRootAccuracy > candidate.meanRootAccuracy
      || otherFragmentation < candidateFragmentation;
    return noWorse && strictlyBetter;
  })).sort((left, right) =>
    right.meanDetailedAccuracy - left.meanDetailedAccuracy
    || right.meanRootAccuracy - left.meanRootAccuracy
    || left.candidate.complexity - right.candidate.complexity);
}

export function selectCalibrationCandidate(
  rows: CandidateSelectionRow[],
  ruleByCapture: Record<string, CandidateCaptureMetrics>,
  constraints: SelectionConstraints,
): CandidateSelectionRow | null {
  const eligible = rows.filter((row) =>
    candidateSatisfiesConstraints(row, ruleByCapture, constraints));
  if (!eligible.length) return null;
  return [...eligible].sort((left, right) =>
    right.meanDetailedAccuracy - left.meanDetailedAccuracy
    || right.meanRootAccuracy - left.meanRootAccuracy
    || right.worstFoldDetailedAccuracy - left.worstFoldDetailedAccuracy
    || left.candidate.complexity - right.candidate.complexity
    || left.candidate.id.localeCompare(right.candidate.id))[0];
}

export function assertFinalValidationIsFrozen(
  frozen: { status?: string; settingsIdentity?: string } | null,
): void {
  if (!frozen
    || frozen.status !== "frozen-before-p00-validation"
    || !/^[a-f0-9]{64}$/.test(frozen.settingsIdentity ?? "")) {
    throw new Error("p00 evaluation cannot execute before calibration settings are frozen");
  }
}

export function assertNonEmptyCalibrationTracks(trackCount: number): void {
  if (!Number.isInteger(trackCount) || trackCount <= 0) {
    throw new Error("Hybrid calibration loaded zero tracks");
  }
}
