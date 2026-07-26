import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import {
  mkdir,
  readFile,
  writeFile,
} from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import process from "node:process";
import {
  createHarmonyObservations,
  decodeHarmonyObservations,
  type HarmonyObservationPackage,
} from "../src/chordAnalysis";
import type { ChordAnalysisResult, ChordEvent } from "../src/types";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  alignLearnedEvidence,
  fuseHybridObservations,
  traceHybridWeighting,
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
import type {
  HybridFusionDiagnostics,
  HybridHarmonySettings,
  HybridWeightStageId,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";
import bundledWeights from "../src/learnedHarmony/model/app-chord-model.weights.json";
import {
  PRODUCTION_RULE_ENGINE,
  buildEvaluationRequest,
  decodeLearnedOnly,
  type EnginePrediction,
  type PreparedEvaluationTrack,
  type TrackEngineComparison,
} from "./hybridComparison";
import {
  aggregateEngineMetrics,
  bootstrapMeanConfidenceInterval,
  canonicalChordLabel,
  scoreTrackComparison,
  type EngineAggregate,
  type EngineTrackMetrics,
} from "./hybridMetrics";
import { assertNoAbsolutePaths } from "./hybridReport";
import {
  loadDataset,
  type DatasetConfig,
} from "./run-hybrid-comparison";
import {
  CachedLearnedHarmonyProvider,
  type LearnedInferenceCacheStats,
} from "./hybridInferenceCache";
import {
  SEALED_FINAL_VALIDATION_PERFORMER,
  assertNonEmptyCalibrationTracks,
  assignGroupedFolds,
  calibrateLearnedResponse,
  candidateSatisfiesConstraints,
  generateCalibrationCandidates,
  paretoFrontier,
  reliabilityReport,
  selectCalibrationCandidate,
  settingsIdentity,
  validateCalibrationSplit,
  type CalibrationCandidate,
  type CalibrationFold,
  type CandidateCaptureMetrics,
  type CandidateSelectionRow,
  type HybridCalibrationSplitManifest,
  type ProbabilityCalibration,
  type ReliabilityPoint,
  type SelectionConstraints,
} from "./hybridCalibration";

interface CalibrationConfig {
  schemaVersion: 1;
  sampleRate: number;
  splitManifest: string;
  dataset: {
    datasetIdentifier: string;
    officialDistributionRecord: string;
    datasetVersion: string;
    annotationArchiveMd5: string;
    audioArchiveMd5ByCapture: Record<string, string>;
    manifestPath: string;
    annotationDirectory: string;
    splitSeed: number;
    sourceType: "solo-guitar";
    sourceMode: "guitar-focused";
    expectedAnnotationCount: number;
    expectedTracksPerPerformer: number;
  };
  search: {
    seed: number;
    randomCandidateCount: number;
    temperatureCandidates: number[];
    shortRegionThresholdSeconds: number;
    constraints: SelectionConstraints;
  };
  outputDirectory: string;
}

interface CalibrationEnvironment extends NodeJS.ProcessEnv {
  TABSMITH_GUITARSET_MANIFEST?: string;
  TABSMITH_GUITARSET_ANNOTATIONS?: string;
  TABSMITH_GUITARSET_MIC_AUDIO?: string;
  TABSMITH_GUITARSET_PICKUP_AUDIO?: string;
  TABSMITH_HYBRID_CALIBRATION_CACHE?: string;
}

interface PreparedCalibrationEvidence {
  captureType: string;
  foldId: string;
  performerId: string;
  track: Omit<PreparedEvaluationTrack, "samples" | "audioHash">;
  observationPackage: HarmonyObservationPackage;
  ruleAnalysis: ChordAnalysisResult;
  response: LearnedHarmonyResponse;
  responseHash: string;
  mlOnlyRegions: ChordEvent[];
}

interface CandidateTrackResult {
  evidence: PreparedCalibrationEvidence;
  metrics: FastTrackMetrics;
  regions: ChordEvent[];
  analysis: ChordAnalysisResult;
  diagnostics: HybridFusionDiagnostics;
  correctOverrides: number;
  incorrectOverrides: number;
  blockedCorrectMl: number;
  protectedCorrectRule: number;
  regionInvariantsValid: boolean;
}

interface FastTrackMetrics {
  evaluatedDurationSeconds: number;
  rootCorrectSeconds: number;
  majorMinorCorrectSeconds: number;
  detailedCorrectSeconds: number;
  fragmentedReferenceRegions: number;
  referenceRegionCount: number;
  predictedRegionCount: number;
  absoluteBoundaryErrorSumMs: number;
  finiteBoundaryCount: number;
}

interface CandidateScreeningRow {
  candidateId: string;
  detailedAccuracy: number;
  rootAccuracy: number;
  observationChangeRate: number;
  score: number;
}

interface TemperatureFoldResult {
  foldId: string;
  validationPerformer: string;
  selectedTemperature: number;
  trainingNll: number;
  validation: {
    uncalibrated: ReturnType<typeof reliabilityReport>;
    temperature: ReturnType<typeof reliabilityReport>;
  };
}

interface GateAccumulator {
  affectedWindows: number;
  sumBefore: number;
  sumAfter: number;
  blockedCorrectMl: number;
  protectedCorrectRule: number;
}

const CAPTURE_ENVIRONMENT = {
  "audio_mono-mic": "TABSMITH_GUITARSET_MIC_AUDIO",
  "audio_mono-pickup_mix": "TABSMITH_GUITARSET_PICKUP_AUDIO",
} as const;

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

function currentCommit(): string {
  return execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: process.cwd(),
    encoding: "utf8",
  }).trim();
}

function responseProbabilityHash(response: LearnedHarmonyResponse): string {
  return sha256(JSON.stringify({
    modelVersion: response.modelVersion,
    modelChecksum: response.modelChecksum,
    featureVersion: response.featureVersion,
    frameTimes: response.frameTimes,
    rootProbabilities: response.rootProbabilities,
    qualityProbabilities: response.qualityProbabilities,
    noChordProbabilities: response.noChordProbabilities,
    boundaryProbabilities: response.boundaryProbabilities,
  }));
}

function splitNameForPerformer(
  performerId: string,
  manifest: HybridCalibrationSplitManifest,
): "training" | "development" {
  const role = manifest.calibrationPerformers.find(
    (performer) => performer.performerId === performerId,
  )?.learnedModelRole;
  if (!role) throw new Error(`${performerId}: missing learned-model role`);
  return role;
}

function sourceAudioRoot(
  captureType: keyof typeof CAPTURE_ENVIRONMENT,
  environment: CalibrationEnvironment,
): string {
  const variable = CAPTURE_ENVIRONMENT[captureType];
  const value = environment[variable]?.trim();
  if (!value) throw new Error(`${variable} is required for hybrid calibration`);
  return path.resolve(value);
}

function referenceAt(
  track: PreparedCalibrationEvidence["track"],
  time: number,
): string {
  const region = track.referenceRegions.find((candidate, index) =>
    time >= candidate.start
    && (time < candidate.end
      || (index === track.referenceRegions.length - 1 && time <= candidate.end)));
  return canonicalChordLabel(region?.label ?? "N");
}

function chordAt(regions: ChordEvent[], time: number): string {
  const region = regions.find((candidate, index) =>
    time >= candidate.start
    && (time < candidate.end
      || (index === regions.length - 1 && time <= candidate.end)));
  return canonicalChordLabel(region?.name ?? "N");
}

function parsedFamily(label: string): { root: string; family: string } {
  const canonical = canonicalChordLabel(label);
  if (canonical === "N") return { root: "N", family: "N" };
  const match = /^([A-G](?:#|b)?)(.*)$/.exec(canonical);
  const suffix = match?.[2] ?? "";
  return {
    root: match?.[1] ?? "N",
    family: suffix.startsWith("m") && !suffix.startsWith("maj")
      ? "min"
      : suffix === "dim" ? "dim" : "maj",
  };
}

function fastTrackMetrics(
  evidence: PreparedCalibrationEvidence,
  regions: ChordEvent[],
): FastTrackMetrics {
  const points = [...new Set([
    ...evidence.track.referenceRegions.flatMap((region) => [region.start, region.end]),
    ...regions.flatMap((region) => [region.start, region.end]),
  ])].sort((left, right) => left - right);
  let evaluatedDurationSeconds = 0;
  let rootCorrectSeconds = 0;
  let majorMinorCorrectSeconds = 0;
  let detailedCorrectSeconds = 0;
  for (let index = 0; index + 1 < points.length; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (end <= start) continue;
    const midpoint = (start + end) / 2;
    const reference = referenceAt(evidence.track, midpoint);
    if (!evidence.track.referenceRegions.some((region) =>
      midpoint >= region.start && midpoint <= region.end)) continue;
    const predicted = chordAt(regions, midpoint);
    const duration = end - start;
    const expectedParts = parsedFamily(reference);
    const predictedParts = parsedFamily(predicted);
    evaluatedDurationSeconds += duration;
    if (expectedParts.root === predictedParts.root) {
      rootCorrectSeconds += duration;
      if (expectedParts.family === predictedParts.family) {
        majorMinorCorrectSeconds += duration;
      }
    }
    if (reference === predicted) detailedCorrectSeconds += duration;
  }
  const fragmentedReferenceRegions = evidence.track.referenceRegions.filter(
    (expected) => regions.filter((actual) =>
      actual.start < expected.end && actual.end > expected.start).length > 1,
  ).length;
  const predictedBoundaries = regions.slice(1).map((region) => region.start);
  const boundaryErrors = evidence.track.referenceRegions.slice(1).flatMap((region) => {
    if (!predictedBoundaries.length) return [];
    const nearest = predictedBoundaries.reduce((best, candidate) =>
      Math.abs(candidate - region.start) < Math.abs(best - region.start)
        ? candidate
        : best);
    return [Math.abs(nearest - region.start) * 1000];
  });
  return {
    evaluatedDurationSeconds,
    rootCorrectSeconds,
    majorMinorCorrectSeconds,
    detailedCorrectSeconds,
    fragmentedReferenceRegions,
    referenceRegionCount: evidence.track.referenceRegions.length,
    predictedRegionCount: regions.length,
    absoluteBoundaryErrorSumMs:
      boundaryErrors.reduce((sum, value) => sum + value, 0),
    finiteBoundaryCount: boundaryErrors.length,
  };
}

function regionInvariants(regions: ChordEvent[], duration: number): boolean {
  if (!regions.length || duration <= 0) return false;
  if (Math.abs(regions[0].start) > 1e-6
    || Math.abs(regions.at(-1)!.end - duration) > 1e-6) return false;
  for (let index = 0; index < regions.length; index += 1) {
    const region = regions[index];
    if (!(region.end > region.start)
      || region.start < -1e-9
      || region.end > duration + 1e-6) return false;
    if (index && Math.abs(regions[index - 1].end - region.start) > 1e-6) {
      return false;
    }
  }
  return true;
}

function emptyTiming(): EnginePrediction["timing"] {
  return {
    observationMilliseconds: 0,
    inferenceMilliseconds: 0,
    decodeMilliseconds: 0,
    endToEndMilliseconds: 0,
    peakHeapBytes: null,
  };
}

function trackComparisonForCandidate(
  evidence: PreparedCalibrationEvidence,
  regions: ChordEvent[],
  analysis: ChordAnalysisResult,
  diagnostics: HybridFusionDiagnostics,
): TrackEngineComparison {
  return {
    track: evidence.track,
    observationWindowCount: evidence.observationPackage.observations.length,
    modelIdentity: {
      modelVersion: LEARNED_MODEL_VERSION,
      modelChecksum: LEARNED_MODEL_CHECKSUM,
      featureVersion: FEATURE_VERSION,
    },
    predictions: {
      "rule-only": {
        engineId: "rule-only",
        regions: evidence.ruleAnalysis.regions,
        chordAnalysis: evidence.ruleAnalysis,
        usedLearned: false,
        timing: emptyTiming(),
      },
      "observation-hybrid": {
        engineId: "observation-hybrid",
        regions,
        chordAnalysis: analysis,
        usedLearned: true,
        probabilitySourceId: evidence.responseHash,
        timing: emptyTiming(),
        diagnostics: {
          averageLearnedEntropy: diagnostics.averageLearnedEntropy,
          ruleLearnedAgreementRate: diagnostics.ruleLearnedAgreementRate,
          ruleLearnedDisagreements: diagnostics.ruleLearnedDisagreements,
          changedTopCandidateWindows: diagnostics.changedTopCandidateWindows,
          learnedBoundaryPeaksConsidered: diagnostics.learnedBoundaryPeaksConsidered,
          effectiveLearnedWeightAverage: diagnostics.effectiveLearnedWeight.average,
          effectiveLearnedWeightMinimum: diagnostics.effectiveLearnedWeight.minimum,
          effectiveLearnedWeightMaximum: diagnostics.effectiveLearnedWeight.maximum,
          alignedLearnedWindows: diagnostics.alignedWindows,
          missingLearnedWindows: diagnostics.missingLearnedWindows,
        },
      },
    },
    ablations: [],
    disagreementWindows: [],
    execution: {
      actualHybridDecoderInvoked: true,
      actualTemporalDecoderCompleted: true,
      sharedLearnedResponseObject: true,
      hybridImplementation:
        "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony",
      productionDecoder:
        "src/chordAnalysis.ts#decodeHarmonyObservations",
      productionObservationBuilder:
        "src/chordAnalysis.ts#createHarmonyObservations",
    },
  };
}

function overrideDiagnostics(
  evidence: PreparedCalibrationEvidence,
  response: LearnedHarmonyResponse,
  hybridRegions: ChordEvent[],
): Pick<CandidateTrackResult,
  "correctOverrides" | "incorrectOverrides" | "blockedCorrectMl" | "protectedCorrectRule"> {
  const aligned = alignLearnedEvidence(
    evidence.observationPackage.observations,
    response,
  );
  let correctOverrides = 0;
  let incorrectOverrides = 0;
  let blockedCorrectMl = 0;
  let protectedCorrectRule = 0;
  evidence.observationPackage.observations.forEach((observation, index) => {
    const midpoint = (observation.start + observation.end) / 2;
    const reference = referenceAt(evidence.track, midpoint);
    const rule = chordAt(evidence.ruleAnalysis.regions, midpoint);
    const hybrid = chordAt(hybridRegions, midpoint);
    const learned = canonicalChordLabel(aligned[index]?.topChord ?? "N");
    if (hybrid !== rule) {
      if (hybrid === reference && rule !== reference) correctOverrides += 1;
      if (rule === reference && hybrid !== reference) incorrectOverrides += 1;
    }
    if (learned === reference && hybrid !== reference) blockedCorrectMl += 1;
    if (rule === reference && learned !== reference && hybrid === rule) {
      protectedCorrectRule += 1;
    }
  });
  return {
    correctOverrides,
    incorrectOverrides,
    blockedCorrectMl,
    protectedCorrectRule,
  };
}

function evaluateCandidateTrack(
  evidence: PreparedCalibrationEvidence,
  candidate: CalibrationCandidate,
  calibration: ProbabilityCalibration,
  shortRegionThresholdSeconds: number,
): CandidateTrackResult {
  const response = calibrateLearnedResponse(evidence.response, calibration);
  const fused = fuseHybridObservations(
    evidence.observationPackage.observations,
    response,
    candidate.settings,
    {
      sourceMode: evidence.track.sourceMode,
      adaptiveWeighting: candidate.adaptiveWeighting,
    },
  );
  const analysis = decodeHarmonyObservations({
    ...evidence.observationPackage,
    observations: fused.observations,
  });
  return {
    evidence,
    metrics: fastTrackMetrics(evidence, analysis.regions),
    regions: analysis.regions,
    analysis,
    diagnostics: fused.diagnostics,
    ...overrideDiagnostics(evidence, response, analysis.regions),
    regionInvariantsValid: regionInvariants(
      analysis.regions,
      evidence.observationPackage.duration,
    ),
  };
}

function screenCandidate(
  evidence: PreparedCalibrationEvidence[],
  candidate: CalibrationCandidate,
  foldTemperatures: Map<string, number>,
): CandidateScreeningRow {
  let duration = 0;
  let detailedCorrect = 0;
  let rootCorrect = 0;
  let changes = 0;
  let windows = 0;
  for (const item of evidence) {
    const response = calibrateLearnedResponse(
      item.response,
      candidate.probabilityCalibration === "temperature"
        ? {
          kind: "temperature",
          value: foldTemperatures.get(item.performerId) ?? 1,
        }
        : { kind: "none", value: 1 },
    );
    const fused = fuseHybridObservations(
      item.observationPackage.observations,
      response,
      candidate.settings,
      {
        sourceMode: item.track.sourceMode,
        adaptiveWeighting: candidate.adaptiveWeighting,
      },
    );
    let previous: string | null = null;
    for (const observation of fused.observations) {
      const midpoint = (observation.start + observation.end) / 2;
      const reference = referenceAt(item.track, midpoint);
      const predicted = canonicalChordLabel(observation.bestChord);
      const windowDuration = observation.end - observation.start;
      duration += windowDuration;
      if (reference === predicted) detailedCorrect += windowDuration;
      if (parsedFamily(reference).root === parsedFamily(predicted).root) {
        rootCorrect += windowDuration;
      }
      if (previous !== null && previous !== predicted) changes += 1;
      previous = predicted;
      windows += 1;
    }
  }
  const detailedAccuracy = duration ? detailedCorrect / duration : 0;
  const rootAccuracy = duration ? rootCorrect / duration : 0;
  const observationChangeRate = windows ? changes / windows : 0;
  return {
    candidateId: candidate.id,
    detailedAccuracy,
    rootAccuracy,
    observationChangeRate,
    // Screening is intentionally only a shortlist heuristic. Frozen selection
    // uses the real decoder metrics and hard constraints below.
    score: detailedAccuracy + rootAccuracy * 0.25 - observationChangeRate * 0.02,
  };
}

function shortlistCandidates(
  candidates: CalibrationCandidate[],
  screening: CandidateScreeningRow[],
  leadingCount = 4,
): CalibrationCandidate[] {
  const mandatory = new Set([
    "current-adaptive",
    "fixed-current-weight",
    "adaptive-without-rule-protection",
    "adaptive-without-entropy-suppression",
    "adaptive-temperature-calibrated",
    "chord-evidence-only",
  ]);
  const ranked = [...screening].sort((left, right) =>
    right.score - left.score || left.candidateId.localeCompare(right.candidateId));
  for (const row of ranked) {
    if (mandatory.has(row.candidateId)) continue;
    mandatory.add(row.candidateId);
    if ([...mandatory].filter((id) => id.startsWith("random-")).length
      >= leadingCount) break;
  }
  return candidates.filter((candidate) => mandatory.has(candidate.id));
}

function probabilityPoints(
  evidence: PreparedCalibrationEvidence[],
  calibration: ProbabilityCalibration,
): ReliabilityPoint[] {
  return evidence.flatMap((item) => {
    const response = calibrateLearnedResponse(item.response, calibration);
    const aligned = alignLearnedEvidence(
      item.observationPackage.observations,
      response,
    );
    return item.observationPackage.observations.flatMap((observation, index) => {
      const learned = aligned[index];
      if (!learned?.topChord || !learned.contributingFrameCount) return [];
      const reference = referenceAt(item.track, (observation.start + observation.end) / 2);
      return [{
        confidence: learned.topChordConfidence,
        correct: canonicalChordLabel(learned.topChord) === reference,
      }];
    });
  });
}

function probabilityNll(
  evidence: PreparedCalibrationEvidence[],
  calibration: ProbabilityCalibration,
): number {
  let total = 0;
  let count = 0;
  for (const item of evidence) {
    const response = calibrateLearnedResponse(item.response, calibration);
    const aligned = alignLearnedEvidence(
      item.observationPackage.observations,
      response,
    );
    item.observationPackage.observations.forEach((observation, index) => {
      const learned = aligned[index];
      if (!learned?.contributingFrameCount) return;
      const reference = referenceAt(item.track, (observation.start + observation.end) / 2);
      const probability = reference === "N"
        ? learned.noChordProbability
        : learned.chordProbabilities[reference] ?? 1e-9;
      total -= Math.log(Math.max(1e-9, probability));
      count += 1;
    });
  }
  return count ? total / count : Number.POSITIVE_INFINITY;
}

function fitTemperature(
  evidence: PreparedCalibrationEvidence[],
  candidates: number[],
): { temperature: number; nll: number } {
  if (!evidence.length) throw new Error("Temperature fitting received zero tracks");
  return candidates.map((temperature) => ({
    temperature,
    nll: probabilityNll(evidence, { kind: "temperature", value: temperature }),
  })).sort((left, right) =>
    left.nll - right.nll || Math.abs(left.temperature - 1) - Math.abs(right.temperature - 1))[0];
}

function fitFoldTemperatures(
  evidence: PreparedCalibrationEvidence[],
  folds: CalibrationFold[],
  temperatureCandidates: number[],
): { folds: TemperatureFoldResult[]; byPerformer: Map<string, number> } {
  const byPerformer = new Map<string, number>();
  const results = folds.map((fold) => {
    const training = evidence.filter((item) =>
      fold.calibrationTrainPerformers.includes(item.performerId));
    const validation = evidence.filter((item) =>
      fold.validationPerformers.includes(item.performerId));
    const fitted = fitTemperature(training, temperatureCandidates);
    const validationPerformer = fold.validationPerformers[0];
    byPerformer.set(validationPerformer, fitted.temperature);
    return {
      foldId: fold.foldId,
      validationPerformer,
      selectedTemperature: fitted.temperature,
      trainingNll: fitted.nll,
      validation: {
        uncalibrated: reliabilityReport(
          probabilityPoints(validation, { kind: "none", value: 1 }),
        ),
        temperature: reliabilityReport(
          probabilityPoints(validation, {
            kind: "temperature",
            value: fitted.temperature,
          }),
        ),
      },
    };
  });
  return { folds: results, byPerformer };
}

function captureMetrics(
  aggregate: EngineAggregate,
  incorrectOverrides: number,
  invariantsValid: boolean,
): CandidateCaptureMetrics {
  return {
    detailedAccuracy: aggregate.durationWeighted.detailedAccuracy,
    rootAccuracy: aggregate.durationWeighted.rootAccuracy,
    fragmentationRate: aggregate.durationWeighted.fragmentationRate,
    regionsPerMinute: aggregate.durationWeighted.regionsPerMinute,
    meanAbsoluteBoundaryErrorMs:
      aggregate.durationWeighted.meanAbsoluteBoundaryErrorMs,
    incorrectOverrides,
    fallbacks: 0,
    regionInvariantsValid: invariantsValid,
  };
}

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function aggregateCandidate(
  candidate: CalibrationCandidate,
  results: CandidateTrackResult[],
  seed: number,
): {
  selection: CandidateSelectionRow;
  aggregates: Record<string, CandidateCaptureMetrics>;
  folds: Array<{
    foldId: string;
    captureType: string;
    performerId: string;
    aggregate: CandidateCaptureMetrics;
  }>;
  overrides: {
    correct: number;
    incorrect: number;
    blockedCorrectMl: number;
    protectedCorrectRule: number;
  };
} {
  const captures = [...new Set(results.map(({ evidence }) => evidence.captureType))];
  const aggregateFast = (selected: CandidateTrackResult[]): CandidateCaptureMetrics => {
    const duration = selected.reduce(
      (sum, result) => sum + result.metrics.evaluatedDurationSeconds,
      0,
    );
    const referenceRegions = selected.reduce(
      (sum, result) => sum + result.metrics.referenceRegionCount,
      0,
    );
    const predictedRegions = selected.reduce(
      (sum, result) => sum + result.metrics.predictedRegionCount,
      0,
    );
    const boundaryCount = selected.reduce(
      (sum, result) => sum + result.metrics.finiteBoundaryCount,
      0,
    );
    return {
      detailedAccuracy: duration ? selected.reduce(
        (sum, result) => sum + result.metrics.detailedCorrectSeconds,
        0,
      ) / duration : 0,
      rootAccuracy: duration ? selected.reduce(
        (sum, result) => sum + result.metrics.rootCorrectSeconds,
        0,
      ) / duration : 0,
      fragmentationRate: referenceRegions ? selected.reduce(
        (sum, result) => sum + result.metrics.fragmentedReferenceRegions,
        0,
      ) / referenceRegions : 0,
      regionsPerMinute: duration ? predictedRegions / (duration / 60) : 0,
      meanAbsoluteBoundaryErrorMs: boundaryCount ? selected.reduce(
        (sum, result) => sum + result.metrics.absoluteBoundaryErrorSumMs,
        0,
      ) / boundaryCount : null,
      incorrectOverrides:
        selected.reduce((sum, result) => sum + result.incorrectOverrides, 0),
      fallbacks: 0,
      regionInvariantsValid:
        selected.every(({ regionInvariantsValid }) => regionInvariantsValid),
    };
  };
  const aggregates = Object.fromEntries(captures.map((capture) => [
    capture,
    aggregateFast(results.filter(({ evidence }) =>
      evidence.captureType === capture)),
  ]));
  const folds = [...new Set(results.map(({ evidence }) => evidence.foldId))]
    .flatMap((foldId) => captures.map((captureType) => {
      const selected = results.filter(({ evidence }) =>
        evidence.foldId === foldId && evidence.captureType === captureType);
      return {
        foldId,
        captureType,
        performerId: selected[0]?.evidence.performerId ?? "unknown",
        aggregate: aggregateFast(selected),
      };
    }));
  const captureSelection = Object.fromEntries(captures.map((capture) => {
    const selected = results.filter(({ evidence }) => evidence.captureType === capture);
    return [
      capture,
      aggregates[capture],
    ];
  }));
  return {
    selection: {
      candidate,
      captures: captureSelection,
      meanDetailedAccuracy: mean(Object.values(captureSelection)
        .map((metrics) => metrics.detailedAccuracy)),
      meanRootAccuracy: mean(Object.values(captureSelection)
        .map((metrics) => metrics.rootAccuracy)),
      worstFoldDetailedAccuracy: Math.min(...folds.map(
        ({ aggregate }) => aggregate.detailedAccuracy,
      )),
    },
    aggregates,
    folds,
    overrides: {
      correct: results.reduce((sum, result) => sum + result.correctOverrides, 0),
      incorrect: results.reduce((sum, result) => sum + result.incorrectOverrides, 0),
      blockedCorrectMl:
        results.reduce((sum, result) => sum + result.blockedCorrectMl, 0),
      protectedCorrectRule:
        results.reduce((sum, result) => sum + result.protectedCorrectRule, 0),
    },
  };
}

function aggregateRuleMetrics(
  evidence: PreparedCalibrationEvidence[],
  shortRegionThresholdSeconds: number,
  seed: number,
): Record<string, CandidateCaptureMetrics> {
  const byCapture: Record<string, CandidateCaptureMetrics> = {};
  for (const capture of [...new Set(evidence.map((item) => item.captureType))]) {
    const selected = evidence.filter((item) => item.captureType === capture);
    const results = selected.map((item): CandidateTrackResult => ({
      evidence: item,
      metrics: fastTrackMetrics(item, item.ruleAnalysis.regions),
      regions: item.ruleAnalysis.regions,
      analysis: item.ruleAnalysis,
      diagnostics: {
        alignedWindows: 0,
        missingLearnedWindows: 0,
        averageLearnedEntropy: 0,
        ruleLearnedAgreementRate: 0,
        ruleLearnedDisagreements: 0,
        changedTopCandidateWindows: 0,
        learnedBoundaryPeaksConsidered: 0,
        effectiveLearnedWeight: { minimum: 0, maximum: 0, average: 0 },
      },
      correctOverrides: 0,
      incorrectOverrides: 0,
      blockedCorrectMl: 0,
      protectedCorrectRule: 0,
      regionInvariantsValid: true,
    }));
    byCapture[capture] = aggregateCandidate(
      {
        id: "rule-only",
        settings: CONSERVATIVE_HYBRID_SETTINGS,
        adaptiveWeighting: true,
        probabilityCalibration: "none",
        description: "rule",
        complexity: 0,
      },
      results,
      seed,
    ).selection.captures[capture];
  }
  return byCapture;
}

function baselineGateDiagnostics(
  evidence: PreparedCalibrationEvidence[],
  baselineResults: CandidateTrackResult[],
): {
  totalWindows: number;
  averageEffectiveLearnedWeight: number;
  minimumEffectiveLearnedWeight: number;
  maximumEffectiveLearnedWeight: number;
  learnedWeightDistribution: Record<string, { count: number; percentage: number }>;
  gates: Record<string, {
    affectedWindows: number;
    percentage: number;
    averageWeightBefore: number;
    averageWeightAfter: number;
    blockedCorrectMl: number;
    protectedCorrectRule: number;
  }>;
  nonWeightLimiters: Record<string, unknown>;
} {
  const accumulators = new Map<HybridWeightStageId, GateAccumulator>();
  const resultByTrack = new Map(baselineResults.map((result) => [
    `${result.evidence.captureType}:${result.evidence.track.trackId}`,
    result,
  ]));
  const weights: number[] = [];
  let bassConflictWindows = 0;
  let noChordProtectionWindows = 0;
  let disagreementWindows = 0;
  for (const item of evidence) {
    const result = resultByTrack.get(`${item.captureType}:${item.track.trackId}`);
    if (!result) throw new Error(`${item.track.trackId}: missing baseline result`);
    const traces = traceHybridWeighting(
      item.observationPackage.observations,
      item.response,
      CONSERVATIVE_HYBRID_SETTINGS,
      { sourceMode: item.track.sourceMode },
    );
    for (const trace of traces) {
      weights.push(trace.effectiveWeight);
      if (trace.bassConflict) bassConflictWindows += 1;
      if (trace.noChordProtectionActive) noChordProtectionWindows += 1;
      if (trace.learnedTopChord
        && canonicalChordLabel(trace.learnedTopChord)
          !== canonicalChordLabel(trace.ruleTopChord)) disagreementWindows += 1;
      const midpoint = (trace.startSeconds + trace.endSeconds) / 2;
      const reference = referenceAt(item.track, midpoint);
      const learned = canonicalChordLabel(trace.learnedTopChord ?? "N");
      const rule = chordAt(item.ruleAnalysis.regions, midpoint);
      const hybrid = chordAt(result.regions, midpoint);
      for (const stage of trace.stages) {
        if (!stage.limited) continue;
        const accumulator = accumulators.get(stage.id) ?? {
          affectedWindows: 0,
          sumBefore: 0,
          sumAfter: 0,
          blockedCorrectMl: 0,
          protectedCorrectRule: 0,
        };
        accumulator.affectedWindows += 1;
        accumulator.sumBefore += stage.before;
        accumulator.sumAfter += stage.after;
        if (learned === reference && hybrid !== reference) {
          accumulator.blockedCorrectMl += 1;
        }
        if (rule === reference && learned !== reference && hybrid === rule) {
          accumulator.protectedCorrectRule += 1;
        }
        accumulators.set(stage.id, accumulator);
      }
    }
  }
  const totalWindows = weights.length;
  const allStages: HybridWeightStageId[] = [
    "availability",
    "minimum-learned-confidence",
    "maximum-learned-entropy",
    "learned-confidence-scaling",
    "learned-entropy-scaling",
    "rule-score-margin",
    "rule-confidence-room",
    "high-rule-confidence-protection",
    "agreement-bonus",
    "learned-flicker-suppression",
    "source-specific-maximum",
  ];
  const buckets: Array<[string, (weight: number) => boolean]> = [
    ["zero", (weight) => weight === 0],
    ["below-0.05", (weight) => weight > 0 && weight < 0.05],
    ["0.05-to-0.15", (weight) => weight >= 0.05 && weight < 0.15],
    ["0.15-to-0.30", (weight) => weight >= 0.15 && weight < 0.30],
    ["above-0.30", (weight) => weight >= 0.30],
  ];
  return {
    totalWindows,
    averageEffectiveLearnedWeight: mean(weights),
    minimumEffectiveLearnedWeight: weights.length ? Math.min(...weights) : 0,
    maximumEffectiveLearnedWeight: weights.length ? Math.max(...weights) : 0,
    learnedWeightDistribution: Object.fromEntries(buckets.map(([label, predicate]) => {
      const count = weights.filter(predicate).length;
      return [label, { count, percentage: totalWindows ? count / totalWindows : 0 }];
    })),
    gates: Object.fromEntries(allStages.map((id) => {
      const value = accumulators.get(id);
      return [
        id,
        value ? {
          affectedWindows: value.affectedWindows,
          percentage: totalWindows ? value.affectedWindows / totalWindows : 0,
          averageWeightBefore: value.sumBefore / value.affectedWindows,
          averageWeightAfter: value.sumAfter / value.affectedWindows,
          blockedCorrectMl: value.blockedCorrectMl,
          protectedCorrectRule: value.protectedCorrectRule,
        } : {
          affectedWindows: 0,
          percentage: 0,
          averageWeightBefore: 0,
          averageWeightAfter: 0,
          blockedCorrectMl: 0,
          protectedCorrectRule: 0,
        },
      ];
    })),
    nonWeightLimiters: {
      disagreement: {
        affectedWindows: disagreementWindows,
        explicitPenaltyApplied: false,
        note: "The current fusion formula has no explicit disagreement weight penalty.",
      },
      bassConflict: {
        affectedWindows: bassConflictWindows,
        note: "Bass conflicts do not reduce learned weight; matching candidates receive a bass bonus.",
      },
      noChordHandling: {
        affectedWindows: noChordProtectionWindows,
        note: "Sustained bass or high rule confidence scales only the learned N candidate term by 0.2.",
      },
    },
  };
}

function fastBaselineByCapture(
  evidence: PreparedCalibrationEvidence[],
  regions: (item: PreparedCalibrationEvidence) => ChordEvent[],
): Record<string, CandidateCaptureMetrics> {
  const output: Record<string, CandidateCaptureMetrics> = {};
  for (const capture of [...new Set(evidence.map(({ captureType }) => captureType))]) {
    const selected = evidence.filter(({ captureType }) => captureType === capture);
    const metrics = selected.map((item) => fastTrackMetrics(item, regions(item)));
    const duration = metrics.reduce(
      (sum, metric) => sum + metric.evaluatedDurationSeconds,
      0,
    );
    const references = metrics.reduce(
      (sum, metric) => sum + metric.referenceRegionCount,
      0,
    );
    const predicted = metrics.reduce(
      (sum, metric) => sum + metric.predictedRegionCount,
      0,
    );
    const boundaries = metrics.reduce(
      (sum, metric) => sum + metric.finiteBoundaryCount,
      0,
    );
    output[capture] = {
      detailedAccuracy: duration ? metrics.reduce(
        (sum, metric) => sum + metric.detailedCorrectSeconds,
        0,
      ) / duration : 0,
      rootAccuracy: duration ? metrics.reduce(
        (sum, metric) => sum + metric.rootCorrectSeconds,
        0,
      ) / duration : 0,
      fragmentationRate: references ? metrics.reduce(
        (sum, metric) => sum + metric.fragmentedReferenceRegions,
        0,
      ) / references : 0,
      regionsPerMinute: duration ? predicted / (duration / 60) : 0,
      meanAbsoluteBoundaryErrorMs: boundaries ? metrics.reduce(
        (sum, metric) => sum + metric.absoluteBoundaryErrorSumMs,
        0,
      ) / boundaries : null,
      incorrectOverrides: 0,
      fallbacks: 0,
      regionInvariantsValid: selected.every((item) =>
        regionInvariants(regions(item), item.observationPackage.duration)),
    };
  }
  return output;
}

function fastAccuracy(metric: FastTrackMetrics, name: "root" | "detailed"): number {
  if (!metric.evaluatedDurationSeconds) return 0;
  return (name === "root"
    ? metric.rootCorrectSeconds
    : metric.detailedCorrectSeconds) / metric.evaluatedDurationSeconds;
}

function fastFragmentation(metric: FastTrackMetrics): number {
  return metric.referenceRegionCount
    ? metric.fragmentedReferenceRegions / metric.referenceRegionCount
    : 0;
}

function fastBoundary(metric: FastTrackMetrics): number | null {
  return metric.finiteBoundaryCount
    ? metric.absoluteBoundaryErrorSumMs / metric.finiteBoundaryCount
    : null;
}

function pairedFastDiagnostics(
  selected: CandidateTrackResult[],
  seed: number,
): Record<string, unknown> {
  const rows = selected.map((result) => ({
    captureType: result.evidence.captureType,
    hybrid: result.metrics,
    rule: fastTrackMetrics(result.evidence, result.evidence.ruleAnalysis.regions),
    mlOnly: fastTrackMetrics(result.evidence, result.evidence.mlOnlyRegions),
  }));
  const output: Record<string, unknown> = {};
  for (const capture of [...new Set(rows.map(({ captureType }) => captureType))]) {
    const captureRows = rows.filter(({ captureType }) => captureType === capture);
    output[capture] = Object.fromEntries(
      (["rule", "mlOnly"] as const).map((comparator) => {
        const comparisons = [
          {
            metric: "rootAccuracy",
            direction: 1,
            values: captureRows.map((row) =>
              fastAccuracy(row.hybrid, "root")
              - fastAccuracy(row[comparator], "root")),
          },
          {
            metric: "detailedAccuracy",
            direction: 1,
            values: captureRows.map((row) =>
              fastAccuracy(row.hybrid, "detailed")
              - fastAccuracy(row[comparator], "detailed")),
          },
          {
            metric: "fragmentationRate",
            direction: -1,
            values: captureRows.map((row) =>
              fastFragmentation(row.hybrid)
              - fastFragmentation(row[comparator])),
          },
          {
            metric: "meanAbsoluteBoundaryErrorMs",
            direction: -1,
            values: captureRows.flatMap((row) => {
              const hybrid = fastBoundary(row.hybrid);
              const compared = fastBoundary(row[comparator]);
              return hybrid === null || compared === null ? [] : [hybrid - compared];
            }),
          },
        ];
        return [
          `hybrid-minus-${comparator === "mlOnly" ? "ml-only" : comparator}`,
          comparisons.map(({ metric, direction, values }, metricIndex) => ({
            metric,
            meanDifference: mean(values),
            bootstrap95ConfidenceInterval:
              bootstrapMeanConfidenceInterval(values, seed + metricIndex, 1000),
            improvedTracks:
              values.filter((value) => value * direction > 1e-9).length,
            tiedTracks:
              values.filter((value) => Math.abs(value) <= 1e-9).length,
            worsenedTracks:
              values.filter((value) => value * direction < -1e-9).length,
            trackCount: values.length,
          })),
        ];
      }),
    );
  }
  return output;
}

function calibrationDiagnostics(
  evidence: PreparedCalibrationEvidence[],
  foldTemperatures: Map<string, number>,
): Record<string, unknown> {
  const sections: Record<string, ReliabilityPoint[]> = {
    all: [],
    microphone: [],
    pickupMix: [],
    agreements: [],
    disagreements: [],
  };
  const entropy = {
    correct: [] as number[],
    incorrect: [] as number[],
  };
  for (const item of evidence) {
    const aligned = alignLearnedEvidence(
      item.observationPackage.observations,
      item.response,
    );
    item.observationPackage.observations.forEach((observation, index) => {
      const learned = aligned[index];
      if (!learned?.topChord) return;
      const reference = referenceAt(item.track, (observation.start + observation.end) / 2);
      const correct = canonicalChordLabel(learned.topChord) === reference;
      const point = { confidence: learned.topChordConfidence, correct };
      sections.all.push(point);
      sections[item.captureType === "audio_mono-mic" ? "microphone" : "pickupMix"]
        .push(point);
      sections[canonicalChordLabel(learned.topChord)
        === canonicalChordLabel(observation.bestChord)
        ? "agreements"
        : "disagreements"].push(point);
      entropy[correct ? "correct" : "incorrect"].push(learned.entropy);
    });
  }
  const calibratedPoints = evidence.flatMap((item) =>
    probabilityPoints([item], {
      kind: "temperature",
      value: foldTemperatures.get(item.performerId) ?? 1,
    }));
  return {
    uncalibrated: Object.fromEntries(Object.entries(sections)
      .map(([name, points]) => [name, reliabilityReport(points)])),
    foldTemperatureCalibrated: reliabilityReport(calibratedPoints),
    entropy: {
      correctAverage: mean(entropy.correct),
      incorrectAverage: mean(entropy.incorrect),
    },
    confidencePower: {
      evaluated: true,
      note: "For normalized categorical probabilities, confidence power p is mathematically identical to temperature 1/p; results therefore share the temperature curve.",
    },
    logitScaling: {
      evaluated: false,
      reason: "The frozen response contract exposes probabilities, not recoverable pre-softmax logits.",
    },
  };
}

function markdownReport(report: Record<string, any>): string {
  const selected = report.selection.selected;
  const lines = [
    "# Hybrid calibration summary",
    "",
    `Status: **${report.status}**`,
    "",
    "## Protocol",
    "",
    `- Calibration performers: ${report.protocol.calibrationPerformers.join(", ")}`,
    `- Sealed validation performer: ${report.protocol.sealedFinalValidationPerformer}`,
    `- Strategy: ${report.protocol.strategy}`,
    `- Track-captures: ${report.protocol.trackCaptureCount}`,
    `- Candidates: ${report.search.candidateCount}`,
    `- Learned-model overlap: ${report.protocol.learnedModelOverlapLimitation}`,
    "",
    "## Baseline gates",
    "",
    "| Gate | Windows | % | Before | After | Blocked correct ML | Protected correct rule |",
    "|---|---:|---:|---:|---:|---:|---:|",
    ...Object.entries(report.baselineWeightDiagnostics.gates)
      .map(([id, value]: [string, any]) =>
        `| ${id} | ${value.affectedWindows} | ${(value.percentage * 100).toFixed(1)}`
          + ` | ${value.averageWeightBefore.toFixed(4)}`
          + ` | ${value.averageWeightAfter.toFixed(4)}`
          + ` | ${value.blockedCorrectMl} | ${value.protectedCorrectRule} |`),
    "",
    "## Selection",
    "",
    selected
      ? `Selected **${selected.candidate.id}** using only non-p00 calibration folds.`
      : "No candidate satisfied the frozen constraints; current defaults remain selected.",
    "",
    "| Candidate | Detailed mean | Root mean | Worst fold detailed | Eligible |",
    "|---|---:|---:|---:|---:|",
    ...report.search.results.slice(0, 15).map((row: any) =>
      `| ${row.candidate.id} | ${(row.meanDetailedAccuracy * 100).toFixed(2)}%`
        + ` | ${(row.meanRootAccuracy * 100).toFixed(2)}%`
        + ` | ${(row.worstFoldDetailedAccuracy * 100).toFixed(2)}%`
        + ` | ${row.eligible ? "yes" : "no"} |`),
    "",
    "## Probability calibration",
    "",
    `Global fitted temperature: ${report.probabilityCalibration.globalTemperature}`,
    `Fold-calibrated ECE: ${report.probabilityCalibration.diagnostics.foldTemperatureCalibrated.expectedCalibrationError.toFixed(4)}`,
    "",
    "## Decision",
    "",
    report.selection.reason,
    "",
    "p00 was not loaded or scored by this command.",
  ];
  return `${lines.join("\n")}\n`;
}

async function prepareEvidence(
  config: CalibrationConfig,
  configPath: string,
  split: HybridCalibrationSplitManifest,
  environment: CalibrationEnvironment,
): Promise<{
  evidence: PreparedCalibrationEvidence[];
  cacheStats: Record<string, LearnedInferenceCacheStats>;
  cacheParityChecks: Array<{ captureType: string; trackId: string; identical: boolean }>;
  manifestChecksums: string[];
}> {
  const weights = bundledWeights as unknown as TcnWeights;
  const baseProvider = new LearnedTcnProvider(weights);
  const cacheDirectory = path.resolve(
    environment.TABSMITH_HYBRID_CALIBRATION_CACHE?.trim()
      ?? path.join(os.tmpdir(), "tabsmith-hybrid-calibration-cache"),
  );
  const evidence: PreparedCalibrationEvidence[] = [];
  const cacheStats: Record<string, LearnedInferenceCacheStats> = {};
  const cacheParityChecks: Array<{
    captureType: string;
    trackId: string;
    identical: boolean;
  }> = [];
  const manifestChecksums = new Set<string>();
  for (const captureType of split.captures as Array<keyof typeof CAPTURE_ENVIRONMENT>) {
    const cachedProvider = new CachedLearnedHarmonyProvider(
      baseProvider,
      cacheDirectory,
      captureType,
    );
    let parityChecked = false;
    for (const performer of split.calibrationPerformers) {
      if (performer.performerId === SEALED_FINAL_VALIDATION_PERFORMER) {
        throw new Error("p00 cannot enter calibration search");
      }
      const trainingRole = splitNameForPerformer(performer.performerId, split);
      const dataset: DatasetConfig = {
        datasetIdentifier: config.dataset.datasetIdentifier,
        officialDistributionRecord: config.dataset.officialDistributionRecord,
        datasetVersion: config.dataset.datasetVersion,
        annotationArchiveMd5: config.dataset.annotationArchiveMd5,
        audioArchiveMd5ByCapture: config.dataset.audioArchiveMd5ByCapture,
        captureType,
        manifestPath: config.dataset.manifestPath,
        annotationDirectory: config.dataset.annotationDirectory,
        splitName: trainingRole,
        splitStrategy: "artist-hash",
        splitSeed: config.dataset.splitSeed,
        heldOutArtists: [performer.performerId],
        sourceType: config.dataset.sourceType,
        sourceMode: config.dataset.sourceMode,
        expectedAnnotationCount: config.dataset.expectedAnnotationCount,
        expectedTrackCount: config.dataset.expectedTracksPerPerformer,
      };
      const loaderEnvironment = {
        TABSMITH_GUITARSET_MANIFEST:
          environment.TABSMITH_GUITARSET_MANIFEST,
        TABSMITH_GUITARSET_ANNOTATIONS:
          environment.TABSMITH_GUITARSET_ANNOTATIONS,
        TABSMITH_GUITARSET_AUDIO:
          sourceAudioRoot(captureType, environment),
        TABSMITH_HYBRID_EVAL_CAPTURE: captureType,
      };
      const loaded = await loadDataset(
        dataset,
        configPath,
        config.sampleRate,
        loaderEnvironment,
      );
      manifestChecksums.add(loaded.identity.manifestChecksum);
      for (const [index, track] of loaded.tracks.entries()) {
        console.log(
          `[calibration] ${captureType}/${performer.performerId}`
            + ` prepare ${index + 1}/${loaded.tracks.length}: ${track.trackId}`,
        );
        const observationPackage = createHarmonyObservations(
          track.samples,
          track.sampleRate,
          {},
          { source: "guitar-only" },
        );
        const ruleAnalysis = decodeHarmonyObservations(observationPackage);
        const request = buildEvaluationRequest(track, observationPackage, {
          modelVersion: LEARNED_MODEL_VERSION,
          modelChecksum: LEARNED_MODEL_CHECKSUM,
          featureVersion: FEATURE_VERSION,
        });
        if (!request) throw new Error(`${track.trackId}: no learned feature request`);
        const response = await cachedProvider.predict(request);
        const responseHash = responseProbabilityHash(response);
        if (!parityChecked) {
          const uncached = await baseProvider.predict(request);
          const identical = responseHash === responseProbabilityHash(uncached);
          cacheParityChecks.push({ captureType, trackId: track.trackId, identical });
          if (!identical) {
            throw new Error(`${track.trackId}: cached and uncached inference differ`);
          }
          parityChecked = true;
        }
        evidence.push({
          captureType,
          foldId: split.folds.find((fold) =>
            fold.validationPerformers.includes(performer.performerId))!.foldId,
          performerId: performer.performerId,
          track: {
            datasetId: track.datasetId,
            splitName: "calibration",
            trackId: track.trackId,
            artist: track.artist,
            title: track.title,
            sourceType: track.sourceType,
            sourceMode: track.sourceMode,
            sampleRate: track.sampleRate,
            duration: track.duration,
            originalOffsetSeconds: track.originalOffsetSeconds,
            bpm: track.bpm,
            referenceRegions: track.referenceRegions,
            referenceChecksum: track.referenceChecksum,
          },
          observationPackage,
          ruleAnalysis,
          response,
          responseHash,
          mlOnlyRegions: decodeLearnedOnly(response, observationPackage.duration),
        });
      }
    }
    cacheStats[captureType] = { ...cachedProvider.stats };
  }
  return {
    evidence,
    cacheStats,
    cacheParityChecks,
    manifestChecksums: [...manifestChecksums],
  };
}

export async function runHybridCalibration(
  configPath: string,
  environment: CalibrationEnvironment = process.env,
): Promise<Record<string, any>> {
  const config = JSON.parse(await readFile(configPath, "utf8")) as CalibrationConfig;
  if (config.schemaVersion !== 1) throw new Error("Unsupported calibration config");
  const splitPath = path.resolve(path.dirname(configPath), config.splitManifest);
  const split = JSON.parse(
    await readFile(splitPath, "utf8"),
  ) as HybridCalibrationSplitManifest;
  validateCalibrationSplit(split);
  if (split.finalValidationPerformer !== SEALED_FINAL_VALIDATION_PERFORMER) {
    throw new Error("p00 seal is missing");
  }
  const prepared = await prepareEvidence(
    config,
    configPath,
    split,
    environment,
  );
  assertNonEmptyCalibrationTracks(prepared.evidence.length);
  if (prepared.evidence.some(({ performerId }) =>
    performerId === SEALED_FINAL_VALIDATION_PERFORMER)) {
    throw new Error("p00 cannot enter calibration search");
  }
  assignGroupedFolds(
    prepared.evidence.map((item) => ({
      trackId: item.track.trackId,
      performerId: item.performerId,
      captureType: item.captureType,
    })),
    split,
  );

  const temperatures = fitFoldTemperatures(
    prepared.evidence,
    split.folds,
    config.search.temperatureCandidates,
  );
  const globalTemperature = fitTemperature(
    prepared.evidence,
    config.search.temperatureCandidates,
  );
  const candidates = generateCalibrationCandidates(
    config.search.seed,
    config.search.randomCandidateCount,
  );
  const ruleByCapture = aggregateRuleMetrics(
    prepared.evidence,
    config.search.shortRegionThresholdSeconds,
    config.search.seed,
  );
  const screening = candidates.map((candidate, index) => {
    console.log(
      `[calibration] screen ${index + 1}/${candidates.length}: ${candidate.id}`,
    );
    return screenCandidate(
      prepared.evidence,
      candidate,
      temperatures.byPerformer,
    );
  });
  const shortlistedCandidates = shortlistCandidates(candidates, screening);
  console.log(
    `[calibration] production-decoder shortlist:`
      + ` ${shortlistedCandidates.map(({ id }) => id).join(", ")}`,
  );
  const evaluated: Array<ReturnType<typeof aggregateCandidate>> = [];
  let baselineResults: CandidateTrackResult[] | null = null;
  for (const [candidateIndex, candidate] of shortlistedCandidates.entries()) {
    console.log(
      `[calibration] full decoder ${candidateIndex + 1}/${shortlistedCandidates.length}:`
        + ` ${candidate.id}`,
    );
    const results = prepared.evidence.map((item) => evaluateCandidateTrack(
      item,
      candidate,
      candidate.probabilityCalibration === "temperature"
        ? {
          kind: "temperature",
          value: temperatures.byPerformer.get(item.performerId) ?? 1,
        }
        : { kind: "none", value: 1 },
      config.search.shortRegionThresholdSeconds,
    ));
    if (candidate.id === "current-adaptive") baselineResults = results;
    evaluated.push(aggregateCandidate(
      candidate,
      results,
      config.search.seed,
    ));
  }
  if (!baselineResults) throw new Error("Current adaptive baseline was not evaluated");
  const temperatureImprovedEceFolds = temperatures.folds.filter((fold) =>
    fold.validation.temperature.expectedCalibrationError
      < fold.validation.uncalibrated.expectedCalibrationError).length;
  const temperatureCalibrationEligible =
    temperatureImprovedEceFolds >= Math.ceil(temperatures.folds.length / 2);
  const selectionRows = evaluated.map(({ selection }) => ({
    ...selection,
    eligible: candidateSatisfiesConstraints(
      selection,
      ruleByCapture,
      config.search.constraints,
    ) && (selection.candidate.probabilityCalibration !== "temperature"
      || temperatureCalibrationEligible),
  }));
  const selected = selectCalibrationCandidate(
    evaluated.flatMap(({ selection }) =>
      selection.candidate.probabilityCalibration === "temperature"
        && !temperatureCalibrationEligible ? [] : [selection]),
    ruleByCapture,
    config.search.constraints,
  );
  const current = evaluated.find(
    ({ selection }) => selection.candidate.id === "current-adaptive",
  )!;
  const improvedSelection = selected
    && selected.candidate.id !== "current-adaptive"
    && selected.meanDetailedAccuracy > current.selection.meanDetailedAccuracy
    && selected.meanRootAccuracy > current.selection.meanRootAccuracy
    ? selected
    : null;
  const selectedImprovesCurrent = Boolean(improvedSelection);
  const finalSelection = improvedSelection ?? current.selection;
  const selectedCalibration: ProbabilityCalibration =
    finalSelection.candidate.probabilityCalibration === "temperature"
      ? { kind: "temperature", value: globalTemperature.temperature }
      : { kind: "none", value: 1 };
  const identity = settingsIdentity(
    finalSelection.candidate.settings,
    selectedCalibration,
  );
  const selectedResults = finalSelection.candidate.id === "current-adaptive"
    ? baselineResults
    : prepared.evidence.map((item) => evaluateCandidateTrack(
      item,
      finalSelection.candidate,
      finalSelection.candidate.probabilityCalibration === "temperature"
        ? {
          kind: "temperature",
          value: temperatures.byPerformer.get(item.performerId) ?? 1,
        }
        : { kind: "none", value: 1 },
      config.search.shortRegionThresholdSeconds,
    ));
  const temperatureCalibrationRetained =
    finalSelection.candidate.probabilityCalibration === "temperature"
    && temperatureImprovedEceFolds >= Math.ceil(temperatures.folds.length / 2);
  const report: Record<string, any> = {
    schemaVersion: 1,
    status: "completed-non-p00-calibration",
    createdAt: new Date().toISOString(),
    commitSha: currentCommit(),
    protocol: {
      splitSeed: split.seed,
      strategy: split.strategy,
      captures: split.captures,
      calibrationPerformers:
        split.calibrationPerformers.map(({ performerId }) => performerId),
      performerRoles: Object.fromEntries(split.calibrationPerformers.map(
        ({ performerId, learnedModelRole }) => [performerId, learnedModelRole],
      )),
      sealedFinalValidationPerformer: split.finalValidationPerformer,
      trackCaptureCount: prepared.evidence.length,
      performanceCount: new Set(prepared.evidence.map(
        ({ track }) => track.trackId,
      )).size,
      durationSecondsByCapture: Object.fromEntries(split.captures.map((capture) => [
        capture,
        prepared.evidence.filter((item) => item.captureType === capture)
          .reduce((sum, item) => sum + item.track.duration, 0),
      ])),
      manifestChecksums: prepared.manifestChecksums,
      learnedModelOverlapLimitation:
        "All non-p00 performers overlap learned-model fitting: p01/p02/p03/p05 training and p04 development. This is integration calibration, not independent model evaluation.",
      p00LoadedOrScored: false,
    },
    model: {
      modelVersion: LEARNED_MODEL_VERSION,
      modelChecksum: LEARNED_MODEL_CHECKSUM,
      featureVersion: FEATURE_VERSION,
      productionRuleEngine: PRODUCTION_RULE_ENGINE,
    },
    inferenceCache: {
      identityIncludesModelChecksum: true,
      identityIncludesFeatureVersion: true,
      identityIncludesCaptureType: true,
      storesAudio: false,
      stats: prepared.cacheStats,
      cachedUncachedParityChecks: prepared.cacheParityChecks,
    },
    probabilityCalibration: {
      temperatureCandidates: config.search.temperatureCandidates,
      folds: temperatures.folds,
      globalTemperature: globalTemperature.temperature,
      globalTrainingNll: globalTemperature.nll,
      heldOutFoldsWithImprovedEce: temperatureImprovedEceFolds,
      retainedInFrozenConfiguration: temperatureCalibrationRetained,
      decision: temperatureCalibrationRetained
        ? "Temperature scaling was retained because it improved held-out calibration across a majority of performer folds."
        : "Temperature scaling was not retained: held-out ECE did not improve consistently across performer folds.",
      diagnostics: calibrationDiagnostics(
        prepared.evidence,
        temperatures.byPerformer,
      ),
    },
    baseline: {
      settings: CONSERVATIVE_HYBRID_SETTINGS,
      settingsIdentity: settingsIdentity(CONSERVATIVE_HYBRID_SETTINGS),
      aggregates: current.aggregates,
      folds: current.folds,
      overrides: current.overrides,
    },
    referenceBaselines: {
      ruleByCapture: fastBaselineByCapture(
        prepared.evidence,
        (item) => item.ruleAnalysis.regions,
      ),
      mlOnlyByCapture: fastBaselineByCapture(
        prepared.evidence,
        (item) => item.mlOnlyRegions,
      ),
    },
    pairedComparisons: pairedFastDiagnostics(
      selectedResults,
      config.search.seed,
    ),
    baselineWeightDiagnostics:
      baselineGateDiagnostics(prepared.evidence, baselineResults),
    search: {
      seed: config.search.seed,
      candidateCount: candidates.length,
      fullyDecodedCandidateCount: shortlistedCandidates.length,
      randomCandidateCount: config.search.randomCandidateCount,
      searchSpace: {
        learnedChordWeight: [0.35, 0.85],
        maximumLearnedWeightGuitarOnly: [0.5, 0.9],
        minimumLearnedConfidence: [0.35, 0.62],
        maximumLearnedEntropy: [0.7, 0.99],
        protectRuleConfidenceAbove: [0.68, 0.88],
        learnedBoundaryWeight: [0, 0.1, 0.25],
        bassRootWeight: [0, 0.15, 0.25],
        adaptiveWeighting: [true, false],
        probabilityCalibration: ["none", "fold-temperature"],
      },
      constraints: config.search.constraints,
      ruleByCapture,
      screening: screening.sort((left, right) => right.score - left.score),
      results: selectionRows.sort((left, right) =>
        right.meanDetailedAccuracy - left.meanDetailedAccuracy),
      detailedResults: evaluated.map((row) => ({
        candidateId: row.selection.candidate.id,
        aggregates: row.aggregates,
        folds: row.folds,
        overrides: row.overrides,
      })),
      paretoFrontier: paretoFrontier(
        evaluated.map(({ selection }) => selection),
      ).map((row) => row.candidate.id),
    },
    selection: {
      selected: finalSelection,
      selectedProbabilityCalibration: selectedCalibration,
      settingsIdentity: identity,
      selectedByConstraints: selected?.candidate.id ?? null,
      changedFromCurrent: finalSelection.candidate.id !== "current-adaptive",
      reason: selectedImprovesCurrent
        ? "Selected the highest detailed-accuracy eligible candidate that also improved root accuracy over the current hybrid; p00 remained sealed."
        : selected
          ? "No eligible candidate clearly improved both detailed and root accuracy over the current hybrid, so the current experimental defaults are retained."
          : "No candidate satisfied all frozen stability constraints; the current experimental defaults are retained and the Pareto frontier is reported.",
    },
    limitations: [
      "Every non-p00 GuitarSet performer overlaps learned-model training or development, so calibration metrics are integration-domain estimates.",
      "GuitarSet contains alternate captures of solo acoustic guitar only; full-band validation remains required.",
      "The calibration references contain no useful no-chord duration.",
      "Logit scaling was not evaluated because the frozen learned response exposes probabilities rather than logits.",
    ],
  };
  assertNoAbsolutePaths(report);
  const outputDirectory = path.resolve(
    path.dirname(configPath),
    config.outputDirectory,
  );
  await mkdir(outputDirectory, { recursive: true });
  const jsonPath = path.join(outputDirectory, "hybrid-calibration-search.json");
  const markdownPath = path.join(outputDirectory, "hybrid-calibration-summary.md");
  const frozenPath = path.join(
    path.dirname(configPath),
    "hybrid-calibration-frozen-settings.json",
  );
  const frozen = {
    schemaVersion: 1,
    status: "frozen-before-p00-validation",
    frozenAt: new Date().toISOString(),
    calibrationReport: "reports/hybrid-calibration-search.json",
    calibrationPerformers:
      split.calibrationPerformers.map(({ performerId }) => performerId),
    sealedFinalValidationPerformer: split.finalValidationPerformer,
    candidateId: finalSelection.candidate.id,
    settings: finalSelection.candidate.settings,
    adaptiveWeighting: finalSelection.candidate.adaptiveWeighting,
    probabilityCalibration: selectedCalibration,
    settingsIdentity: identity,
    changedFromCurrent: finalSelection.candidate.id !== "current-adaptive",
    selectionReason: report.selection.reason,
  };
  assertNoAbsolutePaths(frozen);
  await Promise.all([
    writeFile(jsonPath, `${JSON.stringify(report, null, 2)}\n`, "utf8"),
    writeFile(markdownPath, markdownReport(report), "utf8"),
    writeFile(frozenPath, `${JSON.stringify(frozen, null, 2)}\n`, "utf8"),
  ]);
  console.log(`Calibration JSON: ${jsonPath}`);
  console.log(`Calibration Markdown: ${markdownPath}`);
  console.log(`Frozen settings: ${frozenPath}`);
  console.log("p00 loaded/scored: false");
  return report;
}

function parseConfigPath(argv: string[]): string {
  const index = argv.indexOf("--config");
  const target = index >= 0 ? argv[index + 1] : "evaluation/hybrid-calibration-config.json";
  if (!target) throw new Error("--config requires a path");
  return path.resolve(target);
}

const invokedDirectly = process.env.npm_lifecycle_event === "calibrate:hybrid"
  || process.argv.some((argument) =>
    path.basename(argument).startsWith("run-hybrid-calibration.ts"));
if (invokedDirectly) {
  await runHybridCalibration(parseConfigPath(process.argv.slice(2))).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
