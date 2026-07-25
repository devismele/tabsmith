import {
  decodeHarmonyObservations,
  type ChordObservation,
} from "../chordAnalysis";
import type { ChordEvent } from "../types";
import { CONTRACT, validateResponse } from "./contract";
import { boundaryPeakCount, shannonEntropy } from "./diagnostics";
import type {
  HybridComparisonRow,
  HybridDiagnostics,
  HybridFallbackReason,
  HybridFusionContext,
  HybridFusionDiagnostics,
  HybridHarmonyResult,
  HybridHarmonySettings,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
  LearnedWindowEvidence,
  RuleBasedHarmonyEvidence,
} from "./types";

// This is an internal decoder/fusion revision, deliberately separate from the
// frozen learned-harmony v1 wire contract.
export const HYBRID_DECODER_VERSION = 2;

/**
 * Initial calibration is intentionally conservative. The rule scores already
 * include chroma, bass and key context; the learned term is admitted only when
 * its full distribution is confident and the rule observation has room to move.
 */
export const CONSERVATIVE_HYBRID_SETTINGS: HybridHarmonySettings = {
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
};

const ROOT_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const ROOT_INDEX: Record<string, number> = {
  C: 0, "C#": 1, Db: 1, D: 2, "D#": 3, Eb: 3, E: 4,
  F: 5, "F#": 6, Gb: 6, G: 7, "G#": 8, Ab: 8, A: 9,
  "A#": 10, Bb: 10, B: 11,
};
const EPSILON = 1e-7;
const RULE_SOFTMAX_TEMPERATURE = 0.12;
const LEARNED_LOG_SCALE = 0.1;

function clamp(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function averageRows(rows: number[][], indices: number[]): number[] {
  const width = rows[0]?.length ?? 0;
  if (!width || !indices.length) return [];
  const result = new Array<number>(width).fill(0);
  for (const index of indices) {
    for (let column = 0; column < width; column += 1) {
      result[column] += rows[index][column] ?? 0;
    }
  }
  return result.map((value) => value / indices.length);
}

function normalizedEntropy(probabilities: number[]): number {
  if (probabilities.length <= 1) return 0;
  return clamp(shannonEntropy(probabilities) / Math.log(probabilities.length));
}

function binaryEntropy(probability: number): number {
  const p = clamp(probability, EPSILON, 1 - EPSILON);
  return clamp((-(p * Math.log(p) + (1 - p) * Math.log(1 - p))) / Math.log(2));
}

function chordParts(name: string): { root: number; suffix: string } | null {
  const match = /^([A-G](?:#|b)?)(.*)$/.exec(name);
  if (!match) return null;
  const root = ROOT_INDEX[match[1]];
  return root === undefined ? null : { root, suffix: match[2] };
}

/**
 * Maps the frozen maj/min/7 vocabulary onto every candidate understood by the
 * rule decoder. Unsupported detailed qualities retain a small root-conditioned
 * probability instead of disappearing from Viterbi.
 */
function learnedProbabilityForChord(
  chordName: string,
  roots: number[],
  qualities: number[],
  noChordProbability: number,
): number {
  const parts = chordParts(chordName);
  if (!parts) return EPSILON;
  const rootProbability = roots[parts.root] ?? 0;
  let qualityProbability = 0;
  let detailCalibration = 1;
  if (parts.suffix === "") {
    qualityProbability = qualities[0] ?? 0;
  } else if (parts.suffix === "m") {
    qualityProbability = qualities[1] ?? 0;
  } else if (parts.suffix === "7") {
    qualityProbability = qualities[2] ?? 0;
  } else if (parts.suffix === "maj7") {
    qualityProbability = qualities[2] ?? 0;
    detailCalibration = 0.45;
  } else if (parts.suffix === "m7") {
    qualityProbability = qualities[1] ?? 0;
    detailCalibration = 0.45;
  } else if (parts.suffix === "dim") {
    qualityProbability = qualities[1] ?? 0;
    detailCalibration = 0.15;
  } else {
    qualityProbability = qualities[0] ?? 0;
    detailCalibration = 0.15;
  }
  return Math.max(
    EPSILON,
    (1 - noChordProbability) * rootProbability * qualityProbability * detailCalibration,
  );
}

export function alignLearnedEvidence(
  observations: ChordObservation[],
  learned: LearnedHarmonyResponse,
): LearnedWindowEvidence[] {
  return observations.map((observation, observationIndex) => {
    const indices: number[] = [];
    for (let frame = 0; frame < learned.frameTimes.length; frame += 1) {
      const time = learned.frameTimes[frame];
      const inside = time >= observation.start
        && (time < observation.end
          || (observationIndex === observations.length - 1 && time <= observation.end));
      if (inside) indices.push(frame);
    }
    if (!indices.length) {
      return {
        chordProbabilities: { N: 0 },
        noChordProbability: 0,
        boundaryProbability: 0,
        entropy: 1,
        topChord: null,
        topChordConfidence: 0,
        contributingFrameCount: 0,
      };
    }

    const roots = averageRows(learned.rootProbabilities, indices);
    const qualities = averageRows(learned.qualityProbabilities, indices);
    const noChordProbability = clamp(mean(indices.map(
      (index) => learned.noChordProbabilities[index],
    )));
    // A boundary head is event-like rather than state-like. Max pooling retains a
    // narrow real change; its decoder effect is separately reliability-gated.
    const boundaryProbability = clamp(Math.max(...indices.map(
      (index) => learned.boundaryProbabilities[index],
    )));
    const chordProbabilities = Object.fromEntries(
      Object.keys(observation.candidateScores).map((chordName) => [
        chordName,
        learnedProbabilityForChord(chordName, roots, qualities, noChordProbability),
      ]),
    );
    chordProbabilities.N = noChordProbability;
    const ranked = Object.entries(chordProbabilities)
      .sort((left, right) => right[1] - left[1]);
    return {
      chordProbabilities,
      noChordProbability,
      boundaryProbability,
      entropy: mean([
        normalizedEntropy(roots),
        normalizedEntropy(qualities),
        binaryEntropy(noChordProbability),
      ]),
      topChord: ranked[0]?.[0] ?? null,
      topChordConfidence: ranked[0]?.[1] ?? 0,
      contributingFrameCount: indices.length,
    };
  });
}

function ruleScoreEntries(observation: ChordObservation): Array<[string, number]> {
  return [
    ...Object.entries(observation.candidateScores),
    ["N", observation.noChordScore],
  ];
}

function normalizedRuleScores(observation: ChordObservation): Record<string, number> {
  const entries = ruleScoreEntries(observation);
  const maximum = Math.max(...entries.map(([, score]) => score));
  const exponentials = entries.map(([, score]) =>
    Math.exp((score - maximum) / RULE_SOFTMAX_TEMPERATURE));
  const total = exponentials.reduce((sum, value) => sum + value, 0) || 1;
  // This log-softmax calibration is affine-equivalent to the original template
  // scores, preserving their relative evidence while putting every window
  // through one explicit probabilistic scale.
  return Object.fromEntries(entries.map(([name], index) => [
    name,
    maximum + RULE_SOFTMAX_TEMPERATURE * Math.log(exponentials[index] / total + EPSILON),
  ]));
}

function flickerFactor(evidence: LearnedWindowEvidence[], index: number): number {
  const current = evidence[index].topChord;
  if (!current) return 0;
  const previous = evidence[index - 1]?.topChord;
  const next = evidence[index + 1]?.topChord;
  const changes = Number(Boolean(previous && previous !== current))
    + Number(Boolean(next && next !== current));
  return changes === 2 ? 0.25 : changes === 1 ? 0.65 : 1;
}

function effectiveLearnedWeight(
  observation: ChordObservation,
  evidence: LearnedWindowEvidence,
  settings: HybridHarmonySettings,
  sourceMode: "full-mix" | "guitar-focused",
  flicker: number,
): number {
  if (!evidence.contributingFrameCount
    || evidence.topChordConfidence < settings.minimumLearnedConfidence
    || evidence.entropy >= settings.maximumLearnedEntropy) return 0;

  const confidenceFactor = clamp(
    (evidence.topChordConfidence - settings.minimumLearnedConfidence)
      / Math.max(EPSILON, 1 - settings.minimumLearnedConfidence),
  );
  const entropyFactor = clamp(
    (settings.maximumLearnedEntropy - evidence.entropy)
      / Math.max(EPSILON, settings.maximumLearnedEntropy),
  );
  const marginAmbiguity = 1 - clamp(observation.scoreMargin / 0.1);
  const confidenceAmbiguity = 1 - clamp(
    observation.confidence / Math.max(EPSILON, settings.protectRuleConfidenceAbove),
  );
  const ruleRoom = 0.35 + 0.65 * mean([marginAmbiguity, confidenceAmbiguity]);
  const protection = observation.confidence <= settings.protectRuleConfidenceAbove
    ? 1
    : clamp(
      (0.9 - observation.confidence)
        / Math.max(EPSILON, 0.9 - settings.protectRuleConfidenceAbove),
    );
  const agreement = evidence.topChord === observation.bestChord ? 1.1 : 1;
  const cap = sourceMode === "full-mix"
    ? settings.maximumLearnedWeightFullMix
    : settings.maximumLearnedWeightGuitarOnly;
  return clamp(
    settings.learnedChordWeight
      * (0.35 + confidenceFactor * 0.65)
      * entropyFactor
      * ruleRoom
      * protection
      * agreement
      * flicker,
    0,
    cap,
  );
}

function learnedScore(probability: number, candidateCount: number): number {
  const uniform = 1 / Math.max(2, candidateCount);
  return clamp(
    LEARNED_LOG_SCALE * Math.log(Math.max(EPSILON, probability) / uniform),
    -0.35,
    0.25,
  );
}

function candidateRoot(name: string): number | null {
  return name === "N" ? null : chordParts(name)?.root ?? null;
}

function observedBassRoot(observation: ChordObservation): number | null {
  const label = observation.bassRootClass ?? observation.rootClass;
  return label ? candidateRoot(label) : null;
}

function countBoundaryPeaks(evidence: LearnedWindowEvidence[], threshold = 0.5): number {
  return boundaryPeakCount(evidence.map((window) => window.boundaryProbability), threshold);
}

export function fuseHybridObservations(
  ruleObservations: ChordObservation[],
  learnedResponse: LearnedHarmonyResponse,
  settings: HybridHarmonySettings,
  context: HybridFusionContext = {},
): { observations: ChordObservation[]; diagnostics: HybridFusionDiagnostics } {
  const sourceMode = context.sourceMode ?? "full-mix";
  const evidence = alignLearnedEvidence(ruleObservations, learnedResponse);
  const weights: number[] = [];
  let disagreements = 0;
  let agreements = 0;
  let changedTopCandidates = 0;

  const observations = ruleObservations.map((observation, index) => {
    const learned = evidence[index];
    const weight = effectiveLearnedWeight(
      observation,
      learned,
      settings,
      sourceMode,
      flickerFactor(evidence, index),
    );
    weights.push(weight);
    if (learned.topChord) {
      if (learned.topChord === observation.bestChord) agreements += 1;
      else disagreements += 1;
    }

    const ruleScores = normalizedRuleScores(observation);
    const candidateCount = Object.keys(ruleScores).length;
    const bassRoot = observedBassRoot(observation);
    const bassConfidence = observation.bassConfidence ?? 0;
    const fusedEntries = Object.entries(ruleScores).map(([name, ruleScore]) => {
      const probability = name === "N"
        ? learned.noChordProbability
        : (learned.chordProbabilities[name] ?? EPSILON);
      const noChordProtection = name === "N"
        && (bassConfidence >= 0.18 || observation.confidence >= settings.protectRuleConfidenceAbove)
        ? 0.2
        : 1;
      const bassBonus = name !== "N"
        && bassRoot !== null
        && candidateRoot(name) === bassRoot
        ? weight * settings.bassRootWeight * 0.12 * bassConfidence
        : 0;
      const agreementBonus = name === learned.topChord
        && name === observation.bestChord
        ? weight * 0.015
        : 0;
      return [
        name,
        settings.ruleObservationWeight * ruleScore
          + weight * noChordProtection * learnedScore(probability, candidateCount)
          + bassBonus
          + agreementBonus,
      ] as const;
    }).sort((left, right) => right[1] - left[1]);

    const best = fusedEntries[0] ?? [observation.bestChord, observation.bestScore] as const;
    const second = fusedEntries[1] ?? [observation.secondBestChord, observation.secondBestScore] as const;
    const margin = best[1] - second[1];
    if (best[0] !== observation.bestChord) changedTopCandidates += 1;
    const candidateScores = Object.fromEntries(
      fusedEntries.filter(([name]) => name !== "N"),
    );
    const noChordScore = fusedEntries.find(([name]) => name === "N")?.[1]
      ?? observation.noChordScore;
    const boundaryReliability = settings.learnedChordWeight > 0
      ? clamp(weight / settings.learnedChordWeight)
      : 0;
    return {
      ...observation,
      bestChord: best[0],
      bestScore: best[1],
      secondBestChord: second[0],
      secondBestScore: second[1],
      scoreMargin: margin,
      confidence: clamp(
        observation.confidence + Math.max(0, margin - observation.scoreMargin) * 1.5,
      ),
      uncertain: weight === 0 ? observation.uncertain : margin < 0.01,
      candidateScores,
      noChordScore,
      learnedBoundaryProbability: learned.boundaryProbability,
      learnedBoundaryAdjustment: clamp(
        settings.learnedBoundaryWeight
          * learned.boundaryProbability
          * boundaryReliability
          * 0.25,
        0,
        0.06,
      ),
    };
  });

  const aligned = evidence.filter((window) => window.contributingFrameCount > 0);
  const compared = agreements + disagreements;
  return {
    observations,
    diagnostics: {
      alignedWindows: aligned.length,
      missingLearnedWindows: evidence.length - aligned.length,
      averageLearnedEntropy: mean(aligned.map((window) => window.entropy)),
      ruleLearnedAgreementRate: compared ? agreements / compared : 0,
      ruleLearnedDisagreements: disagreements,
      changedTopCandidateWindows: changedTopCandidates,
      learnedBoundaryPeaksConsidered: countBoundaryPeaks(evidence),
      effectiveLearnedWeight: {
        minimum: weights.length ? Math.min(...weights) : 0,
        maximum: weights.length ? Math.max(...weights) : 0,
        average: mean(weights),
      },
    },
  };
}

function regionAt(regions: ChordEvent[], time: number): ChordEvent | undefined {
  return regions.find((region, index) =>
    time >= region.start
      && (time < region.end || (index === regions.length - 1 && time <= region.end)));
}

function changedRegionCount(ruleRegions: ChordEvent[], hybridRegions: ChordEvent[]): number {
  const count = Math.max(ruleRegions.length, hybridRegions.length);
  let changed = 0;
  for (let index = 0; index < count; index += 1) {
    const rule = ruleRegions[index];
    const hybrid = hybridRegions[index];
    if (!rule || !hybrid
      || rule.name !== hybrid.name
      || Math.abs(rule.start - hybrid.start) > 1e-6
      || Math.abs(rule.end - hybrid.end) > 1e-6) changed += 1;
  }
  return changed;
}

export interface RunHybridParams {
  provider: LearnedHarmonyProvider;
  request: LearnedHarmonyRequest | null;
  ruleBased: RuleBasedHarmonyEvidence;
  settings: HybridHarmonySettings;
  enabled: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
  sourceMode?: "full-mix" | "guitar-focused";
}

/**
 * Orchestrates one atomic hybrid attempt. Every failure returns the exact rule
 * regions and analysis object; learned evidence is never partially decoded.
 */
export async function runHybridHarmony(params: RunHybridParams): Promise<HybridHarmonyResult> {
  const { provider, request, ruleBased, settings, enabled } = params;
  const timeoutMs = params.timeoutMs ?? CONTRACT.limits.inferenceTimeoutMs;
  const diagnostics: HybridDiagnostics = {
    providerId: provider.id,
    providerAvailable: false,
    sourceMode: params.sourceMode ?? "full-mix",
    warnings: [],
    comparison: [],
  };
  const fallback = (reason: HybridFallbackReason): HybridHarmonyResult => ({
    engine: "rule-based",
    regions: ruleBased.regions,
    chordAnalysis: ruleBased.analysisResult,
    usedLearned: false,
    fallbackReason: reason,
    diagnostics: { ...diagnostics, fallbackReason: reason },
  });

  if (!enabled) return fallback("learned-disabled");
  if (!request || !ruleBased.observationPackage) return fallback("no-feature-package");
  if (params.signal?.aborted) return fallback("cancelled");

  try {
    diagnostics.providerAvailable = await provider.isAvailable();
  } catch {
    return fallback("provider-error");
  }
  if (!diagnostics.providerAvailable) return fallback("provider-unavailable");

  try {
    const metadata = await provider.getMetadata();
    diagnostics.modelVersion = metadata.modelVersion;
    diagnostics.modelChecksum = metadata.modelChecksum;
    diagnostics.featureVersion = metadata.featureVersion;
    if (
      metadata.contractVersion !== request.contractVersion
      || metadata.featureVersion !== request.featureVersion
      || metadata.modelVersion !== request.modelMetadata.modelVersion
      || metadata.modelChecksum !== request.modelMetadata.modelChecksum
    ) return fallback("version-mismatch");
  } catch {
    return fallback("provider-error");
  }

  let response: LearnedHarmonyResponse;
  let timedOut = false;
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  params.signal?.addEventListener("abort", onExternalAbort, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  try {
    response = await provider.predict(request, controller.signal);
  } catch (error) {
    if (timedOut) return fallback("timeout");
    if (controller.signal.aborted
      || (error instanceof DOMException && error.name === "AbortError")) {
      return fallback("cancelled");
    }
    return fallback("provider-error");
  } finally {
    clearTimeout(timer);
    params.signal?.removeEventListener("abort", onExternalAbort);
  }
  diagnostics.inferenceMs = response.diagnostics.inferenceMilliseconds;

  const validation = validateResponse(response, request);
  if (!validation.ok) {
    diagnostics.warnings.push(...validation.errors);
    return fallback("invalid-response");
  }
  if (response.modelVersion !== request.modelMetadata.modelVersion
    || response.modelChecksum !== request.modelMetadata.modelChecksum) {
    return fallback("version-mismatch");
  }

  try {
    const fusionStart = performance.now();
    const fused = fuseHybridObservations(
      ruleBased.observationPackage.observations,
      response,
      settings,
      { sourceMode: diagnostics.sourceMode },
    );
    diagnostics.fusionMs = performance.now() - fusionStart;
    const decoderStart = performance.now();
    const chordAnalysis = decodeHarmonyObservations({
      ...ruleBased.observationPackage,
      observations: fused.observations,
    });
    diagnostics.decoderMs = performance.now() - decoderStart;
    diagnostics.inferenceMs = response.diagnostics.inferenceMilliseconds;
    diagnostics.modelVersion = response.modelVersion;
    diagnostics.modelChecksum = response.modelChecksum;
    diagnostics.featureVersion = response.featureVersion;
    diagnostics.learnedProbabilityEntropy = fused.diagnostics.averageLearnedEntropy;
    diagnostics.boundaryPeakCount = fused.diagnostics.learnedBoundaryPeaksConsidered;
    diagnostics.ruleLearnedDisagreements = fused.diagnostics.ruleLearnedDisagreements;
    diagnostics.ruleLearnedAgreementRate = fused.diagnostics.ruleLearnedAgreementRate;
    diagnostics.changedTopCandidateWindows = fused.diagnostics.changedTopCandidateWindows;
    diagnostics.finalRegionsChanged = changedRegionCount(
      ruleBased.regions,
      chordAnalysis.regions,
    );
    diagnostics.effectiveLearnedWeightMinimum =
      fused.diagnostics.effectiveLearnedWeight.minimum;
    diagnostics.effectiveLearnedWeightMaximum =
      fused.diagnostics.effectiveLearnedWeight.maximum;
    diagnostics.effectiveLearnedWeightAverage =
      fused.diagnostics.effectiveLearnedWeight.average;
    diagnostics.warnings.push(...response.diagnostics.warnings);

    const aligned = alignLearnedEvidence(
      ruleBased.observationPackage.observations,
      response,
    );
    const comparisonStride = Math.max(
      1,
      Math.ceil(ruleBased.observationPackage.observations.length / 512),
    );
    diagnostics.comparison = ruleBased.observationPackage.observations.flatMap(
      (observation, index): HybridComparisonRow[] => index % comparisonStride
        ? []
        : [{
        startSeconds: observation.start,
        endSeconds: observation.end,
        ruleChord: regionAt(ruleBased.regions, observation.start)?.name
          ?? observation.bestChord,
        learnedTopChord: aligned[index].topChord,
        hybridChord: regionAt(chordAnalysis.regions, observation.start)?.name
          ?? fused.observations[index].bestChord,
        ruleConfidence: observation.confidence,
        learnedConfidence: aligned[index].topChordConfidence,
        boundaryProbability: aligned[index].boundaryProbability,
        }],
    );
    if (comparisonStride > 1) {
      diagnostics.warnings.push(
        `Comparison diagnostics sampled every ${comparisonStride} windows (512-row cap).`,
      );
    }
    return {
      engine: "hybrid-experimental",
      regions: chordAnalysis.regions,
      chordAnalysis,
      usedLearned: true,
      diagnostics,
    };
  } catch (error) {
    diagnostics.warnings.push(
      `Hybrid fusion/decoding failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return fallback("invalid-response");
  }
}
