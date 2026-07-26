import {
  createHarmonyObservations,
  decodeHarmonyObservations,
  type HarmonyObservationPackage,
} from "../src/chordAnalysis";
import type { ChordAnalysisResult, ChordEvent } from "../src/types";
import { FEATURE_VERSION } from "../src/learnedHarmony/contract";
import { buildFeaturePackage } from "../src/learnedHarmony/featurePackage";
import {
  CONSERVATIVE_HYBRID_SETTINGS,
  alignLearnedEvidence,
  fuseHybridObservations,
  runHybridHarmony,
} from "../src/learnedHarmony/hybridDecoder";
import type {
  HybridFallbackReason,
  HybridFusionDiagnostics,
  HybridHarmonySettings,
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";

export const PRODUCTION_RULE_ENGINE =
  "2026-07-harmonic-context-v3-reduced-latency";
export const LEGACY_REGION_HYBRID_SOURCE_COMMIT =
  "51392820c7c9c1a144069a3ed1371ef2cfe9f5b9";
export const DEFAULT_ML_ONLY_TRANSITION_PENALTY = 4;

export type PrimaryEngineId = "rule-only" | "ml-only" | "observation-hybrid";
export type EngineId = PrimaryEngineId | "legacy-region-hybrid";
export type SourceType = "solo-guitar" | "full-band";

export interface EvaluationReferenceRegion {
  start: number;
  end: number;
  label: string;
}

export interface PreparedEvaluationTrack {
  datasetId: string;
  splitName: string;
  trackId: string;
  artist: string;
  title: string;
  sourceType: SourceType;
  sourceMode: "full-mix" | "guitar-focused";
  sampleRate: number;
  samples: Float32Array;
  duration: number;
  originalOffsetSeconds: number;
  bpm: number | null;
  referenceRegions: EvaluationReferenceRegion[];
  audioHash: string;
  referenceChecksum: string;
}

export interface EvaluationModelIdentity {
  modelVersion: string;
  modelChecksum: string;
  featureVersion: string;
}

export interface EngineTiming {
  observationMilliseconds: number;
  inferenceMilliseconds: number;
  decodeMilliseconds: number;
  endToEndMilliseconds: number;
  peakHeapBytes: number | null;
}

export interface EnginePrediction {
  engineId: EngineId;
  regions: ChordEvent[];
  chordAnalysis?: ChordAnalysisResult;
  usedLearned: boolean;
  fallbackReason?: HybridFallbackReason;
  probabilitySourceId?: string;
  timing: EngineTiming;
  diagnostics?: {
    averageLearnedEntropy?: number;
    ruleLearnedAgreementRate?: number;
    ruleLearnedDisagreements?: number;
    changedTopCandidateWindows?: number;
    learnedBoundaryPeaksConsidered?: number;
    effectiveLearnedWeightAverage?: number;
    effectiveLearnedWeightMinimum?: number;
    effectiveLearnedWeightMaximum?: number;
    alignedLearnedWindows?: number;
    missingLearnedWindows?: number;
  };
}

export type HybridAblationId =
  | "rule-only"
  | "hybrid-chord-probabilities-only"
  | "hybrid-without-boundary"
  | "hybrid-without-adaptive-weighting"
  | "full-hybrid";

export interface AblationPrediction {
  id: HybridAblationId;
  regions: ChordEvent[];
  diagnostics?: HybridFusionDiagnostics;
}

export interface DisagreementWindow {
  trackId: string;
  startSeconds: number;
  endSeconds: number;
  ruleTopCandidate: string;
  learnedTopCandidate: string | null;
  hybridFinalLabel: string;
  ruleConfidence: number;
  learnedConfidence: number;
  learnedEntropy: number;
  boundaryProbability: number;
}

export interface TrackEngineComparison {
  track: Omit<PreparedEvaluationTrack, "samples" | "audioHash">;
  observationWindowCount: number;
  modelIdentity: EvaluationModelIdentity;
  predictions: Partial<Record<EngineId, EnginePrediction>>;
  ablations: AblationPrediction[];
  disagreementWindows: DisagreementWindow[];
  execution: {
    actualHybridDecoderInvoked: boolean;
    actualTemporalDecoderCompleted: boolean;
    sharedLearnedResponseObject: boolean;
    hybridImplementation: "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony";
    productionDecoder: "src/chordAnalysis.ts#decodeHarmonyObservations";
    productionObservationBuilder: "src/chordAnalysis.ts#createHarmonyObservations";
  };
}

export interface TrackComparisonOptions {
  modelIdentity: EvaluationModelIdentity;
  hybridSettings?: HybridHarmonySettings;
  mlOnlyTransitionPenalty?: number;
  timeoutMs?: number;
  includeLegacyRegionHybrid?: boolean;
  runAblations?: boolean;
  maxDisagreementWindows?: number;
}

export type HybridRunner = typeof runHybridHarmony;

export interface TrackComparisonDependencies {
  provider: LearnedHarmonyProvider;
  hybridRunner?: HybridRunner;
  now?: () => number;
  heapUsed?: () => number | null;
}

const ROOT_NAMES = [
  "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B",
];
const QUALITY_SUFFIXES = ["", "m", "7"];
const NO_CHORD_STATE = 36;
const STATE_COUNT = 37;
const EPSILON = 1e-8;

function clamp(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

function mean(values: number[]): number {
  return values.length
    ? values.reduce((sum, value) => sum + value, 0) / values.length
    : 0;
}

function stateLabel(state: number): string {
  if (state === NO_CHORD_STATE) return "N";
  return `${ROOT_NAMES[Math.floor(state / 3)]}${QUALITY_SUFFIXES[state % 3]}`;
}

function stateProbability(response: LearnedHarmonyResponse, frame: number, state: number): number {
  if (state === NO_CHORD_STATE) {
    return response.noChordProbabilities[frame] ?? 0;
  }
  const root = Math.floor(state / 3);
  const quality = state % 3;
  return (1 - (response.noChordProbabilities[frame] ?? 0))
    * (response.rootProbabilities[frame]?.[root] ?? 0)
    * (response.qualityProbabilities[frame]?.[quality] ?? 0);
}

/**
 * Evaluation-only ML decoder. It uses all 37 learned states (12 roots x
 * maj/min/7 plus N), log emissions, and a uniform switch penalty. It does not
 * consume rule scores, bass evidence, beat information, or key context.
 */
export function decodeLearnedOnly(
  response: LearnedHarmonyResponse,
  duration: number,
  transitionPenalty = DEFAULT_ML_ONLY_TRANSITION_PENALTY,
): ChordEvent[] {
  const frameCount = response.frameTimes.length;
  if (!frameCount || duration <= 0) return [];
  let previous = new Float64Array(STATE_COUNT);
  const backPointers = new Int32Array(frameCount * STATE_COUNT);

  for (let state = 0; state < STATE_COUNT; state += 1) {
    previous[state] = Math.log(stateProbability(response, 0, state) + EPSILON);
  }

  for (let frame = 1; frame < frameCount; frame += 1) {
    let bestPriorState = 0;
    for (let state = 1; state < STATE_COUNT; state += 1) {
      if (previous[state] > previous[bestPriorState]) bestPriorState = state;
    }
    const bestSwitchScore = previous[bestPriorState] - transitionPenalty;
    const next = new Float64Array(STATE_COUNT);
    for (let state = 0; state < STATE_COUNT; state += 1) {
      const stay = previous[state];
      const useStay = stay >= bestSwitchScore;
      next[state] = (useStay ? stay : bestSwitchScore)
        + Math.log(stateProbability(response, frame, state) + EPSILON);
      backPointers[frame * STATE_COUNT + state] = useStay ? state : bestPriorState;
    }
    previous = next;
  }

  let state = 0;
  for (let candidate = 1; candidate < STATE_COUNT; candidate += 1) {
    if (previous[candidate] > previous[state]) state = candidate;
  }
  const states = new Int32Array(frameCount);
  states[frameCount - 1] = state;
  for (let frame = frameCount - 1; frame > 0; frame -= 1) {
    states[frame - 1] = backPointers[frame * STATE_COUNT + states[frame]];
  }

  const regions: ChordEvent[] = [];
  let firstFrame = 0;
  for (let frame = 1; frame <= frameCount; frame += 1) {
    if (frame < frameCount && states[frame] === states[firstFrame]) continue;
    const start = firstFrame === 0
      ? 0
      : clamp(response.frameTimes[firstFrame], 0, duration);
    const end = frame === frameCount
      ? duration
      : clamp(response.frameTimes[frame], start, duration);
    if (end > start) {
      const probabilities = Array.from(
        { length: frame - firstFrame },
        (_unused, index) => stateProbability(
          response,
          firstFrame + index,
          states[firstFrame],
        ),
      );
      regions.push({
        start,
        end,
        name: stateLabel(states[firstFrame]),
        confidence: clamp(mean(probabilities)),
      });
    }
    firstFrame = frame;
  }
  return regions;
}

function argmax(values: number[]): number {
  let best = 0;
  for (let index = 1; index < values.length; index += 1) {
    if (values[index] > values[best]) best = index;
  }
  return best;
}

function averageRows(rows: number[][], indices: number[]): number[] {
  const width = rows[0]?.length ?? 0;
  const averaged = new Array<number>(width).fill(0);
  for (const index of indices) {
    for (let column = 0; column < width; column += 1) {
      averaged[column] += rows[index]?.[column] ?? 0;
    }
  }
  return averaged.map((value) => value / Math.max(1, indices.length));
}

/**
 * Faithful evaluation-only reconstruction of the region-level implementation at
 * LEGACY_REGION_HYBRID_SOURCE_COMMIT. It averages ML output inside completed
 * rule regions and chooses between two final labels; it never enters Viterbi.
 */
export function decodeLegacyRegionHybrid(
  response: LearnedHarmonyResponse,
  ruleRegions: ChordEvent[],
  settings: HybridHarmonySettings,
): ChordEvent[] {
  const regions: ChordEvent[] = [];
  ruleRegions.forEach((region, regionIndex) => {
    const indices = response.frameTimes.flatMap((time, index) =>
      time >= region.start && time < region.end ? [index] : []);
    if (!indices.length) {
      regions.push({ ...region });
      return;
    }
    const roots = averageRows(response.rootProbabilities, indices);
    const qualities = averageRows(response.qualityProbabilities, indices);
    const root = argmax(roots);
    const quality = argmax(qualities);
    const learnedConfidence = (roots[root] ?? 0) * (qualities[quality] ?? 0);
    const learnedLabel = `${ROOT_NAMES[root]}${QUALITY_SUFFIXES[quality]}`;
    const ruleScore = settings.ruleObservationWeight * region.confidence;
    const learnedScore = settings.learnedChordWeight * learnedConfidence;
    let start = region.start;
    if (settings.allowLearnedBoundaryBackdating && regionIndex > 0) {
      const preceding = response.frameTimes.flatMap((time, index) =>
        time >= region.start - 0.3 && time < region.start ? [index] : []);
      for (const index of preceding) {
        if ((response.boundaryProbabilities[index] ?? 0) > 0.6
          && response.frameTimes[index] < start) {
          start = response.frameTimes[index];
        }
      }
    }
    if (regions.length && start < regions.at(-1)!.end) {
      regions.at(-1)!.end = start;
    }
    regions.push({
      ...region,
      start,
      name: learnedScore > ruleScore ? learnedLabel : region.name,
    });
  });
  return regions;
}

function regionAt(regions: ChordEvent[], time: number): ChordEvent | undefined {
  return regions.find((region, index) =>
    time >= region.start
      && (time < region.end
        || (index === regions.length - 1 && time <= region.end)));
}

class CapturingProvider implements LearnedHarmonyProvider {
  readonly id: string;
  response: LearnedHarmonyResponse | null = null;

  constructor(private readonly inner: LearnedHarmonyProvider) {
    this.id = inner.id;
  }

  isAvailable(): Promise<boolean> {
    return this.inner.isAvailable();
  }

  getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return this.inner.getMetadata();
  }

  async predict(
    request: LearnedHarmonyRequest,
    signal?: AbortSignal,
  ): Promise<LearnedHarmonyResponse> {
    const response = await this.inner.predict(request, signal);
    this.response = response;
    return response;
  }
}

export function buildEvaluationRequest(
  track: PreparedEvaluationTrack,
  observationPackage: HarmonyObservationPackage,
  identity: EvaluationModelIdentity,
): LearnedHarmonyRequest | null {
  const frames = observationPackage.learnedFeatures;
  if (!frames.frameTimes.length) return null;
  return buildFeaturePackage(
    {
      audioHash: track.audioHash,
      sectionStartSeconds: frames.frameTimes[0],
      sectionEndSeconds: Math.max(
        observationPackage.duration,
        frames.frameTimes.at(-1)! + 0.25,
      ),
      frameTimes: frames.frameTimes,
      harmonicChroma: frames.harmonicChroma,
      bassChroma: frames.bassChroma,
      onsetStrength: frames.onsetStrength,
    },
    {
      modelVersion: identity.modelVersion,
      modelChecksum: identity.modelChecksum,
      expectedFeatureVersion: identity.featureVersion,
    },
    `hybrid-eval-${track.trackId}`,
  ).request;
}

function diagnosticSummary(
  diagnostics: Awaited<ReturnType<HybridRunner>>["diagnostics"],
): EnginePrediction["diagnostics"] {
  return {
    averageLearnedEntropy: diagnostics.learnedProbabilityEntropy,
    ruleLearnedAgreementRate: diagnostics.ruleLearnedAgreementRate,
    ruleLearnedDisagreements: diagnostics.ruleLearnedDisagreements,
    changedTopCandidateWindows: diagnostics.changedTopCandidateWindows,
    learnedBoundaryPeaksConsidered: diagnostics.boundaryPeakCount,
    effectiveLearnedWeightAverage: diagnostics.effectiveLearnedWeightAverage,
    effectiveLearnedWeightMinimum: diagnostics.effectiveLearnedWeightMinimum,
    effectiveLearnedWeightMaximum: diagnostics.effectiveLearnedWeightMaximum,
    alignedLearnedWindows: diagnostics.alignedLearnedWindows,
    missingLearnedWindows: diagnostics.missingLearnedWindows,
  };
}

function runAblations(
  observationPackage: HarmonyObservationPackage,
  ruleAnalysis: ChordAnalysisResult,
  response: LearnedHarmonyResponse,
  settings: HybridHarmonySettings,
  sourceMode: "full-mix" | "guitar-focused",
): AblationPrediction[] {
  const variants: Array<{
    id: Exclude<HybridAblationId, "rule-only" | "full-hybrid">;
    settings: HybridHarmonySettings;
    adaptiveWeighting: boolean;
  }> = [
    {
      id: "hybrid-chord-probabilities-only",
      settings: {
        ...settings,
        learnedBoundaryWeight: 0,
        bassRootWeight: 0,
      },
      adaptiveWeighting: true,
    },
    {
      id: "hybrid-without-boundary",
      settings: { ...settings, learnedBoundaryWeight: 0 },
      adaptiveWeighting: true,
    },
    {
      id: "hybrid-without-adaptive-weighting",
      settings,
      adaptiveWeighting: false,
    },
  ];
  const results: AblationPrediction[] = [{
    id: "rule-only",
    regions: ruleAnalysis.regions,
  }];
  for (const variant of variants) {
    const fused = fuseHybridObservations(
      observationPackage.observations,
      response,
      variant.settings,
      {
        sourceMode,
        adaptiveWeighting: variant.adaptiveWeighting,
      },
    );
    results.push({
      id: variant.id,
      regions: decodeHarmonyObservations({
        ...observationPackage,
        observations: fused.observations,
      }).regions,
      diagnostics: fused.diagnostics,
    });
  }
  const full = fuseHybridObservations(
    observationPackage.observations,
    response,
    settings,
    { sourceMode },
  );
  results.push({
    id: "full-hybrid",
    regions: decodeHarmonyObservations({
      ...observationPackage,
      observations: full.observations,
    }).regions,
    diagnostics: full.diagnostics,
  });
  return results;
}

function makeDisagreementWindows(
  track: PreparedEvaluationTrack,
  observationPackage: HarmonyObservationPackage,
  response: LearnedHarmonyResponse,
  hybridRegions: ChordEvent[],
  limit: number,
): DisagreementWindow[] {
  const aligned = alignLearnedEvidence(observationPackage.observations, response);
  return observationPackage.observations.flatMap((observation, index) => {
    if (index >= aligned.length) return [];
    const learned = aligned[index];
    const finalLabel = regionAt(hybridRegions, observation.start)?.name
      ?? observation.bestChord;
    if (learned.topChord === observation.bestChord
      && finalLabel === observation.bestChord) return [];
    return [{
      trackId: track.trackId,
      startSeconds: observation.start + track.originalOffsetSeconds,
      endSeconds: observation.end + track.originalOffsetSeconds,
      ruleTopCandidate: observation.bestChord,
      learnedTopCandidate: learned.topChord,
      hybridFinalLabel: finalLabel,
      ruleConfidence: observation.confidence,
      learnedConfidence: learned.topChordConfidence,
      learnedEntropy: learned.entropy,
      boundaryProbability: learned.boundaryProbability,
    }];
  }).slice(0, limit);
}

export async function compareTrackEngines(
  track: PreparedEvaluationTrack,
  dependencies: TrackComparisonDependencies,
  options: TrackComparisonOptions,
): Promise<TrackEngineComparison> {
  if (track.sampleRate <= 0 || !track.samples.length || track.duration <= 0) {
    throw new Error(`${track.trackId}: evaluation audio is empty`);
  }
  if (options.modelIdentity.featureVersion !== FEATURE_VERSION) {
    throw new Error(
      `${track.trackId}: model feature version ${options.modelIdentity.featureVersion}`
        + ` does not match ${FEATURE_VERSION}`,
    );
  }

  const now = dependencies.now ?? (() => performance.now());
  const heapUsed = dependencies.heapUsed
    ?? (() => typeof process === "undefined" ? null : process.memoryUsage().heapUsed);
  let peakHeapBytes = heapUsed();
  const sampleHeap = () => {
    const current = heapUsed();
    if (current !== null) peakHeapBytes = Math.max(peakHeapBytes ?? current, current);
  };
  const settings = options.hybridSettings ?? CONSERVATIVE_HYBRID_SETTINGS;
  const hybridRunner = dependencies.hybridRunner ?? runHybridHarmony;

  const observationStart = now();
  const observationPackage = createHarmonyObservations(
    track.samples,
    track.sampleRate,
    {},
    { source: track.sourceMode === "full-mix" ? "full-mix" : "guitar-only" },
  );
  const observationMilliseconds = now() - observationStart;
  sampleHeap();

  const ruleDecodeStart = now();
  const ruleAnalysis = decodeHarmonyObservations(observationPackage);
  const ruleDecodeMilliseconds = now() - ruleDecodeStart;
  sampleHeap();
  const predictions: TrackEngineComparison["predictions"] = {
    "rule-only": {
      engineId: "rule-only",
      regions: ruleAnalysis.regions,
      chordAnalysis: ruleAnalysis,
      usedLearned: false,
      timing: {
        observationMilliseconds,
        inferenceMilliseconds: 0,
        decodeMilliseconds: ruleDecodeMilliseconds,
        endToEndMilliseconds: observationMilliseconds + ruleDecodeMilliseconds,
        peakHeapBytes,
      },
    },
  };

  const request = buildEvaluationRequest(track, observationPackage, options.modelIdentity);
  const capturingProvider = new CapturingProvider(dependencies.provider);
  let actualHybridDecoderInvoked = false;
  const hybridStart = now();
  actualHybridDecoderInvoked = true;
  const hybrid = await hybridRunner({
    provider: capturingProvider,
    request,
    ruleBased: {
      pipelineVersion: PRODUCTION_RULE_ENGINE,
      regions: ruleAnalysis.regions,
      observationPackage,
      analysisResult: ruleAnalysis,
    },
    settings,
    enabled: true,
    timeoutMs: options.timeoutMs,
    sourceMode: track.sourceMode,
  });
  const hybridElapsed = now() - hybridStart;
  sampleHeap();
  predictions["observation-hybrid"] = {
    engineId: "observation-hybrid",
    regions: hybrid.regions,
    chordAnalysis: hybrid.chordAnalysis,
    usedLearned: hybrid.usedLearned,
    fallbackReason: hybrid.fallbackReason,
    probabilitySourceId: capturingProvider.response
      ? `${capturingProvider.response.requestId}:${capturingProvider.response.modelChecksum}`
      : undefined,
    timing: {
      observationMilliseconds,
      inferenceMilliseconds: hybrid.diagnostics.inferenceMs ?? 0,
      decodeMilliseconds: hybrid.diagnostics.decoderMs ?? (
        hybrid.usedLearned ? hybridElapsed : 0
      ),
      // This mirrors the actual application request: rule decode is retained so
      // every learned failure can return the exact already-computed rule result.
      endToEndMilliseconds:
        observationMilliseconds + ruleDecodeMilliseconds + hybridElapsed,
      peakHeapBytes,
    },
    diagnostics: diagnosticSummary(hybrid.diagnostics),
  };

  let sharedLearnedResponseObject = false;
  const response = capturingProvider.response;
  if (hybrid.usedLearned && response) {
    const mlDecodeStart = now();
    const mlOnlyRegions = decodeLearnedOnly(
      response,
      observationPackage.duration,
      options.mlOnlyTransitionPenalty ?? DEFAULT_ML_ONLY_TRANSITION_PENALTY,
    );
    const mlDecodeMilliseconds = now() - mlDecodeStart;
    sampleHeap();
    const probabilitySourceId = `${response.requestId}:${response.modelChecksum}`;
    predictions["ml-only"] = {
      engineId: "ml-only",
      regions: mlOnlyRegions,
      usedLearned: true,
      probabilitySourceId,
      timing: {
        observationMilliseconds,
        inferenceMilliseconds: response.diagnostics.inferenceMilliseconds,
        decodeMilliseconds: mlDecodeMilliseconds,
        endToEndMilliseconds: observationMilliseconds
          + response.diagnostics.inferenceMilliseconds
          + mlDecodeMilliseconds,
        peakHeapBytes,
      },
    };
    // CapturingProvider returns this exact object to runHybridHarmony; ML-only
    // receives the same reference here rather than a second inference or copy.
    sharedLearnedResponseObject = response === capturingProvider.response;

    if (options.includeLegacyRegionHybrid) {
      const legacyStart = now();
      const regions = decodeLegacyRegionHybrid(response, ruleAnalysis.regions, settings);
      const legacyDecodeMilliseconds = now() - legacyStart;
      predictions["legacy-region-hybrid"] = {
        engineId: "legacy-region-hybrid",
        regions,
        usedLearned: true,
        probabilitySourceId,
        timing: {
          observationMilliseconds,
          inferenceMilliseconds: response.diagnostics.inferenceMilliseconds,
          decodeMilliseconds: legacyDecodeMilliseconds,
          endToEndMilliseconds: observationMilliseconds
            + ruleDecodeMilliseconds
            + response.diagnostics.inferenceMilliseconds
            + legacyDecodeMilliseconds,
          peakHeapBytes,
        },
      };
    }
  }

  const ablations = options.runAblations && hybrid.usedLearned && response
    ? runAblations(
      observationPackage,
      ruleAnalysis,
      response,
      settings,
      track.sourceMode,
    )
    : [];
  const disagreementWindows = hybrid.usedLearned && response
    ? makeDisagreementWindows(
      track,
      observationPackage,
      response,
      hybrid.regions,
      options.maxDisagreementWindows ?? 200,
    )
    : [];

  return {
    track: {
      datasetId: track.datasetId,
      splitName: track.splitName,
      trackId: track.trackId,
      artist: track.artist,
      title: track.title,
      sourceType: track.sourceType,
      sourceMode: track.sourceMode,
      sampleRate: track.sampleRate,
      duration: observationPackage.duration,
      originalOffsetSeconds: track.originalOffsetSeconds,
      bpm: track.bpm,
      referenceRegions: track.referenceRegions,
      referenceChecksum: track.referenceChecksum,
    },
    observationWindowCount: observationPackage.observations.length,
    modelIdentity: options.modelIdentity,
    predictions,
    ablations,
    disagreementWindows,
    execution: {
      actualHybridDecoderInvoked,
      actualTemporalDecoderCompleted:
        hybrid.usedLearned && Boolean(hybrid.chordAnalysis),
      sharedLearnedResponseObject,
      hybridImplementation:
        "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony",
      productionDecoder:
        "src/chordAnalysis.ts#decodeHarmonyObservations",
      productionObservationBuilder:
        "src/chordAnalysis.ts#createHarmonyObservations",
    },
  };
}
