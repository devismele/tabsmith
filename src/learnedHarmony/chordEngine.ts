import type { HarmonyObservationPackage } from "../chordAnalysis";
import type { ChordAnalysisResult, ChordEvent } from "../types";
import { isHybridHarmonyBuildEnabled } from "./buildGate";
import { FEATURE_VERSION } from "./contract";
import { buildFeaturePackage, type FeatureSource, type ModelBinding } from "./featurePackage";
import { CONSERVATIVE_HYBRID_SETTINGS, runHybridHarmony } from "./hybridDecoder";
import { LEARNED_MODEL_CHECKSUM, LEARNED_MODEL_VERSION } from "./modelIdentity";
import { LearnedTcnProvider, type TcnWeights } from "./onnxProvider";
import type {
  HybridDiagnostics,
  HybridFallbackReason,
  HybridHarmonySettings,
  LearnedHarmonyProvider,
  RuleBasedHarmonyEvidence,
} from "./types";

let weightsPromise: Promise<TcnWeights> | null = null;
function loadWeights(): Promise<TcnWeights> {
  if (!weightsPromise) {
    weightsPromise = import("./model/app-chord-model.weights.json")
      .then((module) => (module.default ?? module) as unknown as TcnWeights);
  }
  return weightsPromise;
}

export interface HybridChordEngineResult {
  regions: ChordEvent[];
  chordAnalysis: ChordAnalysisResult;
  usedLearned: boolean;
  fallbackReason?: HybridFallbackReason;
  diagnostics: HybridDiagnostics;
}

export interface ApplyHybridChordEngineOptions {
  audioHash: string;
  enabled?: boolean;
  provider?: LearnedHarmonyProvider;
  settings?: HybridHarmonySettings;
  signal?: AbortSignal;
  timeoutMs?: number;
}

function directFallback(
  ruleAnalysis: ChordAnalysisResult,
  reason: HybridFallbackReason,
  warning?: string,
): HybridChordEngineResult {
  return {
    regions: ruleAnalysis.regions,
    chordAnalysis: ruleAnalysis,
    usedLearned: false,
    fallbackReason: reason,
    diagnostics: {
      providerId: "onnx",
      providerAvailable: false,
      fallbackReason: reason,
      sourceMode: ruleAnalysis.diagnostics.harmonyEvidenceSource === "full-mix"
        ? "full-mix"
        : "guitar-focused",
      warnings: warning ? [warning] : [],
      comparison: [],
    },
  };
}

/**
 * Runs the experimental model over rule observations and hands the fused
 * observations back to Tabsmith's production temporal decoder. Weight loading is
 * behind both the compile-time worker gate and this direct-call guard.
 */
export async function applyHybridChordEngine(
  observationPackage: HarmonyObservationPackage,
  ruleAnalysis: ChordAnalysisResult,
  options: ApplyHybridChordEngineOptions,
): Promise<HybridChordEngineResult> {
  const enabled = options.enabled ?? isHybridHarmonyBuildEnabled();
  if (!enabled) return directFallback(ruleAnalysis, "learned-disabled");
  if (!observationPackage.learnedFeatures.frameTimes.length) {
    return directFallback(ruleAnalysis, "no-feature-package");
  }

  let provider = options.provider;
  let binding: ModelBinding;
  try {
    if (provider) {
      const metadata = await provider.getMetadata();
      binding = {
        modelVersion: metadata.modelVersion,
        modelChecksum: metadata.modelChecksum,
        expectedFeatureVersion: FEATURE_VERSION,
      };
    } else {
      const weights = await loadWeights();
      if (weights.modelVersion !== LEARNED_MODEL_VERSION
        || weights.modelChecksum !== LEARNED_MODEL_CHECKSUM
        || weights.featureVersion !== FEATURE_VERSION) {
        return directFallback(
          ruleAnalysis,
          "version-mismatch",
          "Bundled model identity does not match its cache/build metadata.",
        );
      }
      provider = new LearnedTcnProvider(weights);
      binding = {
        modelVersion: weights.modelVersion,
        modelChecksum: weights.modelChecksum,
        expectedFeatureVersion: FEATURE_VERSION,
      };
    }
  } catch (error) {
    return directFallback(
      ruleAnalysis,
      "provider-error",
      error instanceof Error ? error.message : String(error),
    );
  }

  const featureStart = performance.now();
  const frames = observationPackage.learnedFeatures;
  const source: FeatureSource = {
    audioHash: options.audioHash,
    sectionStartSeconds: frames.frameTimes[0],
    sectionEndSeconds: Math.max(
      observationPackage.duration,
      frames.frameTimes[frames.frameTimes.length - 1] + 0.25,
    ),
    frameTimes: frames.frameTimes,
    harmonicChroma: frames.harmonicChroma,
    bassChroma: frames.bassChroma,
    onsetStrength: frames.onsetStrength,
  };
  let request;
  try {
    request = buildFeaturePackage(
      source,
      binding,
      `hybrid-${options.audioHash.slice(0, 8)}-${frames.frameTimes.length}`,
    ).request;
  } catch (error) {
    return directFallback(
      ruleAnalysis,
      "no-feature-package",
      error instanceof Error ? error.message : String(error),
    );
  }
  const featureConstructionMs = performance.now() - featureStart;
  const ruleBased: RuleBasedHarmonyEvidence = {
    pipelineVersion: "2026-07-harmonic-context-v3-reduced-latency",
    regions: ruleAnalysis.regions,
    observationPackage,
    analysisResult: ruleAnalysis,
  };
  const result = await runHybridHarmony({
    provider,
    request,
    ruleBased,
    settings: options.settings ?? CONSERVATIVE_HYBRID_SETTINGS,
    enabled,
    signal: options.signal,
    timeoutMs: options.timeoutMs,
    sourceMode: observationPackage.harmonyEvidenceSource === "full-mix"
      ? "full-mix"
      : "guitar-focused",
  });
  result.diagnostics.featureConstructionMs = featureConstructionMs;
  return {
    regions: result.regions,
    chordAnalysis: result.chordAnalysis ?? ruleAnalysis,
    usedLearned: result.usedLearned,
    fallbackReason: result.fallbackReason,
    diagnostics: result.diagnostics,
  };
}
