// Runs the experimental learned chord engine over an already-computed rule-based
// result, reusing the frozen integration infra (featurePackage + provider +
// hybridDecoder). The hybrid decoder falls back to the rule-based regions on ANY
// failure (unavailable model, timeout, invalid output, version mismatch), so
// selecting this engine can never lose the job.
//
// The 6 MB model weights are dynamic-imported so they are code-split out of the
// default bundle and only fetched when a user opts into this engine.
import { extractHarmonyFrames } from "../chordAnalysis";
import type { ChordEvent } from "../types";
import { FEATURE_VERSION } from "./contract";
import { buildFeaturePackage, type FeatureSource, type ModelBinding } from "./featurePackage";
import { CONSERVATIVE_HYBRID_SETTINGS, runHybridHarmony } from "./hybridDecoder";
import { LearnedTcnProvider, type TcnWeights } from "./onnxProvider";
import type { HybridDiagnostics, RuleBasedHarmonyEvidence } from "./types";

let weightsPromise: Promise<TcnWeights> | null = null;
function loadWeights(): Promise<TcnWeights> {
  if (!weightsPromise) {
    weightsPromise = import("./model/app-chord-model.weights.json")
      .then((module) => (module.default ?? module) as unknown as TcnWeights);
  }
  return weightsPromise;
}

export interface LearnedChordEngineResult {
  regions: ChordEvent[];
  usedLearned: boolean;
  fallbackReason?: string;
  diagnostics: HybridDiagnostics;
}

export async function applyLearnedChordEngine(
  samples: Float32Array,
  sampleRate: number,
  ruleRegions: ChordEvent[],
  audioHash: string,
  durationSeconds: number,
): Promise<LearnedChordEngineResult> {
  const ruleBased: RuleBasedHarmonyEvidence = {
    pipelineVersion: "2026-07-harmonic-context-v3-reduced-latency",
    regions: ruleRegions,
  };
  const frames = extractHarmonyFrames(samples, sampleRate);
  if (!frames.frameTimes.length) {
    return { regions: ruleRegions, usedLearned: false, fallbackReason: "no-frames",
      diagnostics: { providerId: "onnx", providerAvailable: false, warnings: ["no frames"], comparison: [] } };
  }

  const weights = await loadWeights();
  const binding: ModelBinding = {
    modelVersion: weights.modelVersion,
    modelChecksum: weights.modelChecksum,
    expectedFeatureVersion: FEATURE_VERSION,
  };
  const source: FeatureSource = {
    audioHash,
    sectionStartSeconds: frames.frameTimes[0],
    sectionEndSeconds: Math.max(durationSeconds, frames.frameTimes[frames.frameTimes.length - 1] + 0.25),
    frameTimes: frames.frameTimes,
    harmonicChroma: frames.harmonicChroma,
    bassChroma: frames.bassChroma,
    onsetStrength: frames.onsetStrength,
  };
  const pkg = buildFeaturePackage(source, binding, `learned-${audioHash.slice(0, 8)}-${frames.frameTimes.length}`);

  const result = await runHybridHarmony({
    provider: new LearnedTcnProvider(weights),
    request: pkg.request,
    ruleBased,
    settings: CONSERVATIVE_HYBRID_SETTINGS,
    enabled: true,
  });
  return {
    regions: result.regions,
    usedLearned: result.usedLearned,
    fallbackReason: result.fallbackReason,
    diagnostics: result.diagnostics,
  };
}
