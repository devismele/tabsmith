import type { HarmonyObservationPackage } from "../chordAnalysis";
import type { ChordAnalysisResult, ChordEvent } from "../types";

// Frozen learned-harmony v1 request/response contract plus internal hybrid types.
// The model remains development-only and is excluded from release builds.

export interface LearnedHarmonyRequest {
  contractVersion: number;
  requestId: string;

  audioHash: string;
  sectionStartSeconds: number;
  sectionEndSeconds: number;

  featureVersion: string;
  frameTimes: number[];

  harmonicChroma: number[][];
  bassChroma: number[][];
  onsetStrength: number[];

  // Optional context features. Absent => explicitly unavailable, never faked.
  beatPhase?: number[];
  downbeatProbability?: number[];
  melodyPitchClass?: number[][];
  melodyConfidence?: number[];
  keyContext?: number[][];

  modelMetadata: {
    modelVersion: string;
    modelChecksum: string;
    expectedFeatureVersion: string;
  };
}

export interface LearnedHarmonyResponse {
  contractVersion: number;
  requestId: string;

  modelVersion: string;
  modelChecksum: string;
  featureVersion: string;

  frameTimes: number[];

  rootProbabilities: number[][];      // T x 12
  qualityProbabilities: number[][];   // T x qualities
  noChordProbabilities: number[];     // T
  boundaryProbabilities: number[];    // T

  diagnostics: {
    inferenceMilliseconds: number;
    backend: "mock" | "onnx" | "python";
    warnings: string[];
  };
}

export interface LearnedHarmonyModelMetadata {
  modelVersion: string;
  modelChecksum: string;
  contractVersion: number;
  featureVersion: string;
  format: "mock" | "onnx" | "python";
  vocabulary: {
    roots: number;
    qualities: string[];
    hasNoChordHead: boolean;
    hasBoundaryHead: boolean;
  };
  inputShapes: { features: (string | number)[] };
  disclaimer: string;
}

export interface LearnedHarmonyProvider {
  readonly id: string;
  isAvailable(): Promise<boolean>;
  getMetadata(): Promise<LearnedHarmonyModelMetadata>;
  predict(request: LearnedHarmonyRequest, signal?: AbortSignal): Promise<LearnedHarmonyResponse>;
}

/** Evidence already produced by the existing rule-based pipeline. */
export interface RuleBasedHarmonyEvidence {
  pipelineVersion: string;
  regions: ChordEvent[];
  /** Pre-decoder rule evidence. Required for genuine hybrid fusion. */
  observationPackage?: HarmonyObservationPackage;
  /** Exact rule result returned unchanged on every learned failure. */
  analysisResult?: ChordAnalysisResult;
  frameTimes?: number[];
  /** Optional per-frame bass root pitch class (0..11) from the existing tracker. */
  bassRootPerFrame?: number[];
}

export interface HybridHarmonySettings {
  learnedChordWeight: number;
  learnedBoundaryWeight: number;
  bassRootWeight: number;
  ruleObservationWeight: number;
  allowLearnedBoundaryBackdating: boolean;
  minimumLearnedConfidence: number;
  maximumLearnedEntropy: number;
  protectRuleConfidenceAbove: number;
  maximumLearnedWeightFullMix: number;
  maximumLearnedWeightGuitarOnly: number;
}

export interface LearnedWindowEvidence {
  chordProbabilities: Record<string, number>;
  noChordProbability: number;
  boundaryProbability: number;
  /** Normalized to 0..1 across the root, quality and no-chord heads. */
  entropy: number;
  topChord: string | null;
  topChordConfidence: number;
  contributingFrameCount: number;
}

export interface HybridFusionContext {
  sourceMode?: "full-mix" | "guitar-focused";
  /**
   * Evaluation-only ablation switch. Production callers omit this, so adaptive
   * confidence/entropy/rule-protection weighting remains enabled.
   */
  adaptiveWeighting?: boolean;
}

export interface HybridFusionDiagnostics {
  alignedWindows: number;
  missingLearnedWindows: number;
  averageLearnedEntropy: number;
  ruleLearnedAgreementRate: number;
  ruleLearnedDisagreements: number;
  changedTopCandidateWindows: number;
  learnedBoundaryPeaksConsidered: number;
  effectiveLearnedWeight: {
    minimum: number;
    maximum: number;
    average: number;
  };
}

export interface HybridHarmonyInput {
  learned?: LearnedHarmonyResponse;
  ruleBased: RuleBasedHarmonyEvidence;
  settings: HybridHarmonySettings;
}

export type HybridFallbackReason =
  | "learned-disabled"
  | "provider-unavailable"
  | "no-feature-package"
  | "timeout"
  | "cancelled"
  | "invalid-response"
  | "version-mismatch"
  | "provider-error";

export interface HybridHarmonyResult {
  engine: "rule-based" | "hybrid-experimental";
  regions: ChordEvent[];
  chordAnalysis?: ChordAnalysisResult;
  usedLearned: boolean;
  fallbackReason?: HybridFallbackReason;
  diagnostics: HybridDiagnostics;
}

export interface HybridDiagnostics {
  providerId: string;
  providerAvailable: boolean;
  featureConstructionMs?: number;
  inferenceMs?: number;
  fusionMs?: number;
  decoderMs?: number;
  modelVersion?: string;
  modelChecksum?: string;
  featureVersion?: string;
  fallbackReason?: HybridFallbackReason;
  learnedProbabilityEntropy?: number;
  boundaryPeakCount?: number;
  ruleLearnedDisagreements?: number;
  ruleLearnedAgreementRate?: number;
  changedTopCandidateWindows?: number;
  finalRegionsChanged?: number;
  effectiveLearnedWeightMinimum?: number;
  effectiveLearnedWeightMaximum?: number;
  effectiveLearnedWeightAverage?: number;
  alignedLearnedWindows?: number;
  missingLearnedWindows?: number;
  sourceMode?: "full-mix" | "guitar-focused";
  warnings: string[];
  comparison: HybridComparisonRow[];
}

export interface HybridComparisonRow {
  startSeconds: number;
  endSeconds: number;
  ruleChord: string;
  learnedTopChord: string | null;
  hybridChord: string;
  ruleConfidence: number;
  learnedConfidence: number | null;
  boundaryProbability: number | null;
}

export interface ValidationResult {
  ok: boolean;
  errors: string[];
}

export interface ModelManifest {
  modelVersion: string;
  contractVersion: number;
  featureVersion: string;
  checksum: string;
  format: "onnx" | "python";
  vocabulary: {
    roots: number;
    qualities: string[];
    hasNoChordHead: boolean;
    hasBoundaryHead: boolean;
  };
  inputShapes: { features: (string | number)[] };
}
