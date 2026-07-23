import type { ChordEvent } from "../types";

// Application-side integration contract for a FUTURE learned-harmony model.
// The current synthetic TCN is a pipeline-validation model only: nothing here is
// wired into the production UI, no model is bundled, and the default provider is
// disabled. These types define how a real model will one day communicate with
// Tabsmith without restructuring the app.

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
  usedLearned: boolean;
  fallbackReason?: HybridFallbackReason;
  diagnostics: HybridDiagnostics;
}

export interface HybridDiagnostics {
  providerId: string;
  providerAvailable: boolean;
  featureConstructionMs?: number;
  inferenceMs?: number;
  decoderMs?: number;
  modelVersion?: string;
  modelChecksum?: string;
  featureVersion?: string;
  fallbackReason?: HybridFallbackReason;
  learnedProbabilityEntropy?: number;
  boundaryPeakCount?: number;
  ruleLearnedDisagreements?: number;
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
