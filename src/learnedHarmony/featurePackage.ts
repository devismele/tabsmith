import { CONTRACT, CONTRACT_VERSION, FEATURE_VERSION } from "./contract";
import type { LearnedHarmonyRequest } from "./types";

// Converts existing Tabsmith analysis data into an inference request. It does not
// recompute anything already available (harmonic/bass chroma, onset, beats). Any
// optional feature that is not supplied is reported as unavailable and OMITTED —
// never filled with misleading fake data.

export interface FeatureSource {
  audioHash: string;
  sectionStartSeconds: number;
  sectionEndSeconds: number;
  frameTimes: number[];
  harmonicChroma: number[][];
  bassChroma: number[][];
  onsetStrength: number[];
  beatPhase?: number[];
  downbeatProbability?: number[];
  melodyPitchClass?: number[][];
  melodyConfidence?: number[];
  keyContext?: number[][];
}

export interface ModelBinding {
  modelVersion: string;
  modelChecksum: string;
  expectedFeatureVersion: string;
}

export interface FeaturePackage {
  request: LearnedHarmonyRequest;
  availableFeatures: string[];
  unavailableFeatures: string[];
}

const OPTIONAL_KEYS = [
  "beatPhase",
  "downbeatProbability",
  "melodyPitchClass",
  "melodyConfidence",
  "keyContext",
] as const;

export function buildFeaturePackage(source: FeatureSource, binding: ModelBinding, requestId: string): FeaturePackage {
  for (const key of CONTRACT.requiredFeatures) {
    const value = (source as unknown as Record<string, unknown>)[key];
    if (value === undefined || value === null) throw new Error(`Feature package is missing required feature "${key}"`);
  }

  const available: string[] = [...CONTRACT.requiredFeatures];
  const unavailable: string[] = [];
  const request: LearnedHarmonyRequest = {
    contractVersion: CONTRACT_VERSION,
    requestId,
    audioHash: source.audioHash,
    sectionStartSeconds: source.sectionStartSeconds,
    sectionEndSeconds: source.sectionEndSeconds,
    featureVersion: FEATURE_VERSION,
    frameTimes: source.frameTimes,
    harmonicChroma: source.harmonicChroma,
    bassChroma: source.bassChroma,
    onsetStrength: source.onsetStrength,
    modelMetadata: {
      modelVersion: binding.modelVersion,
      modelChecksum: binding.modelChecksum,
      expectedFeatureVersion: binding.expectedFeatureVersion,
    },
  };

  for (const key of OPTIONAL_KEYS) {
    const value = source[key];
    if (value !== undefined) {
      (request as unknown as Record<string, unknown>)[key] = value;
      available.push(key);
    } else {
      unavailable.push(key);
    }
  }

  return { request, availableFeatures: available, unavailableFeatures: unavailable };
}

export function describeFeatureAvailability(pkg: FeaturePackage): { availableFeatures: string[]; unavailableFeatures: string[] } {
  return { availableFeatures: pkg.availableFeatures, unavailableFeatures: pkg.unavailableFeatures };
}
