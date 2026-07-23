import contractJson from "../../shared/learned-harmony-contract.json";
import type {
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
  ModelManifest,
  ValidationResult,
} from "./types";

export const CONTRACT = contractJson;
export const CONTRACT_VERSION = contractJson.contractVersion;
export const FEATURE_VERSION = contractJson.featureVersion;
export const CHROMA_BINS = contractJson.limits.chromaBins;
export const QUALITY_COUNT = contractJson.vocabulary.qualities.length;
export const ENGINES = contractJson.engines;

function isFiniteNumberArray(values: unknown, length?: number): boolean {
  if (!Array.isArray(values)) return false;
  if (length !== undefined && values.length !== length) return false;
  return values.every((value) => typeof value === "number" && Number.isFinite(value));
}

function isMatrix(values: unknown, rows: number, columns: number): boolean {
  return Array.isArray(values) && values.length === rows && values.every((row) => isFiniteNumberArray(row, columns));
}

function isProbabilityArray(values: unknown, length: number): boolean {
  return isFiniteNumberArray(values, length) && (values as number[]).every((value) => value >= 0 && value <= 1);
}

function isProbabilityMatrix(values: unknown, rows: number, columns: number, sumError: number): boolean {
  if (!isMatrix(values, rows, columns)) return false;
  return (values as number[][]).every((row) => {
    if (!row.every((value) => value >= 0 && value <= 1)) return false;
    const sum = row.reduce((total, value) => total + value, 0);
    return Math.abs(sum - 1) <= sumError;
  });
}

function isStrictlyIncreasing(values: number[]): boolean {
  for (let i = 1; i < values.length; i += 1) if (values[i] <= values[i - 1]) return false;
  return true;
}

/** Validates an inference request; rejects incompatible input rather than guessing. */
export function validateRequest(request: LearnedHarmonyRequest): ValidationResult {
  const errors: string[] = [];
  if (request.contractVersion !== CONTRACT_VERSION) {
    errors.push(`contractVersion ${request.contractVersion} != expected ${CONTRACT_VERSION}`);
  }
  if (!request.requestId || typeof request.requestId !== "string") errors.push("requestId is required");
  if (request.featureVersion !== FEATURE_VERSION) {
    errors.push(`featureVersion "${request.featureVersion}" != expected "${FEATURE_VERSION}"`);
  }
  if (request.modelMetadata?.expectedFeatureVersion !== request.featureVersion) {
    errors.push("modelMetadata.expectedFeatureVersion does not match request featureVersion");
  }
  if (!(request.sectionEndSeconds > request.sectionStartSeconds)) {
    errors.push("sectionEndSeconds must be greater than sectionStartSeconds");
  }
  if ((request.sectionEndSeconds - request.sectionStartSeconds) > CONTRACT.limits.maxSectionSeconds) {
    errors.push("section exceeds maxSectionSeconds");
  }

  const frames = request.frameTimes;
  if (!isFiniteNumberArray(frames) || frames.length === 0) {
    errors.push("frameTimes must be a non-empty finite array");
    return { ok: false, errors };
  }
  if (frames.length > CONTRACT.limits.maxFrames) errors.push("frameTimes exceeds maxFrames");
  if (!isStrictlyIncreasing(frames)) errors.push("frameTimes must be strictly increasing");

  const T = frames.length;
  if (!isMatrix(request.harmonicChroma, T, CHROMA_BINS)) errors.push(`harmonicChroma must be ${T}x${CHROMA_BINS} finite`);
  if (!isMatrix(request.bassChroma, T, CHROMA_BINS)) errors.push(`bassChroma must be ${T}x${CHROMA_BINS} finite`);
  if (!isFiniteNumberArray(request.onsetStrength, T)) errors.push(`onsetStrength must be length ${T} finite`);

  // Optional features: if present they must be correctly shaped (never faked).
  if (request.beatPhase !== undefined && !isFiniteNumberArray(request.beatPhase, T)) errors.push("beatPhase length mismatch");
  if (request.downbeatProbability !== undefined && !isProbabilityArray(request.downbeatProbability, T)) errors.push("downbeatProbability invalid");
  if (request.melodyConfidence !== undefined && !isProbabilityArray(request.melodyConfidence, T)) errors.push("melodyConfidence invalid");
  if (request.melodyPitchClass !== undefined && !isMatrix(request.melodyPitchClass, T, CHROMA_BINS)) errors.push("melodyPitchClass shape mismatch");
  if (request.keyContext !== undefined && !isMatrix(request.keyContext, T, CHROMA_BINS)) errors.push("keyContext shape mismatch");

  return { ok: errors.length === 0, errors };
}

/** Validates a model response against the request it answers. */
export function validateResponse(response: LearnedHarmonyResponse, request: LearnedHarmonyRequest): ValidationResult {
  const errors: string[] = [];
  const sumError = CONTRACT.limits.maxProbabilitySumError;
  if (response.contractVersion !== CONTRACT_VERSION) errors.push("response contractVersion mismatch");
  if (response.requestId !== request.requestId) errors.push("response requestId does not match request");
  if (response.featureVersion !== request.featureVersion) errors.push("response featureVersion mismatch");
  if (!response.modelVersion) errors.push("response modelVersion is required");
  if (!response.modelChecksum) errors.push("response modelChecksum is required");

  const frames = response.frameTimes;
  if (!isFiniteNumberArray(frames) || frames.length !== request.frameTimes.length) {
    errors.push("response frameTimes must match request length");
    return { ok: false, errors };
  }
  if (!isStrictlyIncreasing(frames)) errors.push("response frameTimes must be strictly increasing");

  const T = frames.length;
  if (!isProbabilityMatrix(response.rootProbabilities, T, CHROMA_BINS, sumError)) errors.push("rootProbabilities invalid (shape/range/sum)");
  if (!isProbabilityMatrix(response.qualityProbabilities, T, QUALITY_COUNT, sumError)) errors.push("qualityProbabilities invalid (shape/range/sum)");
  if (!isProbabilityArray(response.noChordProbabilities, T)) errors.push("noChordProbabilities invalid (range)");
  if (!isProbabilityArray(response.boundaryProbabilities, T)) errors.push("boundaryProbabilities invalid (range)");

  const backend = response.diagnostics?.backend;
  if (backend !== "mock" && backend !== "onnx" && backend !== "python") errors.push("diagnostics.backend invalid");
  return { ok: errors.length === 0, errors };
}

export function validateModelManifest(manifest: ModelManifest): ValidationResult {
  const errors: string[] = [];
  if (manifest.contractVersion !== CONTRACT_VERSION) errors.push("manifest contractVersion mismatch");
  if (manifest.featureVersion !== FEATURE_VERSION) errors.push("manifest featureVersion mismatch");
  if (!manifest.checksum) errors.push("manifest checksum is required");
  if (manifest.format !== "onnx" && manifest.format !== "python") errors.push("manifest format must be onnx or python");
  if (manifest.vocabulary?.roots !== CONTRACT.vocabulary.roots) errors.push("manifest root vocabulary mismatch");
  return { ok: errors.length === 0, errors };
}
