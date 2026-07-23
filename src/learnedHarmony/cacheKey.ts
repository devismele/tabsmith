import { CONTRACT, CONTRACT_VERSION, FEATURE_VERSION } from "./contract";
import { HYBRID_DECODER_VERSION } from "./hybridDecoder";
import type { HybridHarmonySettings, LearnedHarmonyResponse } from "./types";

// Experimental cache entries are namespaced so they can NEVER collide with or
// contaminate production rule-based transcription cache entries. Mock results are
// never production-cacheable.

export const ENGINE_IDS = {
  ruleBased: CONTRACT.engines.ruleBased,
  learnedExperimental: CONTRACT.engines.learned,
  hybridExperimental: CONTRACT.engines.hybrid,
} as const;

export type ExperimentalEngine = typeof ENGINE_IDS.learnedExperimental | typeof ENGINE_IDS.hybridExperimental;

export interface ExperimentalCacheKeyParams {
  engine: ExperimentalEngine;
  audioHash: string;
  stemHash?: string;
  modelVersion: string;
  modelChecksum: string;
  ruleBasedPipelineVersion: string;
  hybridSettings: HybridHarmonySettings;
  sectionStartSeconds: number;
  sectionEndSeconds: number;
}

function hash(input: string): string {
  // FNV-1a 32-bit — deterministic, dependency-free.
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, "0");
}

/** Builds a namespaced experimental cache key embedding every version + weight input. */
export function buildExperimentalCacheKey(params: ExperimentalCacheKeyParams): string {
  const payload = JSON.stringify({
    engine: params.engine,
    audioHash: params.audioHash,
    stemHash: params.stemHash ?? null,
    featureVersion: FEATURE_VERSION,
    contractVersion: CONTRACT_VERSION,
    modelVersion: params.modelVersion,
    modelChecksum: params.modelChecksum,
    hybridDecoderVersion: HYBRID_DECODER_VERSION,
    hybridSettings: params.hybridSettings,
    ruleBasedPipelineVersion: params.ruleBasedPipelineVersion,
    section: [params.sectionStartSeconds, params.sectionEndSeconds],
  });
  return `experimental:${params.engine}:${hash(payload)}`;
}

/** Only genuine rule-based production results may enter the production cache. */
export function isProductionCacheable(engine: string, backend?: string): boolean {
  if (engine !== ENGINE_IDS.ruleBased) return false;
  return backend === undefined || backend === "onnx" || backend === "python";
}

/** A mock/experimental response must never be persisted as a production transcription. */
export function isMockResult(response: Pick<LearnedHarmonyResponse, "diagnostics">): boolean {
  return response.diagnostics.backend === "mock";
}

/** Throws if a mock/experimental result would be written to the production cache. */
export function assertNotStoredAsProduction(engine: string, response: Pick<LearnedHarmonyResponse, "diagnostics">): void {
  if (isMockResult(response) || !isProductionCacheable(engine, response.diagnostics.backend)) {
    throw new Error(`Refusing to store ${engine}/${response.diagnostics.backend} result as a production transcription.`);
  }
}
