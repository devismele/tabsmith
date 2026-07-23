// Experimental learned-harmony integration boundary.
//
// DISABLED BY DEFAULT. No model is bundled, no ML runtime is started, no UI is
// exposed, and the production chord engine (harmonic-context-v3-reduced-latency)
// is unchanged. This module only defines how a future model will communicate
// with Tabsmith so the real model can replace the mock provider without
// restructuring the desktop application.

export * from "./types";
export {
  CONTRACT,
  CONTRACT_VERSION,
  FEATURE_VERSION,
  ENGINES,
  validateRequest,
  validateResponse,
  validateModelManifest,
} from "./contract";
export {
  DisabledLearnedHarmonyProvider,
  MockLearnedHarmonyProvider,
  type MockProviderOptions,
} from "./provider";
export {
  EXPERIMENTAL_FLAG,
  isLearnedHarmonyEnabled,
  resolveLearnedHarmonyProvider,
  describeDevControl,
  type FlagContext,
} from "./flag";
export {
  buildFeaturePackage,
  describeFeatureAvailability,
  type FeatureSource,
  type FeaturePackage,
  type ModelBinding,
} from "./featurePackage";
export {
  runHybridHarmony,
  combineHybrid,
  CONSERVATIVE_HYBRID_SETTINGS,
  HYBRID_DECODER_VERSION,
  type RunHybridParams,
} from "./hybridDecoder";
export {
  buildExperimentalCacheKey,
  isProductionCacheable,
  isMockResult,
  assertNotStoredAsProduction,
  ENGINE_IDS,
  type ExperimentalEngine,
  type ExperimentalCacheKeyParams,
} from "./cacheKey";
export { summarizeDiagnostics, formatChordLabel, pitchName } from "./diagnostics";
