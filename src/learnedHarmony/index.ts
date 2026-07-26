// Experimental hybrid-harmony integration boundary. Disabled by default and
// compile-time excluded from release execution; ML-only remains evaluation-only.

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
  alignLearnedEvidence,
  fuseHybridObservations,
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
