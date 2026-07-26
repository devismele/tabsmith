/**
 * Cache identity for the development-only bundled experiment. Kept separate
 * from the weight import so computing a cache key never loads model parameters.
 * Export tooling must update these values atomically with the weights file.
 */
export const LEARNED_MODEL_VERSION = "temporal-baseline-real-v1";
export const LEARNED_MODEL_CHECKSUM =
  "a8719a40942eb8afe700fd7ff003353abbc1444344a0eb4c74e2a35a8b51cb34";
