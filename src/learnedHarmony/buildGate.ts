import type { ChordEngine } from "../types";

declare const __TABSMITH_EXPERIMENTAL_HYBRID__: boolean | undefined;

export const HYBRID_HARMONY_BUILD_ENABLED =
  typeof __TABSMITH_EXPERIMENTAL_HYBRID__ !== "undefined"
  && __TABSMITH_EXPERIMENTAL_HYBRID__ === true;

/** Compile-time release gate injected by Vite. Undefined (tests/Node) is off. */
export function isHybridHarmonyBuildEnabled(): boolean {
  return HYBRID_HARMONY_BUILD_ENABLED;
}

/**
 * Sanitizes UI and worker messages. A DOM mutation or hand-crafted worker
 * message cannot select hybrid when the compiled build gate is closed.
 */
export function resolveChordEngineSelection(
  requested: unknown,
  buildEnabled = isHybridHarmonyBuildEnabled(),
): ChordEngine {
  return buildEnabled && requested === "hybrid" ? "hybrid" : "rule";
}
