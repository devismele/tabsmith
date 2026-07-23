import { DisabledLearnedHarmonyProvider, MockLearnedHarmonyProvider } from "./provider";
import type { LearnedHarmonyProvider } from "./types";

export const EXPERIMENTAL_FLAG = "TABSMITH_EXPERIMENTAL_LEARNED_HARMONY";

export interface FlagContext {
  env?: Record<string, string | undefined>;
  isReleaseBuild?: boolean;
}

/**
 * Learned harmony is opt-in for developers only. It is enabled ONLY when the
 * hidden env flag is set AND the build is not a release build. Default: disabled.
 * Release builds can never turn it on, so nothing new is exposed to users.
 */
export function isLearnedHarmonyEnabled(context: FlagContext = {}): boolean {
  if (context.isReleaseBuild) return false;
  const env = context.env ?? {};
  return env[EXPERIMENTAL_FLAG] === "1";
}

/**
 * Resolves the active provider. When disabled (the default), no model files are
 * searched for and no inference runtime is started — the disabled provider is
 * inert. When enabled in development, the mock provider is used.
 */
export function resolveLearnedHarmonyProvider(context: FlagContext = {}): LearnedHarmonyProvider {
  return isLearnedHarmonyEnabled(context)
    ? new MockLearnedHarmonyProvider()
    : new DisabledLearnedHarmonyProvider();
}

export interface DevControlDescriptor {
  visible: boolean;
  label: string;
  subLabel: string;
  provider: string;
}

/** Describes the hidden dev diagnostics control. Never visible in release builds. */
export function describeDevControl(context: FlagContext = {}): DevControlDescriptor {
  const enabled = isLearnedHarmonyEnabled(context);
  return {
    visible: enabled,
    label: "Experimental harmony integration",
    subLabel: "Not for musical evaluation",
    provider: enabled ? "Mock" : "Disabled",
  };
}
