import { LearnedTcnProvider, type TcnWeights } from "./onnxProvider";
import { DisabledLearnedHarmonyProvider, MockLearnedHarmonyProvider } from "./provider";
import type { LearnedHarmonyProvider } from "./types";
import { isHybridHarmonyBuildEnabled } from "./buildGate";

export const EXPERIMENTAL_FLAG = "TABSMITH_EXPERIMENTAL_LEARNED_HARMONY";

export interface FlagContext {
  env?: Record<string, string | undefined>;
  isReleaseBuild?: boolean;
}

/**
 * Hybrid harmony is opt-in for developers only. It is enabled ONLY when the
 * hidden env flag is set AND the build is not a release build. Default: disabled.
 * Release builds can never turn it on, so nothing new is exposed to users.
 */
export function isLearnedHarmonyEnabled(context: FlagContext = {}): boolean {
  if (context.isReleaseBuild) return false;
  // Explicit false is a test/dev-host override. In application code the injected
  // compile-time gate must also be open.
  if (context.isReleaseBuild !== false && !isHybridHarmonyBuildEnabled()) return false;
  const env = context.env ?? {};
  return env[EXPERIMENTAL_FLAG] === "1";
}

/**
 * Resolves the active provider. When disabled (the default), no model files are
 * searched for and no inference runtime is started — the disabled provider is
 * inert. When enabled in development: if exported TCN ``weights`` are supplied the
 * real (pure-TS) learned provider runs; otherwise the mock provider is used. The
 * caller loads the weights JSON (dev-only), keeping this resolver pure/tree-shakeable.
 */
export function resolveLearnedHarmonyProvider(
  context: FlagContext = {},
  weights?: TcnWeights,
): LearnedHarmonyProvider {
  if (!isLearnedHarmonyEnabled(context)) return new DisabledLearnedHarmonyProvider();
  return weights ? new LearnedTcnProvider(weights) : new MockLearnedHarmonyProvider();
}

export interface DevControlDescriptor {
  visible: boolean;
  label: string;
  subLabel: string;
  provider: string;
}

/** Describes the hidden hybrid dev control. Never visible in release builds. */
export function describeDevControl(context: FlagContext = {}): DevControlDescriptor {
  const enabled = isLearnedHarmonyEnabled(context);
  return {
    visible: enabled,
    label: "Hybrid ML — experimental",
    subLabel: "Rule decoder with learned observation evidence",
    provider: enabled ? "Mock" : "Disabled",
  };
}
