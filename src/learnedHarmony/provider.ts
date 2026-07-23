import { CHROMA_BINS, CONTRACT, CONTRACT_VERSION, FEATURE_VERSION, QUALITY_COUNT, validateRequest } from "./contract";
import type {
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "./types";

const MOCK_DISCLAIMER = "Synthetic/mock probabilities — not musical inference";

function abortableDelay(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Cancelled", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => { clearTimeout(timer); reject(new DOMException("Cancelled", "AbortError")); };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const sum = exps.reduce((total, value) => total + value, 0) || 1;
  return exps.map((value) => value / sum);
}

/** Default provider: learned harmony is off. It searches for nothing and runs nothing. */
export class DisabledLearnedHarmonyProvider implements LearnedHarmonyProvider {
  readonly id = "disabled";
  async isAvailable(): Promise<boolean> { return false; }
  async getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    throw new Error("Learned harmony is disabled; no model metadata is available.");
  }
  async predict(): Promise<LearnedHarmonyResponse> {
    throw new Error("Learned harmony is disabled.");
  }
}

export interface MockProviderOptions {
  latencyMs?: number;
  failWith?: string;
  /** Override the checksum a downstream cache key would embed. */
  modelChecksum?: string;
}

/**
 * Dev-only provider returning deterministic, clearly-labelled mock probabilities.
 * It validates the full contract and simulates latency/cancellation/failure so the
 * feature-extraction → request → response → hybrid-decoder → diagnostics path can
 * be exercised. It performs NO musical inference.
 */
export class MockLearnedHarmonyProvider implements LearnedHarmonyProvider {
  readonly id = "mock";
  constructor(private readonly options: MockProviderOptions = {}) {}

  async isAvailable(): Promise<boolean> { return true; }

  async getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return {
      modelVersion: "mock-temporal-baseline-v0",
      modelChecksum: this.options.modelChecksum ?? "mock-0000000000000000",
      contractVersion: CONTRACT_VERSION,
      featureVersion: FEATURE_VERSION,
      format: "mock",
      vocabulary: {
        roots: CONTRACT.vocabulary.roots,
        qualities: CONTRACT.vocabulary.qualities,
        hasNoChordHead: CONTRACT.vocabulary.hasNoChordHead,
        hasBoundaryHead: CONTRACT.vocabulary.hasBoundaryHead,
      },
      inputShapes: { features: ["batch", "frames", CHROMA_BINS * 2 + 1] },
      disclaimer: MOCK_DISCLAIMER,
    };
  }

  async predict(request: LearnedHarmonyRequest, signal?: AbortSignal): Promise<LearnedHarmonyResponse> {
    const validation = validateRequest(request);
    if (!validation.ok) throw new Error(`Mock provider received invalid request: ${validation.errors.join("; ")}`);

    const started = Date.now();
    await abortableDelay(this.options.latencyMs ?? 5, signal);
    if (this.options.failWith) throw new Error(this.options.failWith);

    const T = request.frameTimes.length;
    const rootProbabilities: number[][] = [];
    const qualityProbabilities: number[][] = [];
    const noChordProbabilities: number[] = [];
    const boundaryProbabilities: number[] = [];
    const qualityBias = Array.from({ length: QUALITY_COUNT }, (_v, index) => (index === 0 ? 1.2 : 0.4));

    for (let t = 0; t < T; t += 1) {
      // Deterministic "top chord" derived from the supplied chroma — plausible-looking
      // but explicitly not real inference.
      rootProbabilities.push(softmax(request.harmonicChroma[t].map((value) => value * 6)));
      qualityProbabilities.push(softmax(qualityBias));
      noChordProbabilities.push(Math.min(1, Math.max(0, 0.5 - request.onsetStrength[t])) * 0.1);
      const onset = request.onsetStrength[t] ?? 0;
      boundaryProbabilities.push(1 / (1 + Math.exp(-(onset * 4 - 2))));
    }

    return {
      contractVersion: CONTRACT_VERSION,
      requestId: request.requestId,
      modelVersion: "mock-temporal-baseline-v0",
      modelChecksum: this.options.modelChecksum ?? "mock-0000000000000000",
      featureVersion: FEATURE_VERSION,
      frameTimes: request.frameTimes.slice(),
      rootProbabilities,
      qualityProbabilities,
      noChordProbabilities,
      boundaryProbabilities,
      diagnostics: {
        inferenceMilliseconds: Date.now() - started,
        backend: "mock",
        warnings: [MOCK_DISCLAIMER],
      },
    };
  }
}
