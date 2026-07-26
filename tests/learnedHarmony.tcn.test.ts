import { describe, test, expect } from "vitest";
import fixture from "../src/learnedHarmony/__fixtures__/tcn-parity.json";
import { CONTRACT_VERSION, FEATURE_VERSION, validateResponse } from "../src/learnedHarmony/contract";
import { LearnedTcnProvider, runTcn, type TcnWeights } from "../src/learnedHarmony/onnxProvider";
import type { LearnedHarmonyRequest } from "../src/learnedHarmony/types";

const weights = fixture as unknown as TcnWeights;

function maxAbsDiff(a: number[][], b: number[][]): number {
  let m = 0;
  for (let i = 0; i < a.length; i += 1) {
    for (let j = 0; j < a[i].length; j += 1) m = Math.max(m, Math.abs(a[i][j] - b[i][j]));
  }
  return m;
}

function buildRequest(T: number): LearnedHarmonyRequest {
  const chroma = Array.from({ length: T }, (_v, t) =>
    Array.from({ length: 12 }, (_w, k) => ((t + k) % 12) / 30));
  return {
    contractVersion: CONTRACT_VERSION,
    requestId: "test-req",
    audioHash: "hash",
    sectionStartSeconds: 0,
    sectionEndSeconds: 3,
    featureVersion: FEATURE_VERSION,
    frameTimes: Array.from({ length: T }, (_v, t) => t * 0.25),
    harmonicChroma: chroma,
    bassChroma: chroma,
    onsetStrength: Array.from({ length: T }, () => 0.2),
    modelMetadata: {
      modelVersion: weights.modelVersion,
      modelChecksum: weights.modelChecksum,
      expectedFeatureVersion: FEATURE_VERSION,
    },
  };
}

describe("TCN pure-TS forward pass", () => {
  test("matches PyTorch head logits (parity fixture)", () => {
    const sample = weights.paritySample!;
    const out = runTcn(weights, sample.input);
    expect(maxAbsDiff(out.root, sample.root)).toBeLessThan(1e-4);
    expect(maxAbsDiff(out.quality, sample.quality)).toBeLessThan(1e-4);
    expect(maxAbsDiff([out.nochord], [sample.nochord])).toBeLessThan(1e-4);
    expect(maxAbsDiff([out.boundary], [sample.boundary])).toBeLessThan(1e-4);
  });

  test("provider returns a contract-valid response", async () => {
    const provider = new LearnedTcnProvider(weights);
    expect(await provider.isAvailable()).toBe(true);
    const request = buildRequest(6);
    const response = await provider.predict(request);
    const validation = validateResponse(response, request);
    expect(validation.ok).toBe(true);
    expect(response.diagnostics.backend).toBe("onnx");
    // Root/quality are probability distributions.
    for (const row of response.rootProbabilities) {
      expect(Math.abs(row.reduce((a, b) => a + b, 0) - 1)).toBeLessThan(1e-5);
    }
  });
});
