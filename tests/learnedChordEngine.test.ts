import { test, expect } from "vitest";
import {
  createHarmonyObservations,
  decodeHarmonyObservations,
} from "../src/chordAnalysis";
import { applyHybridChordEngine } from "../src/learnedHarmony/chordEngine";

test("hybrid chord engine runs end-to-end through the temporal decoder", async () => {
  const sr = 22050;
  const samples = new Float32Array(sr * 3);
  // A major-ish content (A + C# + E) so chroma is non-trivial.
  for (let i = 0; i < samples.length; i += 1) {
    const t = i / sr;
    samples[i] = 0.25 * (Math.sin(2 * Math.PI * 220 * t) + Math.sin(2 * Math.PI * 277.18 * t) + Math.sin(2 * Math.PI * 329.63 * t));
  }
  const observationPackage = createHarmonyObservations(samples, sr);
  const ruleAnalysis = decodeHarmonyObservations(observationPackage);
  const result = await applyHybridChordEngine(observationPackage, ruleAnalysis, {
    audioHash: "testhash01",
    enabled: true,
  });

  expect(result.regions.length).toBeGreaterThan(0);
  expect(result.usedLearned).toBe(true);
  expect(result.fallbackReason).toBeUndefined();
  expect(result.chordAnalysis.decoderAlternatives?.reducedLatency).toEqual(result.regions);
}, 30000);

test("disabled hybrid returns the exact rule analysis without loading a provider", async () => {
  const samples = new Float32Array(8000);
  const observationPackage = createHarmonyObservations(samples, 8000);
  const ruleAnalysis = decodeHarmonyObservations(observationPackage);
  const result = await applyHybridChordEngine(observationPackage, ruleAnalysis, {
    audioHash: "disabled-test",
    enabled: false,
  });
  expect(result.regions).toBe(ruleAnalysis.regions);
  expect(result.chordAnalysis).toBe(ruleAnalysis);
  expect(result.fallbackReason).toBe("learned-disabled");
});
