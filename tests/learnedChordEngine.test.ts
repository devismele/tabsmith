import { test, expect } from "vitest";
import { applyLearnedChordEngine } from "../src/learnedHarmony/chordEngine";
import type { ChordEvent } from "../src/types";

test("learned chord engine runs end-to-end and uses the model", async () => {
  const sr = 22050;
  const samples = new Float32Array(sr * 3);
  // A major-ish content (A + C# + E) so chroma is non-trivial.
  for (let i = 0; i < samples.length; i += 1) {
    const t = i / sr;
    samples[i] = 0.25 * (Math.sin(2 * Math.PI * 220 * t) + Math.sin(2 * Math.PI * 277.18 * t) + Math.sin(2 * Math.PI * 329.63 * t));
  }
  const ruleRegions: ChordEvent[] = [{ start: 0, end: 3, name: "A", confidence: 0.5 }];

  const result = await applyLearnedChordEngine(samples, sr, ruleRegions, "testhash01", 3);

  expect(result.regions.length).toBeGreaterThan(0);
  // Versions match the bundled model, so the hybrid decoder should use it (not fall back).
  expect(result.usedLearned).toBe(true);
  expect(result.fallbackReason).toBeUndefined();
}, 30000);
