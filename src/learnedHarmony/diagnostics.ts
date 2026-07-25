import { CONTRACT } from "./contract";
import type { HybridDiagnostics } from "./types";

const PITCH_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const QUALITY_SUFFIX: Record<string, string> = { maj: "", min: "m", "7": "7" };

export function pitchName(index: number): string {
  return PITCH_NAMES[((index % 12) + 12) % 12];
}

export function formatChordLabel(rootIndex: number, qualityIndex: number): string {
  const quality = CONTRACT.vocabulary.qualities[qualityIndex] ?? "maj";
  return `${pitchName(rootIndex)}${QUALITY_SUFFIX[quality] ?? quality}`;
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if (values[i] > values[best]) best = i;
  return best;
}

/** Shannon entropy (nats) of a probability distribution — a learned-uncertainty signal. */
export function shannonEntropy(distribution: number[]): number {
  let entropy = 0;
  for (const p of distribution) if (p > 0) entropy -= p * Math.log(p);
  return entropy;
}

/** Counts boundary-probability peaks above 0.5 (local maxima) — a fragmentation signal. */
export function boundaryPeakCount(boundary: number[], threshold = 0.5): number {
  let count = 0;
  for (let i = 0; i < boundary.length; i += 1) {
    const value = boundary[i];
    if (value <= threshold) continue;
    const leftOk = i === 0 || value >= boundary[i - 1];
    const rightOk = i === boundary.length - 1 || value >= boundary[i + 1];
    if (leftOk && rightOk) count += 1;
  }
  return count;
}

/** Dev-only one-line summary. Diagnostic data only; never a musical claim. */
export function summarizeDiagnostics(diagnostics: HybridDiagnostics): string {
  const parts = [
    `provider=${diagnostics.providerId}`,
    `available=${diagnostics.providerAvailable}`,
    diagnostics.modelVersion ? `model=${diagnostics.modelVersion}` : null,
    diagnostics.fallbackReason ? `fallback=${diagnostics.fallbackReason}` : "fallback=none",
    diagnostics.inferenceMs !== undefined ? `inferMs=${diagnostics.inferenceMs}` : null,
    diagnostics.fusionMs !== undefined ? `fusionMs=${diagnostics.fusionMs.toFixed(1)}` : null,
    diagnostics.decoderMs !== undefined ? `decodeMs=${diagnostics.decoderMs.toFixed(1)}` : null,
    diagnostics.ruleLearnedDisagreements !== undefined ? `disagreements=${diagnostics.ruleLearnedDisagreements}` : null,
    diagnostics.effectiveLearnedWeightAverage !== undefined
      ? `avgMlWeight=${diagnostics.effectiveLearnedWeightAverage.toFixed(3)}`
      : null,
    diagnostics.boundaryPeakCount !== undefined ? `boundaryPeaks=${diagnostics.boundaryPeakCount}` : null,
  ].filter(Boolean);
  return `[learned-harmony diagnostics · dev-only] ${parts.join(" ")}`;
}
