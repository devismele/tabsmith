import { CONTRACT, validateResponse } from "./contract";
import { argmax, boundaryPeakCount, formatChordLabel, shannonEntropy } from "./diagnostics";
import type {
  HybridComparisonRow,
  HybridDiagnostics,
  HybridFallbackReason,
  HybridHarmonyResult,
  HybridHarmonySettings,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
  RuleBasedHarmonyEvidence,
} from "./types";
import type { ChordEvent } from "../types";

export const HYBRID_DECODER_VERSION: number = CONTRACT.hybridDecoderVersion;

/** Conservative defaults: the rule-based observation dominates. Used in tests only. */
export const CONSERVATIVE_HYBRID_SETTINGS: HybridHarmonySettings = {
  learnedChordWeight: 0.2,
  learnedBoundaryWeight: 0.2,
  bassRootWeight: 0.3,
  ruleObservationWeight: 1.0,
  allowLearnedBoundaryBackdating: false,
};

interface CombineOutput {
  regions: ChordEvent[];
  comparison: HybridComparisonRow[];
  entropy: number;
  boundaryPeaks: number;
  disagreements: number;
}

function averageRow(rows: number[][], indices: number[]): number[] {
  const width = rows[0]?.length ?? 0;
  const acc = new Array(width).fill(0);
  for (const i of indices) for (let c = 0; c < width; c += 1) acc[c] += rows[i][c];
  return acc.map((value) => value / (indices.length || 1));
}

/** Combines learned probabilities with rule-based regions; never rewrites the decoder. */
export function combineHybrid(
  learned: LearnedHarmonyResponse,
  ruleBased: RuleBasedHarmonyEvidence,
  settings: HybridHarmonySettings,
): CombineOutput {
  const times = learned.frameTimes;
  const comparison: HybridComparisonRow[] = [];
  const regions: ChordEvent[] = [];
  let disagreements = 0;

  const framesIn = (start: number, end: number): number[] => {
    const out: number[] = [];
    for (let i = 0; i < times.length; i += 1) if (times[i] >= start && times[i] < end) out.push(i);
    return out;
  };

  ruleBased.regions.forEach((region, regionIndex) => {
    const indices = framesIn(region.start, region.end);
    let learnedLabel: string | null = null;
    let learnedConfidence: number | null = null;
    let boundaryProbability: number | null = null;
    let start = region.start;

    if (indices.length) {
      const rootAvg = averageRow(learned.rootProbabilities, indices);
      const qualityAvg = averageRow(learned.qualityProbabilities, indices);
      const rootIndex = argmax(rootAvg);
      const qualityIndex = argmax(qualityAvg);
      learnedLabel = formatChordLabel(rootIndex, qualityIndex);
      learnedConfidence = rootAvg[rootIndex] * qualityAvg[qualityIndex];
      boundaryProbability = Math.max(...indices.map((i) => learned.boundaryProbabilities[i]));

      if (settings.allowLearnedBoundaryBackdating && regionIndex > 0) {
        const preIndices = framesIn(region.start - 0.3, region.start);
        for (const i of preIndices) {
          if (learned.boundaryProbabilities[i] > 0.6 && times[i] < start) start = times[i];
        }
      }
    }

    const ruleScore = settings.ruleObservationWeight * region.confidence;
    const learnedScore = settings.learnedChordWeight * (learnedConfidence ?? 0);
    const hybridLabel = learnedLabel && learnedScore > ruleScore ? learnedLabel : region.name;
    if (learnedLabel && learnedLabel !== region.name) disagreements += 1;

    if (regions.length && start < regions[regions.length - 1].end) {
      regions[regions.length - 1].end = start; // keep contiguity if backdated
    }
    regions.push({ start, end: region.end, name: hybridLabel, confidence: region.confidence });
    comparison.push({
      startSeconds: region.start,
      endSeconds: region.end,
      ruleChord: region.name,
      learnedTopChord: learnedLabel,
      hybridChord: hybridLabel,
      ruleConfidence: region.confidence,
      learnedConfidence,
      boundaryProbability,
    });
  });

  const entropy = learned.rootProbabilities.length
    ? learned.rootProbabilities.reduce((sum, row) => sum + shannonEntropy(row), 0) / learned.rootProbabilities.length
    : 0;

  return { regions, comparison, entropy, boundaryPeaks: boundaryPeakCount(learned.boundaryProbabilities), disagreements };
}

export interface RunHybridParams {
  provider: LearnedHarmonyProvider;
  request: LearnedHarmonyRequest | null;
  ruleBased: RuleBasedHarmonyEvidence;
  settings: HybridHarmonySettings;
  enabled: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
}

/**
 * Orchestrates the learned request with the rule-based engine. In EVERY failure
 * case — disabled, unavailable, timeout, cancellation, invalid output, version
 * mismatch — it falls back to the rule-based regions without losing the job.
 */
export async function runHybridHarmony(params: RunHybridParams): Promise<HybridHarmonyResult> {
  const { provider, request, ruleBased, settings, enabled } = params;
  const timeoutMs = params.timeoutMs ?? CONTRACT.limits.inferenceTimeoutMs;
  const diagnostics: HybridDiagnostics = {
    providerId: provider.id,
    providerAvailable: false,
    warnings: [],
    comparison: [],
  };

  const fallback = (reason: HybridFallbackReason): HybridHarmonyResult => ({
    engine: "rule-based",
    regions: ruleBased.regions,
    usedLearned: false,
    fallbackReason: reason,
    diagnostics: { ...diagnostics, fallbackReason: reason },
  });

  if (!enabled) return fallback("learned-disabled");
  if (!request) return fallback("no-feature-package");

  diagnostics.providerAvailable = await provider.isAvailable();
  if (!diagnostics.providerAvailable) return fallback("provider-unavailable");

  try {
    const metadata = await provider.getMetadata();
    if (
      metadata.contractVersion !== request.contractVersion ||
      metadata.featureVersion !== request.featureVersion ||
      metadata.modelVersion !== request.modelMetadata.modelVersion ||
      metadata.modelChecksum !== request.modelMetadata.modelChecksum
    ) {
      return fallback("version-mismatch");
    }
  } catch {
    return fallback("provider-error");
  }

  let response: LearnedHarmonyResponse;
  let timedOut = false;
  const controller = new AbortController();
  const onExternalAbort = () => controller.abort();
  params.signal?.addEventListener("abort", onExternalAbort, { once: true });
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    response = await provider.predict(request, controller.signal);
  } catch (error) {
    if (timedOut) return fallback("timeout");
    if (error instanceof DOMException && error.name === "AbortError") return fallback("cancelled");
    return fallback("provider-error");
  } finally {
    clearTimeout(timer);
    params.signal?.removeEventListener("abort", onExternalAbort);
  }

  const validation = validateResponse(response, request);
  if (!validation.ok) {
    diagnostics.warnings.push(...validation.errors);
    return fallback("invalid-response");
  }

  const decoderStart = Date.now();
  const combined = combineHybrid(response, ruleBased, settings);
  diagnostics.decoderMs = Date.now() - decoderStart;
  diagnostics.inferenceMs = response.diagnostics.inferenceMilliseconds;
  diagnostics.modelVersion = response.modelVersion;
  diagnostics.modelChecksum = response.modelChecksum;
  diagnostics.featureVersion = response.featureVersion;
  diagnostics.learnedProbabilityEntropy = combined.entropy;
  diagnostics.boundaryPeakCount = combined.boundaryPeaks;
  diagnostics.ruleLearnedDisagreements = combined.disagreements;
  diagnostics.comparison = combined.comparison;
  diagnostics.warnings.push(...response.diagnostics.warnings);

  return { engine: "hybrid-experimental", regions: combined.regions, usedLearned: true, diagnostics };
}
