import {
  bootstrapMeanConfidenceInterval,
  type MainMetricName,
  type PairedDifference,
} from "./hybridMetrics";
import {
  assertNoAbsolutePaths,
  type HybridAccuracyReport,
} from "./hybridReport";
import type { EngineId } from "./hybridComparison";

const ENGINES: EngineId[] = [
  "rule-only",
  "ml-only",
  "observation-hybrid",
  "legacy-region-hybrid",
];
const STABILITY_METRICS = [
  "rootAccuracy",
  "majorMinorAccuracy",
  "detailedAccuracy",
  "noChordF1",
  "fragmentationRate",
  "meanAbsoluteBoundaryErrorMs",
  "regionsPerMinute",
] as const;
type StabilityMetric = typeof STABILITY_METRICS[number];

interface CrossCaptureMetric {
  microphone: number | null;
  pickupMix: number | null;
  pickupMinusMicrophone: number | null;
  pairedTrackMeanDifference: number | null;
  pairedTrackBootstrap95ConfidenceInterval: [number, number] | null;
}

export interface HybridCrossCaptureSummary {
  schemaVersion: 1;
  status: "completed-valid-cross-capture-summary";
  createdAt: string;
  methodology: {
    alternateCapturesNotCombinedAsIndependentTracks: true;
    identicalTrackIds: true;
    identicalReferenceChecksums: true;
    identicalEvaluatedDurations: true;
    bootstrapSeed: number;
    bootstrapIterations: number;
  };
  dataset: {
    internalDatasetId: string;
    officialDistributionRecord: string | null;
    datasetVersion: string | null;
    splitName: string;
    heldOutPerformers: string[];
    trackCount: number;
    evaluatedDurationSeconds: number;
  };
  model: {
    modelVersion: string;
    modelChecksum: string;
    featureVersion: string;
    hybridDecoderVersion: number;
  };
  captures: {
    microphone: {
      captureType: string | null;
      audioArchiveMd5: string | null;
      fallbackCount: number;
      engines: HybridAccuracyReport["aggregate"]["engines"];
      pairedDifferences: PairedDifference[];
    };
    pickupMix: {
      captureType: string | null;
      audioArchiveMd5: string | null;
      fallbackCount: number;
      engines: HybridAccuracyReport["aggregate"]["engines"];
      pairedDifferences: PairedDifference[];
    };
  };
  crossCaptureSensitivity: Partial<Record<
    EngineId,
    Record<StabilityMetric, CrossCaptureMetric>
  >>;
  conclusions: {
    hybridOutperformsRuleOnMicrophone: boolean;
    hybridOutperformsMlOnlyOnMicrophone: boolean;
    hybridOutperformsLegacyOnMicrophone: boolean | null;
    ruleAccuracyImprovementReproducedOnPickupMix: boolean;
    fragmentationImprovesOnBothCaptures: boolean;
    boundaryTimingImprovesOnBothCaptures: boolean;
    totalHybridFallbacks: number;
    evidenceLevel: "held-out-validation";
    recommendation: "keep-experimental" | "limited-beta";
    summary: string;
  };
}

function sorted(values: string[]): string[] {
  return [...values].sort();
}

function assertSameStrings(left: string[], right: string[], label: string): void {
  if (JSON.stringify(sorted(left)) !== JSON.stringify(sorted(right))) {
    throw new Error(`Cross-capture ${label} do not match`);
  }
}

function perTrackRows(
  report: HybridAccuracyReport,
  engine: EngineId,
): Map<string, HybridAccuracyReport["perTrack"][number]> {
  return new Map(
    report.perTrack
      .filter((row) => row.engine === engine)
      .map((row) => [row.trackId, row]),
  );
}

function pairedImprovement(
  report: HybridAccuracyReport,
  comparison: PairedDifference["comparison"],
  metric: MainMetricName,
): boolean {
  const row = report.aggregate.pairedDifferences.find(
    (candidate) => candidate.comparison === comparison
      && candidate.metric === metric,
  );
  const interval = row?.bootstrap95ConfidenceInterval;
  if (!interval) return false;
  return metric === "fragmentationRate"
    || metric === "meanAbsoluteBoundaryErrorMs"
    ? interval[1] < 0
    : interval[0] > 0;
}

function aggregateValue(
  report: HybridAccuracyReport,
  engine: EngineId,
  metric: StabilityMetric,
): number | null {
  const value = report.aggregate.engines[engine]?.durationWeighted[metric];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function trackMetric(
  row: HybridAccuracyReport["perTrack"][number],
  metric: StabilityMetric,
): number | null {
  const value = row.metrics[metric];
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function buildCrossCaptureSummary(
  microphone: HybridAccuracyReport,
  pickupMix: HybridAccuracyReport,
  createdAt: string,
): HybridCrossCaptureSummary {
  const micDataset = microphone.datasets[0];
  const pickupDataset = pickupMix.datasets[0];
  if (!micDataset || !pickupDataset
    || microphone.datasets.length !== 1
    || pickupMix.datasets.length !== 1) {
    throw new Error("Cross-capture comparison requires one dataset per report");
  }
  assertSameStrings(
    microphone.fairness.commonSuccessfulTrackIds,
    pickupMix.fairness.commonSuccessfulTrackIds,
    "successful track IDs",
  );
  if (!microphone.fairness.commonSuccessfulTrackIds.length) {
    throw new Error("Cross-capture comparison has zero successful tracks");
  }
  if (micDataset.datasetIdentifier !== pickupDataset.datasetIdentifier
    || micDataset.splitName !== pickupDataset.splitName
    || microphone.engineIdentities["ml-only"].modelChecksum
      !== pickupMix.engineIdentities["ml-only"].modelChecksum
    || microphone.engineIdentities["ml-only"].featureVersion
      !== pickupMix.engineIdentities["ml-only"].featureVersion
    || microphone.engineIdentities["observation-hybrid"].hybridDecoderVersion
      !== pickupMix.engineIdentities["observation-hybrid"].hybridDecoderVersion
    || JSON.stringify(
      microphone.engineIdentities["observation-hybrid"].fusionSettings,
    ) !== JSON.stringify(
      pickupMix.engineIdentities["observation-hybrid"].fusionSettings,
    )) {
    throw new Error("Cross-capture dataset, model, decoder, or settings differ");
  }

  const sensitivity: HybridCrossCaptureSummary["crossCaptureSensitivity"] = {};
  for (const engine of ENGINES) {
    const micRows = perTrackRows(microphone, engine);
    const pickupRows = perTrackRows(pickupMix, engine);
    if (!micRows.size && !pickupRows.size) continue;
    assertSameStrings([...micRows.keys()], [...pickupRows.keys()], `${engine} track IDs`);
    for (const [trackId, micRow] of micRows) {
      const pickupRow = pickupRows.get(trackId)!;
      if (micRow.referenceChecksum !== pickupRow.referenceChecksum) {
        throw new Error(
          `Cross-capture reference checksum differs for ${trackId}`,
        );
      }
      if (Math.abs(micRow.durationSeconds - pickupRow.durationSeconds) > 1e-9) {
        throw new Error(
          `Cross-capture evaluated duration differs for ${trackId}`,
        );
      }
    }
    const engineMetrics = {} as Record<StabilityMetric, CrossCaptureMetric>;
    for (const metric of STABILITY_METRICS) {
      const differences = [...micRows.keys()].flatMap((trackId) => {
        const mic = trackMetric(micRows.get(trackId)!, metric);
        const pickup = trackMetric(pickupRows.get(trackId)!, metric);
        return mic === null || pickup === null ? [] : [pickup - mic];
      });
      const microphoneValue = aggregateValue(microphone, engine, metric);
      const pickupValue = aggregateValue(pickupMix, engine, metric);
      engineMetrics[metric] = {
        microphone: microphoneValue,
        pickupMix: pickupValue,
        pickupMinusMicrophone:
          microphoneValue === null || pickupValue === null
            ? null
            : pickupValue - microphoneValue,
        pairedTrackMeanDifference: differences.length
          ? differences.reduce((sum, value) => sum + value, 0) / differences.length
          : null,
        pairedTrackBootstrap95ConfidenceInterval:
          bootstrapMeanConfidenceInterval(
            differences,
            microphone.methodology.bootstrap.seed
              + ENGINES.indexOf(engine) * 101
              + STABILITY_METRICS.indexOf(metric),
            microphone.methodology.bootstrap.iterations,
          ),
      };
    }
    sensitivity[engine] = engineMetrics;
  }

  const hybridRuleMic = pairedImprovement(
    microphone,
    "hybrid-minus-rule",
    "rootAccuracy",
  );
  const hybridMlMic = pairedImprovement(
    microphone,
    "hybrid-minus-ml-only",
    "rootAccuracy",
  );
  const hasLegacy = Boolean(microphone.aggregate.engines["legacy-region-hybrid"]);
  const hybridLegacyMic = hasLegacy
    ? pairedImprovement(
      microphone,
      "hybrid-minus-legacy-region-hybrid",
      "rootAccuracy",
    )
    : null;
  const hybridRulePickup = pairedImprovement(
    pickupMix,
    "hybrid-minus-rule",
    "rootAccuracy",
  );
  const fragmentationBoth = pairedImprovement(
    microphone,
    "hybrid-minus-rule",
    "fragmentationRate",
  ) && pairedImprovement(
    pickupMix,
    "hybrid-minus-rule",
    "fragmentationRate",
  );
  const boundaryBoth = pairedImprovement(
    microphone,
    "hybrid-minus-rule",
    "meanAbsoluteBoundaryErrorMs",
  ) && pairedImprovement(
    pickupMix,
    "hybrid-minus-rule",
    "meanAbsoluteBoundaryErrorMs",
  );
  const totalFallbacks = microphone.fairness.fallbackTrackCount
    + pickupMix.fairness.fallbackTrackCount;
  const micHybridFragmentation = aggregateValue(
    microphone,
    "observation-hybrid",
    "fragmentationRate",
  );
  const micRuleFragmentation = aggregateValue(
    microphone,
    "rule-only",
    "fragmentationRate",
  );
  const pickupHybridFragmentation = aggregateValue(
    pickupMix,
    "observation-hybrid",
    "fragmentationRate",
  );
  const pickupRuleFragmentation = aggregateValue(
    pickupMix,
    "rule-only",
    "fragmentationRate",
  );
  const limitedBeta = hybridRuleMic && hybridRulePickup
    && totalFallbacks === 0
    && micHybridFragmentation !== null
    && micRuleFragmentation !== null
    && pickupHybridFragmentation !== null
    && pickupRuleFragmentation !== null
    && micHybridFragmentation <= micRuleFragmentation + 0.02
    && pickupHybridFragmentation <= pickupRuleFragmentation + 0.02;
  const summary: HybridCrossCaptureSummary = {
    schemaVersion: 1,
    status: "completed-valid-cross-capture-summary",
    createdAt,
    methodology: {
      alternateCapturesNotCombinedAsIndependentTracks: true,
      identicalTrackIds: true,
      identicalReferenceChecksums: true,
      identicalEvaluatedDurations: true,
      bootstrapSeed: microphone.methodology.bootstrap.seed,
      bootstrapIterations: microphone.methodology.bootstrap.iterations,
    },
    dataset: {
      internalDatasetId: micDataset.datasetIdentifier,
      officialDistributionRecord: micDataset.officialDistributionRecord,
      datasetVersion: micDataset.datasetVersion,
      splitName: micDataset.splitName,
      heldOutPerformers: micDataset.leakageAudit.heldOutPerformers,
      trackCount: microphone.fairness.commonSuccessfulTrackIds.length,
      evaluatedDurationSeconds: micDataset.totalEvaluatedDurationSeconds,
    },
    model: {
      modelVersion: microphone.engineIdentities["ml-only"].modelVersion,
      modelChecksum: microphone.engineIdentities["ml-only"].modelChecksum,
      featureVersion: microphone.engineIdentities["ml-only"].featureVersion,
      hybridDecoderVersion:
        microphone.engineIdentities["observation-hybrid"].hybridDecoderVersion,
    },
    captures: {
      microphone: {
        captureType: micDataset.captureType,
        audioArchiveMd5: micDataset.audioArchiveMd5,
        fallbackCount: microphone.fairness.fallbackTrackCount,
        engines: microphone.aggregate.engines,
        pairedDifferences: microphone.aggregate.pairedDifferences,
      },
      pickupMix: {
        captureType: pickupDataset.captureType,
        audioArchiveMd5: pickupDataset.audioArchiveMd5,
        fallbackCount: pickupMix.fairness.fallbackTrackCount,
        engines: pickupMix.aggregate.engines,
        pairedDifferences: pickupMix.aggregate.pairedDifferences,
      },
    },
    crossCaptureSensitivity: sensitivity,
    conclusions: {
      hybridOutperformsRuleOnMicrophone: hybridRuleMic,
      hybridOutperformsMlOnlyOnMicrophone: hybridMlMic,
      hybridOutperformsLegacyOnMicrophone: hybridLegacyMic,
      ruleAccuracyImprovementReproducedOnPickupMix: hybridRulePickup,
      fragmentationImprovesOnBothCaptures: fragmentationBoth,
      boundaryTimingImprovesOnBothCaptures: boundaryBoth,
      totalHybridFallbacks: totalFallbacks,
      evidenceLevel: "held-out-validation",
      recommendation: limitedBeta ? "limited-beta" : "keep-experimental",
      summary: limitedBeta
        ? "The paired validation improvement reproduces across both solo-guitar captures without a material fragmentation regression. This supports a limited experimental beta, not production promotion."
        : "The paired validation evidence does not establish a replicated accuracy/stability win across both captures. Keep the engine experimental.",
    },
  };
  assertNoAbsolutePaths(summary);
  return summary;
}

function percent(value: number | null): string {
  return value === null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function milliseconds(value: number | null): string {
  return value === null ? "—" : `${value.toFixed(1)} ms`;
}

function engineName(engine: EngineId): string {
  return {
    "rule-only": "Rule v3",
    "ml-only": "ML-only",
    "observation-hybrid": "New hybrid",
    "legacy-region-hybrid": "Legacy region hybrid",
  }[engine];
}

function captureTable(
  engines: HybridAccuracyReport["aggregate"]["engines"],
): string {
  const rows = ENGINES.flatMap((engine) => {
    const values = engines[engine]?.durationWeighted;
    return values ? [
      `| ${engineName(engine)} | ${percent(values.rootAccuracy)}`
        + ` | ${percent(values.majorMinorAccuracy)}`
        + ` | ${percent(values.detailedAccuracy)}`
        + ` | ${percent(values.noChordF1)}`
        + ` | ${percent(values.fragmentationRate)}`
        + ` | ${values.regionsPerMinute.toFixed(2)}`
        + ` | ${milliseconds(values.meanAbsoluteBoundaryErrorMs)} |`,
    ] : [];
  });
  return [
    "| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Regions/min | Boundary error |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
    ...rows,
  ].join("\n");
}

export function crossCaptureSummaryMarkdown(
  summary: HybridCrossCaptureSummary,
): string {
  return `# Hybrid harmony cross-capture comparison

## Executive summary

${summary.conclusions.summary}

This is held-out **validation** evidence on ${summary.dataset.trackCount} paired
GuitarSet performances. Microphone and pickup recordings are alternate captures
and are never combined as independent tracks.

## Microphone capture

${captureTable(summary.captures.microphone.engines)}

Fallbacks: ${summary.captures.microphone.fallbackCount}.

## Pickup-mix capture

${captureTable(summary.captures.pickupMix.engines)}

Fallbacks: ${summary.captures.pickupMix.fallbackCount}.

## Decision questions

- Hybrid beats rule on microphone with a root-accuracy CI above zero: **${summary.conclusions.hybridOutperformsRuleOnMicrophone ? "yes" : "no"}**
- Hybrid beats ML-only on microphone with a root-accuracy CI above zero: **${summary.conclusions.hybridOutperformsMlOnlyOnMicrophone ? "yes" : "no"}**
- Hybrid beats legacy region fusion on microphone: **${summary.conclusions.hybridOutperformsLegacyOnMicrophone === null ? "not available" : summary.conclusions.hybridOutperformsLegacyOnMicrophone ? "yes" : "no"}**
- Rule-relative accuracy improvement reproduces on pickup mix: **${summary.conclusions.ruleAccuracyImprovementReproducedOnPickupMix ? "yes" : "no"}**
- Fragmentation improves significantly on both captures: **${summary.conclusions.fragmentationImprovesOnBothCaptures ? "yes" : "no"}**
- Boundary timing improves significantly on both captures: **${summary.conclusions.boundaryTimingImprovesOnBothCaptures ? "yes" : "no"}**
- Hybrid fallbacks across both runs: **${summary.conclusions.totalHybridFallbacks}**

## Recommendation

**${summary.conclusions.recommendation}**

No production-promotion conclusion is possible from validation-only,
solo-acoustic-guitar evidence without representative full-band raw audio.
`;
}
