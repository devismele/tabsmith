import type { HybridHarmonySettings } from "../src/learnedHarmony/types";
import {
  DEFAULT_ML_ONLY_TRANSITION_PENALTY,
  LEGACY_REGION_HYBRID_SOURCE_COMMIT,
  PRODUCTION_RULE_ENGINE,
  type DisagreementWindow,
  type EngineId,
  type EvaluationModelIdentity,
  type SourceType,
} from "./hybridComparison";
import {
  aggregateComparison,
  canonicalChordLabel,
  publicTrackMetrics,
  type AggregateComparison,
  type EngineAggregate,
  type ScoredTrackComparison,
} from "./hybridMetrics";

export interface DatasetReportIdentity {
  datasetIdentifier: string;
  manifestChecksum: string;
  annotationChecksum: string;
  splitName: string;
  splitDefinition: string;
  sourceType: SourceType;
  trackCount: number;
  totalEvaluatedDurationSeconds: number;
  trackIds: string[];
  splitLeakageWarnings: string[];
}

export interface HybridReportInput {
  commitSha: string;
  createdAt: string;
  datasets: DatasetReportIdentity[];
  scoredTracks: ScoredTrackComparison[];
  modelIdentity: EvaluationModelIdentity;
  hybridDecoderVersion: number;
  hybridSettings: HybridHarmonySettings;
  bootstrapSeed: number;
  bootstrapIterations: number;
  mlOnlyTransitionPenalty?: number;
  includeLegacyRegionHybrid: boolean;
  combineSourceTypes: boolean;
  ablationSplitDescription: string | null;
  limitations: string[];
}

export interface RepresentativeExample {
  category:
    | "hybrid-fixes-ml-flicker"
    | "hybrid-corrects-uncertain-rule"
    | "hybrid-worsens-rule"
    | "rule-protection"
    | "learned-boundary-timing";
  trackId: string;
  startSeconds: number;
  endSeconds: number;
  referenceChord: string | null;
  ruleChord: string | null;
  learnedChord: string | null;
  hybridChord: string | null;
  note: string;
}

export interface HybridAccuracyReport {
  schemaVersion: 1;
  status: "completed-valid-raw-audio-evaluation";
  createdAt: string;
  commitSha: string;
  methodology: {
    primaryComparisonUsesIdenticalTracks: true;
    primaryComparisonUsesIdenticalReferenceIntervals: true;
    primaryComparisonUsesSharedFeatureFrames: true;
    mlOnlyAndHybridShareOneLearnedResponseObject: boolean;
    actualTypeScriptHybridDecoderExecuted: boolean;
    actualTemporalDecoderCompletedTracks: number;
    typeScriptHybridPath: string[];
    metricImplementation: string;
    mlOnlyDecoder: {
      stateVocabulary: string;
      emissions: string;
      transitionPenalty: number;
      durationModel: string;
      consumesRuleEvidence: false;
      consumesBoundaryHead: false;
    };
    shortRegionThresholdSeconds: number;
    bootstrap: { seed: number; iterations: number; unit: "track" };
  };
  datasets: DatasetReportIdentity[];
  engineIdentities: {
    "rule-only": { version: string };
    "ml-only": EvaluationModelIdentity & { productExposed: false };
    "observation-hybrid": EvaluationModelIdentity & {
      hybridDecoderVersion: number;
      fusionSettings: HybridHarmonySettings;
      implementation: string;
    };
    "legacy-region-hybrid"?: {
      sourceCommit: string;
      implementation: "evaluation-only reconstruction";
    };
  };
  fairness: {
    commonSuccessfulTrackIds: string[];
    allRequestedHybridTrackIds: string[];
    requestedHybridTrackCount: number;
    successfulHybridTrackCount: number;
    fallbackTrackCount: number;
    fallbackReasons: Record<string, number>;
    sourceTypesCombined: boolean;
    note: string;
  };
  aggregate: AggregateComparison;
  combinedPrimaryAggregate: AggregateComparison["engines"] | null;
  perTrack: Array<{
    datasetId: string;
    splitName: string;
    sourceType: SourceType;
    trackId: string;
    durationSeconds: number;
    referenceChecksum: string;
    actualTypeScriptHybridDecoderInvoked: boolean;
    actualTemporalDecoderCompleted: boolean;
    engine: EngineId;
    usedLearned: boolean;
    fallbackReason: string | null;
    probabilitySourceId: string | null;
    metrics: ReturnType<typeof publicTrackMetrics>;
  }>;
  representativeExamples: RepresentativeExample[];
  missingRepresentativeExampleCategories: RepresentativeExample["category"][];
  ablationSplitDescription: string | null;
  limitations: string[];
  recommendation: {
    keepExperimental: boolean;
    supportsBroaderBetaTesting: boolean;
    supportsProductionPromotion: boolean;
    summary: string;
  };
}

function regionAt(
  regions: Array<{ start: number; end: number; name: string }>,
  time: number,
): string | null {
  const region = regions.find((candidate, index) =>
    time >= candidate.start
      && (time < candidate.end
        || (index === regions.length - 1 && time <= candidate.end)));
  return region ? canonicalChordLabel(region.name) : null;
}

function referenceAt(
  track: ScoredTrackComparison["comparison"]["track"],
  time: number,
): string | null {
  const region = track.referenceRegions.find((candidate, index) =>
    time >= candidate.start
      && (time < candidate.end
        || (index === track.referenceRegions.length - 1 && time <= candidate.end)));
  return region ? canonicalChordLabel(region.label) : null;
}

function exampleFromWindow(
  category: RepresentativeExample["category"],
  track: ScoredTrackComparison,
  window: DisagreementWindow,
  note: string,
): RepresentativeExample {
  const localStart = window.startSeconds - track.comparison.track.originalOffsetSeconds;
  return {
    category,
    trackId: track.comparison.track.trackId,
    startSeconds: window.startSeconds,
    endSeconds: window.endSeconds,
    referenceChord: referenceAt(track.comparison.track, localStart),
    ruleChord: canonicalChordLabel(window.ruleTopCandidate),
    learnedChord: window.learnedTopCandidate
      ? canonicalChordLabel(window.learnedTopCandidate)
      : null,
    hybridChord: canonicalChordLabel(window.hybridFinalLabel),
    note,
  };
}

function representativeExamples(
  tracks: ScoredTrackComparison[],
): RepresentativeExample[] {
  const examples: RepresentativeExample[] = [];
  const add = (example: RepresentativeExample | undefined) => {
    if (example && !examples.some((candidate) => candidate.category === example.category)) {
      examples.push(example);
    }
  };

  for (const track of tracks) {
    const ml = track.comparison.predictions["ml-only"];
    const hybrid = track.comparison.predictions["observation-hybrid"];
    if (ml && hybrid) {
      for (let index = 1; index + 1 < ml.regions.length; index += 1) {
        const previous = canonicalChordLabel(ml.regions[index - 1].name);
        const current = canonicalChordLabel(ml.regions[index].name);
        const next = canonicalChordLabel(ml.regions[index + 1].name);
        const duration = ml.regions[index].end - ml.regions[index].start;
        const midpoint = (ml.regions[index].start + ml.regions[index].end) / 2;
        if (previous === next && previous !== current && duration <= 0.5
          && regionAt(hybrid.regions, midpoint) === previous) {
          add({
            category: "hybrid-fixes-ml-flicker",
            trackId: track.comparison.track.trackId,
            startSeconds: ml.regions[index].start
              + track.comparison.track.originalOffsetSeconds,
            endSeconds: ml.regions[index].end
              + track.comparison.track.originalOffsetSeconds,
            referenceChord: referenceAt(track.comparison.track, midpoint),
            ruleChord: regionAt(
              track.comparison.predictions["rule-only"]?.regions ?? [],
              midpoint,
            ),
            learnedChord: current,
            hybridChord: previous,
            note: "ML-only made a short A-B-A excursion that the production temporal decoder suppressed.",
          });
          break;
        }
      }
    }

    for (const window of track.comparison.disagreementWindows) {
      const localStart = window.startSeconds
        - track.comparison.track.originalOffsetSeconds;
      const reference = referenceAt(track.comparison.track, localStart);
      const rule = canonicalChordLabel(window.ruleTopCandidate);
      const learned = canonicalChordLabel(window.learnedTopCandidate ?? "N");
      const hybridLabel = canonicalChordLabel(window.hybridFinalLabel);
      if (window.ruleConfidence < 0.72
        && rule !== reference
        && hybridLabel === reference
        && learned === reference) {
        add(exampleFromWindow(
          "hybrid-corrects-uncertain-rule",
          track,
          window,
          "Confident learned evidence corrected a low-confidence rule candidate before temporal decoding.",
        ));
      }
      if (rule === reference && hybridLabel !== reference) {
        add(exampleFromWindow(
          "hybrid-worsens-rule",
          track,
          window,
          "The hybrid changed a rule candidate that matched the annotation.",
        ));
      }
      if (rule !== learned && hybridLabel === rule) {
        add(exampleFromWindow(
          "rule-protection",
          track,
          window,
          "Rule/ML disagreement was resolved in favor of the protected rule result.",
        ));
      }
    }

    const full = track.comparison.ablations.find(
      (ablation) => ablation.id === "full-hybrid",
    );
    const noBoundary = track.comparison.ablations.find(
      (ablation) => ablation.id === "hybrid-without-boundary",
    );
    if (full && noBoundary) {
      const candidate = full.regions.slice(1).find((region) =>
        noBoundary.regions.slice(1).some((without) =>
          canonicalChordLabel(without.name) === canonicalChordLabel(region.name)
            && Math.abs(without.start - region.start) >= 0.001));
      if (candidate) {
        const without = noBoundary.regions.slice(1).find((region) =>
          canonicalChordLabel(region.name) === canonicalChordLabel(candidate.name)
            && Math.abs(region.start - candidate.start) >= 0.001)!;
        add({
          category: "learned-boundary-timing",
          trackId: track.comparison.track.trackId,
          startSeconds: Math.min(candidate.start, without.start)
            + track.comparison.track.originalOffsetSeconds,
          endSeconds: Math.max(candidate.start, without.start)
            + track.comparison.track.originalOffsetSeconds,
          referenceChord: referenceAt(track.comparison.track, candidate.start),
          ruleChord: regionAt(
            track.comparison.predictions["rule-only"]?.regions ?? [],
            candidate.start,
          ),
          learnedChord: null,
          hybridChord: canonicalChordLabel(candidate.name),
          note: `Removing the learned boundary term moved this transition by ${Math.round(
            Math.abs(candidate.start - without.start) * 1000,
          )} ms.`,
        });
      }
    }
  }
  return examples;
}

function mainAggregate(
  aggregate: AggregateComparison,
): {
  hybrid?: EngineAggregate;
  rule?: EngineAggregate;
  ml?: EngineAggregate;
} {
  return {
    hybrid: aggregate.engines["observation-hybrid"],
    rule: aggregate.engines["rule-only"],
    ml: aggregate.engines["ml-only"],
  };
}

function recommendation(
  aggregate: AggregateComparison,
  datasets: DatasetReportIdentity[],
): HybridAccuracyReport["recommendation"] {
  const { hybrid, rule } = mainAggregate(aggregate);
  const hasFullBand = datasets.some((dataset) =>
    dataset.sourceType === "full-band" && dataset.trackCount > 0);
  if (!hybrid || !rule) {
    return {
      keepExperimental: true,
      supportsBroaderBetaTesting: false,
      supportsProductionPromotion: false,
      summary: "No complete common-track primary comparison was available.",
    };
  }
  const accuracyImproved = hybrid.durationWeighted.rootAccuracy
      > rule.durationWeighted.rootAccuracy
    && hybrid.durationWeighted.majorMinorAccuracy
      >= rule.durationWeighted.majorMinorAccuracy;
  const fragmentationAcceptable = hybrid.durationWeighted.fragmentationRate
    <= rule.durationWeighted.fragmentationRate + 0.02;
  if (!hasFullBand) {
    return {
      keepExperimental: true,
      supportsBroaderBetaTesting: accuracyImproved && fragmentationAcceptable,
      supportsProductionPromotion: false,
      summary: accuracyImproved && fragmentationAcceptable
        ? "Positive guitar-domain evidence, but no representative full-band raw-audio result exists; the production gate remains open."
        : "The guitar-domain result does not establish a clear accuracy/stability win, and no representative full-band raw-audio result exists.",
    };
  }
  if (accuracyImproved && fragmentationAcceptable) {
    return {
      keepExperimental: true,
      supportsBroaderBetaTesting: true,
      supportsProductionPromotion: false,
      summary: "The result supports broader beta testing, but production promotion still requires broader replicated evaluation and latency review.",
    };
  }
  return {
    keepExperimental: true,
    supportsBroaderBetaTesting: false,
    supportsProductionPromotion: false,
    summary: "The hybrid did not show an accuracy gain without a material stability trade-off; do not promote it.",
  };
}

export function buildHybridAccuracyReport(
  input: HybridReportInput,
): HybridAccuracyReport {
  const aggregate = aggregateComparison(
    input.scoredTracks,
    input.bootstrapSeed,
    input.bootstrapIterations,
  );
  const sourceTypes = new Set(input.datasets.map((dataset) => dataset.sourceType));
  const combinedValid = sourceTypes.size <= 1 || input.combineSourceTypes;
  const examples = representativeExamples(input.scoredTracks);
  const categories: RepresentativeExample["category"][] = [
    "hybrid-fixes-ml-flicker",
    "hybrid-corrects-uncertain-rule",
    "hybrid-worsens-rule",
    "rule-protection",
    "learned-boundary-timing",
  ];
  const sharedResponses = input.scoredTracks
    .filter((track) =>
      track.comparison.predictions["observation-hybrid"]?.usedLearned)
    .every((track) => track.comparison.execution.sharedLearnedResponseObject);
  const actualTypeScript = input.scoredTracks.every(
    (track) => track.comparison.execution.actualHybridDecoderInvoked,
  );
  const completed = input.scoredTracks.filter(
    (track) => track.comparison.execution.actualTemporalDecoderCompleted,
  ).length;
  const perTrack = input.scoredTracks.flatMap((track) =>
    Object.entries(track.engineMetrics).flatMap(([engineId, metrics]) => {
      const prediction = track.comparison.predictions[engineId as EngineId];
      return metrics && prediction ? [{
        datasetId: track.comparison.track.datasetId,
        splitName: track.comparison.track.splitName,
        sourceType: track.comparison.track.sourceType,
        trackId: track.comparison.track.trackId,
        durationSeconds: track.comparison.track.duration,
        referenceChecksum: track.comparison.track.referenceChecksum,
        actualTypeScriptHybridDecoderInvoked:
          track.comparison.execution.actualHybridDecoderInvoked,
        actualTemporalDecoderCompleted:
          track.comparison.execution.actualTemporalDecoderCompleted,
        engine: engineId as EngineId,
        usedLearned: prediction.usedLearned,
        fallbackReason: prediction.fallbackReason ?? null,
        probabilitySourceId: prediction.probabilitySourceId ?? null,
        metrics: publicTrackMetrics(metrics),
      }] : [];
    }));
  const fallbackTrackCount = Object.values(aggregate.fallbackCounts)
    .reduce((sum, count) => sum + count, 0);
  const report: HybridAccuracyReport = {
    schemaVersion: 1,
    status: "completed-valid-raw-audio-evaluation",
    createdAt: input.createdAt,
    commitSha: input.commitSha,
    methodology: {
      primaryComparisonUsesIdenticalTracks: true,
      primaryComparisonUsesIdenticalReferenceIntervals: true,
      primaryComparisonUsesSharedFeatureFrames: true,
      mlOnlyAndHybridShareOneLearnedResponseObject: sharedResponses,
      actualTypeScriptHybridDecoderExecuted: actualTypeScript,
      actualTemporalDecoderCompletedTracks: completed,
      typeScriptHybridPath: [
        "src/chordAnalysis.ts#createHarmonyObservations",
        "src/learnedHarmony/onnxProvider.ts#LearnedTcnProvider.predict",
        "src/learnedHarmony/hybridDecoder.ts#alignLearnedEvidence",
        "src/learnedHarmony/hybridDecoder.ts#fuseHybridObservations",
        "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony",
        "src/chordAnalysis.ts#decodeHarmonyObservations",
        "src/chordAnalysis.ts#decodeChordSequence",
        "src/chordAnalysis.ts#decodeReducedLatencySequence",
      ],
      metricImplementation:
        "server/evaluation.mjs#evaluateChordReference plus documented fragmentation/boundary supplements",
      mlOnlyDecoder: {
        stateVocabulary: "12 roots x {major, minor, dominant-7} plus N",
        emissions:
          "log(rootProbability * qualityProbability * (1 - noChordProbability)); N uses log(noChordProbability)",
        transitionPenalty:
          input.mlOnlyTransitionPenalty ?? DEFAULT_ML_ONLY_TRANSITION_PENALTY,
        durationModel:
          "global Viterbi with a uniform chord-switch penalty; no rule, beat, bass, key, hysteresis, or cleanup evidence",
        consumesRuleEvidence: false,
        consumesBoundaryHead: false,
      },
      shortRegionThresholdSeconds: 0.5,
      bootstrap: {
        seed: input.bootstrapSeed,
        iterations: input.bootstrapIterations,
        unit: "track",
      },
    },
    datasets: input.datasets,
    engineIdentities: {
      "rule-only": { version: PRODUCTION_RULE_ENGINE },
      "ml-only": { ...input.modelIdentity, productExposed: false },
      "observation-hybrid": {
        ...input.modelIdentity,
        hybridDecoderVersion: input.hybridDecoderVersion,
        fusionSettings: input.hybridSettings,
        implementation:
          "src/learnedHarmony/hybridDecoder.ts#runHybridHarmony",
      },
      ...(input.includeLegacyRegionHybrid ? {
        "legacy-region-hybrid": {
          sourceCommit: LEGACY_REGION_HYBRID_SOURCE_COMMIT,
          implementation: "evaluation-only reconstruction" as const,
        },
      } : {}),
    },
    fairness: {
      commonSuccessfulTrackIds: aggregate.commonSuccessfulTrackIds,
      allRequestedHybridTrackIds: aggregate.allRequestedHybridTrackIds,
      requestedHybridTrackCount: aggregate.allRequestedHybridTrackIds.length,
      successfulHybridTrackCount: aggregate.commonSuccessfulTrackIds.length,
      fallbackTrackCount,
      fallbackReasons: aggregate.fallbackCounts,
      sourceTypesCombined: combinedValid,
      note: combinedValid
        ? "All displayed combined primary aggregates use one common successful track set."
        : "Solo-guitar and full-band aggregates are kept separate because the configuration did not authorize a combined domain result.",
    },
    aggregate,
    combinedPrimaryAggregate: combinedValid ? aggregate.engines : null,
    perTrack,
    representativeExamples: examples,
    missingRepresentativeExampleCategories: categories.filter(
      (category) => !examples.some((example) => example.category === category),
    ),
    ablationSplitDescription: input.ablationSplitDescription,
    limitations: input.limitations,
    recommendation: recommendation(aggregate, input.datasets),
  };
  assertNoAbsolutePaths(report);
  return report;
}

function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(1)}%`;
}

function milliseconds(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : `${value.toFixed(1)} ms`;
}

function engineName(engine: EngineId): string {
  return {
    "rule-only": "Rule v3",
    "ml-only": "ML-only",
    "observation-hybrid": "New hybrid",
    "legacy-region-hybrid": "Legacy region hybrid",
  }[engine];
}

function aggregateTable(engines: Partial<Record<EngineId, EngineAggregate>>): string {
  const rows = ([
    "rule-only",
    "ml-only",
    "observation-hybrid",
    "legacy-region-hybrid",
  ] as EngineId[]).flatMap((engine) => {
    const aggregate = engines[engine];
    if (!aggregate) return [];
    const values = aggregate.durationWeighted;
    return [
      `| ${engineName(engine)} | ${percent(values.rootAccuracy)}`
        + ` | ${percent(values.majorMinorAccuracy)}`
        + ` | ${percent(values.detailedAccuracy)}`
        + ` | ${percent(values.noChordF1)}`
        + ` | ${percent(values.fragmentationRate)}`
        + ` | ${milliseconds(values.meanAbsoluteBoundaryErrorMs)} |`,
    ];
  });
  return [
    "| Engine | Root | Maj/min | Detailed | N F1 | Fragmentation | Boundary error |",
    "|---|---:|---:|---:|---:|---:|---:|",
    ...rows,
  ].join("\n");
}

export function hybridAccuracyReportMarkdown(
  report: HybridAccuracyReport,
): string {
  const datasetLines = report.datasets.map((dataset) =>
    `- ${dataset.datasetIdentifier}, split \`${dataset.splitName}\`:`
      + ` ${dataset.trackCount} tracks,`
      + ` ${(dataset.totalEvaluatedDurationSeconds / 60).toFixed(1)} minutes,`
      + ` source type \`${dataset.sourceType}\`,`
      + ` split definition \`${dataset.splitDefinition}\`,`
      + ` manifest SHA-256 \`${dataset.manifestChecksum}\`.`);
  const pairedRows = report.aggregate.pairedDifferences.map((difference) => {
    const interval = difference.bootstrap95ConfidenceInterval
      ? `[${difference.bootstrap95ConfidenceInterval[0].toFixed(4)},`
        + ` ${difference.bootstrap95ConfidenceInterval[1].toFixed(4)}]`
      : "—";
    return `| ${difference.comparison} | ${difference.metric}`
      + ` | ${difference.meanDifference?.toFixed(4) ?? "—"}`
      + ` | ${interval} | ${difference.improvedTracks}`
      + ` / ${difference.tiedTracks} / ${difference.worsenedTracks} |`;
  });
  const sourceSections = Object.entries(report.aggregate.bySourceType)
    .map(([sourceType, engines]) =>
      `### ${sourceType}\n\n${aggregateTable(engines)}`)
    .join("\n\n");
  const exampleLines = report.representativeExamples.length
    ? report.representativeExamples.map((example) =>
      `- **${example.category}** — \`${example.trackId}\``
        + ` ${example.startSeconds.toFixed(3)}–${example.endSeconds.toFixed(3)} s:`
        + ` ${example.note}`)
    : ["- No representative examples could be derived without fabricating evidence."];
  const ablationRows = Object.entries(report.aggregate.ablations).map(
    ([id, aggregate]) =>
      `| ${id} | ${percent(aggregate.durationWeighted.rootAccuracy)}`
        + ` | ${percent(aggregate.durationWeighted.majorMinorAccuracy)}`
        + ` | ${percent(aggregate.durationWeighted.fragmentationRate)}`
        + ` | ${milliseconds(aggregate.durationWeighted.meanAbsoluteBoundaryErrorMs)} |`,
  );
  return `# Hybrid harmony accuracy comparison

## Executive summary

${report.recommendation.summary}

The actual TypeScript hybrid decoder was executed: **${report.methodology.actualTypeScriptHybridDecoderExecuted ? "yes" : "no"}**.
Learned inference completed through the production temporal decoder on ${report.methodology.actualTemporalDecoderCompletedTracks} of ${report.fairness.requestedHybridTrackCount} requested tracks. Fallbacks: ${report.fairness.fallbackTrackCount}.

## Methodology

All primary engines used identical raw-audio intervals, annotations, sample rates,
feature frames, and chord normalization. ML-only and hybrid used the same captured
learned response object per successful track. Hybrid called the production
\`runHybridHarmony()\` path, which aligns the full probability distribution,
performs adaptive fusion, and returns the fused observations to
\`decodeHarmonyObservations()\`.

ML-only used a 37-state log-emission Viterbi decoder with a uniform switch penalty
of ${report.methodology.mlOnlyDecoder.transitionPenalty}. It received no rule,
bass, beat, key, boundary-head, hysteresis, or cleanup evidence and remains
evaluation-only.

Paired bootstrap confidence intervals resample tracks with seed
${report.methodology.bootstrap.seed} for ${report.methodology.bootstrap.iterations}
iterations.

## Dataset and split

${datasetLines.join("\n")}

## Engine identities

- Rule-only: \`${report.engineIdentities["rule-only"].version}\`.
- ML-only and hybrid model: \`${report.engineIdentities["ml-only"].modelVersion}\`,
  checksum \`${report.engineIdentities["ml-only"].modelChecksum}\`,
  feature version \`${report.engineIdentities["ml-only"].featureVersion}\`.
- Observation hybrid decoder version:
  \`${report.engineIdentities["observation-hybrid"].hybridDecoderVersion}\`.
${report.engineIdentities["legacy-region-hybrid"]
    ? `- Legacy region hybrid: evaluation-only reconstruction from \`${report.engineIdentities["legacy-region-hybrid"].sourceCommit}\`.`
    : "- Legacy region hybrid: omitted."}

## Aggregate results

${report.combinedPrimaryAggregate
    ? aggregateTable(report.combinedPrimaryAggregate)
    : "Combined-domain results are intentionally omitted; see the source-specific tables."}

${sourceSections}

## Paired differences and confidence intervals

Positive accuracy/N-F1 differences favor hybrid. Negative fragmentation and
boundary-error differences favor hybrid.

| Comparison | Metric | Mean difference | Bootstrap 95% CI | Improved / tied / worsened |
|---|---|---:|---:|---:|
${pairedRows.join("\n")}

## Fragmentation and runtime

${Object.entries(report.aggregate.engines).map(([engine, aggregate]) =>
    `- ${engineName(engine as EngineId)}:`
      + ` ${aggregate.durationWeighted.regionsPerMinute.toFixed(2)} regions/min,`
      + ` ${percent(aggregate.durationWeighted.oneWindowRegionRate)} one-window regions,`
      + ` ${percent(aggregate.durationWeighted.veryShortRegionRate)} very-short regions,`
      + ` ${aggregate.durationWeighted.flickerEventCount} flicker events,`
      + ` ${aggregate.durationWeighted.runtimePerAudioMinuteSeconds.toFixed(3)} runtime seconds/audio minute.`
  ).join("\n")}

## Ablation results

${report.ablationSplitDescription
    ? `${report.ablationSplitDescription}\n\n| Variant | Root | Maj/min | Fragmentation | Boundary error |\n|---|---:|---:|---:|---:|\n${ablationRows.join("\n")}`
    : "No separate valid development/validation ablation run was configured."}

## Representative successes and regressions

${exampleLines.join("\n")}

Missing evidence categories: ${report.missingRepresentativeExampleCategories.length
    ? report.missingRepresentativeExampleCategories.map((value) => `\`${value}\``).join(", ")
    : "none"}.

## Limitations

${report.limitations.map((limitation) => `- ${limitation}`).join("\n")}

## Recommendation

- Keep experimental: **${report.recommendation.keepExperimental ? "yes" : "no"}**
- Supports broader beta testing: **${report.recommendation.supportsBroaderBetaTesting ? "yes" : "no"}**
- Supports production promotion: **${report.recommendation.supportsProductionPromotion ? "yes" : "no"}**

${report.recommendation.summary}
`;
}

function csvCell(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

export function hybridAccuracyPerTrackCsv(report: HybridAccuracyReport): string {
  const headers = [
    "datasetId", "splitName", "sourceType", "trackId", "engine",
    "durationSeconds", "usedLearned", "fallbackReason", "rootAccuracy",
    "majorMinorAccuracy", "detailedAccuracy", "noChordPrecision",
    "noChordRecall", "noChordF1", "fragmentationRate",
    "predictedRegionCount", "regionsPerMinute", "veryShortRegionCount",
    "oneWindowRegionCount", "oneWindowRegionRate", "veryShortRegionRate",
    "flickerEventCount", "meanAbsoluteBoundaryErrorMs",
    "medianAbsoluteBoundaryErrorMs", "meanSignedBoundaryErrorMs",
    "boundariesWithin100ms", "boundariesWithin250ms",
    "boundariesWithin500ms", "boundariesWithin1000ms",
    "mlOverrideCountVsRule", "percentRuleWindowsChangedByMl",
    "ruleMlAgreementRate", "averageEffectiveLearnedWeight",
    "averageLearnedEntropy", "runtimePerAudioMinuteSeconds", "peakMemoryMiB",
  ];
  const rows = report.perTrack.map((row) => {
    const values: Record<string, unknown> = {
      ...row,
      ...row.metrics,
    };
    return headers.map((header) => csvCell(values[header])).join(",");
  });
  return `${headers.join(",")}\n${rows.join("\n")}\n`;
}

export function hybridDisagreementReport(
  report: HybridAccuracyReport,
  tracks: ScoredTrackComparison[],
): {
  schemaVersion: 1;
  modelVersion: string;
  modelChecksum: string;
  windows: DisagreementWindow[];
  representativeExamples: RepresentativeExample[];
} {
  return {
    schemaVersion: 1,
    modelVersion: report.engineIdentities["ml-only"].modelVersion,
    modelChecksum: report.engineIdentities["ml-only"].modelChecksum,
    windows: tracks.flatMap((track) => track.comparison.disagreementWindows),
    representativeExamples: report.representativeExamples,
  };
}

function looksAbsolutePath(value: string): boolean {
  return /^[A-Za-z]:[\\/]/.test(value)
    || /^file:\/\//i.test(value)
    || /^\/(?:Users|home|root|mnt)\//.test(value);
}

export function assertNoAbsolutePaths(value: unknown, location = "$"): void {
  if (typeof value === "string") {
    if (looksAbsolutePath(value)) {
      throw new Error(`Absolute path found in saved report at ${location}`);
    }
    return;
  }
  if (Array.isArray(value)) {
    value.forEach((item, index) => assertNoAbsolutePaths(item, `${location}[${index}]`));
    return;
  }
  if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      assertNoAbsolutePaths(item, `${location}.${key}`);
    }
  }
}
