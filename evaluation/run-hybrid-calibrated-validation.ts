import {
  mkdir,
  readFile,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import type { HybridHarmonySettings } from "../src/learnedHarmony/types";
import {
  assertFinalValidationIsFrozen,
  settingsIdentity,
  type ProbabilityCalibration,
} from "./hybridCalibration";
import { assertNoAbsolutePaths } from "./hybridReport";
import { runHybridComparison } from "./run-hybrid-comparison";

interface FrozenCalibrationSettings {
  schemaVersion: 1;
  status: "frozen-before-p00-validation";
  sealedFinalValidationPerformer: "guitarset-p00";
  candidateId: string;
  settings: HybridHarmonySettings;
  adaptiveWeighting: boolean;
  probabilityCalibration: ProbabilityCalibration;
  settingsIdentity: string;
  changedFromCurrent: boolean;
  selectionReason: string;
}

interface ValidationEnvironment extends NodeJS.ProcessEnv {
  TABSMITH_GUITARSET_MIC_AUDIO?: string;
  TABSMITH_GUITARSET_PICKUP_AUDIO?: string;
}

function requiredPath(
  environment: ValidationEnvironment,
  name: "TABSMITH_GUITARSET_MIC_AUDIO" | "TABSMITH_GUITARSET_PICKUP_AUDIO",
): string {
  const value = environment[name]?.trim();
  if (!value) throw new Error(`${name} is required for calibrated p00 validation`);
  return path.resolve(value);
}

function metricIdentity(report: any): unknown {
  return {
    datasets: report.datasets.map((dataset: any) => ({
      trackIds: dataset.trackIds,
      totalEvaluatedDurationSeconds: dataset.totalEvaluatedDurationSeconds,
    })),
    engines: Object.fromEntries(Object.entries(report.aggregate.engines)
      .map(([engine, aggregate]: [string, any]) => {
        const {
          runtimePerAudioMinuteSeconds: _runtime,
          ...durationWeighted
        } = aggregate.durationWeighted;
        const {
          peakMemoryMiB: _peakMemory,
          ...diagnostics
        } = aggregate.diagnostics;
        return [engine, {
          trackCount: aggregate.trackCount,
          totalEvaluatedDurationSeconds:
            aggregate.totalEvaluatedDurationSeconds,
          durationWeighted,
          trackLevel: aggregate.trackLevel,
          diagnostics,
          errorTaxonomy: aggregate.errorTaxonomy,
        }];
      })),
    pairedDifferences: report.aggregate.pairedDifferences,
  };
}

function markdown(summary: any): string {
  const lines = [
    "# Frozen hybrid p00 cross-capture validation",
    "",
    `Frozen candidate: **${summary.frozen.candidateId}**`,
    "",
    `Changed from current defaults: **${summary.frozen.changedFromCurrent ? "yes" : "no"}**`,
    "",
    "| Capture | Tracks | Duration | Current/calibrated identical | Hybrid fallbacks |",
    "|---|---:|---:|---:|---:|",
    ...Object.entries(summary.captures).map(([capture, value]: [string, any]) =>
      `| ${capture} | ${value.trackCount}`
        + ` | ${value.durationSeconds.toFixed(3)} s`
        + ` | ${value.identicalToCurrentBenchmark ? "yes" : "no"}`
        + ` | ${value.fallbackCount} |`),
    "",
    "## Decision",
    "",
    summary.decision,
    "",
    "The final validation was executed only after the non-p00 settings freeze.",
    "The original benchmark reports remain unchanged.",
  ];
  return `${lines.join("\n")}\n`;
}

export async function runCalibratedP00Validation(
  repositoryRoot: string,
  environment: ValidationEnvironment = process.env,
): Promise<Record<string, any>> {
  const evaluationDirectory = path.join(repositoryRoot, "evaluation");
  const reportsDirectory = path.join(evaluationDirectory, "reports");
  const frozenPath = path.join(
    evaluationDirectory,
    "hybrid-calibration-frozen-settings.json",
  );
  const frozen = JSON.parse(
    await readFile(frozenPath, "utf8"),
  ) as FrozenCalibrationSettings;
  assertFinalValidationIsFrozen(frozen);
  if (frozen.sealedFinalValidationPerformer !== "guitarset-p00") {
    throw new Error("Frozen calibration does not seal p00 as final validation");
  }
  if (settingsIdentity(frozen.settings, frozen.probabilityCalibration)
    !== frozen.settingsIdentity) {
    throw new Error("Frozen hybrid settings identity is inconsistent");
  }
  if (!frozen.adaptiveWeighting) {
    throw new Error(
      "The application runner supports the production adaptive path only;"
        + " a non-adaptive candidate cannot be promoted silently",
    );
  }
  if (frozen.probabilityCalibration.kind !== "none") {
    throw new Error(
      "Probability calibration was selected but is not implemented in the"
        + " production hybrid path; p00 validation aborted",
    );
  }

  const configPath = path.join(
    evaluationDirectory,
    "hybrid-accuracy-config.json",
  );
  const captures = [
    {
      captureType: "audio_mono-mic",
      tag: "calibrated-mono-mic",
      audioRoot: requiredPath(environment, "TABSMITH_GUITARSET_MIC_AUDIO"),
      oldReport: "hybrid-accuracy-mono-mic.json",
      newReport: "hybrid-calibrated-mono-mic.json",
    },
    {
      captureType: "audio_mono-pickup_mix",
      tag: "calibrated-pickup-mix",
      audioRoot: requiredPath(environment, "TABSMITH_GUITARSET_PICKUP_AUDIO"),
      oldReport: "hybrid-accuracy-pickup-mix.json",
      newReport: "hybrid-calibrated-pickup-mix.json",
    },
  ];
  const summaryCaptures: Record<string, unknown> = {};
  for (const capture of captures) {
    const report = await runHybridComparison(
      configPath,
      reportsDirectory,
      {
        ...environment,
        TABSMITH_GUITARSET_AUDIO: capture.audioRoot,
        TABSMITH_HYBRID_EVAL_CAPTURE: capture.captureType,
        TABSMITH_HYBRID_EVAL_OUTPUT_TAG: capture.tag,
      },
      frozen.settings,
    );
    const old = JSON.parse(await readFile(
      path.join(reportsDirectory, capture.oldReport),
      "utf8",
    ));
    const identicalToCurrentBenchmark =
      JSON.stringify(metricIdentity(report)) === JSON.stringify(metricIdentity(old));
    if (!frozen.changedFromCurrent && !identicalToCurrentBenchmark) {
      throw new Error(
        `${capture.captureType}: unchanged frozen settings did not reproduce`
          + " the current benchmark metrics",
      );
    }
    const hybridAggregate = report.aggregate.engines["observation-hybrid"];
    if (!hybridAggregate) {
      throw new Error(`${capture.captureType}: hybrid aggregate is missing`);
    }
    summaryCaptures[capture.captureType] = {
      report: capture.newReport,
      previousBenchmark: capture.oldReport,
      trackCount: report.datasets[0]?.trackCount ?? 0,
      durationSeconds:
        report.datasets[0]?.totalEvaluatedDurationSeconds ?? 0,
      actualTypeScriptHybridDecoderExecuted:
        report.methodology.actualTypeScriptHybridDecoderExecuted,
      sharedLearnedResponse:
        report.methodology.mlOnlyAndHybridShareOneLearnedResponseObject,
      fallbackCount: report.fairness.fallbackTrackCount,
      identicalToCurrentBenchmark,
      hybridMetrics: hybridAggregate.durationWeighted,
      pairedDifferences: report.aggregate.pairedDifferences,
    };
  }
  const summary = {
    schemaVersion: 1,
    status: "completed-frozen-p00-validation",
    frozen: {
      candidateId: frozen.candidateId,
      settingsIdentity: frozen.settingsIdentity,
      changedFromCurrent: frozen.changedFromCurrent,
      settings: frozen.settings,
      probabilityCalibration: frozen.probabilityCalibration,
      selectionReason: frozen.selectionReason,
    },
    methodology: {
      settingsFrozenBeforeP00: true,
      finalValidationPerformer: "guitarset-p00",
      alternateCapturesNotCombinedAsIndependentTracks: true,
      actualTypeScriptHybridPathRequired: true,
    },
    captures: summaryCaptures,
    decision: frozen.changedFromCurrent
      ? "Use the calibrated reports to decide whether the frozen candidate should replace the previous experimental defaults."
      : "No eligible calibration candidate replaced the current settings; the p00 rerun reproduces the existing benchmark and the current experimental defaults remain unchanged.",
  };
  assertNoAbsolutePaths(summary);
  await mkdir(reportsDirectory, { recursive: true });
  await Promise.all([
    writeFile(
      path.join(reportsDirectory, "hybrid-calibrated-cross-capture.json"),
      `${JSON.stringify(summary, null, 2)}\n`,
      "utf8",
    ),
    writeFile(
      path.join(reportsDirectory, "hybrid-calibrated-cross-capture.md"),
      markdown(summary),
      "utf8",
    ),
  ]);
  return summary;
}

if (process.env.npm_lifecycle_event === "evaluate:hybrid:calibrated") {
  await runCalibratedP00Validation(process.cwd()).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
