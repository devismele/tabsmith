import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import {
  buildCrossCaptureSummary,
  crossCaptureSummaryMarkdown,
} from "./hybridCrossCapture";
import {
  assertNoAbsolutePaths,
  type HybridAccuracyReport,
} from "./hybridReport";

function argument(name: string, fallback: string): string {
  const index = process.argv.indexOf(name);
  return path.resolve(index >= 0 && process.argv[index + 1]
    ? process.argv[index + 1]
    : fallback);
}

const microphonePath = argument(
  "--microphone",
  "evaluation/reports/hybrid-accuracy-mono-mic.json",
);
const pickupPath = argument(
  "--pickup",
  "evaluation/reports/hybrid-accuracy-pickup-mix.json",
);
const outputDirectory = argument("--output", "evaluation/reports");

const microphone = JSON.parse(
  await readFile(microphonePath, "utf8"),
) as HybridAccuracyReport;
const pickupMix = JSON.parse(
  await readFile(pickupPath, "utf8"),
) as HybridAccuracyReport;
const summary = buildCrossCaptureSummary(
  microphone,
  pickupMix,
  new Date().toISOString(),
);
assertNoAbsolutePaths(summary);
await Promise.all([
  writeFile(
    path.join(outputDirectory, "hybrid-accuracy-cross-capture-summary.json"),
    `${JSON.stringify(summary, null, 2)}\n`,
    "utf8",
  ),
  writeFile(
    path.join(outputDirectory, "hybrid-accuracy-cross-capture-summary.md"),
    crossCaptureSummaryMarkdown(summary),
    "utf8",
  ),
]);
