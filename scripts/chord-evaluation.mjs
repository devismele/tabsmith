import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import {
  aggregateChordEvaluationReports,
  chordEvaluationReportMarkdown,
  evaluateAblations,
  evaluateChordReference,
  summarizeAblationResults,
} from "../server/evaluation.mjs";

const manifestPath = path.resolve(process.argv[2] || "evaluation/chord-reference-songs.json");
const manifestRoot = path.dirname(manifestPath);
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
const songs = Array.isArray(manifest) ? manifest : manifest.songs;
if (!Array.isArray(songs)) throw new Error("Chord evaluation manifest must contain a song array.");

const reports = [];
const ablations = [];
for (const song of songs) {
  if (!song.referencePath || !song.predictionPath) continue;
  const reference = JSON.parse(await readFile(path.resolve(manifestRoot, song.referencePath), "utf8"));
  const prediction = JSON.parse(await readFile(path.resolve(manifestRoot, song.predictionPath), "utf8"));
  const analysis = prediction.chordAnalysis ? prediction : prediction.result || prediction;
  const report = evaluateChordReference(reference, analysis, {
    context: {
      recordingMismatch: Boolean(song.recordingMismatch),
      referenceSimplified: Boolean(song.referenceSimplified),
    },
  });
  reports.push(report);

  if (song.ablationPredictionPaths) {
    const predictions = {};
    for (const [id, file] of Object.entries(song.ablationPredictionPaths)) {
      const value = JSON.parse(await readFile(path.resolve(manifestRoot, file), "utf8"));
      predictions[id] = value.chordAnalysis ? value : value.result || value;
    }
    ablations.push({
      song: { artist: reference.artist, title: reference.title },
      configurations: evaluateAblations(reference, predictions),
    });
  }
}

const aggregate = aggregateChordEvaluationReports(reports);
aggregate.ablationResults = ablations;
aggregate.ablationSummary = summarizeAblationResults(ablations);
const outputRoot = path.resolve("evaluation");
await mkdir(outputRoot, { recursive: true });
await writeFile(
  path.join(outputRoot, "chord-evaluation-report.json"),
  `${JSON.stringify(aggregate, null, 2)}\n`,
  "utf8",
);
await writeFile(
  path.join(outputRoot, "chord-evaluation-report.md"),
  chordEvaluationReportMarkdown(aggregate),
  "utf8",
);

if (!reports.length) {
  console.log("No authorized local reference/prediction pairs were configured; no songs were evaluated.");
  console.log("Add private referencePath and predictionPath fields, then rerun this command.");
} else {
  console.table(reports.map((report) => ({
    song: `${report.song.artist} — ${report.song.title}`,
    root: report.metrics.rootAccuracy,
    quality: report.metrics.majorMinorAccuracy,
    detailed: report.metrics.detailedAccuracy,
    medianBoundaryMs: report.metrics.medianBoundaryErrorMs,
    falseN: report.metrics.falseNoChordDurationPercent,
  })));
}
console.log(`Reports: ${path.join(outputRoot, "chord-evaluation-report.json")}`);
