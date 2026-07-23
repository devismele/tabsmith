import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { evaluateTranscription } from "../server/evaluation.mjs";

const manifestPath = path.resolve(process.argv[2] || "benchmarks/manifest.json");
const manifest = JSON.parse(await readFile(manifestPath, "utf8"));
if (!Array.isArray(manifest.songs) || !manifest.songs.length) throw new Error("Benchmark manifest contains no songs.");
const root = path.dirname(manifestPath);
const rows = [];
for (const song of manifest.songs) {
  const reference = JSON.parse(await readFile(path.resolve(root, song.reference), "utf8"));
  const prediction = JSON.parse(await readFile(path.resolve(root, song.prediction), "utf8"));
  rows.push({ name: song.name, ...evaluateTranscription(reference, prediction, manifest.onsetTolerance ?? 0.08) });
}
const average = {};
for (const field of ["notePrecision", "noteRecall", "noteF1", "onsetMaeMs", "fretPositionAccuracy", "chordTimeAccuracy"]) {
  const values = rows.map((row) => row[field]).filter(Number.isFinite);
  average[field] = values.length ? Math.round(values.reduce((sum, value) => sum + value, 0) / values.length * 10_000) / 10_000 : null;
}
const report = { createdAt: new Date().toISOString(), manifest: manifestPath, songs: rows, average };
await mkdir(path.resolve("benchmark-results"), { recursive: true });
const outputPath = path.resolve("benchmark-results", `benchmark-${Date.now()}.json`);
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.table(rows);
console.log(`Report: ${outputPath}`);
