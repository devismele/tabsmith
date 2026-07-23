import { randomUUID } from "node:crypto";
import {
  mkdir,
  readFile,
  readdir,
  rename,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import {
  aggregateChordEvaluationReports,
  chordEvaluationReportMarkdown,
  createManualAlignment,
  createManualReference,
  evaluateChordReference,
} from "./evaluation.mjs";

function evaluationRoot() {
  return process.env.TABSMITH_EVALUATION_DIR || path.resolve("evaluation", "local");
}

function referencesRoot() {
  return path.join(evaluationRoot(), "references");
}

function reportsRoot() {
  return path.join(evaluationRoot(), "song-reports");
}

function safeId(value) {
  if (!/^[a-zA-Z0-9-]{8,80}$/.test(String(value || ""))) throw new Error("Invalid evaluation identifier.");
  return value;
}

async function writeJsonAtomic(filePath, value) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporary, filePath);
}

export async function importChordReference(input) {
  const reference = createManualReference(input);
  reference.id = randomUUID();
  await writeJsonAtomic(path.join(referencesRoot(), `${reference.id}.json`), reference);
  return reference;
}

export async function listChordReferences() {
  await mkdir(referencesRoot(), { recursive: true });
  const files = (await readdir(referencesRoot())).filter((file) => file.endsWith(".json"));
  const references = [];
  for (const file of files) {
    try {
      references.push(JSON.parse(await readFile(path.join(referencesRoot(), file), "utf8")));
    } catch {
      // One malformed local evaluation file must not hide the other references.
    }
  }
  return references.sort((a, b) =>
    String(a.artist).localeCompare(String(b.artist)) || String(a.title).localeCompare(String(b.title)));
}

export async function loadChordReference(id) {
  const reference = JSON.parse(await readFile(path.join(referencesRoot(), `${safeId(id)}.json`), "utf8"));
  if (!reference?.id || !Array.isArray(reference.chords)) throw new Error("Stored chord reference is invalid.");
  return reference;
}

export async function saveManualReferenceAlignment(id, marks, duration) {
  const reference = await loadChordReference(id);
  reference.alignment = createManualAlignment(reference, marks, duration);
  reference.updatedAt = new Date().toISOString();
  await writeJsonAtomic(path.join(referencesRoot(), `${reference.id}.json`), reference);
  return reference;
}

export async function compareStoredReference(id, prediction, context = {}) {
  const reference = await loadChordReference(id);
  const report = evaluateChordReference(reference, prediction, { context });
  await writeJsonAtomic(path.join(reportsRoot(), `${reference.id}.json`), report);
  return { report, aggregate: await rebuildAggregateReports() };
}

export async function rebuildAggregateReports() {
  await mkdir(reportsRoot(), { recursive: true });
  const files = (await readdir(reportsRoot())).filter((file) => file.endsWith(".json"));
  const reports = [];
  for (const file of files) {
    try {
      reports.push(JSON.parse(await readFile(path.join(reportsRoot(), file), "utf8")));
    } catch {
      // Preserve all usable per-song results if one local file was interrupted.
    }
  }
  const aggregate = aggregateChordEvaluationReports(reports);
  await writeJsonAtomic(path.join(evaluationRoot(), "chord-evaluation-report.json"), aggregate);
  await writeFile(
    path.join(evaluationRoot(), "chord-evaluation-report.md"),
    chordEvaluationReportMarkdown(aggregate),
    "utf8",
  );
  return aggregate;
}

export async function latestAggregateReport() {
  try {
    return JSON.parse(await readFile(path.join(evaluationRoot(), "chord-evaluation-report.json"), "utf8"));
  } catch {
    return rebuildAggregateReports();
  }
}

export function evaluationStorageInfo() {
  return {
    directory: evaluationRoot(),
    productionProjectDataSeparated: true,
    networkImportEnabled: false,
    storedFields: "normalized chord labels, source metadata, alignment marks, and evaluation statistics",
  };
}
