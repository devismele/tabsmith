import { readFile } from "node:fs/promises";
import { processDetectedNotes } from "../src/noteCleanup";
import type { CachedTranscriptionEntry } from "../src/transcriptionCache";

const input = process.argv[2];
if (!input) throw new Error("Pass a transcription cache JSON path.");
const entry = JSON.parse(await readFile(input, "utf8")) as CachedTranscriptionEntry;
const original = entry.result;
const processed = processDetectedNotes(
  original.noteAnalysis.rawNotes,
  original.duration,
  original.bpm,
  original.chords,
  original.chordAnalysis.diagnostics.rawChordChanges,
  { mode: "lead", preset: "balanced", separation: original.separation },
);
const nearSection = processed.notes.filter(
  (note) => note.end > 92 && note.start < 105,
);
process.stdout.write(`${JSON.stringify({
  diagnostics: processed.noteAnalysis.diagnostics,
  sectionNotes: nearSection.map((note) => ({
    start: note.start,
    end: note.end,
    name: note.name,
    string: note.string,
    fret: note.fret,
  })),
}, null, 2)}\n`);
