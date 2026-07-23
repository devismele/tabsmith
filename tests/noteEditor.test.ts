import { describe, expect, it } from "vitest";
import {
  noteEditorWindowKey,
  selectNoteEditorRows,
} from "../src/noteEditor";
import type { AnalysisResult } from "../src/types";

const result = {
  duration: 120,
  bpm: 120,
  notes: [
    { start: 0, end: 0.5, midi: 60, name: "C4", confidence: 0.9, string: 2, fret: 1 },
    { start: 98.1, end: 98.5, midi: 64, name: "E4", confidence: 0.9, string: 1, fret: 0 },
    { start: 99.2, end: 99.5, midi: 65, name: "F4", confidence: 0.4, string: 1, fret: 1 },
  ],
  chords: [
    { start: 0, end: 96, name: "E", confidence: 0.9 },
    { start: 96, end: 104, name: "A", confidence: 0.9 },
  ],
  noteAnalysis: {
    rawNotes: [
      { start: 0, end: 0.1, midi: 48, name: "C3", confidence: 0.3 },
      { start: 98.05, end: 98.2, midi: 64, name: "E4", confidence: 0.7 },
    ],
    settings: { minimumConfidence: 0.42 },
  },
} as AnalysisResult;

describe("playback-following note editor", () => {
  it("shows nearby final arranged notes after seeking to 1:38", () => {
    const rows = selectNoteEditorRows(result, 98, "follow");
    expect(rows.every((row) => row.kind === "final")).toBe(true);
    expect(rows.map((row) => row.note.start)).toEqual([98.1, 99.2]);
    expect(rows[0].note.start).toBeGreaterThanOrEqual(98);
  });

  it("recomputes its follow window after seeking", () => {
    expect(noteEditorWindowKey(result, 0, "follow"))
      .not.toBe(noteEditorWindowKey(result, 98, "follow"));
  });

  it("shows raw events only in the explicit raw diagnostic mode", () => {
    expect(selectNoteEditorRows(result, 98, "all").every(
      (row) => row.kind === "final",
    )).toBe(true);
    expect(selectNoteEditorRows(result, 98, "raw").every(
      (row) => row.kind === "raw",
    )).toBe(true);
  });
});
