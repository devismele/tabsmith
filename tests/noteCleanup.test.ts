import { describe, expect, it } from "vitest";
import {
  arrangeDetectedNotes,
  cleanupDetectedNotes,
  clusterNoteOnsets,
  maximumPolyphony,
  NOTE_CLEANUP_PRESETS,
  processDetectedNotes,
} from "../src/noteCleanup";
import type { ChordEvent, RawNoteEvent } from "../src/types";

function raw(
  midi: number,
  start: number,
  end: number,
  confidence = 0.9,
): RawNoteEvent {
  return { midi, start, end, confidence, name: `m${midi}` };
}

const eChord: ChordEvent[] = [{ start: 0, end: 4, name: "E", confidence: 0.95 }];

describe("note cleanup", () => {
  it("merges split same-pitch detections across a tiny gap", () => {
    const result = cleanupDetectedNotes([
      raw(64, 1, 1.22),
      raw(64, 1.25, 1.47),
    ]);
    expect(result.notes).toHaveLength(1);
    expect(result.notes[0]).toMatchObject({ start: 1, end: 1.47, midi: 64 });
    expect(result.counts.duplicatesMerged).toBe(1);
  });

  it("removes a very short low-confidence artifact", () => {
    const result = cleanupDetectedNotes([raw(60, 0, 0.04, 0.45)]);
    expect(result.notes).toEqual([]);
  });

  it("preserves strong short notes in a coherent fast phrase", () => {
    const result = cleanupDetectedNotes([
      raw(60, 0, 0.06, 0.94),
      raw(62, 0.1, 0.16, 0.93),
      raw(64, 0.2, 0.26, 0.95),
    ]);
    expect(result.notes).toHaveLength(3);
  });

  it("suppresses a weak simultaneous octave ghost", () => {
    const result = cleanupDetectedNotes([
      raw(52, 0, 0.3, 0.92),
      raw(64, 0.01, 0.31, 0.3),
    ]);
    expect(result.notes.map((note) => note.midi)).toEqual([52]);
    expect(result.counts.octaveGhostsRemoved).toBe(1);
  });

  it("clusters detector-jittered onsets into one musical attack", () => {
    const result = cleanupDetectedNotes([
      raw(52, 1, 1.5),
      raw(56, 1.035, 1.5),
      raw(59, 1.039, 1.5),
    ]);
    expect(clusterNoteOnsets(result.notes, 0.04)).toHaveLength(1);
    expect(new Set(result.notes.map((note) => note.start))).toEqual(new Set([1]));
  });
});

describe("arrangement selection", () => {
  it("Lead mode never exceeds two simultaneous notes", () => {
    const notes = Array.from({ length: 5 }, (_, onset) => (
      [60, 64, 67, 71].map((midi, voice) => raw(
        midi + onset,
        onset * 0.1 + voice * 0.002,
        onset * 0.1 + 0.5,
        0.95 - voice * 0.04,
      ))
    )).flat();
    const arranged = arrangeDetectedNotes(
      notes,
      eChord,
      120,
      "lead",
      NOTE_CLEANUP_PRESETS.balanced,
    );
    expect(maximumPolyphony(arranged.notes)).toBeLessThanOrEqual(2);
  });

  it("Lead mode skips an isolated detector onset that would force a huge register jump", () => {
    const arranged = arrangeDetectedNotes([
      raw(69, 0, 0.18, 0.94),
      raw(40, 0.2, 0.34, 0.9),
      raw(70, 0.4, 0.58, 0.94),
    ], eChord, 120, "lead", NOTE_CLEANUP_PRESETS.balanced);
    expect(arranged.notes.map((note) => note.midi)).toEqual([69, 70]);
  });

  it("Lead mode can reset register when a new passage sustains the change", () => {
    const notes = [
      raw(69, 0, 0.18, 0.94),
      ...Array.from({ length: 12 }, (_, index) => (
        raw(43 + (index % 3), 0.2 + index * 0.2, 0.36 + index * 0.2, 0.92)
      )),
    ];
    const arranged = arrangeDetectedNotes(
      notes,
      eChord,
      120,
      "lead",
      NOTE_CLEANUP_PRESETS.balanced,
    );
    expect(arranged.notes.some((note) => note.midi < 50)).toBe(true);
  });

  it("Rhythm mode preserves a playable, aligned E chord attack", () => {
    const notes = [40, 47, 52, 56, 59, 64].map(
      (midi, index) => raw(midi, index * 0.004, 0.8, 0.95),
    );
    const result = processDetectedNotes(notes, 2, 120, eChord, 0, {
      mode: "rhythm",
      preset: "balanced",
    });
    expect(result.notes.length).toBeGreaterThanOrEqual(3);
    expect(new Set(result.notes.map((note) => note.string)).size).toBe(result.notes.length);
    const activeFrets = result.notes.map((note) => note.fret).filter((fret) => fret > 0);
    expect(Math.max(...activeFrets) - Math.min(...activeFrets)).toBeLessThanOrEqual(4);
  });

  it("keeps raw detections available while exposing only arranged notes", () => {
    const rawNotes = [
      raw(60, 0, 0.3, 0.95),
      raw(64, 0, 0.3, 0.88),
      raw(67, 0, 0.3, 0.84),
      raw(71, 0, 0.3, 0.8),
    ];
    const result = processDetectedNotes(rawNotes, 2, 120, eChord, 0, {
      mode: "lead",
      preset: "balanced",
    });
    expect(result.noteAnalysis.rawNotes).toHaveLength(4);
    expect(result.notes.length).toBeLessThanOrEqual(2);
    expect(result.noteAnalysis.diagnostics.rawNoteCount).toBe(4);
    expect(result.noteAnalysis.diagnostics.arrangedNoteCount).toBe(result.notes.length);
  });

  it("Automatic mode selects a lead part for implausibly dense polyphony", () => {
    const rawNotes = Array.from({ length: 8 }, (_, index) => (
      raw(52 + index, 0, 0.4, 0.9)
    ));
    const result = processDetectedNotes(rawNotes, 1, 120, eChord, 0, {
      mode: "automatic",
      preset: "balanced",
    });
    expect(result.noteAnalysis.resolvedMode).toBe("lead");
    expect(result.noteAnalysis.diagnostics.maximumPolyphonyCleaned).toBeLessThanOrEqual(2);
  });

  it("reports density, duration, polyphony, and fret-movement diagnostics", () => {
    const result = processDetectedNotes([
      raw(60, 0, 0.05, 0.3),
      raw(64, 0.2, 0.5, 0.95),
      raw(65, 0.6, 0.9, 0.95),
    ], 1, 120, eChord, 12, { mode: "lead" });
    expect(result.noteAnalysis.diagnostics).toMatchObject({
      rawNoteCount: 3,
      notesBelow100msRaw: 1,
      maximumPolyphonyRaw: 1,
    });
    expect(result.noteAnalysis.diagnostics.averageFretMovementFinal).toBeGreaterThanOrEqual(0);
    expect(result.noteAnalysis.warnings.some((warning) => warning.code === "unstable-chords")).toBe(true);
  });
});
