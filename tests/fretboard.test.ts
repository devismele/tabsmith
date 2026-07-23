import { describe, expect, it } from "vitest";
import {
  clipNoteSustains,
  estimateGuitarSetup,
  fretMovementDiagnostics,
  mapNotesToFretboard,
  mapPolyphonicNotesToFretboard,
  positionsForMidi,
} from "../src/fretboard";

describe("fretboard mapping", () => {
  it("finds every playable standard-tuning position", () => {
    expect(positionsForMidi(64)).toEqual([
      { string: 6, fret: 24 },
      { string: 5, fret: 19 },
      { string: 4, fret: 14 },
      { string: 3, fret: 9 },
      { string: 2, fret: 5 },
      { string: 1, fret: 0 },
    ].filter((position) => position.fret <= 20));
  });

  it("conservatively detects repeated Drop D open-string evidence", () => {
    const events = [38, 45, 50, 38, 45, 50].map((midi, index) => ({ start: index, end: index + 0.8, midi, name: "note", confidence: 1 }));
    expect(estimateGuitarSetup(events).name).toBe("Drop D");
  });

  it("chooses a playable low-movement path", () => {
    const notes = [60, 62, 64].map((midi, index) => ({
      start: index * 0.5,
      end: index * 0.5 + 0.4,
      midi,
      name: "note",
      confidence: 0.9,
    }));
    const mapped = mapNotesToFretboard(notes);
    expect(mapped).toHaveLength(3);
    expect(mapped.every((note) => note.fret >= 0 && note.fret <= 20)).toBe(true);
    expect(Math.max(...mapped.map((note) => note.fret)) - Math.min(...mapped.map((note) => note.fret))).toBeLessThanOrEqual(4);
  });

  it("assigns a simultaneous E major chord to six distinct strings", () => {
    const notes = [40, 47, 52, 56, 59, 64].map((midi) => ({
      start: 0,
      end: 1,
      midi,
      name: "note",
      confidence: 0.9,
    }));
    const mapped = mapPolyphonicNotesToFretboard(notes);
    expect(new Set(mapped.map((note) => note.string)).size).toBe(6);
    expect(mapped.map((note) => [note.string, note.fret])).toEqual([
      [6, 0], [5, 2], [4, 2], [3, 1], [2, 0], [1, 0],
    ]);
  });

  it("keeps a repeated melody in a stable hand position", () => {
    const notes = [64, 65, 67, 64, 65, 67].map((midi, index) => ({
      start: index * 0.25,
      end: index * 0.25 + 0.2,
      midi,
      name: "note",
      confidence: 0.95,
    }));
    const mapped = mapPolyphonicNotesToFretboard(notes, 0.04);
    expect(fretMovementDiagnostics(mapped).maximum).toBeLessThanOrEqual(3);
  });

  it("respects maximum stretch and distinct strings for noisy chords", () => {
    const notes = [48, 52, 55, 60, 64, 67].map((midi) => ({
      start: 0,
      end: 1,
      midi,
      name: "note",
      confidence: 0.9,
    }));
    const mapped = mapPolyphonicNotesToFretboard(notes, 0.04);
    const frets = mapped.map((note) => note.fret).filter((fret) => fret > 0);
    expect(new Set(mapped.map((note) => note.string)).size).toBe(mapped.length);
    expect(Math.max(...frets) - Math.min(...frets)).toBeLessThanOrEqual(4);
  });

  it("clips sustain when another note begins on the same string", () => {
    const notes = [
      { start: 0, end: 1, midi: 64, name: "E4", confidence: 0.9, string: 1, fret: 0 },
      { start: 0.4, end: 0.8, midi: 65, name: "F4", confidence: 0.9, string: 1, fret: 1 },
    ];
    const clipped = clipNoteSustains(notes, 120);
    expect(clipped.notes[0].end).toBe(0.4);
    expect(clipped.overlapsResolved).toBe(1);
  });
});
