import { describe, expect, it } from "vitest";
import { buildScore, measureIndexAtTime, rhythmForSpan } from "../src/score";
import type { NoteEvent } from "../src/types";

const base = { midi: 64, name: "E4", confidence: 1 };

function note(partial: Partial<NoteEvent>): NoteEvent {
  return { start: 0, end: 0.2, string: 1, fret: 0, ...base, ...partial } as NoteEvent;
}

describe("buildScore", () => {
  it("splits notes into four-beat measures using the tempo", () => {
    const score = buildScore({
      notes: [note({ start: 0, string: 6, fret: 0 }), note({ start: 2.1, string: 1, fret: 0 })],
      duration: 4,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    expect(score.barDuration).toBeCloseTo(2, 6);
    expect(score.measures).toHaveLength(2);
    expect(score.measures[0].columns[0].notes[0].string).toBe(6);
    expect(score.measures[1].columns[0].notes[0].noteIndex).toBe(1);
  });

  it("stacks simultaneous notes into one column ordered high-to-low string", () => {
    const score = buildScore({
      notes: [note({ start: 0, string: 1, fret: 0 }), note({ start: 0, string: 6, fret: 3 })],
      duration: 2,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    const column = score.measures[0].columns[0];
    expect(column.notes.map((n) => n.string)).toEqual([1, 6]);
  });

  it("derives rhythmic values and sustain from onset spacing", () => {
    // barDuration 2s, slotDuration 0.125s: onsets at slot 0 and slot 8.
    const score = buildScore({
      notes: [note({ start: 0, end: 1, string: 1 }), note({ start: 1, end: 1.2, string: 1 })],
      duration: 2,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    const [a, b] = score.measures[0].columns;
    expect(a.slot).toBe(0);
    expect(b.slot).toBe(8);
    expect(a.value).toBe("half");
    expect(a.notes[0].sustainSlots).toBe(8);
    expect(b.value).toBe("half");
  });

  it("keeps the most confident note per string within a slot", () => {
    const score = buildScore({
      notes: [
        note({ start: 0, string: 1, fret: 5, confidence: 0.3 }),
        note({ start: 0.01, string: 1, fret: 7, confidence: 0.9 }),
      ],
      duration: 2,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    const column = score.measures[0].columns[0];
    expect(column.notes).toHaveLength(1);
    expect(column.notes[0].fret).toBe(7);
  });

  it("uses a coarser eighth-note grid when onset confidence is weak", () => {
    const score = buildScore({
      notes: [note({ start: 0.14, confidence: 0.3 })],
      duration: 2,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    expect(score.measures[0].columns[0].slot % 2).toBe(0);
  });

  it("stops sustain at the next attack on the same string", () => {
    const score = buildScore({
      notes: [
        note({ start: 0, end: 1.5, string: 1 }),
        note({ start: 0.5, end: 0.8, string: 1, fret: 2 }),
      ],
      duration: 2,
      bpm: 120,
      tuning: [],
      capo: 0,
    });
    expect(score.measures[0].columns[0].notes[0].sustainSlots).toBe(4);
  });
});

describe("rhythmForSpan", () => {
  it("maps slot spans to readable values", () => {
    expect(rhythmForSpan(16)).toEqual({ value: "whole", dotted: false });
    expect(rhythmForSpan(12)).toEqual({ value: "half", dotted: true });
    expect(rhythmForSpan(4)).toEqual({ value: "quarter", dotted: false });
    expect(rhythmForSpan(3)).toEqual({ value: "eighth", dotted: true });
    expect(rhythmForSpan(2)).toEqual({ value: "eighth", dotted: false });
    expect(rhythmForSpan(1)).toEqual({ value: "sixteenth", dotted: false });
  });
});

describe("measureIndexAtTime", () => {
  it("locates the sounding measure", () => {
    const score = buildScore({ notes: [], duration: 6, bpm: 120, tuning: [], capo: 0 });
    expect(measureIndexAtTime(score, 0)).toBe(0);
    expect(measureIndexAtTime(score, 2.5)).toBe(1);
    expect(measureIndexAtTime(score, 99)).toBe(score.measures.length - 1);
  });
});
