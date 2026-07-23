import { describe, expect, it } from "vitest";
import { buildTablature, tablatureToText } from "../src/tablature";
import type { NoteEvent } from "../src/types";

describe("tablature transcription", () => {
  it("renders simultaneous notes on conventional high-to-low string lines", () => {
    const notes: NoteEvent[] = [
      { start: 0, end: 1, midi: 40, name: "E2", confidence: 1, string: 6, fret: 0 },
      { start: 0, end: 1, midi: 47, name: "B2", confidence: 1, string: 5, fret: 2 },
      { start: 0, end: 1, midi: 64, name: "E4", confidence: 1, string: 1, fret: 0 },
    ];
    const [measure] = buildTablature(notes, 2, 120);
    expect(measure.lines[0]).toMatch(/^e\|-0-/);
    expect(measure.lines[4]).toMatch(/^A\|-2-/);
    expect(measure.lines[5]).toMatch(/^E\|-0-/);
    expect(measure.lines[5]).toContain("~~~");
    expect(measure.guide).toContain("1  e  &  a");
  });

  it("splits notes into four-beat measures and exports text", () => {
    const notes: NoteEvent[] = [
      { start: 2.1, end: 2.4, midi: 64, name: "E4", confidence: 1, string: 1, fret: 0 },
    ];
    const measures = buildTablature(notes, 4, 120);
    expect(measures).toHaveLength(2);
    expect(tablatureToText(measures, "Demo", 120)).toContain("Bar 2");
  });

  it("does not print sustain through a new attack on the same string", () => {
    const notes: NoteEvent[] = [
      { start: 0, end: 1.5, midi: 64, name: "E4", confidence: 1, string: 1, fret: 0 },
      { start: 0.5, end: 0.51, midi: 65, name: "F4", confidence: 1, string: 1, fret: 1 },
    ];
    const [measure] = buildTablature(notes, 2, 120);
    const tokens = measure.lines[0].slice(2, -1).match(/.{3}/g)!;
    expect(tokens[4]).toBe("-1-");
    expect(tokens.slice(5, 12)).not.toContain("~~~");
  });
});
