import { describe, expect, it } from "vitest";
import { buildScore, type Score } from "../src/score";
import { cursorAtTime, layoutScore, ROW_HEIGHT, visibleRowRange } from "../src/tabLayout";

function scoreWithMeasures(count: number): Score {
  // One note per measure so every bar exists; 120 bpm => 2s bars.
  const notes = Array.from({ length: count }, (_, i) => ({
    start: i * 2,
    end: i * 2 + 0.5,
    midi: 64,
    name: "E4",
    confidence: 1,
    string: 1,
    fret: 0,
  }));
  return buildScore({ notes, duration: count * 2, bpm: 120, tuning: [], capo: 0 });
}

describe("layoutScore", () => {
  it("wraps measures into multiple rows when width is constrained", () => {
    const score = scoreWithMeasures(6);
    const layout = layoutScore(score, { availableWidth: 200, zoom: 1 });
    expect(layout.rows.length).toBeGreaterThan(1);
    // Every measure is placed exactly once.
    const placed = layout.rows.flatMap((row) => row.measures.map((m) => m.measureIndex)).sort((a, b) => a - b);
    expect(placed).toEqual([0, 1, 2, 3, 4, 5]);
    expect(layout.measureLocation.size).toBe(6);
  });

  it("fits more measures per row when width is generous", () => {
    const score = scoreWithMeasures(6);
    const narrow = layoutScore(score, { availableWidth: 200, zoom: 1 });
    const wide = layoutScore(score, { availableWidth: 2000, zoom: 1 });
    expect(wide.rows.length).toBeLessThan(narrow.rows.length);
    expect(wide.rows[0].measures.length).toBeGreaterThan(narrow.rows[0].measures.length);
  });
});

describe("cursorAtTime", () => {
  it("advances the playhead monotonically across a measure", () => {
    const score = scoreWithMeasures(4);
    const layout = layoutScore(score, { availableWidth: 2000, zoom: 1 });
    const early = cursorAtTime(score, layout, 0.1)!;
    const late = cursorAtTime(score, layout, 1.9)!;
    expect(early.rowIndex).toBe(0);
    expect(late.rowIndex).toBe(0);
    expect(late.x).toBeGreaterThan(early.x);
  });

  it("moves to the correct row for wrapped measures", () => {
    const score = scoreWithMeasures(6);
    const layout = layoutScore(score, { availableWidth: 200, zoom: 1 });
    const start = cursorAtTime(score, layout, 0)!;
    const laterMeasure = cursorAtTime(score, layout, 9)!; // measure index 4
    expect(start.rowIndex).toBe(0);
    expect(laterMeasure.measureIndex).toBe(4);
    expect(laterMeasure.rowY).toBe(laterMeasure.rowIndex * ROW_HEIGHT);
  });

  it("clamps to the score bounds", () => {
    const score = scoreWithMeasures(2);
    const layout = layoutScore(score, { availableWidth: 2000, zoom: 1 });
    expect(cursorAtTime(score, layout, -5)!.measureIndex).toBe(0);
    expect(cursorAtTime(score, layout, 999)!.measureIndex).toBe(1);
  });
});

describe("visibleRowRange", () => {
  it("returns only rows overlapping the viewport plus overscan", () => {
    const score = scoreWithMeasures(20);
    const layout = layoutScore(score, { availableWidth: 200, zoom: 1 });
    const range = visibleRowRange(layout, ROW_HEIGHT * 5, ROW_HEIGHT * 2, 1);
    expect(range.first).toBeLessThanOrEqual(4);
    expect(range.last).toBeGreaterThanOrEqual(7);
    expect(range.last - range.first).toBeLessThan(layout.rows.length);
  });
});
