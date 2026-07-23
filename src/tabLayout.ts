import type { Score, ScoreMeasure } from "./score";

// Pure geometry for the tab renderer. No DOM access, so it is unit-testable in
// the node test environment and shared by rendering, hit-testing and the cursor.

export type LayoutOptions = {
  /** Usable width in CSS pixels for a row of measures. */
  availableWidth: number;
  /** Zoom multiplier applied to horizontal spacing (>0). */
  zoom: number;
};

export const STRING_COUNT = 6;
export const STRING_GAP = 12; // px between adjacent strings
export const STRING_BLOCK = STRING_GAP * (STRING_COUNT - 1); // 60px
export const ROW_TOP_PAD = 30; // technique + measure-number band
export const ROW_BOTTOM_PAD = 40; // rhythm stems / beams band
export const ROW_GAP = 26;
export const ROW_HEIGHT = ROW_TOP_PAD + STRING_BLOCK + ROW_BOTTOM_PAD + ROW_GAP;

const MEASURE_MIN_WIDTH = 118;
const COLUMN_SPACING = 30; // min px reserved per rhythmic column
const MEASURE_INSET = 16; // padding inside a bar before the first column
const ROW_SIDE_PAD = 8;

export type MeasureBox = {
  measureIndex: number;
  /** X of the measure's opening bar line within its row. */
  x: number;
  width: number;
};

export type RowBox = {
  index: number;
  y: number;
  height: number;
  measures: MeasureBox[];
};

export type TabLayout = {
  rows: RowBox[];
  width: number;
  height: number;
  zoom: number;
  /** measureIndex -> { rowIndex, box } for O(1) lookup. */
  measureLocation: Map<number, { rowIndex: number; box: MeasureBox }>;
};

/** Natural (unwrapped) width a measure needs given its column count and zoom. */
export function measureWidth(measure: ScoreMeasure, zoom: number): number {
  const columns = Math.max(measure.columns.length, 1);
  const natural = MEASURE_INSET * 2 + columns * COLUMN_SPACING;
  return Math.max(MEASURE_MIN_WIDTH, natural) * zoom;
}

/** Greedily packs measures into rows that fit the available width. */
export function layoutScore(score: Score, options: LayoutOptions): TabLayout {
  const zoom = Math.max(0.1, options.zoom);
  const usable = Math.max(MEASURE_MIN_WIDTH * zoom, options.availableWidth - ROW_SIDE_PAD * 2);
  const rows: RowBox[] = [];
  const measureLocation = new Map<number, { rowIndex: number; box: MeasureBox }>();

  let current: MeasureBox[] = [];
  let cursorX = 0;
  let maxWidth = 0;

  const flush = () => {
    if (!current.length) return;
    const row: RowBox = { index: rows.length, y: rows.length * ROW_HEIGHT, height: ROW_HEIGHT, measures: current };
    for (const box of current) measureLocation.set(box.measureIndex, { rowIndex: row.index, box });
    maxWidth = Math.max(maxWidth, cursorX);
    rows.push(row);
    current = [];
    cursorX = 0;
  };

  for (const measure of score.measures) {
    const width = measureWidth(measure, zoom);
    if (current.length && cursorX + width > usable) flush();
    current.push({ measureIndex: measure.index, x: cursorX, width });
    cursorX += width;
  }
  flush();

  return {
    rows,
    width: Math.max(maxWidth, usable) + ROW_SIDE_PAD * 2,
    height: rows.length * ROW_HEIGHT,
    zoom,
    measureLocation,
  };
}

/** Absolute X (within a row) of a fractional position across a measure. */
export function columnXInBox(box: MeasureBox, fraction: number): number {
  const inset = Math.min(MEASURE_INSET, box.width * 0.14);
  const innerLeft = box.x + ROW_SIDE_PAD + inset;
  const innerWidth = box.width - inset * 2;
  return innerLeft + fraction * innerWidth;
}

export type CursorPosition = {
  x: number;
  rowIndex: number;
  rowY: number;
  measureIndex: number;
};

/** Continuous playhead position for a given time — smooth across a measure. */
export function cursorAtTime(score: Score, layout: TabLayout, time: number): CursorPosition | null {
  if (!score.measures.length || !layout.rows.length) return null;
  const clamped = Math.max(0, Math.min(time, score.duration));
  let measure = score.measures.find((m) => clamped >= m.start && clamped < m.end);
  if (!measure) measure = clamped >= score.duration ? score.measures[score.measures.length - 1] : score.measures[0];
  const located = layout.measureLocation.get(measure.index);
  if (!located) return null;
  const span = Math.max(1e-6, measure.end - measure.start);
  const fraction = Math.max(0, Math.min(1, (clamped - measure.start) / span));
  return {
    x: columnXInBox(located.box, fraction),
    rowIndex: located.rowIndex,
    rowY: located.rowIndex * ROW_HEIGHT,
    measureIndex: measure.index,
  };
}

/** Y offset of a string line (string 1 = high e at top). */
export function stringY(string: number): number {
  return ROW_TOP_PAD + (string - 1) * STRING_GAP;
}

/** Indices of rows overlapping a scroll viewport, with a small overscan. */
export function visibleRowRange(
  layout: TabLayout,
  scrollTop: number,
  viewportHeight: number,
  overscan = 1,
): { first: number; last: number } {
  if (!layout.rows.length) return { first: 0, last: -1 };
  const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - overscan);
  const last = Math.min(layout.rows.length - 1, Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT) + overscan);
  return { first, last };
}
