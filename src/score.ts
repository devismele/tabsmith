import type { AnalysisResult } from "./types";

// Canonical, render-agnostic musical model derived from AnalysisResult.
// The renderer draws from this; it never parses formatted tab text.

export type RhythmValue = "whole" | "half" | "quarter" | "eighth" | "sixteenth";

export type ScoreNote = {
  /** Index into AnalysisResult.notes so edits can map back to canonical data. */
  noteIndex: number;
  string: number;
  fret: number;
  midi: number;
  name: string;
  start: number;
  end: number;
  confidence: number;
  /** How many grid slots the note sustains within its measure (>= 1). */
  sustainSlots: number;
};

export type ScoreColumn = {
  /** Slot position within the measure, 0..slotsPerBar-1. */
  slot: number;
  /** Absolute onset time in seconds. */
  time: number;
  /** Position within the measure, 0..1. */
  fraction: number;
  /** Slots until the next column (or the bar end) — drives the rhythmic value. */
  slotSpan: number;
  value: RhythmValue;
  dotted: boolean;
  /** Notes sharing this onset, one per string at most. */
  notes: ScoreNote[];
};

export type ScoreMeasure = {
  index: number;
  start: number;
  end: number;
  beatsPerBar: number;
  slotsPerBeat: number;
  columns: ScoreColumn[];
};

export type Score = {
  measures: ScoreMeasure[];
  bpm: number;
  duration: number;
  tuning: string[];
  capo: number;
  slotsPerBar: number;
  slotDuration: number;
  barDuration: number;
};

export const SLOTS_PER_BAR = 16;
const BEATS_PER_BAR = 4;

/** Maps a run of grid slots to a readable rhythmic value. */
export function rhythmForSpan(slotSpan: number): { value: RhythmValue; dotted: boolean } {
  if (slotSpan >= 16) return { value: "whole", dotted: false };
  if (slotSpan >= 12) return { value: "half", dotted: true };
  if (slotSpan >= 8) return { value: "half", dotted: false };
  if (slotSpan >= 6) return { value: "quarter", dotted: true };
  if (slotSpan >= 4) return { value: "quarter", dotted: false };
  if (slotSpan >= 3) return { value: "eighth", dotted: true };
  if (slotSpan >= 2) return { value: "eighth", dotted: false };
  return { value: "sixteenth", dotted: false };
}

/** Number of flags/beams a value carries (quarter and longer: 0). */
export function flagsForValue(value: RhythmValue): number {
  return value === "sixteenth" ? 2 : value === "eighth" ? 1 : 0;
}

export function safeBpm(bpm: number | null): number {
  return bpm && bpm >= 40 && bpm <= 240 ? bpm : 120;
}

export function buildScore(analysis: Pick<AnalysisResult, "notes" | "duration" | "bpm" | "tuning" | "capo">): Score {
  const bpm = safeBpm(analysis.bpm);
  const barDuration = (60 / bpm) * BEATS_PER_BAR;
  const slotDuration = barDuration / SLOTS_PER_BAR;
  const duration = Math.max(analysis.duration, barDuration);
  const measureCount = Math.max(1, Math.ceil(duration / barDuration));
  const measures: ScoreMeasure[] = [];
  const nextStartByIndex = new Map<number, number>();
  for (let string = 1; string <= 6; string += 1) {
    const stringNotes = analysis.notes
      .map((note, index) => ({ note, index }))
      .filter(({ note }) => note.string === string)
      .sort((a, b) => a.note.start - b.note.start);
    stringNotes.forEach(({ index }, position) => {
      const next = stringNotes[position + 1];
      if (next) nextStartByIndex.set(index, next.note.start);
    });
  }

  for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
    const start = measureIndex * barDuration;
    const end = Math.min(duration, start + barDuration);
    const barEnd = start + barDuration;

    // Bucket note onsets by quantized slot, keeping the most confident note per string.
    const slotMap = new Map<number, Map<number, ScoreNote>>();
    for (let i = 0; i < analysis.notes.length; i += 1) {
      const note = analysis.notes[i];
      if (note.start < start || note.start >= barEnd) continue;
      const stringIndex = note.string;
      if (stringIndex < 1 || stringIndex > 6) continue;
      const rawSlot = Math.max(
        0,
        Math.min(SLOTS_PER_BAR - 1, Math.round(((note.start - start) / barDuration) * SLOTS_PER_BAR)),
      );
      // Weak timing evidence is displayed on an eighth-note grid. Confident
      // attacks retain the full sixteenth grid.
      const slot = note.confidence < 0.45
        ? Math.min(SLOTS_PER_BAR - 1, Math.round(rawSlot / 2) * 2)
        : rawSlot;
      const effectiveEnd = Math.min(
        note.end,
        barEnd,
        nextStartByIndex.get(i) ?? Infinity,
      );
      const sustainSlots = Math.max(
        1,
        Math.round((effectiveEnd - (start + slot * slotDuration)) / slotDuration),
      );
      const scoreNote: ScoreNote = {
        noteIndex: i,
        string: note.string,
        fret: note.fret,
        midi: note.midi,
        name: note.name,
        start: note.start,
        end: note.end,
        confidence: note.confidence,
        sustainSlots,
      };
      let strings = slotMap.get(slot);
      if (!strings) slotMap.set(slot, (strings = new Map()));
      const existing = strings.get(note.string);
      if (!existing || note.confidence > existing.confidence) strings.set(note.string, scoreNote);
    }

    const slots = [...slotMap.keys()].sort((a, b) => a - b);
    const columns: ScoreColumn[] = slots.map((slot, columnIndex) => {
      const nextSlot = columnIndex + 1 < slots.length ? slots[columnIndex + 1] : SLOTS_PER_BAR;
      const slotSpan = Math.max(1, nextSlot - slot);
      const { value, dotted } = rhythmForSpan(slotSpan);
      const notes = [...slotMap.get(slot)!.values()].sort((a, b) => a.string - b.string);
      return {
        slot,
        time: start + slot * slotDuration,
        fraction: slot / SLOTS_PER_BAR,
        slotSpan,
        value,
        dotted,
        notes,
      };
    });

    measures.push({
      index: measureIndex,
      start,
      end,
      beatsPerBar: BEATS_PER_BAR,
      slotsPerBeat: SLOTS_PER_BAR / BEATS_PER_BAR,
      columns,
    });
  }

  return {
    measures,
    bpm,
    duration,
    tuning: analysis.tuning,
    capo: analysis.capo,
    slotsPerBar: SLOTS_PER_BAR,
    slotDuration,
    barDuration,
  };
}

/** Index of the measure sounding at a given time, or -1. */
export function measureIndexAtTime(score: Score, time: number): number {
  if (!score.measures.length) return -1;
  for (const measure of score.measures) {
    if (time >= measure.start && time < measure.end) return measure.index;
  }
  return time >= score.duration ? score.measures.length - 1 : -1;
}
