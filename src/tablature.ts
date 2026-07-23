import type { NoteEvent } from "./types";

export type TabMeasure = {
  index: number;
  start: number;
  end: number;
  lines: string[];
  guide: string;
};

const STRING_LABELS = ["e", "B", "G", "D", "A", "E"];
const SUBDIVISIONS_PER_BAR = 16;

export function buildTablature(
  notes: NoteEvent[],
  duration: number,
  bpm: number | null,
): TabMeasure[] {
  const safeBpm = bpm && bpm >= 40 && bpm <= 240 ? bpm : 120;
  const barDuration = (60 / safeBpm) * 4;
  const measureCount = Math.max(1, Math.ceil(duration / barDuration));
  const measures: TabMeasure[] = [];
  const nextStart = new Map<NoteEvent, number>();
  for (let string = 1; string <= 6; string += 1) {
    const stringNotes = notes
      .filter((note) => note.string === string)
      .sort((a, b) => a.start - b.start);
    stringNotes.forEach((note, index) => {
      if (stringNotes[index + 1]) nextStart.set(note, stringNotes[index + 1].start);
    });
  }

  for (let measureIndex = 0; measureIndex < measureCount; measureIndex += 1) {
    const start = measureIndex * barDuration;
    const end = Math.min(duration, start + barDuration);
    const slots = Array.from({ length: 6 }, () => Array<number | "hold" | null>(SUBDIVISIONS_PER_BAR).fill(null));
    const confidence = Array.from({ length: 6 }, () => Array<number>(SUBDIVISIONS_PER_BAR).fill(-1));
    for (const note of [...notes].sort((a, b) => a.start - b.start)) {
      if (note.start < start || note.start >= start + barDuration) continue;
      const stringIndex = note.string - 1;
      if (stringIndex < 0 || stringIndex > 5) continue;
      const position = Math.max(
        0,
        Math.min(SUBDIVISIONS_PER_BAR - 1, Math.round(((note.start - start) / barDuration) * SUBDIVISIONS_PER_BAR)),
      );
      if (note.confidence >= confidence[stringIndex][position]) {
        slots[stringIndex][position] = note.fret;
        confidence[stringIndex][position] = note.confidence;
        const effectiveEnd = Math.min(
          note.end,
          start + barDuration,
          nextStart.get(note) ?? Infinity,
        );
        const lastSustainSlot = Math.min(SUBDIVISIONS_PER_BAR - 1, Math.floor(((effectiveEnd - start) / barDuration) * SUBDIVISIONS_PER_BAR));
        for (let sustain = position + 1; sustain <= lastSustainSlot; sustain += 1) {
          if (slots[stringIndex][sustain] === null) slots[stringIndex][sustain] = "hold";
        }
      }
    }
    const lines = STRING_LABELS.map((label, stringIndex) => {
      const tokens = slots[stringIndex].map((fret) => fret === null ? "---" : fret === "hold" ? "~~~" : `${String(fret).padStart(2, "-")}-`);
      return `${label}|${tokens.join("")}|`;
    });
    measures.push({ index: measureIndex, start, end, lines, guide: "   1  e  &  a  2  e  &  a  3  e  &  a  4  e  &  a" });
  }
  return measures;
}

export function tablatureToText(
  measures: TabMeasure[],
  title: string,
  bpm: number | null,
  tuning: string[] = ["E2", "A2", "D3", "G3", "B3", "E4"],
  capo = 0,
): string {
  const heading = `${title}\nTuning: ${tuning.join(" ")}${capo ? ` · capo ${capo}` : ""} · ${bpm ?? "unknown"} BPM\n`;
  const body = measures.map((measure) => [
    `Bar ${measure.index + 1}  (${formatTime(measure.start)}–${formatTime(measure.end)})`,
    measure.guide,
    ...measure.lines,
  ].join("\n")).join("\n\n");
  return `${heading}\n${body}\n`;
}

function formatTime(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}
