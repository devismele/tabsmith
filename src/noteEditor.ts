import { safeBpm } from "./score";
import type { AnalysisResult, NoteEvent, RawNoteEvent } from "./types";

export type NoteEditorView =
  | "follow"
  | "measure"
  | "section"
  | "low-confidence"
  | "all"
  | "raw";

export type FinalEditorRow = {
  kind: "final";
  index: number;
  note: NoteEvent;
};

export type RawEditorRow = {
  kind: "raw";
  index: number;
  note: RawNoteEvent;
};

export type NoteEditorRow = FinalEditorRow | RawEditorRow;

export function noteEditorWindowKey(
  analysis: AnalysisResult,
  playbackTime: number,
  view: NoteEditorView,
): string {
  const barDuration = (60 / safeBpm(analysis.bpm)) * 4;
  const measureIndex = Math.floor(playbackTime / barDuration);
  const sectionIndex = analysis.chords.findIndex(
    (chord) => playbackTime >= chord.start && playbackTime < chord.end,
  );
  if (view === "all" || view === "low-confidence" || view === "raw") return view;
  return view === "section"
    ? `${view}:${sectionIndex}`
    : `${view}:${measureIndex}`;
}

/** Selects final arranged notes unless the user explicitly requests raw diagnostics. */
export function selectNoteEditorRows(
  analysis: AnalysisResult,
  playbackTime: number,
  view: NoteEditorView,
): NoteEditorRow[] {
  if (view === "raw") {
    return analysis.noteAnalysis.rawNotes.map((note, index) => ({
      kind: "raw",
      index,
      note,
    }));
  }

  const indexed: FinalEditorRow[] = analysis.notes.map((note, index) => ({
    kind: "final",
    index,
    note,
  }));
  if (view === "all") return indexed;
  if (view === "low-confidence") {
    return indexed.filter(({ note }) => (
      note.confidence < analysis.noteAnalysis.settings.minimumConfidence + 0.12
    ));
  }

  if (view === "section") {
    const section = analysis.chords.find(
      (chord) => playbackTime >= chord.start && playbackTime < chord.end,
    );
    return section
      ? indexed.filter(({ note }) => note.end > section.start && note.start < section.end)
      : [];
  }

  const barDuration = (60 / safeBpm(analysis.bpm)) * 4;
  const measureStart = Math.floor(playbackTime / barDuration) * barDuration;
  return indexed.filter(({ note }) => (
    note.end > measureStart && note.start < measureStart + barDuration
  ));
}
