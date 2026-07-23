import { processDetectedNotes } from "./noteCleanup";
import { BASIC_PITCH_MODEL_VERSION } from "./transcriptionCache";
import type { NoteEvent, QualityPreset, RawNoteEvent } from "./types";

export type ModelNote = {
  startTimeSeconds: number;
  durationSeconds: number;
  pitchMidi: number;
  amplitude: number;
  pitchBends?: number[];
};

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

export const BASIC_PITCH_THRESHOLDS: Record<QualityPreset, {
  onset: number;
  frame: number;
  minimumLengthFrames: number;
  minimumFrequencyHz: number;
  maximumFrequencyHz: number;
}> = {
  clean: {
    onset: 0.55,
    frame: 0.35,
    minimumLengthFrames: 7,
    minimumFrequencyHz: 75,
    maximumFrequencyHz: 1400,
  },
  balanced: {
    onset: 0.5,
    frame: 0.32,
    minimumLengthFrames: 6,
    minimumFrequencyHz: 75,
    maximumFrequencyHz: 1400,
  },
  detailed: {
    onset: 0.42,
    frame: 0.26,
    minimumLengthFrames: 5,
    minimumFrequencyHz: 75,
    maximumFrequencyHz: 1400,
  },
  raw: {
    onset: 0.35,
    frame: 0.2,
    minimumLengthFrames: 3,
    minimumFrequencyHz: 75,
    maximumFrequencyHz: 1400,
  },
};

export { BASIC_PITCH_MODEL_VERSION };

function noteName(midi: number): string {
  return `${NOTE_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/** Converts model output without discarding detector evidence. */
export function modelNotesToRaw(notes: ModelNote[]): RawNoteEvent[] {
  return notes
    .map((note) => ({
      start: Math.max(0, note.startTimeSeconds),
      end: Math.max(0, note.startTimeSeconds) + note.durationSeconds,
      midi: note.pitchMidi,
      name: noteName(note.pitchMidi),
      confidence: Math.max(0, Math.min(1, note.amplitude)),
      pitchBends: note.pitchBends ? [...note.pitchBends] : undefined,
    }))
    .sort((a, b) => a.start - b.start || a.midi - b.midi);
}

/** Backward-compatible helper; production code keeps the raw stage separate. */
export function modelNotesToTab(notes: ModelNote[]): NoteEvent[] {
  const raw = modelNotesToRaw(notes);
  return processDetectedNotes(
    raw,
    raw.reduce((maximum, note) => Math.max(maximum, note.end), 0),
    null,
    [],
    0,
    { mode: "automatic", preset: "balanced" },
  ).notes;
}

export async function transcribeWithBasicPitch(
  samplesAt22050Hz: Float32Array,
  onProgress: (progress: number) => void,
  signal?: AbortSignal,
  preset: QualityPreset = "balanced",
): Promise<RawNoteEvent[]> {
  if (signal?.aborted) throw new DOMException("Processing was cancelled.", "AbortError");
  const {
    BasicPitch,
    addPitchBendsToNoteEvents,
    noteFramesToTime,
    outputToNotesPoly,
  } = await import("@spotify/basic-pitch");

  const frames: number[][] = [];
  const onsets: number[][] = [];
  const contours: number[][] = [];
  const model = new BasicPitch("/basic-pitch-model/model.json");
  await model.evaluateModel(
    samplesAt22050Hz,
    (frameBatch, onsetBatch, contourBatch) => {
      frames.push(...frameBatch);
      onsets.push(...onsetBatch);
      contours.push(...contourBatch);
    },
    (progress) => {
      if (signal?.aborted) throw new DOMException("Processing was cancelled.", "AbortError");
      onProgress(progress);
    },
  );
  if (signal?.aborted) throw new DOMException("Processing was cancelled.", "AbortError");

  // Presets tune detector sensitivity, then the context-aware cleanup stage
  // makes the final decision. Raw mode deliberately exposes more evidence.
  const threshold = BASIC_PITCH_THRESHOLDS[preset];
  const modelNotes = noteFramesToTime(
    addPitchBendsToNoteEvents(
      contours,
      outputToNotesPoly(
        frames,
        onsets,
        threshold.onset,
        threshold.frame,
        threshold.minimumLengthFrames,
        true,
        threshold.maximumFrequencyHz,
        threshold.minimumFrequencyHz,
      ),
    ),
  ) as ModelNote[];
  return modelNotesToRaw(modelNotes);
}
