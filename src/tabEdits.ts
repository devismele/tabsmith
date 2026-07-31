import { STANDARD_TUNING } from "./fretboard";
import type { FretboardSetup } from "./fretboard";
import type { NoteEvent } from "./types";

/**
 * Phase 2 editing model: user corrections to transcribed notes, with undo/redo.
 *
 * Pure and DOM-free so it stays testable in the node vitest environment, in the
 * same spirit as `score.ts` and `tabLayout.ts`. The renderer and the editor UI
 * read from the session; they never mutate notes directly.
 *
 * Identity is the load-bearing detail. `NoteEvent` has no id, and index-based
 * identity breaks the moment a note is deleted or inserted, so a session assigns
 * stable ids once and keeps them across every edit.
 */

export const MAX_HISTORY = 100;

export type EditableNote = NoteEvent & { id: string };

export type NoteEdit =
  /** String/fret correction: same pitch, played somewhere else on the neck. */
  | { kind: "reposition"; id: string; string: number; fret: number }
  /** Drag timing: move or resize a note in time. */
  | { kind: "retime"; id: string; start: number; end: number }
  /** Pitch correction, remapped onto the neck by the caller-supplied position. */
  | { kind: "repitch"; id: string; midi: number; name: string; string: number; fret: number }
  | { kind: "delete"; id: string }
  | { kind: "insert"; note: NoteEvent };

export type EditSession = {
  readonly notes: readonly EditableNote[];
  readonly past: readonly (readonly EditableNote[])[];
  readonly future: readonly (readonly EditableNote[])[];
  readonly setup: FretboardSetup;
  readonly nextId: number;
};

export type EditResult =
  | { ok: true; session: EditSession }
  | { ok: false; reason: string; session: EditSession };

export const DEFAULT_SETUP: FretboardSetup = {
  tuning: [...STANDARD_TUNING],
  tuningNames: ["E2", "A2", "D3", "G3", "B3", "E4"],
  capo: 0,
  name: "standard",
};

const MAX_FRET = 24;

function sortNotes(notes: readonly EditableNote[]): EditableNote[] {
  return [...notes].sort((a, b) => (
    a.start - b.start || a.string - b.string || a.midi - b.midi
  ));
}

/** Pitch produced by a position under a tuning/capo, or null if out of range. */
export function pitchAt(setup: FretboardSetup, stringIndex: number, fret: number): number | null {
  if (!Number.isInteger(stringIndex) || stringIndex < 0 || stringIndex >= setup.tuning.length) {
    return null;
  }
  if (!Number.isInteger(fret) || fret < 0 || fret > MAX_FRET) return null;
  // A capo raises every open string, so a fretted note sounds capo semitones
  // higher than the same fret number on an uncapoed neck.
  return setup.tuning[stringIndex] + setup.capo + fret;
}

export function createSession(
  notes: readonly NoteEvent[],
  setup: FretboardSetup = DEFAULT_SETUP,
): EditSession {
  const editable = notes.map((note, index) => ({ ...note, id: `n${index}` }));
  return {
    notes: sortNotes(editable),
    past: [],
    future: [],
    setup,
    nextId: notes.length,
  };
}

/** Canonical notes for rendering/export, with editing ids stripped. */
export function sessionNotes(session: EditSession): NoteEvent[] {
  return session.notes.map(({ id: _id, ...note }) => note);
}

export function canUndo(session: EditSession): boolean {
  return session.past.length > 0;
}

export function canRedo(session: EditSession): boolean {
  return session.future.length > 0;
}

function commit(session: EditSession, notes: EditableNote[], nextId = session.nextId): EditSession {
  // A new edit invalidates the redo branch, which is what users expect after
  // undoing and then taking a different action.
  const past = [...session.past, session.notes].slice(-MAX_HISTORY);
  return { ...session, notes: sortNotes(notes), past, future: [], nextId };
}

function fail(session: EditSession, reason: string): EditResult {
  return { ok: false, reason, session };
}

/**
 * Applies one edit, returning a new session or a reason it was refused.
 *
 * Refusing is deliberate: a silent no-op would leave the UI showing a correction
 * the model never accepted, and a silently *applied* invalid edit would put the
 * tab out of sync with the pitch it claims to play.
 */
export function applyEdit(session: EditSession, edit: NoteEdit): EditResult {
  if (edit.kind === "insert") {
    const id = `n${session.nextId}`;
    const pitch = pitchAt(session.setup, edit.note.string, edit.note.fret);
    if (pitch === null) return fail(session, "position is off the fretboard");
    if (pitch !== edit.note.midi) {
      return fail(session, "position does not sound the note's pitch");
    }
    if (!(edit.note.end > edit.note.start)) return fail(session, "note must end after it starts");
    return {
      ok: true,
      session: commit(session, [...session.notes, { ...edit.note, id }], session.nextId + 1),
    };
  }

  const index = session.notes.findIndex((note) => note.id === edit.id);
  if (index < 0) return fail(session, "note not found");
  const target = session.notes[index];

  if (edit.kind === "delete") {
    const remaining = session.notes.filter((note) => note.id !== edit.id);
    return { ok: true, session: commit(session, remaining) };
  }

  if (edit.kind === "retime") {
    if (!(edit.end > edit.start)) return fail(session, "note must end after it starts");
    if (edit.start < 0) return fail(session, "note cannot start before zero");
    const moved = { ...target, start: edit.start, end: edit.end };
    return { ok: true, session: commit(session, replace(session.notes, index, moved)) };
  }

  if (edit.kind === "reposition") {
    const pitch = pitchAt(session.setup, edit.string, edit.fret);
    if (pitch === null) return fail(session, "position is off the fretboard");
    // Repositioning is an alternative fingering, not a transposition: the pitch
    // is what the transcription heard, so a position that changes it is a bug
    // in the caller rather than a correction.
    if (pitch !== target.midi) return fail(session, "position does not sound the note's pitch");
    const moved = { ...target, string: edit.string, fret: edit.fret };
    return { ok: true, session: commit(session, replace(session.notes, index, moved)) };
  }

  const pitch = pitchAt(session.setup, edit.string, edit.fret);
  if (pitch === null) return fail(session, "position is off the fretboard");
  if (pitch !== edit.midi) return fail(session, "position does not sound the requested pitch");
  const repitched = {
    ...target,
    midi: edit.midi,
    name: edit.name,
    string: edit.string,
    fret: edit.fret,
  };
  return { ok: true, session: commit(session, replace(session.notes, index, repitched)) };
}

function replace(
  notes: readonly EditableNote[],
  index: number,
  note: EditableNote,
): EditableNote[] {
  const copy = [...notes];
  copy[index] = note;
  return copy;
}

export function undo(session: EditSession): EditSession {
  if (!canUndo(session)) return session;
  const previous = session.past[session.past.length - 1];
  return {
    ...session,
    notes: previous,
    past: session.past.slice(0, -1),
    future: [session.notes, ...session.future].slice(0, MAX_HISTORY),
  };
}

export function redo(session: EditSession): EditSession {
  if (!canRedo(session)) return session;
  const next = session.future[0];
  return {
    ...session,
    notes: next,
    past: [...session.past, session.notes].slice(-MAX_HISTORY),
    future: session.future.slice(1),
  };
}

export type RetuneResult = {
  session: EditSession;
  /** Notes that cannot be played under the new setup, left at their old position. */
  unplayable: EditableNote[];
};

/**
 * Manual tuning/capo change, preserving what each note sounds.
 *
 * Every note keeps its pitch and is re-fingered for the new setup; the lowest
 * playable string is preferred, matching how a player would rather fret near the
 * bass strings than reach up the neck. Notes with no position under the new
 * setup are reported rather than dropped or silently transposed - losing a note
 * because the user tried a capo would be worse than telling them.
 */
export function retune(session: EditSession, setup: FretboardSetup): RetuneResult {
  const unplayable: EditableNote[] = [];
  const remapped = session.notes.map((note) => {
    let best: { string: number; fret: number } | null = null;
    for (let stringIndex = 0; stringIndex < setup.tuning.length; stringIndex += 1) {
      const fret = note.midi - setup.tuning[stringIndex] - setup.capo;
      if (fret < 0 || fret > MAX_FRET) continue;
      if (!best || fret < best.fret) best = { string: stringIndex, fret };
    }
    if (!best) {
      unplayable.push(note);
      return note;
    }
    return { ...note, string: best.string, fret: best.fret };
  });
  const past = [...session.past, session.notes].slice(-MAX_HISTORY);
  return {
    session: { ...session, notes: sortNotes(remapped), past, future: [], setup },
    unplayable,
  };
}

/**
 * Confidence bands for visualisation (Phase 2).
 *
 * Thresholds are deliberately coarse: the useful question for a user correcting
 * a tab is "should I look at this note", not the exact detector score.
 */
export type ConfidenceBand = "high" | "medium" | "low";

export function confidenceBand(note: Pick<NoteEvent, "confidence">): ConfidenceBand {
  if (note.confidence >= 0.75) return "high";
  if (note.confidence >= 0.45) return "medium";
  return "low";
}

/** Notes worth reviewing first, lowest confidence leading. */
export function reviewQueue(session: EditSession, limit = 20): EditableNote[] {
  return [...session.notes]
    .filter((note) => confidenceBand(note) !== "high")
    .sort((a, b) => a.confidence - b.confidence || a.start - b.start)
    .slice(0, limit);
}
