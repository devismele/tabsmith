import { describe, expect, it } from "vitest";

import {
  DEFAULT_SETUP,
  MAX_HISTORY,
  applyEdit,
  canRedo,
  canUndo,
  confidenceBand,
  createSession,
  pitchAt,
  redo,
  retune,
  reviewQueue,
  sessionNotes,
  undo,
} from "../src/tabEdits";
import type { NoteEvent } from "../src/types";

function note(overrides: Partial<NoteEvent> = {}): NoteEvent {
  return {
    start: 0,
    end: 0.5,
    midi: 40,
    name: "E2",
    confidence: 0.9,
    string: 0,
    fret: 0,
    ...overrides,
  };
}

/** Open low E, and the same pitch nowhere else; A string fret 7 is E3 (52). */
const openE = note({ midi: 40, string: 0, fret: 0 });
const aString = note({ midi: 52, string: 1, fret: 7, name: "E3", start: 1, end: 1.5 });

describe("pitchAt", () => {
  it("returns the sounding pitch for a position", () => {
    expect(pitchAt(DEFAULT_SETUP, 0, 0)).toBe(40);
    expect(pitchAt(DEFAULT_SETUP, 1, 7)).toBe(52);
  });

  it("raises every string by the capo", () => {
    const capo3 = { ...DEFAULT_SETUP, capo: 3 };
    expect(pitchAt(capo3, 0, 0)).toBe(43);
    expect(pitchAt(capo3, 5, 5)).toBe(72);
  });

  it("rejects positions off the neck", () => {
    expect(pitchAt(DEFAULT_SETUP, -1, 0)).toBeNull();
    expect(pitchAt(DEFAULT_SETUP, 6, 0)).toBeNull();
    expect(pitchAt(DEFAULT_SETUP, 0, -1)).toBeNull();
    expect(pitchAt(DEFAULT_SETUP, 0, 25)).toBeNull();
  });

  it("rejects fractional positions", () => {
    expect(pitchAt(DEFAULT_SETUP, 0, 1.5)).toBeNull();
    expect(pitchAt(DEFAULT_SETUP, 0.5, 1)).toBeNull();
  });
});

describe("session identity", () => {
  it("keeps ids stable across edits so indices cannot drift", () => {
    const session = createSession([openE, aString]);
    const target = session.notes[1].id;
    const afterDelete = applyEdit(session, { kind: "delete", id: session.notes[0].id });
    expect(afterDelete.ok).toBe(true);
    if (!afterDelete.ok) return;
    expect(afterDelete.session.notes[0].id).toBe(target);
  });

  it("strips ids from exported notes", () => {
    const session = createSession([openE]);
    expect(sessionNotes(session)[0]).not.toHaveProperty("id");
    expect(sessionNotes(session)[0].midi).toBe(40);
  });

  it("keeps notes sorted by time", () => {
    const session = createSession([aString, openE]);
    expect(session.notes.map((n) => n.start)).toEqual([0, 1]);
  });
});

describe("reposition", () => {
  it("accepts an alternative fingering of the same pitch", () => {
    const session = createSession([aString]);
    const result = applyEdit(session, {
      kind: "reposition",
      id: session.notes[0].id,
      string: 0,
      fret: 12,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.notes[0].string).toBe(0);
    expect(result.session.notes[0].fret).toBe(12);
    expect(result.session.notes[0].midi).toBe(52);
  });

  it("refuses a position that would change the pitch", () => {
    const session = createSession([aString]);
    const result = applyEdit(session, {
      kind: "reposition",
      id: session.notes[0].id,
      string: 0,
      fret: 5,
    });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toMatch(/pitch/);
    expect(result.session.notes[0].fret).toBe(7);
  });

  it("refuses a position off the fretboard", () => {
    const session = createSession([aString]);
    const result = applyEdit(session, {
      kind: "reposition",
      id: session.notes[0].id,
      string: 9,
      fret: 3,
    });
    expect(result.ok).toBe(false);
  });

  it("refuses an unknown note", () => {
    const session = createSession([aString]);
    const result = applyEdit(session, { kind: "reposition", id: "missing", string: 0, fret: 12 });
    expect(result.ok).toBe(false);
    if (result.ok) return;
    expect(result.reason).toMatch(/not found/);
  });
});

describe("retime", () => {
  it("moves a note in time", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "retime",
      id: session.notes[0].id,
      start: 2,
      end: 2.75,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.notes[0].start).toBeCloseTo(2);
    expect(result.session.notes[0].end).toBeCloseTo(2.75);
  });

  it("refuses a note that ends before it starts", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "retime",
      id: session.notes[0].id,
      start: 2,
      end: 1,
    });
    expect(result.ok).toBe(false);
  });

  it("refuses a negative start", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "retime",
      id: session.notes[0].id,
      start: -1,
      end: 1,
    });
    expect(result.ok).toBe(false);
  });

  it("re-sorts after a note is dragged past another", () => {
    const session = createSession([openE, aString]);
    const result = applyEdit(session, {
      kind: "retime",
      id: session.notes[0].id,
      start: 5,
      end: 5.5,
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.notes.map((n) => n.start)).toEqual([1, 5]);
  });
});

describe("insert and delete", () => {
  it("inserts a valid note", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "insert",
      note: note({ midi: 52, string: 1, fret: 7, start: 3, end: 3.5 }),
    });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.notes).toHaveLength(2);
  });

  it("gives an inserted note an id that collides with nothing", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "insert",
      note: note({ midi: 52, string: 1, fret: 7, start: 3, end: 3.5 }),
    });
    if (!result.ok) return;
    const ids = result.session.notes.map((n) => n.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("refuses an inserted note whose position does not sound its pitch", () => {
    const session = createSession([openE]);
    const result = applyEdit(session, {
      kind: "insert",
      note: note({ midi: 99, string: 1, fret: 7, start: 3, end: 3.5 }),
    });
    expect(result.ok).toBe(false);
  });

  it("deletes by id", () => {
    const session = createSession([openE, aString]);
    const result = applyEdit(session, { kind: "delete", id: session.notes[0].id });
    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.session.notes).toHaveLength(1);
    expect(result.session.notes[0].midi).toBe(52);
  });
});

describe("undo and redo", () => {
  it("starts with nothing to undo or redo", () => {
    const session = createSession([openE]);
    expect(canUndo(session)).toBe(false);
    expect(canRedo(session)).toBe(false);
  });

  it("restores the previous notes", () => {
    const session = createSession([openE, aString]);
    const edited = applyEdit(session, { kind: "delete", id: session.notes[0].id });
    if (!edited.ok) return;
    expect(canUndo(edited.session)).toBe(true);
    const back = undo(edited.session);
    expect(back.notes).toHaveLength(2);
    expect(canRedo(back)).toBe(true);
  });

  it("redoes what was undone", () => {
    const session = createSession([openE, aString]);
    const edited = applyEdit(session, { kind: "delete", id: session.notes[0].id });
    if (!edited.ok) return;
    const again = redo(undo(edited.session));
    expect(again.notes).toHaveLength(1);
  });

  it("survives many edits in sequence", () => {
    let session = createSession([openE]);
    for (let i = 1; i <= 5; i += 1) {
      const result = applyEdit(session, {
        kind: "retime",
        id: session.notes[0].id,
        start: i,
        end: i + 0.5,
      });
      if (!result.ok) throw new Error(result.reason);
      session = result.session;
    }
    expect(session.notes[0].start).toBe(5);
    for (let i = 0; i < 5; i += 1) session = undo(session);
    expect(session.notes[0].start).toBe(0);
    expect(canUndo(session)).toBe(false);
  });

  it("drops the redo branch once a new edit is made", () => {
    const session = createSession([openE, aString]);
    const edited = applyEdit(session, { kind: "delete", id: session.notes[0].id });
    if (!edited.ok) return;
    const reverted = undo(edited.session);
    const diverged = applyEdit(reverted, { kind: "delete", id: reverted.notes[1].id });
    if (!diverged.ok) return;
    expect(canRedo(diverged.session)).toBe(false);
  });

  it("does not record history for a refused edit", () => {
    const session = createSession([aString]);
    const refused = applyEdit(session, {
      kind: "reposition",
      id: session.notes[0].id,
      string: 0,
      fret: 5,
    });
    expect(canUndo(refused.session)).toBe(false);
  });

  it("bounds history so a long session cannot grow without limit", () => {
    let session = createSession([openE]);
    for (let i = 1; i <= MAX_HISTORY + 20; i += 1) {
      const result = applyEdit(session, {
        kind: "retime",
        id: session.notes[0].id,
        start: i,
        end: i + 0.5,
      });
      if (!result.ok) throw new Error(result.reason);
      session = result.session;
    }
    expect(session.past.length).toBe(MAX_HISTORY);
  });

  it("undoing with nothing to undo is a no-op", () => {
    const session = createSession([openE]);
    expect(undo(session)).toBe(session);
    expect(redo(session)).toBe(session);
  });
});

describe("retune", () => {
  it("preserves pitch and re-fingers for a capo", () => {
    const session = createSession([aString]);
    const { session: retuned, unplayable } = retune(session, { ...DEFAULT_SETUP, capo: 2 });
    expect(unplayable).toHaveLength(0);
    const moved = retuned.notes[0];
    expect(moved.midi).toBe(52);
    expect(pitchAt(retuned.setup, moved.string, moved.fret)).toBe(52);
  });

  it("re-fingers for drop D", () => {
    const dropD = { ...DEFAULT_SETUP, tuning: [38, 45, 50, 55, 59, 64], name: "drop-d" };
    const session = createSession([note({ midi: 38, string: 0, fret: 0, name: "D2" })]);
    const { session: retuned, unplayable } = retune(session, dropD);
    expect(unplayable).toHaveLength(0);
    expect(retuned.notes[0].fret).toBe(0);
    expect(retuned.notes[0].string).toBe(0);
  });

  it("reports notes that no longer fit rather than dropping them", () => {
    const session = createSession([openE]);
    const { session: retuned, unplayable } = retune(session, { ...DEFAULT_SETUP, capo: 5 });
    expect(unplayable).toHaveLength(1);
    expect(retuned.notes).toHaveLength(1);
  });

  it("is undoable", () => {
    const session = createSession([aString]);
    const { session: retuned } = retune(session, { ...DEFAULT_SETUP, capo: 2 });
    expect(canUndo(retuned)).toBe(true);
    expect(undo(retuned).notes[0].fret).toBe(7);
  });
});

describe("confidence", () => {
  it("bands notes for visualisation", () => {
    expect(confidenceBand({ confidence: 0.9 })).toBe("high");
    expect(confidenceBand({ confidence: 0.5 })).toBe("medium");
    expect(confidenceBand({ confidence: 0.1 })).toBe("low");
  });

  it("queues the least certain notes first", () => {
    const session = createSession([
      note({ confidence: 0.95, start: 0, midi: 40, string: 0, fret: 0 }),
      note({ confidence: 0.2, start: 1, midi: 52, string: 1, fret: 7 }),
      note({ confidence: 0.5, start: 2, midi: 52, string: 1, fret: 7 }),
    ]);
    const queue = reviewQueue(session);
    expect(queue.map((n) => n.confidence)).toEqual([0.2, 0.5]);
  });

  it("respects the limit", () => {
    const notes = Array.from({ length: 30 }, (_v, i) => note({
      confidence: 0.1,
      start: i,
      midi: 52,
      string: 1,
      fret: 7,
    }));
    expect(reviewQueue(createSession(notes), 5)).toHaveLength(5);
  });
});
