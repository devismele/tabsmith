import { describe, expect, it } from "vitest";

import {
  DEFAULT_PROGRAM,
  TICKS_PER_QUARTER,
  buildMidiFile,
  encodeVariableLength,
} from "../src/midiExport";
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

function ascii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.slice(offset, offset + length));
}

function readUint32(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] << 24) | (bytes[offset + 1] << 16) | (bytes[offset + 2] << 8) | bytes[offset + 3]
  ) >>> 0;
}

/** Walks the track chunk and returns note events in order. */
function readNoteEvents(bytes: Uint8Array): { tick: number; status: number; pitch: number }[] {
  let offset = 14 + 8; // header chunk + MTrk marker and length
  let tick = 0;
  const events: { tick: number; status: number; pitch: number }[] = [];
  while (offset < bytes.length) {
    let delta = 0;
    for (;;) {
      const byte = bytes[offset];
      offset += 1;
      delta = delta * 128 + (byte & 0x7f);
      if ((byte & 0x80) === 0) break;
    }
    tick += delta;
    const status = bytes[offset];
    offset += 1;
    if (status === 0xff) {
      const type = bytes[offset];
      offset += 1;
      const length = bytes[offset];
      offset += 1;
      offset += length;
      if (type === 0x2f) break;
      continue;
    }
    if (status === 0xc0) {
      offset += 1;
      continue;
    }
    const pitch = bytes[offset];
    offset += 2; // pitch + velocity
    events.push({ tick, status, pitch });
  }
  return events;
}

describe("encodeVariableLength", () => {
  it("encodes small values in one byte", () => {
    expect(encodeVariableLength(0)).toEqual([0x00]);
    expect(encodeVariableLength(127)).toEqual([0x7f]);
  });

  it("encodes the multi-byte boundary correctly", () => {
    expect(encodeVariableLength(128)).toEqual([0x81, 0x00]);
    expect(encodeVariableLength(8192)).toEqual([0xc0, 0x00]);
  });

  it("sets the continuation bit on every byte but the last", () => {
    const bytes = encodeVariableLength(1_000_000);
    expect(bytes.slice(0, -1).every((b) => (b & 0x80) !== 0)).toBe(true);
    expect(bytes[bytes.length - 1] & 0x80).toBe(0);
  });

  it("rejects negative deltas", () => {
    expect(() => encodeVariableLength(-1)).toThrow(RangeError);
  });
});

describe("buildMidiFile", () => {
  it("writes a valid header chunk", () => {
    const bytes = buildMidiFile([note()], { bpm: 120 });
    expect(ascii(bytes, 0, 4)).toBe("MThd");
    expect(readUint32(bytes, 4)).toBe(6);
    expect((bytes[8] << 8) | bytes[9]).toBe(0); // format 0
    expect((bytes[10] << 8) | bytes[11]).toBe(1); // one track
    expect((bytes[12] << 8) | bytes[13]).toBe(TICKS_PER_QUARTER);
  });

  it("declares a track length matching the bytes that follow", () => {
    const bytes = buildMidiFile([note()], { bpm: 120 });
    expect(ascii(bytes, 14, 4)).toBe("MTrk");
    expect(readUint32(bytes, 18)).toBe(bytes.length - 22);
  });

  it("ends with an end-of-track meta event", () => {
    const bytes = buildMidiFile([note()], { bpm: 120 });
    expect([...bytes.slice(-3)]).toEqual([0xff, 0x2f, 0x00]);
  });

  it("writes the tempo the caller asked for", () => {
    const bytes = buildMidiFile([note()], { bpm: 120 });
    const index = bytes.indexOf(0x51);
    const microseconds = (bytes[index + 2] << 16) | (bytes[index + 3] << 8) | bytes[index + 4];
    expect(microseconds).toBe(500_000); // 120 bpm
  });

  it("places a note on the quarter-note grid", () => {
    // At 120 bpm a quarter note is 0.5 s, so a note at 1.0 s starts on beat 3.
    const bytes = buildMidiFile([note({ start: 1, end: 1.5 })], { bpm: 120 });
    const events = readNoteEvents(bytes);
    expect(events[0]).toMatchObject({ tick: TICKS_PER_QUARTER * 2, status: 0x90 });
    expect(events[1]).toMatchObject({ tick: TICKS_PER_QUARTER * 3, status: 0x80 });
  });

  it("emits note-off before note-on when a pitch repeats at the same tick", () => {
    const bytes = buildMidiFile(
      [note({ start: 0, end: 0.5, midi: 40 }), note({ start: 0.5, end: 1, midi: 40 })],
      { bpm: 120 },
    );
    const atBoundary = readNoteEvents(bytes).filter((e) => e.tick === TICKS_PER_QUARTER);
    expect(atBoundary.map((e) => e.status)).toEqual([0x80, 0x90]);
  });

  it("writes the default guitar program", () => {
    const bytes = buildMidiFile([note()], { bpm: 120 });
    const index = bytes.indexOf(0xc0);
    expect(bytes[index + 1]).toBe(DEFAULT_PROGRAM);
  });

  it("maps confidence onto velocity", () => {
    const quiet = buildMidiFile([note({ confidence: 0 })], { bpm: 120 });
    const loud = buildMidiFile([note({ confidence: 1 })], { bpm: 120 });
    const velocityOf = (bytes: Uint8Array) => bytes[bytes.indexOf(0x90) + 2];
    expect(velocityOf(quiet)).toBeLessThan(velocityOf(loud));
  });

  it("keeps every velocity inside the MIDI range", () => {
    const bytes = buildMidiFile(
      [note({ confidence: 5 }), note({ confidence: -3, start: 1, end: 1.5 })],
      { bpm: 120, minimumVelocity: 400, maximumVelocity: 900 },
    );
    for (const byte of bytes) expect(byte).toBeLessThanOrEqual(255);
    const events = readNoteEvents(bytes);
    expect(events.length).toBe(4);
  });

  it("orders events by tick even when notes are supplied out of order", () => {
    const bytes = buildMidiFile(
      [note({ start: 2, end: 2.5, midi: 45 }), note({ start: 0, end: 0.5, midi: 40 })],
      { bpm: 120 },
    );
    const events = readNoteEvents(bytes);
    expect(events.map((e) => e.tick)).toEqual([...events.map((e) => e.tick)].sort((a, b) => a - b));
  });

  it("gives a zero-length note at least one tick so it is audible", () => {
    const bytes = buildMidiFile([note({ start: 1, end: 1 })], { bpm: 120 });
    const events = readNoteEvents(bytes);
    expect(events[1].tick).toBeGreaterThan(events[0].tick);
  });

  it("skips notes outside the MIDI pitch range rather than corrupting the file", () => {
    const bytes = buildMidiFile(
      [note({ midi: 200 }), note({ midi: -5 }), note({ midi: 40, start: 1, end: 1.5 })],
      { bpm: 120 },
    );
    const events = readNoteEvents(bytes);
    expect(events).toHaveLength(2);
    expect(events[0].pitch).toBe(40);
  });

  it("produces a header-only-plus-meta file for no notes", () => {
    const bytes = buildMidiFile([], { bpm: 120 });
    expect(ascii(bytes, 0, 4)).toBe("MThd");
    expect(readNoteEvents(bytes)).toHaveLength(0);
  });

  it("falls back to a safe tempo when bpm is unusable", () => {
    const bytes = buildMidiFile([note()], { bpm: 0 });
    const index = bytes.indexOf(0x51);
    const microseconds = (bytes[index + 2] << 16) | (bytes[index + 3] << 8) | bytes[index + 4];
    expect(microseconds).toBeGreaterThan(0);
  });

  it("scales tick positions with tempo", () => {
    const slow = readNoteEvents(buildMidiFile([note({ start: 1, end: 2 })], { bpm: 60 }));
    const fast = readNoteEvents(buildMidiFile([note({ start: 1, end: 2 })], { bpm: 120 }));
    expect(fast[0].tick).toBeGreaterThan(slow[0].tick);
  });
});
