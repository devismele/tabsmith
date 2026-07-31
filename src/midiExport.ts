import { safeBpm } from "./score";
import type { NoteEvent } from "./types";

/**
 * Phase 4 export: a standard MIDI file built from the canonical timed notes.
 *
 * Pure and dependency-free so it stays testable in the node vitest environment
 * and adds nothing to the renderer bundle beyond the bytes it writes. It reads
 * `NoteEvent` directly rather than the rendered tab, keeping the rule that
 * canonical timed data is the source of truth for every output.
 */

export const TICKS_PER_QUARTER = 480;

/** General MIDI 25 = Acoustic Guitar (steel), the closest default for Tabsmith. */
export const DEFAULT_PROGRAM = 25;

export type MidiExportOptions = {
  /** Accepts null so an analysis with no detected tempo can be passed straight through. */
  bpm?: number | null;
  /** General MIDI program number, 0-127. */
  program?: number;
  /** Velocity floor/ceiling mapped from note confidence. */
  minimumVelocity?: number;
  maximumVelocity?: number;
  trackName?: string;
};

/** MIDI variable-length quantity: 7 bits per byte, high bit marks continuation. */
export function encodeVariableLength(value: number): number[] {
  if (!Number.isFinite(value) || value < 0) throw new RangeError("delta time must be >= 0");
  const rounded = Math.round(value);
  const bytes = [rounded & 0x7f];
  let rest = Math.floor(rounded / 128);
  while (rest > 0) {
    bytes.unshift((rest & 0x7f) | 0x80);
    rest = Math.floor(rest / 128);
  }
  return bytes;
}

function pushUint32(out: number[], value: number): void {
  out.push((value >>> 24) & 0xff, (value >>> 16) & 0xff, (value >>> 8) & 0xff, value & 0xff);
}

function pushAscii(out: number[], text: string): void {
  for (const char of text) out.push(char.charCodeAt(0) & 0x7f);
}

type MidiEvent = {
  tick: number;
  /** Note-off sorts before note-on at the same tick so a repeated pitch retriggers. */
  order: number;
  bytes: number[];
};

function velocityFor(confidence: number, minimum: number, maximum: number): number {
  const clamped = Math.min(1, Math.max(0, Number.isFinite(confidence) ? confidence : 0.5));
  return Math.round(minimum + (maximum - minimum) * clamped);
}

/**
 * Builds a format-0 standard MIDI file.
 *
 * Notes are emitted in tick order with note-off preceding note-on at any shared
 * tick: without that ordering a repeated pitch would have its second strike
 * silenced by the first note's release.
 */
export function buildMidiFile(
  notes: readonly NoteEvent[],
  options: MidiExportOptions = {},
): Uint8Array<ArrayBuffer> {
  const bpm = safeBpm(options.bpm ?? null);
  const program = clampByte(options.program ?? DEFAULT_PROGRAM);
  const minimumVelocity = clampByte(options.minimumVelocity ?? 55);
  const maximumVelocity = clampByte(options.maximumVelocity ?? 112);
  const secondsPerQuarter = 60 / bpm;
  const toTicks = (seconds: number) => Math.max(0, Math.round(
    (seconds / secondsPerQuarter) * TICKS_PER_QUARTER,
  ));

  const events: MidiEvent[] = [];
  for (const note of notes) {
    if (!Number.isFinite(note.midi) || note.midi < 0 || note.midi > 127) continue;
    const start = toTicks(note.start);
    const end = Math.max(start + 1, toTicks(note.end));
    const pitch = Math.round(note.midi);
    events.push({
      tick: start,
      order: 1,
      bytes: [0x90, pitch, velocityFor(note.confidence, minimumVelocity, maximumVelocity)],
    });
    events.push({ tick: end, order: 0, bytes: [0x80, pitch, 0x40] });
  }
  events.sort((a, b) => a.tick - b.tick || a.order - b.order);

  const track: number[] = [];
  // Tempo, then track name, then program: all at tick zero.
  const microsecondsPerQuarter = Math.round(60_000_000 / bpm);
  track.push(...encodeVariableLength(0), 0xff, 0x51, 0x03,
    (microsecondsPerQuarter >>> 16) & 0xff,
    (microsecondsPerQuarter >>> 8) & 0xff,
    microsecondsPerQuarter & 0xff);
  const name = options.trackName ?? "Tabsmith";
  const nameBytes: number[] = [];
  pushAscii(nameBytes, name);
  track.push(...encodeVariableLength(0), 0xff, 0x03, ...encodeVariableLength(nameBytes.length),
    ...nameBytes);
  track.push(...encodeVariableLength(0), 0xc0, program);

  let previousTick = 0;
  for (const event of events) {
    track.push(...encodeVariableLength(event.tick - previousTick), ...event.bytes);
    previousTick = event.tick;
  }
  track.push(...encodeVariableLength(0), 0xff, 0x2f, 0x00);

  const out: number[] = [];
  pushAscii(out, "MThd");
  pushUint32(out, 6);
  out.push(0x00, 0x00); // format 0
  out.push(0x00, 0x01); // one track
  out.push((TICKS_PER_QUARTER >>> 8) & 0xff, TICKS_PER_QUARTER & 0xff);
  pushAscii(out, "MTrk");
  pushUint32(out, track.length);
  out.push(...track);
  // Backed by a plain ArrayBuffer (not SharedArrayBuffer) so the result is
  // directly usable as a BlobPart for download.
  const bytes = new Uint8Array(new ArrayBuffer(out.length));
  bytes.set(out);
  return bytes;
}

function clampByte(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(127, Math.max(0, Math.round(value)));
}
