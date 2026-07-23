import { describe, expect, it } from "vitest";
import { modelNotesToTab } from "../src/mlTranscription";

describe("Basic Pitch note conversion", () => {
  it("keeps playable guitar notes and maps them to strings and frets", () => {
    const notes = modelNotesToTab([
      { startTimeSeconds: 0, durationSeconds: 0.5, pitchMidi: 40, amplitude: 0.9 },
      { startTimeSeconds: 0.5, durationSeconds: 0.5, pitchMidi: 64, amplitude: 0.8 },
      { startTimeSeconds: 1, durationSeconds: 0.5, pitchMidi: 100, amplitude: 0.9 },
    ]);
    expect(notes).toHaveLength(2);
    expect(notes[0]).toMatchObject({ name: "E2", string: 6, fret: 0 });
    expect(notes[1].midi).toBe(64);
  });

  it("removes extremely short model artifacts", () => {
    expect(modelNotesToTab([
      { startTimeSeconds: 0, durationSeconds: 0.01, pitchMidi: 60, amplitude: 0.5 },
    ])).toEqual([]);
  });
});
