import { describe, expect, it } from "vitest";
import { getPlaybackState } from "../src/playback";

const chords = [
  { start: 0, end: 2, name: "C", confidence: 0.9 },
  { start: 2, end: 5, name: "Am", confidence: 0.8 },
  { start: 5, end: 8, name: "F", confidence: 0.85 },
];

describe("playback chord progression", () => {
  it("selects the current and next chords at a timestamp", () => {
    const state = getPlaybackState(chords, 3.5, 8);
    expect(state.currentIndex).toBe(1);
    expect(state.nextIndex).toBe(2);
    expect(state.songProgress).toBeCloseTo(0.4375);
    expect(state.chordProgress).toBeCloseTo(0.5);
  });

  it("handles the end of the progression", () => {
    const state = getPlaybackState(chords, 7.9, 8);
    expect(state.currentIndex).toBe(2);
    expect(state.nextIndex).toBe(-1);
  });
});
