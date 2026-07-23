import type { ChordEvent } from "./types";

export type PlaybackState = {
  currentIndex: number;
  nextIndex: number;
  songProgress: number;
  chordProgress: number;
};

function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}

export function getPlaybackState(
  chords: ChordEvent[],
  currentTime: number,
  duration: number,
): PlaybackState {
  const currentIndex = chords.findIndex(
    (chord) => currentTime >= chord.start && currentTime < chord.end,
  );
  const nextIndex = currentIndex >= 0
    ? (currentIndex + 1 < chords.length ? currentIndex + 1 : -1)
    : chords.findIndex((chord) => chord.start > currentTime);
  const current = currentIndex >= 0 ? chords[currentIndex] : null;
  return {
    currentIndex,
    nextIndex,
    songProgress: duration > 0 ? clamp(currentTime / duration) : 0,
    chordProgress: current && current.end > current.start
      ? clamp((currentTime - current.start) / (current.end - current.start))
      : 0,
  };
}
