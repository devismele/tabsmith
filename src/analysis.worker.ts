/// <reference lib="webworker" />
import { analyzeAudio, analyzeHarmony } from "./analysis";
import type { NoteProcessingOptions } from "./noteCleanup";
import type { ChordSmoothingSettings } from "./types";

self.onmessage = (event: MessageEvent<{
  samples: Float32Array;
  sampleRate: number;
  mode?: "full" | "harmony";
  noteOptions?: NoteProcessingOptions;
  bassSamples?: Float32Array;
  harmonySource?: "separated-harmonic-mix" | "full-mix" | "guitar-only";
  chordSettings?: Partial<ChordSmoothingSettings>;
}>) => {
  try {
    const result = event.data.mode === "harmony"
      ? analyzeHarmony(
        event.data.samples,
        event.data.sampleRate,
        event.data.chordSettings,
        {
          bassSamples: event.data.bassSamples,
          source: event.data.harmonySource,
        },
      )
      : analyzeAudio(
        event.data.samples,
        event.data.sampleRate,
        event.data.chordSettings,
        event.data.noteOptions,
      );
    self.postMessage({ type: "done", result });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : "Analysis failed" });
  }
};
