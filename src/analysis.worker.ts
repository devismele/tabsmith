/// <reference lib="webworker" />
import { analyzeAudio, analyzeHarmony } from "./analysis";
import type { NoteProcessingOptions } from "./noteCleanup";
import type { ChordSmoothingSettings } from "./types";

self.onmessage = async (event: MessageEvent<{
  samples: Float32Array;
  sampleRate: number;
  mode?: "full" | "harmony";
  noteOptions?: NoteProcessingOptions;
  bassSamples?: Float32Array;
  harmonySource?: "separated-harmonic-mix" | "full-mix" | "guitar-only";
  chordSettings?: Partial<ChordSmoothingSettings>;
  chordEngine?: "rule" | "learned" | "onehotchord";
  audioHash?: string;
}>) => {
  try {
    const data = event.data;
    const result = data.mode === "harmony"
      ? analyzeHarmony(data.samples, data.sampleRate, data.chordSettings, {
        bassSamples: data.bassSamples,
        source: data.harmonySource,
      })
      : analyzeAudio(data.samples, data.sampleRate, data.chordSettings, data.noteOptions);

    // Experimental learned engine (opt-in). Dynamic-imported so no ML code or the
    // 6 MB model loads unless a user selects it; it falls back to rule-based on failure.
    if (data.chordEngine === "learned") {
      const { applyLearnedChordEngine } = await import("./learnedHarmony/chordEngine");
      const learned = await applyLearnedChordEngine(
        data.samples, data.sampleRate, result.chords, data.audioHash ?? "", result.duration,
      );
      result.chords = learned.regions;
      result.learnedEngine = { usedLearned: learned.usedLearned, fallbackReason: learned.fallbackReason };
    }

    self.postMessage({ type: "done", result });
  } catch (error) {
    self.postMessage({ type: "error", message: error instanceof Error ? error.message : "Analysis failed" });
  }
};
