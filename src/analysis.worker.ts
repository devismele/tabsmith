/// <reference lib="webworker" />
import { analyzeAudio, analyzeAudioWithHarmony, analyzeHarmony } from "./analysis";
import {
  createHarmonyObservations,
  decodeHarmonyObservations,
  estimateBeatGrid,
} from "./chordAnalysis";
import {
  resolveChordEngineSelection,
} from "./learnedHarmony/buildGate";
import { TUNING_NAMES } from "./fretboard";
import type { NoteProcessingOptions } from "./noteCleanup";
import type { ChordEngine, ChordSmoothingSettings, HarmonyResult } from "./types";

declare const __TABSMITH_EXPERIMENTAL_HYBRID__: boolean;

self.onmessage = async (event: MessageEvent<{
  samples: Float32Array;
  sampleRate: number;
  mode?: "full" | "harmony";
  noteOptions?: NoteProcessingOptions;
  bassSamples?: Float32Array;
  harmonySource?: "separated-harmonic-mix" | "full-mix" | "guitar-only";
  chordSettings?: Partial<ChordSmoothingSettings>;
  chordEngine?: ChordEngine | "onehotchord";
  audioHash?: string;
}>) => {
  try {
    const data = event.data;
    const requestedEngine = resolveChordEngineSelection(data.chordEngine);

    // Use the injected literal directly here. Vite removes this whole branch,
    // including the dynamic weight import, before release chunk discovery.
    if (__TABSMITH_EXPERIMENTAL_HYBRID__ && requestedEngine === "hybrid") {
      const evidence = {
        bassSamples: data.bassSamples,
        source: data.harmonySource,
      };
      const beatGrid = estimateBeatGrid(data.samples, data.sampleRate);
      const observationPackage = createHarmonyObservations(
        data.samples,
        data.sampleRate,
        data.chordSettings,
        evidence,
        beatGrid,
      );
      const ruleChordAnalysis = decodeHarmonyObservations(observationPackage);
      const ruleHarmony: HarmonyResult = {
        duration: observationPackage.duration,
        bpm: beatGrid.bpm,
        tuning: TUNING_NAMES,
        capo: 0,
        chords: ruleChordAnalysis.regions,
        chordAnalysis: ruleChordAnalysis,
      };
      const { applyHybridChordEngine } = await import("./learnedHarmony/chordEngine");
      const hybrid = await applyHybridChordEngine(
        observationPackage,
        ruleChordAnalysis,
        {
          audioHash: data.audioHash ?? "",
          // The direct-call guard duplicates the worker compile-time gate.
          enabled: __TABSMITH_EXPERIMENTAL_HYBRID__,
        },
      );
      const harmony: HarmonyResult = {
        ...ruleHarmony,
        chords: hybrid.regions,
        chordAnalysis: hybrid.chordAnalysis,
        hybridEngine: {
          usedLearned: hybrid.usedLearned,
          fallbackReason: hybrid.fallbackReason,
          diagnostics: hybrid.diagnostics,
        },
      };
      const result = data.mode === "harmony"
        ? harmony
        : analyzeAudioWithHarmony(
          data.samples,
          data.sampleRate,
          harmony,
          data.noteOptions,
        );
      self.postMessage({ type: "done", result });
      return;
    }

    const result = data.mode === "harmony"
      ? analyzeHarmony(data.samples, data.sampleRate, data.chordSettings, {
        bassSamples: data.bassSamples,
        source: data.harmonySource,
      })
      : analyzeAudio(
        data.samples,
        data.sampleRate,
        data.chordSettings,
        data.noteOptions,
      );
    self.postMessage({ type: "done", result });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : "Analysis failed",
    });
  }
};
