import { TUNING_NAMES } from "./fretboard";
import { processDetectedNotes } from "./noteCleanup";
import type { NoteProcessingOptions } from "./noteCleanup";
import {
  analyzeChordProgression,
  chromaForChordFrame,
  classifyChordChroma,
  estimateBeatGrid,
  type HarmonyEvidenceOptions,
} from "./chordAnalysis";
import type {
  AnalysisResult,
  ChordEvent,
  ChordSmoothingSettings,
  HarmonyResult,
  RawNoteEvent,
} from "./types";

const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

export function midiToName(midi: number): string {
  return `${NOTE_NAMES[((midi % 12) + 12) % 12]}${Math.floor(midi / 12) - 1}`;
}

/** YIN-style difference estimator, tuned for the guitar range. */
export function detectPitch(frame: Float32Array, sampleRate: number): { frequency: number; confidence: number } | null {
  let energy = 0;
  for (const value of frame) energy += value * value;
  const rms = Math.sqrt(energy / frame.length);
  if (rms < 0.008) return null;

  const minLag = Math.floor(sampleRate / 1200);
  const maxLag = Math.min(Math.floor(sampleRate / 70), Math.floor(frame.length / 2));
  const difference = new Float64Array(maxLag + 1);
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let sum = 0;
    for (let i = 0; i < frame.length - lag; i += 1) {
      const delta = frame[i] - frame[i + lag];
      sum += delta * delta;
    }
    difference[lag] = sum;
  }

  let running = 0;
  let bestLag = -1;
  let bestValue = 1;
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    running += difference[lag];
    const normalized = running ? (difference[lag] * (lag - minLag + 1)) / running : 1;
    if (normalized < bestValue) {
      bestValue = normalized;
      bestLag = lag;
    }
    if (normalized < 0.16 && difference[lag] <= difference[lag + 1]) {
      bestLag = lag;
      bestValue = normalized;
      break;
    }
  }
  if (bestLag < 0 || bestValue > 0.38) return null;
  return { frequency: sampleRate / bestLag, confidence: clamp(1 - bestValue, 0, 1) };
}

export function chromaForFrame(frame: Float32Array, sampleRate: number): number[] {
  return chromaForChordFrame(frame, sampleRate);
}

export function classifyChord(chroma: number[]): { name: string; confidence: number } {
  return classifyChordChroma(chroma);
}

/**
 * Removes short chord-label flicker with a local mode filter. Chord-recognition
 * systems commonly smooth frame-by-frame template matches before producing a
 * progression; keeping the window small preserves real chord changes.
 */
export function smoothChordSequence(frames: ChordEvent[], radius = 1): ChordEvent[] {
  if (radius < 1 || frames.length < 3) return frames.map((frame) => ({ ...frame }));
  return frames.map((frame, index) => {
    const votes = new Map<string, { count: number; confidence: number }>();
    const start = Math.max(0, index - radius);
    const end = Math.min(frames.length - 1, index + radius);
    for (let nearby = start; nearby <= end; nearby += 1) {
      const candidate = frames[nearby];
      const vote = votes.get(candidate.name) ?? { count: 0, confidence: 0 };
      vote.count += 1;
      vote.confidence += candidate.confidence;
      votes.set(candidate.name, vote);
    }
    const ranked = [...votes.entries()].sort((a, b) => {
      if (b[1].count !== a[1].count) return b[1].count - a[1].count;
      if (a[0] === frame.name) return -1;
      if (b[0] === frame.name) return 1;
      return b[1].confidence - a[1].confidence;
    });
    const [name, vote] = ranked[0];
    return { ...frame, name, confidence: vote.confidence / vote.count };
  });
}

function detectRawNotes(samples: Float32Array, sampleRate: number): RawNoteEvent[] {
  const frameSize = 2048;
  const hop = 1024;
  const observations: { time: number; midi: number; confidence: number }[] = [];
  for (let start = 0; start + frameSize <= samples.length; start += hop) {
    const pitch = detectPitch(samples.subarray(start, start + frameSize), sampleRate);
    if (!pitch) continue;
    const midi = Math.round(69 + 12 * Math.log2(pitch.frequency / 440));
    if (midi >= 40 && midi <= 88) observations.push({ time: start / sampleRate, midi, confidence: pitch.confidence });
  }

  const raw: RawNoteEvent[] = [];
  let group: typeof observations = [];
  const flush = () => {
    if (!group.length) return;
    const midiCounts = new Map<number, number>();
    group.forEach((item) => midiCounts.set(item.midi, (midiCounts.get(item.midi) ?? 0) + 1));
    const midi = [...midiCounts].sort((a, b) => b[1] - a[1])[0][0];
    raw.push({
      start: group[0].time,
      end: group.at(-1)!.time + hop / sampleRate,
      midi,
      name: midiToName(midi),
      confidence: group.reduce((sum, item) => sum + item.confidence, 0) / group.length,
    });
    group = [];
  };
  for (const observation of observations) {
    const prior = group.at(-1);
    if (prior && (observation.time - prior.time > (hop / sampleRate) * 1.6 || Math.abs(observation.midi - prior.midi) > 1)) flush();
    group.push(observation);
  }
  flush();
  return raw;
}

export function analyzeHarmony(
  samples: Float32Array,
  sampleRate: number,
  settings: Partial<ChordSmoothingSettings> = {},
  evidence: HarmonyEvidenceOptions = {},
): HarmonyResult {
  const beatGrid = estimateBeatGrid(samples, sampleRate);
  const chordAnalysis = analyzeChordProgression(
    samples,
    sampleRate,
    beatGrid,
    settings,
    evidence,
  );
  return {
    duration: samples.length / sampleRate,
    bpm: beatGrid.bpm,
    tuning: TUNING_NAMES,
    capo: 0,
    chords: chordAnalysis.regions,
    chordAnalysis,
  };
}

export function analyzeAudio(
  samples: Float32Array,
  sampleRate: number,
  settings: Partial<ChordSmoothingSettings> = {},
  noteOptions: NoteProcessingOptions = {},
): AnalysisResult {
  const harmony = analyzeHarmony(samples, sampleRate, settings);
  return analyzeAudioWithHarmony(samples, sampleRate, harmony, noteOptions);
}

/**
 * Runs note detection against a precomputed harmony result. The hybrid worker
 * uses this to fuse and decode chords before note cleanup without repeating the
 * rule feature extraction.
 */
export function analyzeAudioWithHarmony(
  samples: Float32Array,
  sampleRate: number,
  harmony: HarmonyResult,
  noteOptions: NoteProcessingOptions = {},
): AnalysisResult {
  const noteProcessing = processDetectedNotes(
    detectRawNotes(samples, sampleRate),
    harmony.duration,
    harmony.bpm,
    harmony.chords,
    harmony.chordAnalysis.diagnostics.rawChordChanges,
    noteOptions,
  );
  return {
    ...harmony,
    ...noteProcessing,
    engine: "dsp",
    separation: noteOptions.separation ?? "none",
  };
}
