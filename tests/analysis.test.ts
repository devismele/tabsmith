import { describe, expect, it } from "vitest";
import {
  analyzeAudio,
  analyzeHarmony,
  chromaForFrame,
  classifyChord,
  detectPitch,
  midiToName,
  smoothChordSequence,
} from "../src/analysis";
import { readFileSync } from "node:fs";
import type { ChordEvent } from "../src/types";

function readMonoPcm16Wav(path: URL): { samples: Float32Array; sampleRate: number } {
  const wav = readFileSync(path);
  const sampleRate = wav.readUInt32LE(24);
  const dataBytes = wav.readUInt32LE(40);
  const samples = new Float32Array(dataBytes / 2);
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = wav.readInt16LE(44 + index * 2) / 32768;
  }
  return { samples, sampleRate };
}

function legacyChordRegions(samples: Float32Array, sampleRate: number): ChordEvent[] {
  const roots = ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"];
  const classifyLegacy = (chroma: number[]) => {
    let best = { name: "N", score: -Infinity };
    for (let root = 0; root < 12; root += 1) {
      for (const quality of ["major", "minor"] as const) {
        const tones = [root, (root + (quality === "major" ? 4 : 3)) % 12, (root + 7) % 12];
        const toneScore = tones.reduce((sum, pitchClass) => sum + chroma[pitchClass], 0);
        const offScore = chroma.reduce(
          (sum, value, pitchClass) => sum + (tones.includes(pitchClass) ? 0 : value),
          0,
        );
        const score = toneScore - offScore * 0.32 + chroma[root] * 0.18;
        if (score > best.score) {
          best = {
            name: `${roots[root]}${quality === "minor" ? "m" : ""}`,
            score,
          };
        }
      }
    }
    return { name: best.score < 0.18 ? "N" : best.name, confidence: Math.max(0, Math.min(1, best.score)) };
  };

  const frameSize = 4096;
  const hopSeconds = 0.4;
  const hop = Math.floor(hopSeconds * sampleRate);
  const raw: ChordEvent[] = [];
  for (let start = 0; start + frameSize <= samples.length; start += hop) {
    const frame = samples.subarray(start, start + frameSize);
    // Use the public chroma primitive so this helper captures only the legacy
    // independent-frame selection and radius-one mode smoothing.
    const chroma = chromaForFrame(frame, sampleRate);
    const chord = classifyLegacy(chroma);
    raw.push({
      start: start / sampleRate,
      end: Math.min(samples.length / sampleRate, start / sampleRate + hopSeconds),
      ...chord,
    });
  }
  const merged: ChordEvent[] = [];
  for (const chord of smoothChordSequence(raw)) {
    const previous = merged.at(-1);
    if (previous?.name === chord.name) previous.end = chord.end;
    else merged.push({ ...chord });
  }
  return merged.filter((chord) => chord.name !== "N" || chord.end - chord.start >= 1);
}

describe("audio analysis primitives", () => {
  it("detects a clean A4 sine wave", () => {
    const sampleRate = 22050;
    const frame = Float32Array.from({ length: 2048 }, (_, index) => 0.5 * Math.sin((2 * Math.PI * 440 * index) / sampleRate));
    const pitch = detectPitch(frame, sampleRate);
    expect(pitch).not.toBeNull();
    expect(pitch!.frequency).toBeCloseTo(440, -1);
    expect(midiToName(69)).toBe("A4");
  });

  it("classifies C major and A minor templates", () => {
    const cMajor = Array(12).fill(0); cMajor[0] = 0.4; cMajor[4] = 0.3; cMajor[7] = 0.3;
    const aMinor = Array(12).fill(0); aMinor[9] = 0.4; aMinor[0] = 0.3; aMinor[4] = 0.3;
    expect(classifyChord(cMajor).name).toBe("C");
    expect(classifyChord(aMinor).name).toBe("Am");
  });

  it("turns synthetic audio into a playable note event", () => {
    const sampleRate = 22050;
    const samples = Float32Array.from(
      { length: sampleRate },
      (_, index) => 0.35 * Math.sin((2 * Math.PI * 440 * index) / sampleRate),
    );
    const result = analyzeAudio(samples, sampleRate);
    expect(result.duration).toBe(1);
    expect(result.notes.length).toBeGreaterThan(0);
    expect(result.notes.some((note) => note.name === "A4")).toBe(true);
    expect(result.notes.every((note) => note.string >= 1 && note.string <= 6)).toBe(true);
    expect(result.noteAnalysis.rawNotes.length).toBeGreaterThan(0);
    expect(result.noteAnalysis.arrangedNotes).toEqual(result.notes);
    expect(result.noteAnalysis.diagnostics.arrangedNoteCount).toBe(result.notes.length);
    const exported = JSON.parse(JSON.stringify(result)) as typeof result;
    expect(exported.notes).toEqual(result.notes);
    expect(exported.noteAnalysis.rawNotes).toEqual(result.noteAnalysis.rawNotes);
    expect(result.engine).toBe("dsp");
    expect(result.chords).toEqual(result.chordAnalysis.regions);
    expect(result.chordAnalysis.rawFrames.length).toBeGreaterThan(0);
    expect(result.chordAnalysis.rawFrames[0]).toEqual(expect.objectContaining({
      bestChord: expect.any(String),
      bestScore: expect.any(Number),
      secondBestChord: expect.any(String),
      secondBestScore: expect.any(Number),
      confidence: expect.any(Number),
      scoreMargin: expect.any(Number),
    }));
    expect(result.chordAnalysis.diagnostics.finalChordRegions).toBe(result.chords.length);
    expect(result.chordAnalysis.windows[0]).toEqual(expect.objectContaining({
      topCandidateChord: expect.any(String),
      secondBestChord: expect.any(String),
      scoreMargin: expect.any(Number),
      beatStrength: expect.any(Number),
      finalLabel: expect.any(String),
      usedSmoothingOverride: expect.any(Boolean),
      mergedFromNearbyWindows: expect.any(Boolean),
    }));
    expect(result.chordAnalysis.windows[0]).toHaveProperty("bassRootEstimate");
    expect(result.chordAnalysis.windows[0]).toHaveProperty("keyEstimate");
    expect(result.chords[0].diagnostics).toEqual(expect.objectContaining({
      finalLabel: result.chords[0].name,
      confidenceLevel: expect.stringMatching(/high|medium|low/),
      usedBassSupport: expect.any(Boolean),
      usedSmoothingOverride: expect.any(Boolean),
      simplifiedFromExtendedChord: expect.any(Boolean),
      mergedFromShortRegions: expect.any(Boolean),
    }));
  });

  it("smooths a one-frame chord detection glitch", () => {
    const names = ["C", "C", "G", "C", "C"];
    const frames = names.map((name, index) => ({ start: index, end: index + 1, name, confidence: 0.8 }));
    expect(smoothChordSequence(frames).map((frame) => frame.name)).toEqual(["C", "C", "C", "C", "C"]);
  });

  it("keeps the bundled E-A-B7-E demo near four meaningful regions", () => {
    const { samples, sampleRate } = readMonoPcm16Wav(
      new URL("../public/tabsmith-demo.wav", import.meta.url),
    );
    const before = legacyChordRegions(samples, sampleRate);
    const after = analyzeHarmony(samples, sampleRate);
    expect(before).toHaveLength(5);
    expect(after.chords).toHaveLength(4);
    expect(after.chords.map((chord) => chord.name)).toEqual(["E", "A", "B7", "E"]);
    expect(after.bpm).toBeGreaterThanOrEqual(110);
    expect(after.bpm).toBeLessThanOrEqual(130);
    after.chords.forEach((chord, index) => {
      expect(chord.start).toBeCloseTo(index * 2, 0);
    });
    expect(after.chordAnalysis.diagnostics.finalChordRegions).toBe(4);
  });
});
