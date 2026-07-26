import { describe, expect, it } from "vitest";
import {
  analyzeChordProgression,
  classifyChordEvidence,
  createHarmonyObservations,
  decodeHarmonyObservations,
  decodeReducedLatencySequence,
  estimateKeyFromChroma,
  smoothChordObservations,
  smoothChordObservationsReducedLatency,
  type ChordObservation,
} from "../src/chordAnalysis";
import { getPlaybackState } from "../src/playback";

const WINDOW_SECONDS = 0.5;

function observation(
  name: string,
  index: number,
  options: {
    scores?: Record<string, number>;
    confidence?: number;
    scoreMargin?: number;
    uncertain?: boolean;
    seventhEvidence?: number;
    boundaryStrength?: number;
    bassAttackRootClass?: string | null;
    bassAttackConfidence?: number;
    bassIsPassingTone?: boolean;
  } = {},
): ChordObservation {
  const scores = {
    C: 0.05,
    D: 0.05,
    E: 0.05,
    E7: 0.01,
    A: 0.05,
    B: 0.05,
    B7: 0.01,
    Bm: 0.05,
    ...options.scores,
  };
  if (name !== "N" && !options.scores?.[name]) scores[name as keyof typeof scores] = 0.94;
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const scoreMargin = options.scoreMargin ?? ranked[0][1] - ranked[1][1];
  return {
    start: index * WINDOW_SECONDS,
    end: (index + 1) * WINDOW_SECONDS,
    bestChord: name,
    bestScore: name === "N" ? 0.08 : ranked[0][1],
    secondBestChord: ranked[1][0],
    secondBestScore: ranked[1][1],
    confidence: options.confidence ?? (name === "N" ? 0.95 : 0.94),
    scoreMargin,
    uncertain: options.uncertain ?? false,
    candidateScores: scores,
    noChordScore: name === "N" ? 0.95 : -0.5,
    seventhEvidence: options.seventhEvidence ?? 0,
    boundaryStrength: options.boundaryStrength
      ?? (index > 0 && index % 4 === 0 ? 1 : index > 0 && index % 2 === 0 ? 0.86 : 0.72),
    bassAttackRootClass: options.bassAttackRootClass,
    bassAttackConfidence: options.bassAttackConfidence,
    bassIsPassingTone: options.bassIsPassingTone,
  };
}

function sequence(
  names: string[],
  optionsForIndex: (name: string, index: number) => Parameters<typeof observation>[2] = () => ({}),
): ChordObservation[] {
  return names.map((name, index) => observation(name, index, optionsForIndex(name, index)));
}

describe("chord sequence smoothing", () => {
  it("keeps a stable E chord as one region despite noisy frames", () => {
    const observations = sequence(
      ["E", "E", "E", "F", "E", "E", "E", "E"],
      (name) => name === "F"
        ? { scores: { F: 0.76, E: 0.73 }, scoreMargin: 0.03, uncertain: true }
        : {},
    );
    const result = smoothChordObservations(observations);
    expect(result.regions.map((region) => region.name)).toEqual(["E"]);
    expect(result.regions[0]).toMatchObject({ start: 0, end: 4 });
  });

  it("collapses Bm to weak B to Bm into one Bm region", () => {
    const observations = sequence(
      ["Bm", "Bm", "Bm", "B", "Bm", "Bm", "Bm"],
      (name) => name === "B"
        ? { scores: { B: 0.79, Bm: 0.76 }, scoreMargin: 0.03, uncertain: true }
        : {},
    );
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(["Bm"]);
  });

  it("absorbs a short no-chord gap between matching D regions", () => {
    const observations = sequence(["D", "D", "D", "N", "D", "D", "D"]);
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(["D"]);
  });

  it("preserves a real sustained chord change", () => {
    const observations = sequence(["E", "E", "E", "E", "A", "A", "A", "A"]);
    const regions = smoothChordObservations(observations).regions;
    expect(regions.map((region) => region.name)).toEqual(["E", "A"]);
    expect(regions.map((region) => [region.start, region.end])).toEqual([[0, 2], [2, 4]]);
  });

  it("rejects a one-window chord fluctuation", () => {
    const observations = sequence(["E", "E", "E", "A", "E", "E", "E"]);
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(["E"]);
  });

  it("merges all adjacent identical chord windows", () => {
    const observations = sequence(["A", "A", "A", "A", "A", "A"]);
    const regions = smoothChordObservations(observations).regions;
    expect(regions).toHaveLength(1);
    expect(regions[0]).toMatchObject({ name: "A", start: 0, end: 3 });
  });

  it("falls back to a triad when seventh evidence is weak", () => {
    const observations = sequence(
      ["E7", "E7", "E7", "E7"],
      () => ({
        scores: { E7: 0.95, E: 0.82 },
        scoreMargin: 0.13,
        seventhEvidence: 0.05,
      }),
    );
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(["E"]);
  });

  it("preserves a high-confidence short chord that is sustained on a strong beat", () => {
    const observations = sequence(
      ["E", "E", "E", "E", "A", "A", "E", "E", "E", "E"],
      (name, index) => name === "A"
        ? {
          scores: { A: 0.99, E: 0.15 },
          confidence: 0.99,
          scoreMargin: 0.84,
          boundaryStrength: index === 4 ? 1 : 0.72,
        }
        : {},
    );
    const regions = smoothChordObservations(observations).regions;
    expect(regions.map((region) => region.name)).toEqual(["E", "A", "E"]);
    expect(regions[1].end - regions[1].start).toBe(1);
  });

  it("keeps chord timestamps monotonic, contiguous, and non-overlapping", () => {
    const observations = sequence([
      "E", "E", "E", "E",
      "A", "A", "A", "A",
      "B7", "B7", "B7", "B7",
      "E", "E", "E", "E",
    ], (name) => name === "B7"
      ? { scores: { B7: 0.96, B: 0.8 }, scoreMargin: 0.16, seventhEvidence: 0.25 }
      : {});
    const regions = smoothChordObservations(observations).regions;
    expect(regions[0].start).toBe(0);
    expect(regions.at(-1)!.end).toBe(8);
    for (let index = 1; index < regions.length; index += 1) {
      expect(regions[index].start).toBe(regions[index - 1].end);
      expect(regions[index].start).toBeGreaterThanOrEqual(regions[index - 1].start);
      expect(regions[index].end).toBeGreaterThan(regions[index].start);
    }
  });

  it("keeps cleaned regions synchronized with playback lookup", () => {
    const regions = smoothChordObservations(
      sequence(["E", "E", "E", "E", "A", "A", "A", "A"]),
    ).regions;
    const state = getPlaybackState(regions, 2.25, 4);
    expect(state.currentIndex).toBe(1);
    expect(state.nextIndex).toBe(-1);
    expect(state.chordProgress).toBeCloseTo(0.125);
  });

  it("decodes E to A to B7 to E as four meaningful regions", () => {
    const observations = sequence([
      "E", "E", "E", "E",
      "A", "A", "A", "A",
      "B7", "B7", "B7", "B7",
      "E", "E", "E", "E",
    ], (name) => name === "B7"
      ? { scores: { B7: 0.96, B: 0.8 }, scoreMargin: 0.16, seventhEvidence: 0.25 }
      : {});
    const regions = smoothChordObservations(observations).regions;
    expect(regions.map((region) => region.name)).toEqual(["E", "A", "B7", "E"]);
    expect(regions).toHaveLength(4);
  });

  it("keeps a stable minor chord through ornamental passing tones", () => {
    const observations = sequence(
      ["Am", "Am", "C", "Am", "Am", "Am"],
      (name) => name === "C"
        ? { scores: { C: 0.74, Am: 0.72 }, scoreMargin: 0.02, uncertain: true }
        : {},
    );
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(["Am"]);
  });

  it("uses dedicated bass-root evidence to resolve C versus A minor ambiguity", () => {
    const chroma = Array(12).fill(0);
    chroma[0] = 0.32;
    chroma[4] = 0.3;
    chroma[7] = 0.2;
    chroma[9] = 0.18;
    const aBass = Array(12).fill(0);
    aBass[9] = 0.8;
    aBass[0] = 0.1;
    aBass[4] = 0.1;

    expect(classifyChordEvidence(chroma).name).toBe("C");
    expect(classifyChordEvidence(chroma, { bassChroma: aBass }).name).toBe("Am");
  });

  it("does not promote a transient seventh but preserves strong sustained evidence", () => {
    const weak = Array(12).fill(0);
    weak[4] = 0.36;
    weak[8] = 0.28;
    weak[11] = 0.23;
    weak[2] = 0.05;
    weak[0] = 0.08;
    const strong = [...weak];
    strong[2] = 0.18;
    strong[0] = 0;

    expect(classifyChordEvidence(weak).name).toBe("E");
    expect(classifyChordEvidence(strong).name).toBe("E7");
  });

  it("allows strong extended labels only in detailed mode", () => {
    const cMajorSeven = Array(12).fill(0);
    cMajorSeven[0] = 0.3;
    cMajorSeven[4] = 0.25;
    cMajorSeven[7] = 0.22;
    cMajorSeven[11] = 0.23;

    expect(classifyChordEvidence(cMajorSeven, {
      settings: { chordDisplayMode: "simple" },
    }).name).toBe("C");
    expect(classifyChordEvidence(cMajorSeven, {
      settings: { chordDisplayMode: "detailed" },
    }).name).toBe("Cmaj7");
  });

  it("estimates a soft song key without treating it as a hard constraint", () => {
    const aMajorContext = Array(12).fill(0);
    for (const [pitchClass, value] of [[9, 0.25], [1, 0.18], [4, 0.2], [2, 0.12], [6, 0.1], [11, 0.08], [8, 0.07]] as const) {
      aMajorContext[pitchClass] = value;
    }
    const estimate = estimateKeyFromChroma(aMajorContext);
    expect(estimate).not.toBeNull();
    expect(estimate!.name).toBe("A major");
  });

  it("makes repeated four-window progressions label-consistent under a weak perturbation", () => {
    const chromas = {
      A: [0, 0.05, 0, 0, 0.32, 0, 0, 0.2, 0, 0.43, 0, 0],
      D: [0, 0, 0.4, 0, 0, 0.2, 0, 0, 0, 0.2, 0, 0.2],
      E: [0, 0, 0, 0, 0.42, 0, 0, 0, 0.22, 0, 0, 0.36],
    };
    const names = ["A", "D", "E", "A", "Am", "D", "E", "A"];
    const observations = names.map((name, index) => observation(name, index, {
      scores: index === 4 ? { Am: 0.8, A: 0.78 } : { [name]: 0.94 },
      scoreMargin: index === 4 ? 0.02 : 0.5,
      uncertain: index === 4,
    }));
    observations.forEach((item, index) => {
      item.chroma = chromas[(index % 4 === 1 ? "D" : index % 4 === 2 ? "E" : "A")];
    });
    const regions = smoothChordObservations(observations, {
      minimumChordDurationSeconds: 0,
      minimumChordDurationBeats: 0,
      requiredConsecutiveWindows: 1,
      changeMargin: 0,
      chordChangePenalty: 0,
      shortRegionPenalty: 0,
      offBeatChangePenalty: 0,
    }).regions;
    expect(regions.map((region) => region.name)).toEqual([
      "A", "D", "E", "A", "D", "E", "A",
    ]);
  });

  it.each([
    [["A", "D", "E", "A"], ["A", "D", "E", "A"]],
    [["Am", "F", "C", "G"], ["Am", "F", "C", "G"]],
    [["E", "E7", "A", "A"], ["E", "E7", "A"]],
  ])("decodes noisy synthetic progression %j", (progression, expected) => {
    const observations = progression.flatMap((name, regionIndex) => (
      Array.from({ length: 4 }, (_, offset) => observation(
        name,
        regionIndex * 4 + offset,
        name === "E7"
          ? {
            scores: { E7: 0.96, E: 0.79 },
            scoreMargin: 0.17,
            seventhEvidence: 0.2,
          }
          : { scores: { [name]: 0.94 } },
      ))
    ));
    expect(smoothChordObservations(observations).regions.map((region) => region.name))
      .toEqual(expected);
  });

  it("backdates a conservatively confirmed change to its evidence beat", () => {
    const observations = sequence(["E", "E", "E", "A", "A", "A"]);
    const result = smoothChordObservationsReducedLatency(observations);
    const decision = result.decisions.find((candidate) => candidate.toChord === "A");
    expect(decision).toBeDefined();
    expect(decision!.candidateEvidenceStart).toBe(1.5);
    expect(decision!.confirmationTime).toBeGreaterThan(decision!.candidateEvidenceStart);
    expect(decision!.finalBoundaryTime).toBeLessThan(decision!.confirmationTime);
    expect(result.regions.find((region) => region.name === "A")?.start)
      .toBe(decision!.finalBoundaryTime);
  });

  it("keeps G to D to Em as three regions at two chords per bar", () => {
    const observations = sequence([
      "G", "G", "G",
      "D", "D",
      "Em", "Em", "Em",
    ], (name) => ({
      scores: { [name]: 0.72 },
      confidence: 0.72,
      scoreMargin: 0.16,
    }));
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["G", "D", "Em"]);
    expect(smoothChordObservations(observations).regions.map(
      (region) => region.name,
    )).toEqual(["G", "D", "Em"]);
  });

  it("still rejects a one-window false chord", () => {
    const observations = sequence(["E", "E", "A", "E", "E"]);
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["E"]);
  });

  it("preserves a valid one-beat chord with strong bass-attack support", () => {
    const observations = sequence(
      ["E", "E", "A", "E", "E"],
      (name, index) => index === 2
        ? {
          scores: { A: 0.98, E: 0.05 },
          confidence: 0.95,
          scoreMargin: 0.9,
          boundaryStrength: 1,
          bassAttackRootClass: "A",
          bassAttackConfidence: 0.9,
        }
        : {},
    );
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["E", "A", "E"]);
  });

  it("merges a weak one-beat chord without bass support", () => {
    const observations = sequence(
      ["E", "E", "A", "E", "E"],
      (name, index) => index === 2
        ? {
          scores: { A: 0.61, E: 0.56 },
          confidence: 0.5,
          scoreMargin: 0.05,
          boundaryStrength: 1,
        }
        : {},
    );
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["E"]);
  });

  it("can select a preceding beat instead of always snapping forward", () => {
    const observations = sequence(["E", "E", "D", "A", "A", "A"], (name, index) => (
      index === 2
        ? {
          scores: { D: 0.57, A: 0.55, E: 0.2 },
          confidence: 0.55,
          scoreMargin: 0.02,
        }
        : {}
    ));
    const decoded = decodeReducedLatencySequence(observations, {
      changeEvidenceThreshold: 0.04,
    });
    const decision = decoded.decisions.find((candidate) => candidate.toChord === "A");
    expect(decision).toBeDefined();
    expect(decision!.finalBoundaryTime).toBeLessThanOrEqual(
      decision!.candidateEvidenceStart,
    );
  });

  it("caps repeated-section reinforcement below strong local evidence", () => {
    const observations = sequence(
      ["E", "E", "E", "E", "A", "A", "A", "A"],
      (name, index) => index >= 4
        ? {
          scores: { A: 0.84, E: 0.7 },
          confidence: 0.82,
          scoreMargin: 0.14,
        }
        : {},
    );
    observations.forEach((item) => {
      item.chroma = [0.1, 0, 0, 0, 0.35, 0, 0, 0, 0, 0.35, 0, 0.2];
    });
    expect(smoothChordObservationsReducedLatency(observations, {
      repeatedSectionConsistencyWeight: 0.2,
    }).regions.at(-1)?.name).toBe("A");
  });

  it("does not let a passing bass attack create a false chord change", () => {
    const observations = sequence(
      ["E", "E", "E", "E", "E"],
      (_name, index) => index === 2
        ? {
          scores: { A: 0.58, E: 0.54 },
          confidence: 0.52,
          scoreMargin: 0.04,
          bassAttackRootClass: "A",
          bassAttackConfidence: 0.8,
          bassIsPassingTone: true,
        }
        : {},
    );
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["E"]);
  });

  it("retains existing false-N and weak-seventh cleanup behavior", () => {
    const observations = sequence(
      ["E", "E7", "N", "E", "E"],
      (name, index) => index === 1
        ? {
          scores: { E7: 0.82, E: 0.79 },
          confidence: 0.75,
          scoreMargin: 0.03,
          seventhEvidence: 0.04,
        }
        : {},
    );
    expect(smoothChordObservationsReducedLatency(observations).regions.map(
      (region) => region.name,
    )).toEqual(["E"]);
  });
});

describe("staged rule observation API", () => {
  it("is exactly equivalent to the existing rule-only entry point", () => {
    const sampleRate = 8000;
    const samples = new Float32Array(sampleRate * 2);
    for (let index = 0; index < samples.length; index += 1) {
      const time = index / sampleRate;
      samples[index] = 0.2 * (
        Math.sin(2 * Math.PI * 196 * time)
        + Math.sin(2 * Math.PI * 246.94 * time)
        + Math.sin(2 * Math.PI * 293.66 * time)
      );
    }
    const beatGrid = { bpm: null, beatDuration: null, phase: 0 };
    const direct = analyzeChordProgression(samples, sampleRate, beatGrid);
    const staged = decodeHarmonyObservations(createHarmonyObservations(
      samples,
      sampleRate,
      {},
      {},
      beatGrid,
    ));
    expect(staged).toEqual(direct);
  });
});
