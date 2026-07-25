import type {
  ChordAnalysisResult,
  ChordBoundaryDecision,
  ChordDiagnostics,
  ChordEvent,
  ChordRegionDiagnostics,
  ChordSmoothingSettings,
  ChordWindowDiagnostics,
  KeyEstimate,
  RawChordFrame,
} from "./types";

const CHORD_ROOTS = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
const SHARP_ROOTS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const FLAT_ROOTS = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"];
// Long enough to capture a typical guitar strum/arpeggio before beat-level
// aggregation, while retaining a substantially shorter diagnostic hop.
const FRAME_SIZE = 8192;
const SILENCE_RMS = 0.008;
const BEATS_PER_BAR = 4;

export const BALANCED_CHORD_SMOOTHING_SETTINGS: ChordSmoothingSettings = {
  minimumChordDurationSeconds: 1.5,
  minimumChordDurationBeats: 2,
  requiredConsecutiveWindows: 2,
  changeMargin: 0.04,
  chordChangePenalty: 0.09,
  shortRegionPenalty: 0.12,
  weakTransitionPenalty: 0.1,
  offBeatChangePenalty: 0.08,
  minimumChordConfidence: 0.16,
  minimumScoreMargin: 0.01,
  preserveSeventhThreshold: 0.085,
  shortNoChordMergeSeconds: 1,
  fallbackWindowSeconds: 1,
  rawFrameHopSeconds: 0.25,
  highConfidenceShortChord: 0.82,
  chordDisplayMode: "simple",
  rootSupportWeight: 0.2,
  bassAgreementWeight: 0.28,
  keyCompatibilityWeight: 0.055,
  commonTransitionBonus: 0.035,
  weakExtensionPenalty: 0.06,
  unnecessaryNoChordPenalty: 0.18,
  strongSeventhThreshold: 0.1,
  repeatedSectionConsistencyWeight: 0.035,
  changeEvidenceThreshold: 0.045,
  maximumRepeatedSectionBonus: 0.015,
  bassAttackWindowFraction: 0.4,
  validShortChordScoreMargin: 0.018,
};

type ChordTemplate = {
  name: string;
  root: number;
  tones: number[];
  requiredColorTones: number[];
  quality: "major" | "minor" | "dominant7" | "maj7" | "min7" | "sus2" | "sus4" | "add9" | "dim" | "aug";
  tier: "simple" | "detailed";
  seventhPitchClass: number | null;
};

type ScoredTemplate = ChordTemplate & { score: number };

type InternalRawFrame = RawChordFrame & {
  chroma: number[];
  rootChroma: number[];
  bassChroma: number[];
  rms: number;
};

export type BeatGrid = {
  bpm: number | null;
  beatDuration: number | null;
  phase: number;
};

export type HarmonyEvidenceOptions = {
  /** Dedicated Demucs bass evidence, sample-aligned with the harmonic signal. */
  bassSamples?: Float32Array;
  source?: "separated-harmonic-mix" | "full-mix" | "guitar-only";
};

export type ChordObservation = {
  start: number;
  end: number;
  bestChord: string;
  bestScore: number;
  secondBestChord: string;
  secondBestScore: number;
  confidence: number;
  scoreMargin: number;
  uncertain: boolean;
  candidateScores: Record<string, number>;
  noChordScore: number;
  seventhEvidence: number;
  seventhEvidenceByChord?: Record<string, number>;
  chroma?: number[];
  rootChroma?: number[];
  bassChroma?: number[];
  rootClass?: string | null;
  bassRootClass?: string | null;
  bassConfidence?: number;
  bassAttackRootClass?: string | null;
  bassAttackConfidence?: number;
  bassSustainRootClass?: string | null;
  bassSustainConfidence?: number;
  bassIsPassingTone?: boolean;
  keyEstimate?: KeyEstimate | null;
  usedBassSupport?: boolean;
  /** 1 = inferred bar boundary, 0 = no beat grid. */
  boundaryStrength: number;
};

type InternalRegion = ChordEvent & {
  firstWindow: number;
  lastWindow: number;
  windowCount: number;
  scoreMargin: number;
  seventhEvidence: number;
  boundaryStrength: number;
  bestScore: number;
  secondBestScore: number;
  rootClass: string | null;
  keyEstimate: string | null;
  usedBassSupport: boolean;
  usedSmoothingOverride: boolean;
  simplifiedFromExtendedChord: boolean;
  mergedFromShortRegions: boolean;
  candidateEvidenceStart: number | null;
  confirmationTime: number | null;
  finalBoundaryTime: number | null;
};

type CleanupCounts = {
  shortRegionsRemoved: number;
  regionsMerged: number;
};

const TEMPLATES: ChordTemplate[] = [];
for (let root = 0; root < 12; root += 1) {
  TEMPLATES.push({
    name: CHORD_ROOTS[root],
    root,
    tones: [root, (root + 4) % 12, (root + 7) % 12],
    requiredColorTones: [(root + 4) % 12],
    quality: "major",
    tier: "simple",
    seventhPitchClass: null,
  });
  TEMPLATES.push({
    name: `${CHORD_ROOTS[root]}m`,
    root,
    tones: [root, (root + 3) % 12, (root + 7) % 12],
    requiredColorTones: [(root + 3) % 12],
    quality: "minor",
    tier: "simple",
    seventhPitchClass: null,
  });
  TEMPLATES.push({
    name: `${CHORD_ROOTS[root]}7`,
    root,
    tones: [root, (root + 4) % 12, (root + 7) % 12, (root + 10) % 12],
    requiredColorTones: [(root + 4) % 12, (root + 10) % 12],
    quality: "dominant7",
    tier: "simple",
    seventhPitchClass: (root + 10) % 12,
  });
  TEMPLATES.push(
    {
      name: `${CHORD_ROOTS[root]}maj7`,
      root,
      tones: [root, (root + 4) % 12, (root + 7) % 12, (root + 11) % 12],
      requiredColorTones: [(root + 4) % 12, (root + 11) % 12],
      quality: "maj7",
      tier: "detailed",
      seventhPitchClass: (root + 11) % 12,
    },
    {
      name: `${CHORD_ROOTS[root]}m7`,
      root,
      tones: [root, (root + 3) % 12, (root + 7) % 12, (root + 10) % 12],
      requiredColorTones: [(root + 3) % 12, (root + 10) % 12],
      quality: "min7",
      tier: "detailed",
      seventhPitchClass: (root + 10) % 12,
    },
    {
      name: `${CHORD_ROOTS[root]}sus2`,
      root,
      tones: [root, (root + 2) % 12, (root + 7) % 12],
      requiredColorTones: [(root + 2) % 12],
      quality: "sus2",
      tier: "detailed",
      seventhPitchClass: null,
    },
    {
      name: `${CHORD_ROOTS[root]}sus4`,
      root,
      tones: [root, (root + 5) % 12, (root + 7) % 12],
      requiredColorTones: [(root + 5) % 12],
      quality: "sus4",
      tier: "detailed",
      seventhPitchClass: null,
    },
    {
      name: `${CHORD_ROOTS[root]}add9`,
      root,
      tones: [root, (root + 2) % 12, (root + 4) % 12, (root + 7) % 12],
      requiredColorTones: [(root + 2) % 12, (root + 4) % 12],
      quality: "add9",
      tier: "detailed",
      seventhPitchClass: null,
    },
    {
      name: `${CHORD_ROOTS[root]}dim`,
      root,
      tones: [root, (root + 3) % 12, (root + 6) % 12],
      requiredColorTones: [(root + 3) % 12, (root + 6) % 12],
      quality: "dim",
      tier: "detailed",
      seventhPitchClass: null,
    },
    {
      name: `${CHORD_ROOTS[root]}aug`,
      root,
      tones: [root, (root + 4) % 12, (root + 8) % 12],
      requiredColorTones: [(root + 4) % 12, (root + 8) % 12],
      quality: "aug",
      tier: "detailed",
      seventhPitchClass: null,
    },
  );
}

function templatesFor(settings: ChordSmoothingSettings): ChordTemplate[] {
  return settings.chordDisplayMode === "detailed"
    ? TEMPLATES
    : TEMPLATES.filter((template) => template.tier === "simple");
}

function stateNames(settings: ChordSmoothingSettings): string[] {
  return [...templatesFor(settings).map((template) => template.name), "N"];
}

const MAJOR_KEY_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR_KEY_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];
const MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11];
const MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10];

function profileScore(chroma: number[], profile: number[], tonic: number): number {
  const profileTotal = profile.reduce((sum, value) => sum + value, 0);
  let score = 0;
  for (let pitchClass = 0; pitchClass < 12; pitchClass += 1) {
    const degree = (pitchClass - tonic + 12) % 12;
    score += chroma[pitchClass] * (profile[degree] / profileTotal);
  }
  return score;
}

export function estimateKeyFromChroma(chroma: number[]): KeyEstimate | null {
  if (chroma.reduce((sum, value) => sum + value, 0) <= 0) return null;
  const candidates: Array<KeyEstimate & { score: number }> = [];
  for (let tonic = 0; tonic < 12; tonic += 1) {
    for (const mode of ["major", "minor"] as const) {
      const score = profileScore(
        chroma,
        mode === "major" ? MAJOR_KEY_PROFILE : MINOR_KEY_PROFILE,
        tonic,
      );
      candidates.push({
        name: `${CHORD_ROOTS[tonic]} ${mode}`,
        tonic,
        mode,
        confidence: 0,
        score,
      });
    }
  }
  candidates.sort((a, b) => b.score - a.score);
  const best = candidates[0];
  const margin = best.score - (candidates[1]?.score ?? 0);
  return {
    name: best.name,
    tonic: best.tonic,
    mode: best.mode,
    // Key is deliberately a weak context feature. Confidence expresses
    // separation from the runner-up rather than pretending certainty.
    confidence: clamp(margin / 0.018),
  };
}

function diatonicTriads(key: KeyEstimate): Array<{ root: number; quality: "major" | "minor" | "dim" }> {
  const scale = key.mode === "major" ? MAJOR_SCALE : MINOR_SCALE;
  const qualities = key.mode === "major"
    ? ["major", "minor", "minor", "major", "major", "minor", "dim"] as const
    : ["minor", "dim", "major", "minor", "minor", "major", "major"] as const;
  return scale.map((degree, index) => ({
    root: (key.tonic + degree) % 12,
    quality: qualities[index],
  }));
}

function keyCompatibility(template: ChordTemplate, key: KeyEstimate | null): number {
  if (!key) return 0;
  const basicQuality = template.quality === "minor" || template.quality === "min7"
    ? "minor"
    : template.quality === "dim"
      ? "dim"
      : "major";
  const triads = diatonicTriads(key);
  if (triads.some((triad) => triad.root === template.root && triad.quality === basicQuality)) {
    return key.confidence;
  }
  // Dominants that resolve by a fifth are common even when chromatic.
  if (template.quality === "dominant7"
    && triads.some((triad) => (template.root + 5) % 12 === triad.root)) {
    return key.confidence * 0.45;
  }
  const scale = key.mode === "major" ? MAJOR_SCALE : MINOR_SCALE;
  return scale.includes((template.root - key.tonic + 12) % 12)
    ? key.confidence * 0.12
    : -key.confidence * 0.12;
}

function spellPitchClass(pitchClass: number, key: KeyEstimate | null): string {
  if (!key) return CHORD_ROOTS[pitchClass];
  const preferFlats = [1, 3, 5, 8, 10].includes(key.tonic);
  return (preferFlats ? FLAT_ROOTS : SHARP_ROOTS)[pitchClass];
}

function spellChordName(name: string, keyName: string | null): string {
  if (name === "N") return name;
  const template = TEMPLATES.find((candidate) => candidate.name === name);
  if (!template) return name;
  const key = keyName
    ? (() => {
      const [rootName, mode] = keyName.split(" ");
      const tonic = [...SHARP_ROOTS, ...FLAT_ROOTS].findIndex((root) => root === rootName) % 12;
      return tonic >= 0 && (mode === "major" || mode === "minor")
        ? { name: keyName, tonic, mode, confidence: 1 } as KeyEstimate
        : null;
    })()
    : null;
  const suffix = name.slice(CHORD_ROOTS[template.root].length);
  return `${spellPitchClass(template.root, key)}${suffix}`;
}

function confidenceLevel(confidence: number): "high" | "medium" | "low" {
  if (confidence >= 0.72) return "high";
  if (confidence >= 0.42) return "medium";
  return "low";
}

function clamp(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

function frameRms(frame: Float32Array): number {
  if (!frame.length) return 0;
  let energy = 0;
  for (const value of frame) energy += value * value;
  return Math.sqrt(energy / frame.length);
}

function fftMagnitudes(input: Float32Array): Float64Array {
  const n = input.length;
  const real = new Float64Array(n);
  const imag = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    real[i] = input[i] * (0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1)));
  }

  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) [real[i], real[j], imag[i], imag[j]] = [real[j], real[i], imag[j], imag[i]];
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    for (let start = 0; start < n; start += length) {
      for (let offset = 0; offset < length / 2; offset += 1) {
        const phase = angle * offset;
        const wr = Math.cos(phase);
        const wi = Math.sin(phase);
        const even = start + offset;
        const odd = even + length / 2;
        const tr = wr * real[odd] - wi * imag[odd];
        const ti = wr * imag[odd] + wi * real[odd];
        real[odd] = real[even] - tr;
        imag[odd] = imag[even] - ti;
        real[even] += tr;
        imag[even] += ti;
      }
    }
  }
  return Float64Array.from({ length: n / 2 }, (_, index) => Math.hypot(real[index], imag[index]));
}

function chromaFromMagnitudes(
  magnitudes: Float64Array,
  frameLength: number,
  sampleRate: number,
  lowHz: number,
  highHz: number,
  lowFrequencyBias = false,
): number[] {
  const chroma = Array(12).fill(0) as number[];
  const binHz = sampleRate / frameLength;
  for (let bin = 1; bin < magnitudes.length; bin += 1) {
    const frequency = bin * binHz;
    if (frequency < lowHz || frequency > highHz) continue;
    const midi = 69 + 12 * Math.log2(frequency / 440);
    const pitchClass = ((Math.round(midi) % 12) + 12) % 12;
    const frequencyWeight = lowFrequencyBias
      ? Math.sqrt(lowHz / Math.max(lowHz, frequency))
      : 1;
    chroma[pitchClass] += Math.sqrt(magnitudes[bin]) * frequencyWeight;
  }
  const total = chroma.reduce((sum, value) => sum + value, 0);
  return total > 0 ? chroma.map((value) => value / total) : chroma;
}

function spectralChroma(
  frame: Float32Array,
  sampleRate: number,
): { chroma: number[]; rootChroma: number[] } {
  const magnitudes = fftMagnitudes(frame);
  return {
    chroma: chromaFromMagnitudes(magnitudes, frame.length, sampleRate, 70, 1400),
    rootChroma: chromaFromMagnitudes(magnitudes, frame.length, sampleRate, 42, 320, true),
  };
}

export function chromaForChordFrame(frame: Float32Array, sampleRate: number): number[] {
  return spectralChroma(frame, sampleRate).chroma;
}

/**
 * Per-frame harmony features (harmony-features-v1) for the learned engine: the
 * same treble chroma, low-band root chroma, and RMS energy the ML model trained
 * on. Frame size and 0.25 s hop match the training pipeline (ml app_features.py),
 * so the learned provider receives exactly the representation it expects.
 */
export function extractHarmonyFrames(
  samples: Float32Array,
  sampleRate: number,
  hopSeconds = 0.25,
): { frameTimes: number[]; harmonicChroma: number[][]; bassChroma: number[][]; onsetStrength: number[] } {
  const hop = Math.max(1, Math.round(hopSeconds * sampleRate));
  const frameTimes: number[] = [];
  const harmonicChroma: number[][] = [];
  const bassChroma: number[][] = [];
  const onsetStrength: number[] = [];
  for (let start = 0; start < samples.length; start += hop) {
    const frame = new Float32Array(FRAME_SIZE);
    frame.set(samples.subarray(start, Math.min(samples.length, start + FRAME_SIZE)));
    const spectral = spectralChroma(frame, sampleRate);
    frameTimes.push(start / sampleRate);
    harmonicChroma.push(spectral.chroma);
    bassChroma.push(spectral.rootChroma);
    onsetStrength.push(frameRms(frame));  // the model's energy channel
  }
  return { frameTimes, harmonicChroma, bassChroma, onsetStrength };
}

function strongestPitchClass(chroma: number[]): { pitchClass: number | null; confidence: number } {
  const ranked = chroma
    .map((value, pitchClass) => ({ value, pitchClass }))
    .sort((a, b) => b.value - a.value);
  if (!ranked.length || ranked[0].value <= 0) return { pitchClass: null, confidence: 0 };
  const confidence = clamp(
    ranked[0].value * 1.8 + Math.max(0, ranked[0].value - (ranked[1]?.value ?? 0)) * 1.8,
  );
  return { pitchClass: ranked[0].pitchClass, confidence };
}

function baseTriadName(template: ChordTemplate): string {
  if (template.quality === "minor" || template.quality === "min7") {
    return `${CHORD_ROOTS[template.root]}m`;
  }
  return CHORD_ROOTS[template.root];
}

function isExtended(template: ChordTemplate): boolean {
  return ["dominant7", "maj7", "min7", "add9"].includes(template.quality);
}

function scoreTemplates(
  chroma: number[],
  rootChroma: number[],
  bassChroma: number[],
  keyEstimate: KeyEstimate | null,
  settings: ChordSmoothingSettings,
): ScoredTemplate[] {
  const triadScores = new Map<string, number>();
  const available = templatesFor(settings);
  for (const template of TEMPLATES.filter(
    (candidate) => candidate.quality === "major" || candidate.quality === "minor",
  )) {
    const toneEnergy = template.tones.reduce((sum, pitchClass) => sum + chroma[pitchClass], 0);
    const offEnergy = Math.max(0, 1 - toneEnergy);
    const score = toneEnergy
      - offEnergy * 0.34
      + chroma[template.root] * 0.1
      + rootChroma[template.root] * settings.rootSupportWeight
      + bassChroma[template.root] * settings.bassAgreementWeight
      + keyCompatibility(template, keyEstimate) * settings.keyCompatibilityWeight;
    triadScores.set(template.name, score);
  }

  return available.map((template) => {
    if (template.quality === "major" || template.quality === "minor") {
      return { ...template, score: triadScores.get(template.name)! };
    }
    if (isExtended(template)) {
      const baseScore = triadScores.get(baseTriadName(template))!;
      const addedPitch = template.quality === "add9"
        ? (template.root + 2) % 12
        : template.seventhPitchClass!;
      const colorEvidence = chroma[addedPitch];
      const threshold = template.quality === "dominant7"
        ? settings.preserveSeventhThreshold
        : settings.strongSeventhThreshold;
      const weakPenalty = colorEvidence < settings.strongSeventhThreshold
        ? settings.weakExtensionPenalty
        : 0;
      return {
        ...template,
        // Extensions must beat their underlying triad with independent,
        // sustained color-tone evidence; complexity never receives a free tone.
        score: baseScore + (colorEvidence - threshold) * 1.35 - weakPenalty,
      };
    }
    const toneEnergy = template.tones.reduce((sum, pitchClass) => sum + chroma[pitchClass], 0);
    const offEnergy = Math.max(0, 1 - toneEnergy);
    const colorSupport = Math.min(...template.requiredColorTones.map((tone) => chroma[tone]));
    return {
      ...template,
      score: toneEnergy
        - offEnergy * 0.36
        + rootChroma[template.root] * settings.rootSupportWeight
        + bassChroma[template.root] * settings.bassAgreementWeight
        + keyCompatibility(template, keyEstimate) * settings.keyCompatibilityWeight
        - (colorSupport < 0.09 ? settings.weakExtensionPenalty : 0),
    };
  }).sort((a, b) => b.score - a.score);
}

function confidenceFor(bestScore: number, scoreMargin: number): number {
  return clamp(bestScore * 0.8 + clamp(scoreMargin / 0.12) * 0.2);
}

function classifyDetailed(
  chroma: number[],
  rms: number,
  settings: ChordSmoothingSettings,
  rootChroma: number[] = chroma,
  bassChroma: number[] = Array(12).fill(0) as number[],
  keyEstimate: KeyEstimate | null = null,
): {
  ranked: ScoredTemplate[];
  confidence: number;
  scoreMargin: number;
  uncertain: boolean;
  noChord: boolean;
} {
  const ranked = scoreTemplates(chroma, rootChroma, bassChroma, keyEstimate, settings);
  const bestScore = ranked[0]?.score ?? 0;
  const secondBestScore = ranked[1]?.score ?? 0;
  const scoreMargin = bestScore - secondBestScore;
  const confidence = confidenceFor(bestScore, scoreMargin);
  return {
    ranked,
    confidence,
    scoreMargin,
    uncertain: confidence < settings.minimumChordConfidence
      || scoreMargin < settings.minimumScoreMargin,
    // Moderate ambiguity is not absence of harmony. `N` is reserved for
    // genuine low energy or diffuse evidence without a stable low root.
    noChord: rms < SILENCE_RMS * 0.5
      || (bestScore < 0.075
        && strongestPitchClass(rootChroma).confidence < 0.2
        && strongestPitchClass(bassChroma).confidence < 0.2),
  };
}

export function classifyChordChroma(
  chroma: number[],
  settings: Partial<ChordSmoothingSettings> = {},
): { name: string; confidence: number } {
  const resolved = resolveSettings(settings);
  const classification = classifyDetailed(chroma, 1, resolved);
  return {
    name: classification.noChord ? "N" : classification.ranked[0].name,
    confidence: classification.confidence,
  };
}

export function classifyChordEvidence(
  chroma: number[],
  {
    rootChroma = chroma,
    bassChroma = Array(12).fill(0) as number[],
    keyEstimate = null,
    settings = {},
  }: {
    rootChroma?: number[];
    bassChroma?: number[];
    keyEstimate?: KeyEstimate | null;
    settings?: Partial<ChordSmoothingSettings>;
  } = {},
): {
  name: string;
  confidence: number;
  bestScore: number;
  secondBestChord: string;
  secondBestScore: number;
} {
  const resolved = resolveSettings(settings);
  const classification = classifyDetailed(
    chroma,
    1,
    resolved,
    rootChroma,
    bassChroma,
    keyEstimate,
  );
  const [best, second] = classification.ranked;
  return {
    name: classification.noChord ? "N" : best.name,
    confidence: classification.confidence,
    bestScore: best.score,
    secondBestChord: second?.name ?? "N",
    secondBestScore: second?.score ?? 0,
  };
}

export function resolveSettings(
  settings: Partial<ChordSmoothingSettings> = {},
): ChordSmoothingSettings {
  const resolved = { ...BALANCED_CHORD_SMOOTHING_SETTINGS, ...settings };
  return {
    ...resolved,
    minimumChordDurationSeconds: Math.max(0, resolved.minimumChordDurationSeconds),
    minimumChordDurationBeats: Math.max(0, resolved.minimumChordDurationBeats),
    requiredConsecutiveWindows: Math.max(1, Math.round(resolved.requiredConsecutiveWindows)),
    fallbackWindowSeconds: clamp(resolved.fallbackWindowSeconds, 0.25, 2),
    rawFrameHopSeconds: clamp(resolved.rawFrameHopSeconds, 0.05, 1),
    preserveSeventhThreshold: clamp(resolved.preserveSeventhThreshold),
    strongSeventhThreshold: clamp(resolved.strongSeventhThreshold),
    rootSupportWeight: clamp(resolved.rootSupportWeight, 0, 1),
    bassAgreementWeight: clamp(resolved.bassAgreementWeight, 0, 1),
    keyCompatibilityWeight: clamp(resolved.keyCompatibilityWeight, 0, 0.25),
    commonTransitionBonus: clamp(resolved.commonTransitionBonus, 0, 0.2),
    weakExtensionPenalty: clamp(resolved.weakExtensionPenalty, 0, 0.3),
    unnecessaryNoChordPenalty: clamp(resolved.unnecessaryNoChordPenalty, 0, 0.5),
    repeatedSectionConsistencyWeight: clamp(
      resolved.repeatedSectionConsistencyWeight,
      0,
      0.2,
    ),
    changeEvidenceThreshold: clamp(resolved.changeEvidenceThreshold, 0.005, 0.5),
    maximumRepeatedSectionBonus: clamp(resolved.maximumRepeatedSectionBonus, 0, 0.1),
    bassAttackWindowFraction: clamp(resolved.bassAttackWindowFraction, 0.15, 0.75),
    validShortChordScoreMargin: clamp(resolved.validShortChordScoreMargin, 0, 0.2),
    chordDisplayMode: resolved.chordDisplayMode === "detailed" ? "detailed" : "simple",
  };
}

/**
 * Estimates both tempo and the phase of its beat grid. Tabsmith still exposes
 * only BPM publicly; the phase is used internally to make chord observations
 * beat-synchronous.
 */
export function estimateBeatGrid(
  samples: Float32Array,
  sampleRate: number,
): BeatGrid {
  const hop = 1024;
  const envelope: number[] = [];
  let previous = 0;
  for (let start = 0; start + hop <= samples.length; start += hop) {
    let energy = 0;
    for (let index = start; index < start + hop; index += 1) {
      energy += samples[index] * samples[index];
    }
    const current = Math.sqrt(energy / hop);
    envelope.push(Math.max(0, current - previous));
    previous = current;
  }
  if (envelope.length < 16) return { bpm: null, beatDuration: null, phase: 0 };

  let bestBpm = 0;
  let bestScore = 0;
  let bestRawScore = 0;
  let bestLag = 0;
  const tempoCandidates: { bpm: number; lag: number; rawScore: number }[] = [];
  for (let bpm = 60; bpm <= 190; bpm += 1) {
    const lag = Math.round((60 * sampleRate) / (bpm * hop));
    let score = 0;
    for (let index = lag; index < envelope.length; index += 1) {
      score += envelope[index] * envelope[index - lag];
    }
    // Autocorrelation often gives equal peaks at half/double tempo. A small
    // mid-tempo prior breaks near-ties without overriding a clearly stronger
    // slow-song pulse.
    const midTempoPreference = 1 + 0.08 * (1 - Math.abs(bpm - 120) / 130);
    const weightedScore = score * midTempoPreference;
    tempoCandidates.push({ bpm, lag, rawScore: score });
    if (weightedScore > bestScore) {
      [bestBpm, bestScore, bestRawScore, bestLag] = [bpm, weightedScore, score, lag];
    }
  }
  if (bestBpm <= 95) {
    const doubled = tempoCandidates
      .filter((candidate) => Math.abs(candidate.bpm - bestBpm * 2) <= 1)
      .sort((a, b) => b.rawScore - a.rawScore)[0];
    if (doubled && doubled.rawScore >= bestRawScore * 0.72) {
      [bestBpm, bestRawScore, bestLag] = [doubled.bpm, doubled.rawScore, doubled.lag];
      bestScore = doubled.rawScore;
    }
  }
  if (!bestBpm || !bestLag || bestScore <= 1e-10) {
    return { bpm: null, beatDuration: null, phase: 0 };
  }

  let phaseIndex = 0;
  let phaseScore = -Infinity;
  for (let phase = 0; phase < bestLag; phase += 1) {
    let score = 0;
    for (let index = phase; index < envelope.length; index += bestLag) {
      score += envelope[index];
    }
    if (score > phaseScore) [phaseIndex, phaseScore] = [phase, score];
  }
  return {
    bpm: bestBpm,
    beatDuration: 60 / bestBpm,
    phase: (phaseIndex * hop) / sampleRate,
  };
}

function createRawFrames(
  samples: Float32Array,
  sampleRate: number,
  settings: ChordSmoothingSettings,
  bassSamples?: Float32Array,
): InternalRawFrame[] {
  const duration = samples.length / sampleRate;
  const hop = Math.max(1, Math.round(settings.rawFrameHopSeconds * sampleRate));
  const frames: InternalRawFrame[] = [];
  for (let start = 0; start < samples.length; start += hop) {
    const padded = new Float32Array(FRAME_SIZE);
    padded.set(samples.subarray(start, Math.min(samples.length, start + FRAME_SIZE)));
    const spectral = spectralChroma(padded, sampleRate);
    const chroma = spectral.chroma;
    let bassChroma = Array(12).fill(0) as number[];
    if (bassSamples?.length) {
      const bassFrame = new Float32Array(FRAME_SIZE);
      bassFrame.set(bassSamples.subarray(start, Math.min(bassSamples.length, start + FRAME_SIZE)));
      bassChroma = spectralChroma(bassFrame, sampleRate).rootChroma;
    }
    const rms = frameRms(padded);
    const classification = classifyDetailed(
      chroma,
      rms,
      settings,
      spectral.rootChroma,
      bassChroma,
    );
    const best = classification.ranked[0];
    const second = classification.ranked[1];
    const root = strongestPitchClass(spectral.rootChroma);
    const bassRoot = strongestPitchClass(bassChroma);
    frames.push({
      start: start / sampleRate,
      end: Math.min(duration, (start + hop) / sampleRate),
      bestChord: classification.noChord ? "N" : best.name,
      bestScore: best?.score ?? 0,
      secondBestChord: second?.name ?? "N",
      secondBestScore: second?.score ?? 0,
      confidence: classification.confidence,
      scoreMargin: classification.scoreMargin,
      uncertain: classification.uncertain,
      chroma,
      rootChroma: spectral.rootChroma,
      bassChroma,
      rms,
      rootClass: root.pitchClass === null ? null : CHORD_ROOTS[root.pitchClass],
      bassRootClass: bassRoot.pitchClass === null ? null : CHORD_ROOTS[bassRoot.pitchClass],
      bassConfidence: bassRoot.confidence,
      keyEstimate: null,
    });
  }
  return frames;
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function robustChroma(
  frames: InternalRawFrame[],
  feature: "chroma" | "rootChroma" | "bassChroma" = "chroma",
): number[] {
  if (!frames.length) return Array(12).fill(0) as number[];
  const peakRms = Math.max(...frames.map((frame) => frame.rms));
  const activeFrames = frames.filter(
    (frame) => frame.rms >= Math.max(SILENCE_RMS * 0.5, peakRms * 0.15),
  );
  const evidence = activeFrames.length ? activeFrames : frames;
  const chroma = Array.from({ length: 12 }, (_, pitchClass) => {
    const values = evidence.map((frame) => frame[feature][pitchClass]);
    const totalWeight = evidence.reduce((sum, frame) => sum + frame.rms, 0);
    const weightedMean = totalWeight > 0
      ? evidence.reduce(
        (sum, frame) => sum + frame[feature][pitchClass] * frame.rms,
        0,
      ) / totalWeight
      : values.reduce((sum, value) => sum + value, 0) / values.length;
    // The median suppresses passing notes; the energy-weighted mean preserves
    // arpeggiated chord tones that do not sound in every raw frame.
    return median(values) * 0.35 + weightedMean * 0.65;
  });
  const total = chroma.reduce((sum, value) => sum + value, 0);
  return total > 0 ? chroma.map((value) => value / total) : chroma;
}

function observationBoundaries(
  duration: number,
  beatGrid: BeatGrid,
  settings: ChordSmoothingSettings,
): { times: number[]; beatAligned: boolean } {
  if (!beatGrid.beatDuration || !beatGrid.bpm) {
    const times = [0];
    for (
      let time = settings.fallbackWindowSeconds;
      time < duration - 1e-6;
      time += settings.fallbackWindowSeconds
    ) {
      times.push(time);
    }
    times.push(duration);
    return { times, beatAligned: false };
  }

  const times = [0];
  const beatDuration = beatGrid.beatDuration;
  let first = beatGrid.phase % beatDuration;
  while (first > 1e-6) first -= beatDuration;
  while (first + beatDuration <= 1e-6) first += beatDuration;
  for (let time = first + beatDuration; time < duration - 1e-6; time += beatDuration) {
    // Fold tiny phase fragments into the adjacent beat window. A sliver at the
    // beginning or end otherwise behaves like a spurious low-energy `N` frame.
    if (time >= beatDuration * 0.35 && duration - time >= beatDuration * 0.35) {
      times.push(time);
    }
  }
  times.push(duration);
  return { times, beatAligned: true };
}

function boundaryStrength(index: number, beatAligned: boolean): number {
  if (!beatAligned || index === 0) return 0;
  const beatIndex = index - 1;
  if (beatIndex % BEATS_PER_BAR === 0) return 1;
  if (beatIndex % 2 === 0) return 0.86;
  return 0.72;
}

function aggregateObservations(
  rawFrames: InternalRawFrame[],
  duration: number,
  beatGrid: BeatGrid,
  settings: ChordSmoothingSettings,
): {
  observations: ChordObservation[];
  beatAlignedBoundaries: number;
  keyEstimate: KeyEstimate | null;
} {
  if (duration <= 0) {
    return { observations: [], beatAlignedBoundaries: 0, keyEstimate: null };
  }
  const boundaries = observationBoundaries(duration, beatGrid, settings);
  const observations: ChordObservation[] = [];
  let beatAlignedBoundaries = 0;
  const globalKey = estimateKeyFromChroma(robustChroma(rawFrames));

  for (let index = 0; index + 1 < boundaries.times.length; index += 1) {
    const start = boundaries.times[index];
    const end = boundaries.times[index + 1];
    const contained = rawFrames.filter((frame) => {
      const midpoint = (frame.start + frame.end) / 2;
      return midpoint >= start && midpoint < end;
    });
    const fallback = contained.length
      ? contained
      : rawFrames.filter((frame) => frame.end > start && frame.start < end);
    const chroma = robustChroma(fallback);
    const rootChroma = robustChroma(fallback, "rootChroma");
    const bassChroma = robustChroma(fallback, "bassChroma");
    const attackCutoff = start
      + (end - start) * settings.bassAttackWindowFraction;
    const bassAttackFrames = fallback.filter(
      (frame) => (frame.start + frame.end) / 2 < attackCutoff,
    );
    const bassSustainFrames = fallback.filter(
      (frame) => (frame.start + frame.end) / 2 >= attackCutoff,
    );
    const bassAttackChroma = robustChroma(
      bassAttackFrames.length ? bassAttackFrames : fallback,
      "bassChroma",
    );
    const bassSustainChroma = robustChroma(
      bassSustainFrames.length ? bassSustainFrames : fallback,
      "bassChroma",
    );
    const localFrames = rawFrames.filter((frame) => (
      frame.end > start - 8 && frame.start < end + 8
    ));
    const localKey = estimateKeyFromChroma(robustChroma(localFrames));
    const keyEstimate = localKey && (
      localKey.confidence >= 0.25 || !globalKey
    ) ? localKey : globalKey;
    const rms = Math.sqrt(
      fallback.reduce((sum, frame) => sum + frame.rms * frame.rms, 0)
        / Math.max(1, fallback.length),
    );
    const classification = classifyDetailed(
      chroma,
      rms,
      settings,
      rootChroma,
      bassChroma,
      keyEstimate,
    );
    const best = classification.ranked[0];
    const second = classification.ranked[1];
    const scores = Object.fromEntries(
      classification.ranked.map((candidate) => [candidate.name, candidate.score]),
    );
    const strength = boundaryStrength(index, boundaries.beatAligned);
    const root = strongestPitchClass(rootChroma);
    const bassRoot = strongestPitchClass(bassChroma);
    const bassAttackRoot = strongestPitchClass(bassAttackChroma);
    const bassSustainRoot = strongestPitchClass(bassSustainChroma);
    const bassIsPassingTone = bassAttackRoot.pitchClass !== null
      && bassSustainRoot.pitchClass !== null
      && bassAttackRoot.pitchClass !== bassSustainRoot.pitchClass
      && bassAttackRoot.confidence < bassSustainRoot.confidence * 0.75;
    const usedBassSupport = bassRoot.pitchClass !== null && bassRoot.confidence >= 0.18;
    if (strength > 0) beatAlignedBoundaries += 1;
    observations.push({
      start,
      end,
      bestChord: classification.noChord ? "N" : best.name,
      bestScore: best?.score ?? 0,
      secondBestChord: second?.name ?? "N",
      secondBestScore: second?.score ?? 0,
      confidence: classification.confidence,
      scoreMargin: classification.scoreMargin,
      uncertain: classification.uncertain,
      candidateScores: scores,
      noChordScore: classification.noChord
        ? 0.9
        : clamp(
          0.02
            - (best.score - 0.08) * 1.2
            - root.confidence * settings.unnecessaryNoChordPenalty
            - bassRoot.confidence * settings.unnecessaryNoChordPenalty,
          -0.75,
          0.02,
        ),
      seventhEvidence: best && isExtended(best)
        ? chroma[
          best.quality === "add9"
            ? (best.root + 2) % 12
            : best.seventhPitchClass!
        ]
        : 0,
      seventhEvidenceByChord: Object.fromEntries(
        TEMPLATES
          .filter(isExtended)
          .map((template) => [
            template.name,
            chroma[
              template.quality === "add9"
                ? (template.root + 2) % 12
                : template.seventhPitchClass!
            ],
          ]),
      ),
      chroma,
      rootChroma,
      bassChroma,
      rootClass: root.pitchClass === null ? null : spellPitchClass(root.pitchClass, keyEstimate),
      bassRootClass: bassRoot.pitchClass === null
        ? null
        : spellPitchClass(bassRoot.pitchClass, keyEstimate),
      bassConfidence: usedBassSupport ? bassRoot.confidence : root.confidence,
      bassAttackRootClass: bassAttackRoot.pitchClass === null
        ? null
        : spellPitchClass(bassAttackRoot.pitchClass, keyEstimate),
      bassAttackConfidence: bassAttackRoot.confidence,
      bassSustainRootClass: bassSustainRoot.pitchClass === null
        ? null
        : spellPitchClass(bassSustainRoot.pitchClass, keyEstimate),
      bassSustainConfidence: bassSustainRoot.confidence,
      bassIsPassingTone,
      keyEstimate,
      usedBassSupport,
      boundaryStrength: strength,
    });
  }
  return { observations, beatAlignedBoundaries, keyEstimate: globalKey };
}

function scoreFor(observation: ChordObservation, stateName: string): number {
  return stateName === "N"
    ? observation.noChordScore
    : (observation.candidateScores[stateName] ?? -1);
}

function transitionPlausibility(
  previousName: string,
  nextName: string,
  key: KeyEstimate | null | undefined,
): number {
  if (previousName === "N" || nextName === "N") return 0;
  const previous = TEMPLATES.find((template) => template.name === previousName);
  const next = TEMPLATES.find((template) => template.name === nextName);
  if (!previous || !next) return 0;
  const rootMotion = (next.root - previous.root + 12) % 12;
  let plausibility = rootMotion === 5 || rootMotion === 7 ? 0.7 : 0;
  if (!key) return plausibility;
  const previousDegree = (previous.root - key.tonic + 12) % 12;
  const nextDegree = (next.root - key.tonic + 12) % 12;
  const commonPairs = key.mode === "major"
    ? [[0, 7], [7, 0], [0, 5], [9, 5], [2, 7], [7, 9], [5, 7]]
    : [[0, 7], [7, 0], [0, 5], [10, 0], [5, 7], [7, 8], [0, 10]];
  if (commonPairs.some(([from, to]) => from === previousDegree && to === nextDegree)) {
    plausibility = 1;
  }
  return plausibility * (0.6 + key.confidence * 0.4);
}

function transitionPenalty(
  previousName: string,
  nextName: string,
  observation: ChordObservation,
  settings: ChordSmoothingSettings,
): number {
  if (previousName === nextName) return 0;
  let penalty = settings.chordChangePenalty
    + settings.offBeatChangePenalty * (1 - observation.boundaryStrength);
  const improvement = scoreFor(observation, nextName) - scoreFor(observation, previousName);
  if (improvement < settings.changeMargin) {
    const scale = settings.changeMargin > 0
      ? 1 + (settings.changeMargin - improvement) / settings.changeMargin
      : 1;
    penalty += settings.weakTransitionPenalty * Math.min(3, scale);
  }
  if (observation.uncertain && nextName !== "N") {
    penalty += settings.weakTransitionPenalty;
  }
  if (nextName === "N"
    && ((observation.bassConfidence ?? 0) >= 0.18
      || strongestPitchClass(observation.rootChroma ?? []).confidence >= 0.2)) {
    penalty += settings.unnecessaryNoChordPenalty;
  }
  penalty -= settings.commonTransitionBonus * transitionPlausibility(
    previousName,
    nextName,
    observation.keyEstimate,
  );
  return Math.max(0, penalty);
}

/**
 * Duration-aware Viterbi decoder. Each state tracks how many consecutive
 * observations the chord has occupied (capped at the minimum-duration bucket).
 * Leaving early pays shortRegionPenalty; changing labels also pays beat-aware
 * and weak-transition costs.
 */
export function decodeChordSequence(
  observations: ChordObservation[],
  partialSettings: Partial<ChordSmoothingSettings> = {},
): string[] {
  if (!observations.length) return [];
  const settings = resolveSettings(partialSettings);
  const names = stateNames(settings);
  const durations = observations.map((observation) => observation.end - observation.start);
  const typicalDuration = median(durations.filter((duration) => duration > 0))
    || settings.fallbackWindowSeconds;
  const durationBuckets = Math.max(
    settings.requiredConsecutiveWindows,
    Math.ceil(settings.minimumChordDurationSeconds / typicalDuration),
  );
  const stateCount = names.length * durationBuckets;
  let previousScores = new Float64Array(stateCount);
  previousScores.fill(-Infinity);
  const backPointers = new Int32Array(observations.length * stateCount);
  backPointers.fill(-1);

  for (let label = 0; label < names.length; label += 1) {
    previousScores[label * durationBuckets] = scoreFor(observations[0], names[label]);
  }

  for (let time = 1; time < observations.length; time += 1) {
    const nextScores = new Float64Array(stateCount);
    nextScores.fill(-Infinity);
    for (let previousState = 0; previousState < stateCount; previousState += 1) {
      const previousScore = previousScores[previousState];
      if (!Number.isFinite(previousScore)) continue;
      const previousLabel = Math.floor(previousState / durationBuckets);
      const previousRun = previousState % durationBuckets;
      const previousName = names[previousLabel];

      const stayedRun = Math.min(durationBuckets - 1, previousRun + 1);
      const stayedState = previousLabel * durationBuckets + stayedRun;
      const stayedScore = previousScore + scoreFor(observations[time], previousName);
      if (stayedScore > nextScores[stayedState]) {
        nextScores[stayedState] = stayedScore;
        backPointers[time * stateCount + stayedState] = previousState;
      }

      for (let nextLabel = 0; nextLabel < names.length; nextLabel += 1) {
        if (nextLabel === previousLabel) continue;
        const nextName = names[nextLabel];
        const nextState = nextLabel * durationBuckets;
        const missingBuckets = Math.max(0, durationBuckets - (previousRun + 1));
        const earlyExitPenalty = missingBuckets
          ? settings.shortRegionPenalty * (missingBuckets / durationBuckets)
          : 0;
        const changedScore = previousScore
          + scoreFor(observations[time], nextName)
          - transitionPenalty(previousName, nextName, observations[time], settings)
          - earlyExitPenalty;
        if (changedScore > nextScores[nextState]) {
          nextScores[nextState] = changedScore;
          backPointers[time * stateCount + nextState] = previousState;
        }
      }
    }
    previousScores = nextScores;
  }

  let finalState = 0;
  for (let state = 1; state < stateCount; state += 1) {
    if (previousScores[state] > previousScores[finalState]) finalState = state;
  }
  const labels = Array<string>(observations.length);
  let state = finalState;
  for (let time = observations.length - 1; time >= 0; time -= 1) {
    labels[time] = names[Math.floor(state / durationBuckets)];
    state = backPointers[time * stateCount + state];
    if (time > 0 && state < 0) state = finalState;
  }
  return labels;
}

function applyHysteresis(
  observations: ChordObservation[],
  decoded: string[],
  settings: ChordSmoothingSettings,
): string[] {
  if (!decoded.length) return [];
  const stable = [...decoded];
  let current = decoded[0];
  let pending = "";
  let pendingStart = -1;
  let pendingCount = 0;

  for (let index = 1; index < decoded.length; index += 1) {
    const target = decoded[index];
    if (target === current) {
      pending = "";
      pendingStart = -1;
      pendingCount = 0;
      stable[index] = current;
      continue;
    }
    const improvement = scoreFor(observations[index], target)
      - scoreFor(observations[index], current);
    const supported = improvement >= settings.changeMargin
      && (target === "N"
        || observations[index].confidence >= settings.minimumChordConfidence)
      && (target === "N"
        || observations[index].scoreMargin >= settings.minimumScoreMargin);
    if (!supported) {
      stable[index] = current;
      pending = "";
      pendingStart = -1;
      pendingCount = 0;
      continue;
    }
    if (pending !== target) {
      pending = target;
      pendingStart = index;
      pendingCount = 1;
    } else {
      pendingCount += 1;
    }
    stable[index] = current;
    if (pendingCount >= settings.requiredConsecutiveWindows) {
      current = target;
      for (let pendingIndex = pendingStart; pendingIndex <= index; pendingIndex += 1) {
        stable[pendingIndex] = current;
      }
      pending = "";
      pendingStart = -1;
      pendingCount = 0;
    }
  }
  return stable;
}

type PendingChangeEvidence = {
  name: string;
  evidence: number;
  firstSupportIndex: number;
  supportingWindows: number;
  lastSupportIndex: number;
  topWinnerWindows: number;
  consecutiveTopWindows: number;
};

function chordRoot(name: string): string | null {
  return chordParts(name)?.root ?? null;
}

function bassAttackSupports(
  observation: ChordObservation,
  chordName: string,
): boolean {
  const root = chordRoot(chordName);
  return Boolean(
    root
      && observation.bassAttackRootClass
      && chordParts(observation.bassAttackRootClass)?.root === root
      && (observation.bassAttackConfidence ?? 0) >= 0.22
      && !observation.bassIsPassingTone,
  );
}

function boundaryFit(
  observations: ChordObservation[],
  boundaryIndex: number,
  fromChord: string,
  toChord: string,
  evidenceIndex: number,
): number {
  const before = observations.slice(Math.max(0, boundaryIndex - 2), boundaryIndex);
  const after = observations.slice(boundaryIndex, Math.min(observations.length, boundaryIndex + 2));
  const average = (values: number[]) =>
    values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
  const beforeSeparation = average(before.map(
    (observation) => scoreFor(observation, fromChord) - scoreFor(observation, toChord),
  ));
  const afterSeparation = average(after.map(
    (observation) => scoreFor(observation, toChord) - scoreFor(observation, fromChord),
  ));
  const boundaryObservation = observations[boundaryIndex];
  const bassAttackBonus = boundaryObservation && bassAttackSupports(boundaryObservation, toChord)
    ? 0.025 * (boundaryObservation.bassAttackConfidence ?? 0)
    : 0;
  const beatBonus = (boundaryObservation?.boundaryStrength ?? 0) * 0.012;
  // Nearest and preceding beats are preferred when their evidence is
  // comparable. This prevents confirmation from automatically snapping ahead.
  const distancePenalty = Math.abs(boundaryIndex - evidenceIndex) * 0.006
    + (boundaryIndex > evidenceIndex ? 0.004 : 0);
  return beforeSeparation + afterSeparation + bassAttackBonus + beatBonus - distancePenalty;
}

function selectBackdatedBoundary(
  observations: ChordObservation[],
  evidenceIndex: number,
  confirmationIndex: number,
  lastBoundaryIndex: number,
  fromChord: string,
  toChord: string,
): number {
  const candidates = [
    evidenceIndex - 1,
    evidenceIndex,
    evidenceIndex + 1,
  ].filter((index) =>
    index > lastBoundaryIndex
      && index >= 1
      && index <= confirmationIndex);
  if (!candidates.length) return Math.max(lastBoundaryIndex + 1, evidenceIndex);
  return candidates.reduce((best, candidate) =>
    boundaryFit(observations, candidate, fromChord, toChord, evidenceIndex)
      > boundaryFit(observations, best, fromChord, toChord, evidenceIndex)
      ? candidate
      : best);
}

function futureConfirmedQuality(
  decoded: string[],
  index: number,
  candidateName: string,
): string {
  const root = chordRoot(candidateName);
  if (!root) return candidateName;
  for (
    let lookahead = index;
    lookahead <= Math.min(decoded.length - 1, index + 4);
    lookahead += 1
  ) {
    const decodedName = decoded[lookahead];
    if (chordRoot(decodedName) === root) return decodedName;
  }
  return candidateName;
}

/**
 * Conservative online-style confirmation with retrospective boundary
 * placement. Candidate evidence accumulates across noisy beat windows; once a
 * change is accepted, its label begins at the best nearby beat rather than at
 * the later confirmation window.
 */
export function decodeReducedLatencySequence(
  observations: ChordObservation[],
  partialSettings: Partial<ChordSmoothingSettings> = {},
  viterbiLabels?: string[],
): { labels: string[]; decisions: ChordBoundaryDecision[] } {
  if (!observations.length) return { labels: [], decisions: [] };
  const settings = resolveSettings(partialSettings);
  const decoded = viterbiLabels ?? decodeChordSequence(observations, settings);
  const names = stateNames(settings);
  const labels = Array<string>(observations.length).fill(decoded[0]);
  const decisions: ChordBoundaryDecision[] = [];
  let current = decoded[0];
  let departureIndex = -1;
  let lastBoundaryIndex = -1;
  const pending = new Map<string, PendingChangeEvidence>();

  for (let index = 1; index < observations.length; index += 1) {
    const observation = observations[index];
    const currentScore = scoreFor(observation, current);
    const currentStillWins = observation.bestChord === current
      || currentScore >= observation.bestScore - settings.minimumScoreMargin * 0.5;
    if (currentStillWins) {
      departureIndex = -1;
    } else if (departureIndex < 0) {
      departureIndex = index;
    }

    for (const [name, entry] of pending) {
      entry.evidence *= currentStillWins ? 0.32 : 0.68;
      if (name !== observation.bestChord) entry.consecutiveTopWindows = 0;
      if (index - entry.lastSupportIndex > 1) {
        entry.supportingWindows = Math.max(0, entry.supportingWindows - 1);
      }
      if (entry.evidence < 0.002) pending.delete(name);
    }

    for (const name of names) {
      if (name === current) continue;
      const sameRootAsCurrent = chordRoot(name)
        && chordRoot(name) === chordRoot(current);
      const template = TEMPLATES.find((candidate) => candidate.name === name);
      const strongSameRootExtension = Boolean(
        sameRootAsCurrent
          && template
          && isExtended(template)
          && (observation.seventhEvidenceByChord?.[name]
            ?? (observation.bestChord === name ? observation.seventhEvidence : 0))
            >= settings.strongSeventhThreshold,
      );
      if (sameRootAsCurrent && !strongSameRootExtension) continue;
      const score = scoreFor(observation, name);
      const advantage = score - currentScore;
      const root = chordRoot(name);
      const rootVisible = Boolean(
        root
          && [observation.bestChord, observation.secondBestChord]
            .some((candidate) => chordRoot(candidate) === root),
      );
      const locallyVisible = name === observation.bestChord
        || name === observation.secondBestChord
        || rootVisible
        || name === decoded[index]
        || bassAttackSupports(observation, name);
      if (!locallyVisible || advantage < -settings.minimumScoreMargin) continue;
      const beatWeight = 0.85 + observation.boundaryStrength * 0.3;
      const confidenceWeight = 0.65 + observation.confidence;
      const fitEvidence = Math.max(0, advantage - settings.minimumScoreMargin * 0.25)
        * beatWeight
        * confidenceWeight;
      const separationEvidence = name === observation.bestChord
        ? Math.max(0, observation.scoreMargin) * 0.12
        : 0;
      const bassEvidence = bassAttackSupports(observation, name)
        ? 0.018 * (observation.bassAttackConfidence ?? 0)
        : 0;
      const decodedEvidence = decoded[index] === name ? 0.003 : 0;
      const contribution = fitEvidence + separationEvidence + bassEvidence + decodedEvidence;
      if (contribution <= 0) continue;
      const entry = pending.get(name) ?? {
        name,
        evidence: 0,
        firstSupportIndex: index,
        supportingWindows: 0,
        lastSupportIndex: index,
        topWinnerWindows: 0,
        consecutiveTopWindows: 0,
      };
      entry.evidence += contribution;
      entry.supportingWindows += 1;
      entry.lastSupportIndex = index;
      if (name === observation.bestChord) {
        entry.topWinnerWindows += 1;
        entry.consecutiveTopWindows += 1;
      }
      pending.set(name, entry);
    }

    const accepted = [...pending.values()]
      .filter((entry) => {
        const singleBeatException = entry.supportingWindows >= 1
          && entry.topWinnerWindows >= 1
          && entry.name === observations[index].bestChord
          && observations[index].confidence >= settings.minimumChordConfidence
          && observations[index].boundaryStrength >= 0.86
          && observations[index].scoreMargin >= settings.minimumScoreMargin * 0.75
          && bassAttackSupports(observations[index], entry.name);
        const sustainedTopWinner = entry.consecutiveTopWindows
          >= settings.requiredConsecutiveWindows;
        return entry.evidence >= settings.changeEvidenceThreshold
          && entry.topWinnerWindows >= 1
          && (sustainedTopWinner || singleBeatException);
      })
      .sort((left, right) =>
        right.consecutiveTopWindows - left.consecutiveTopWindows
          || right.topWinnerWindows - left.topWinnerWindows
          || right.evidence - left.evidence)[0];

    if (!accepted) {
      labels[index] = current;
      continue;
    }

    const evidenceIndex = departureIndex >= 0
      ? Math.min(departureIndex, accepted.firstSupportIndex)
      : accepted.firstSupportIndex;
    const acceptedName = futureConfirmedQuality(decoded, index, accepted.name);
    const boundaryIndex = selectBackdatedBoundary(
      observations,
      evidenceIndex,
      index,
      lastBoundaryIndex,
      current,
      acceptedName,
    );
    for (let labelIndex = boundaryIndex; labelIndex <= index; labelIndex += 1) {
      labels[labelIndex] = acceptedName;
    }
    decisions.push({
      fromChord: current,
      toChord: acceptedName,
      candidateEvidenceStart: observations[evidenceIndex].start,
      confirmationTime: observations[index].end,
      finalBoundaryTime: observations[boundaryIndex].start,
      accumulatedEvidence: accepted.evidence,
      supportingWindows: accepted.supportingWindows,
    });
    current = acceptedName;
    lastBoundaryIndex = boundaryIndex;
    departureIndex = -1;
    pending.clear();
    labels[index] = current;
  }
  return { labels, decisions };
}

function labelsToRegions(
  labels: string[],
  observations: ChordObservation[],
  decisions: ChordBoundaryDecision[] = [],
): InternalRegion[] {
  const regions: InternalRegion[] = [];
  for (let index = 0; index < labels.length; index += 1) {
    const label = labels[index];
    const observation = observations[index];
    const seventhEvidence = observation.seventhEvidenceByChord?.[label]
      ?? observation.seventhEvidence;
    const confidence = label === "N"
      ? clamp(observation.noChordScore)
      : confidenceFor(scoreFor(observation, label), observation.scoreMargin);
    const previous = regions.at(-1);
    if (previous?.name === label) {
      const totalWindows = previous.windowCount + 1;
      previous.end = observation.end;
      previous.lastWindow = index;
      previous.confidence = (
        previous.confidence * previous.windowCount + confidence
      ) / totalWindows;
      previous.scoreMargin = (
        previous.scoreMargin * previous.windowCount + observation.scoreMargin
      ) / totalWindows;
      previous.seventhEvidence = (
        previous.seventhEvidence * previous.windowCount + seventhEvidence
      ) / totalWindows;
      previous.bestScore = (
        previous.bestScore * previous.windowCount + observation.bestScore
      ) / totalWindows;
      previous.secondBestScore = (
        previous.secondBestScore * previous.windowCount + observation.secondBestScore
      ) / totalWindows;
      previous.usedBassSupport ||= Boolean(observation.usedBassSupport);
      previous.usedSmoothingOverride ||= label !== observation.bestChord;
      previous.windowCount = totalWindows;
    } else {
      const decision = decisions.find(
        (candidate) => candidate.toChord === label
          && Math.abs(candidate.finalBoundaryTime - observation.start) < 1e-6,
      );
      regions.push({
        start: observation.start,
        end: observation.end,
        name: label,
        confidence,
        firstWindow: index,
        lastWindow: index,
        windowCount: 1,
        scoreMargin: observation.scoreMargin,
        seventhEvidence,
        boundaryStrength: observation.boundaryStrength,
        bestScore: observation.bestScore,
        secondBestScore: observation.secondBestScore,
        rootClass: observation.bassRootClass ?? observation.rootClass ?? null,
        keyEstimate: observation.keyEstimate?.name ?? null,
        usedBassSupport: Boolean(observation.usedBassSupport),
        usedSmoothingOverride: label !== observation.bestChord,
        simplifiedFromExtendedChord: false,
        mergedFromShortRegions: false,
        candidateEvidenceStart: decision?.candidateEvidenceStart ?? null,
        confirmationTime: decision?.confirmationTime ?? null,
        finalBoundaryTime: decision?.finalBoundaryTime ?? observation.start,
      });
    }
  }
  return regions;
}

const ENHARMONIC_ROOTS: Record<string, string> = {
  "B#": "C",
  "Db": "C#",
  "D♭": "C#",
  "D#": "Eb",
  "D♯": "Eb",
  "Fb": "E",
  "E#": "F",
  "Gb": "F#",
  "G♭": "F#",
  "G#": "Ab",
  "G♯": "Ab",
  "A#": "Bb",
  "A♯": "Bb",
  "Cb": "B",
};

function chordParts(name: string): { root: string; quality: string } | null {
  if (name === "N") return null;
  const match = /^([A-G](?:[#♯b♭])?)(maj7|m7|sus2|sus4|add9|dim|aug|m|7)?$/.exec(name);
  if (!match) return { root: name, quality: "" };
  const asciiRoot = match[1].replace("♯", "#").replace("♭", "b");
  return {
    root: ENHARMONIC_ROOTS[match[1]]
      ?? ENHARMONIC_ROOTS[asciiRoot]
      ?? asciiRoot,
    quality: match[2] ?? "",
  };
}

function equivalentChord(first: string, second: string): boolean {
  const a = chordParts(first);
  const b = chordParts(second);
  return Boolean(a && b && a.root === b.root && a.quality === b.quality)
    || first === second;
}

function sameRootFamily(first: string, second: string): boolean {
  const a = chordParts(first);
  const b = chordParts(second);
  return Boolean(a && b && a.root === b.root);
}

function mergeAt(
  regions: InternalRegion[],
  leftIndex: number,
  counts: CleanupCounts,
): void {
  const left = regions[leftIndex];
  const right = regions[leftIndex + 1];
  const totalWindows = left.windowCount + right.windowCount;
  left.end = right.end;
  left.lastWindow = right.lastWindow;
  left.confidence = (
    left.confidence * left.windowCount + right.confidence * right.windowCount
  ) / totalWindows;
  left.scoreMargin = (
    left.scoreMargin * left.windowCount + right.scoreMargin * right.windowCount
  ) / totalWindows;
  left.seventhEvidence = (
    left.seventhEvidence * left.windowCount + right.seventhEvidence * right.windowCount
  ) / totalWindows;
  left.bestScore = (
    left.bestScore * left.windowCount + right.bestScore * right.windowCount
  ) / totalWindows;
  left.secondBestScore = (
    left.secondBestScore * left.windowCount + right.secondBestScore * right.windowCount
  ) / totalWindows;
  left.usedBassSupport ||= right.usedBassSupport;
  left.usedSmoothingOverride ||= right.usedSmoothingOverride;
  left.simplifiedFromExtendedChord ||= right.simplifiedFromExtendedChord;
  left.mergedFromShortRegions = true;
  left.windowCount = totalWindows;
  regions.splice(leftIndex + 1, 1);
  counts.regionsMerged += 1;
}

function mergeAdjacent(
  regions: InternalRegion[],
  counts: CleanupCounts,
): void {
  for (let index = 0; index + 1 < regions.length;) {
    if (equivalentChord(regions[index].name, regions[index + 1].name)) {
      mergeAt(regions, index, counts);
    } else {
      index += 1;
    }
  }
}

function relabelRegion(
  region: InternalRegion,
  name: string,
  observations: ChordObservation[],
): void {
  if (region.name !== name) region.usedSmoothingOverride = true;
  region.name = name;
  let confidence = 0;
  for (let index = region.firstWindow; index <= region.lastWindow; index += 1) {
    const observation = observations[index];
    confidence += name === "N"
      ? clamp(observation.noChordScore)
      : confidenceFor(scoreFor(observation, name), observation.scoreMargin);
  }
  region.confidence = confidence / region.windowCount;
}

function simplifiedTriadName(name: string): string {
  if (name.endsWith("m7")) return name.slice(0, -1);
  if (name.endsWith("maj7")) return name.slice(0, -4);
  if (name.endsWith("add9")) return name.slice(0, -4);
  if (name.endsWith("7")) return name.slice(0, -1);
  return name;
}

function neighbouringScore(
  region: InternalRegion,
  candidate: InternalRegion | undefined,
  observations: ChordObservation[],
): number {
  if (!candidate) return -Infinity;
  let score = 0;
  for (let index = region.firstWindow; index <= region.lastWindow; index += 1) {
    score += scoreFor(observations[index], candidate.name);
  }
  return score / region.windowCount + candidate.confidence * 0.08;
}

function meaningfulShortRegion(
  region: InternalRegion,
  before: InternalRegion | undefined,
  after: InternalRegion | undefined,
  observations: ChordObservation[],
  settings: ChordSmoothingSettings,
  localHarmonicRhythmWindows: 1 | 2 | 4,
): boolean {
  if (region.name === "N") return false;
  const durations = observations
    .slice(region.firstWindow, region.lastWindow + 1)
    .map((observation) => observation.end - observation.start)
    .filter((duration) => duration > 0);
  const oneBeatDuration = median(durations) || settings.fallbackWindowSeconds;
  if (region.end - region.start < oneBeatDuration * 0.8) return false;
  const regionObservations = observations.slice(region.firstWindow, region.lastWindow + 1);
  const bassSupportedWindows = regionObservations.filter(
    (observation) => bassAttackSupports(observation, region.name),
  ).length;
  const localAdvantage = regionObservations.reduce((sum, observation) => {
    const neighbourScore = Math.max(
      before ? scoreFor(observation, before.name) : -1,
      after ? scoreFor(observation, after.name) : -1,
    );
    return sum + scoreFor(observation, region.name) - neighbourScore;
  }, 0) / Math.max(1, regionObservations.length);
  const surroundedBySameChord = Boolean(
    before
      && after
      && equivalentChord(before.name, after.name),
  );
  if (surroundedBySameChord) {
    return bassSupportedWindows === region.windowCount
      && region.scoreMargin >= settings.validShortChordScoreMargin * 2
      && localAdvantage >= settings.changeMargin
      && region.confidence >= settings.highConfidenceShortChord;
  }

  // A run of two beat windows is normal when local harmonic rhythm reaches
  // two chords per bar. One-beat chords require dedicated attack-root support
  // or unusually clear local evidence.
  if (region.windowCount >= localHarmonicRhythmWindows) {
    return region.scoreMargin >= settings.validShortChordScoreMargin * 0.4
      || bassSupportedWindows > 0
      || localAdvantage >= settings.changeMargin * 0.5;
  }
  if (region.windowCount >= 2) {
    return bassSupportedWindows > 0
      || localAdvantage >= settings.changeMargin * 0.5;
  }
  return bassSupportedWindows > 0
    && region.confidence >= settings.minimumChordConfidence
    && region.scoreMargin >= settings.minimumScoreMargin;
}

function estimateLocalHarmonicRhythmWindows(
  regions: InternalRegion[],
  regionIndex: number,
): 1 | 2 | 4 {
  const nearbyRunLengths = regions
    .slice(Math.max(0, regionIndex - 2), regionIndex + 3)
    .filter((region) => region.name !== "N")
    .map((region) => region.windowCount);
  const localMedian = median(nearbyRunLengths);
  if (localMedian <= 1.5) return 1;
  if (localMedian <= 3) return 2;
  return 4;
}

function cleanupRegions(
  initial: InternalRegion[],
  observations: ChordObservation[],
  settings: ChordSmoothingSettings,
  mode: "production" | "reduced-latency" = "production",
): { regions: InternalRegion[]; counts: CleanupCounts } {
  const regions = initial.map((region) => ({ ...region }));
  const counts: CleanupCounts = { shortRegionsRemoved: 0, regionsMerged: 0 };

  // Complex labels survive only with consistent independent color-tone
  // evidence. Otherwise the harmonic guide uses the compatible triad.
  for (const region of regions) {
    const simplified = simplifiedTriadName(region.name);
    if (simplified !== region.name
      && region.seventhEvidence < settings.strongSeventhThreshold) {
      relabelRegion(region, simplified, observations);
      region.simplifiedFromExtendedChord = true;
    }
  }
  mergeAdjacent(regions, counts);

  let changed = true;
  while (changed && regions.length > 1) {
    changed = false;
    for (let index = 0; index < regions.length; index += 1) {
      const region = regions[index];
      const duration = region.end - region.start;
      const before = regions[index - 1];
      const after = regions[index + 1];

      if (region.name === "N") {
        if (duration >= settings.shortNoChordMergeSeconds) continue;
      } else if (duration >= settings.minimumChordDurationSeconds) {
        continue;
      } else {
        const highConfidenceBeatException = region.windowCount
          >= settings.requiredConsecutiveWindows
          && region.confidence >= settings.highConfidenceShortChord
          && region.boundaryStrength >= 0.86;
        if (highConfidenceBeatException) continue;
        if (mode === "reduced-latency"
          && meaningfulShortRegion(
            region,
            before,
            after,
            observations,
            settings,
            estimateLocalHarmonicRhythmWindows(regions, index),
          )) {
          continue;
        }
      }

      let replacement: InternalRegion | undefined;
      if (before && after && (
        equivalentChord(before.name, after.name)
        || sameRootFamily(before.name, after.name)
      )) {
        replacement = before.confidence >= after.confidence ? before : after;
      } else if (before?.name === "N" && after?.name !== "N") {
        replacement = after;
      } else if (after?.name === "N" && before?.name !== "N") {
        replacement = before;
      } else {
        replacement = neighbouringScore(region, before, observations)
          >= neighbouringScore(region, after, observations)
          ? before
          : after;
      }
      if (!replacement) continue;
      relabelRegion(region, replacement.name, observations);
      region.mergedFromShortRegions = true;
      counts.shortRegionsRemoved += 1;
      mergeAdjacent(regions, counts);
      changed = true;
      break;
    }
  }

  // A remaining weak same-root major/minor or triad/seventh boundary is often
  // a passing third/seventh. Do not collapse two sustained, confident regions.
  for (let index = 0; index + 1 < regions.length;) {
    const left = regions[index];
    const right = regions[index + 1];
    const weakFamilyChange = sameRootFamily(left.name, right.name)
      && (
        left.end - left.start < settings.minimumChordDurationSeconds
        || right.end - right.start < settings.minimumChordDurationSeconds
        || left.scoreMargin < settings.minimumScoreMargin
        || right.scoreMargin < settings.minimumScoreMargin
      );
    if (!weakFamilyChange) {
      index += 1;
      continue;
    }
    const preferred = left.confidence * left.windowCount
      >= right.confidence * right.windowCount
      ? left.name
      : right.name;
    relabelRegion(left, preferred, observations);
    relabelRegion(right, preferred, observations);
    mergeAt(regions, index, counts);
  }
  mergeAdjacent(regions, counts);
  return { regions, counts };
}

function rawChangeCount(frames: InternalRawFrame[]): number {
  let changes = 0;
  for (let index = 1; index < frames.length; index += 1) {
    if (frames[index].bestChord !== frames[index - 1].bestChord) changes += 1;
  }
  return changes;
}

function chromaSimilarity(first: number[] | undefined, second: number[] | undefined): number {
  if (!first?.length || !second?.length) return 0;
  let dot = 0;
  let firstNorm = 0;
  let secondNorm = 0;
  for (let index = 0; index < Math.min(first.length, second.length); index += 1) {
    dot += first[index] * second[index];
    firstNorm += first[index] ** 2;
    secondNorm += second[index] ** 2;
  }
  return firstNorm && secondNorm ? dot / Math.sqrt(firstNorm * secondNorm) : 0;
}

function reinforceRepeatedWindows(
  observations: ChordObservation[],
  firstPassLabels: string[],
  settings: ChordSmoothingSettings,
  maximumBonus = Number.POSITIVE_INFINITY,
): ChordObservation[] {
  if (settings.repeatedSectionConsistencyWeight <= 0) return observations;
  return observations.map((observation, index) => {
    let bestPrior = -1;
    let bestSimilarity = 0;
    // Repetition candidates span one to sixteen four-beat bars. Requiring high
    // chroma similarity keeps the prior from flattening unrelated sections.
    for (let lag = 4; lag <= Math.min(64, index); lag += 4) {
      const prior = index - lag;
      const similarity = chromaSimilarity(observation.chroma, observations[prior].chroma);
      if (similarity > bestSimilarity) [bestPrior, bestSimilarity] = [prior, similarity];
    }
    if (bestPrior < 0 || bestSimilarity < 0.9) return observation;
    const priorLabel = firstPassLabels[bestPrior];
    if (!priorLabel || priorLabel === "N") return observation;
    const localCap = Number.isFinite(maximumBonus)
      ? Math.max(
        0,
        Math.min(
          maximumBonus,
          settings.maximumRepeatedSectionBonus,
          observation.scoreMargin * 0.5,
        ),
      )
      : Number.POSITIVE_INFINITY;
    const bonus = Math.min(
      settings.repeatedSectionConsistencyWeight * bestSimilarity,
      localCap,
    );
    if (bonus <= 0) return observation;
    return {
      ...observation,
      candidateScores: {
        ...observation.candidateScores,
        [priorLabel]: (observation.candidateScores[priorLabel] ?? -1)
          + bonus,
      },
    };
  });
}

function presentRegion(region: InternalRegion): ChordEvent {
  const name = spellChordName(region.name, region.keyEstimate);
  const diagnostics: ChordRegionDiagnostics = {
    finalLabel: name,
    confidence: region.confidence,
    confidenceLevel: confidenceLevel(region.confidence),
    bestScore: region.bestScore,
    secondBestScore: region.secondBestScore,
    scoreMargin: region.scoreMargin,
    rootClass: region.rootClass,
    keyEstimate: region.keyEstimate,
    usedBassSupport: region.usedBassSupport,
    usedSmoothingOverride: region.usedSmoothingOverride,
    simplifiedFromExtendedChord: region.simplifiedFromExtendedChord,
    mergedFromShortRegions: region.mergedFromShortRegions,
    candidateEvidenceStart: region.candidateEvidenceStart,
    confirmationTime: region.confirmationTime,
    finalBoundaryTime: region.finalBoundaryTime,
  };
  return {
    start: region.start,
    end: region.end,
    name,
    confidence: region.confidence,
    diagnostics,
  };
}

function runProductionSmoothing(
  observations: ChordObservation[],
  settings: ChordSmoothingSettings,
) {
  const firstPass = decodeChordSequence(observations, settings);
  const reinforced = reinforceRepeatedWindows(observations, firstPass, settings);
  const decoded = decodeChordSequence(reinforced, settings);
  const hysteretic = applyHysteresis(reinforced, decoded, settings);
  const initial = labelsToRegions(hysteretic, reinforced);
  const cleaned = cleanupRegions(initial, reinforced, settings);
  return { firstPass, reinforced, decoded, hysteretic, initial, cleaned };
}

function runReducedLatencySmoothing(
  observations: ChordObservation[],
  settings: ChordSmoothingSettings,
) {
  const firstPass = decodeChordSequence(observations, settings);
  const reinforced = reinforceRepeatedWindows(
    observations,
    firstPass,
    settings,
    settings.maximumRepeatedSectionBonus,
  );
  const decoded = decodeChordSequence(reinforced, settings);
  const reduced = decodeReducedLatencySequence(reinforced, settings, decoded);
  const initial = labelsToRegions(reduced.labels, reinforced, reduced.decisions);
  const cleaned = cleanupRegions(initial, reinforced, settings, "reduced-latency");
  return { reinforced, decoded, reduced, initial, cleaned };
}

export function smoothChordObservations(
  observations: ChordObservation[],
  partialSettings: Partial<ChordSmoothingSettings> = {},
): { regions: ChordEvent[]; shortRegionsRemoved: number; regionsMerged: number } {
  const settings = resolveSettings(partialSettings);
  const result = runReducedLatencySmoothing(observations, settings);
  return {
    regions: result.cleaned.regions.map(presentRegion),
    ...result.cleaned.counts,
  };
}

export function smoothChordObservationsReducedLatency(
  observations: ChordObservation[],
  partialSettings: Partial<ChordSmoothingSettings> = {},
): {
  regions: ChordEvent[];
  decisions: ChordBoundaryDecision[];
  shortRegionsRemoved: number;
  regionsMerged: number;
} {
  const settings = resolveSettings(partialSettings);
  const result = runReducedLatencySmoothing(observations, settings);
  return {
    regions: result.cleaned.regions.map(presentRegion),
    decisions: result.reduced.decisions,
    ...result.cleaned.counts,
  };
}

export function analyzeChordProgression(
  samples: Float32Array,
  sampleRate: number,
  beatGrid: BeatGrid,
  partialSettings: Partial<ChordSmoothingSettings> = {},
  evidence: HarmonyEvidenceOptions = {},
): ChordAnalysisResult {
  const configuredSettings = resolveSettings(partialSettings);
  // With a reliable pulse, musical duration is a better stability unit than
  // wall-clock seconds. Fall back to the configured seconds for weak tempo.
  const settings = beatGrid.beatDuration
    ? {
      ...configuredSettings,
      minimumChordDurationSeconds: Math.max(
        0.15,
        configuredSettings.minimumChordDurationBeats * beatGrid.beatDuration,
      ),
    }
    : configuredSettings;
  const duration = samples.length / sampleRate;
  if (!samples.length || sampleRate <= 0) {
    const diagnostics: ChordDiagnostics = {
      rawChordChanges: 0,
      finalChordRegions: 0,
      averageRegionDuration: 0,
      shortRegionsRemoved: 0,
      regionsMerged: 0,
      lowConfidenceRegions: 0,
      analysisWindows: 0,
      beatAlignedBoundaries: 0,
      noChordRegions: 0,
      keyEstimate: null,
      harmonyEvidenceSource: evidence.source ?? "full-mix",
      bassSupportedWindows: 0,
      smoothingOverrides: 0,
    };
    return { rawFrames: [], windows: [], regions: [], settings, diagnostics };
  }

  const internalRawFrames = createRawFrames(
    samples,
    sampleRate,
    settings,
    evidence.bassSamples,
  );
  const aggregation = aggregateObservations(
    internalRawFrames,
    duration,
    beatGrid,
    settings,
  );
  for (const frame of internalRawFrames) {
    frame.keyEstimate = aggregation.keyEstimate?.name ?? null;
  }
  const productionStages = runProductionSmoothing(aggregation.observations, settings);
  const reducedStages = runReducedLatencySmoothing(aggregation.observations, settings);
  const reducedLatencyRegions = reducedStages.cleaned.regions.map(presentRegion);
  const cleaned = {
    regions: reducedLatencyRegions,
    ...reducedStages.cleaned.counts,
  };
  const beatLevelWinner = labelsToRegions(
    aggregation.observations.map((observation) => observation.bestChord),
    aggregation.observations,
  ).map(presentRegion);
  const viterbiRegions = labelsToRegions(
    productionStages.decoded,
    productionStages.reinforced,
  ).map(presentRegion);
  const postHysteresisRegions = productionStages.initial.map(presentRegion);
  const windows: ChordWindowDiagnostics[] = aggregation.observations.map((observation) => {
    const region = cleaned.regions.find(
      (candidate) => observation.start >= candidate.start
        && observation.start < candidate.end,
    );
    const reducedRegion = reducedLatencyRegions.find(
      (candidate) => observation.start >= candidate.start
        && observation.start < candidate.end,
    );
    const finalLabel = region?.name ?? "N";
    return {
      start: observation.start,
      end: observation.end,
      topCandidateChord: observation.bestChord,
      topCandidateScore: observation.bestScore,
      secondBestChord: observation.secondBestChord,
      secondBestScore: observation.secondBestScore,
      scoreMargin: observation.scoreMargin,
      bassRootEstimate: observation.bassRootClass ?? observation.rootClass ?? null,
      bassRootConfidence: observation.bassConfidence ?? 0,
      keyEstimate: observation.keyEstimate?.name ?? null,
      beatStrength: observation.boundaryStrength,
      finalLabel,
      usedBassSupport: Boolean(observation.usedBassSupport),
      usedSmoothingOverride: finalLabel !== observation.bestChord,
      mergedFromNearbyWindows: Boolean(region?.diagnostics?.mergedFromShortRegions),
      bassAttackRootEstimate: observation.bassAttackRootClass ?? null,
      bassAttackConfidence: observation.bassAttackConfidence ?? 0,
      bassSustainRootEstimate: observation.bassSustainRootClass ?? null,
      bassSustainConfidence: observation.bassSustainConfidence ?? 0,
      bassIsPassingTone: Boolean(observation.bassIsPassingTone),
      reducedLatencyLabel: reducedRegion?.name ?? "N",
      candidateScores: { ...observation.candidateScores },
      noChordScore: observation.noChordScore,
      chroma: observation.chroma ? [...observation.chroma] : undefined,
      observationConfidence: observation.confidence,
      uncertain: observation.uncertain,
      seventhEvidence: observation.seventhEvidence,
      seventhEvidenceByChord: observation.seventhEvidenceByChord
        ? { ...observation.seventhEvidenceByChord }
        : undefined,
      rootClass: observation.rootClass ?? null,
    };
  });
  const diagnostics: ChordDiagnostics = {
    rawChordChanges: rawChangeCount(internalRawFrames),
    finalChordRegions: cleaned.regions.length,
    averageRegionDuration: cleaned.regions.length
      ? cleaned.regions.reduce((sum, region) => sum + region.end - region.start, 0)
        / cleaned.regions.length
      : 0,
    shortRegionsRemoved: cleaned.shortRegionsRemoved,
    regionsMerged: cleaned.regionsMerged,
    lowConfidenceRegions: cleaned.regions.filter(
      (region) => region.name !== "N"
        && region.confidence < settings.minimumChordConfidence,
    ).length,
    analysisWindows: aggregation.observations.length,
    beatAlignedBoundaries: aggregation.beatAlignedBoundaries,
    noChordRegions: cleaned.regions.filter((region) => region.name === "N").length,
    keyEstimate: aggregation.keyEstimate,
    harmonyEvidenceSource: evidence.source ?? "full-mix",
    bassSupportedWindows: aggregation.observations.filter(
      (observation) => observation.usedBassSupport,
    ).length,
    smoothingOverrides: windows.filter((window) => window.usedSmoothingOverride).length,
  };
  const rawFrames: RawChordFrame[] = internalRawFrames.map(({
    start,
    end,
    bestChord,
    bestScore,
    secondBestChord,
    secondBestScore,
    confidence,
    scoreMargin,
    uncertain,
    rootClass,
    bassRootClass,
    bassConfidence,
    keyEstimate,
  }) => ({
    start,
    end,
    bestChord,
    bestScore,
    secondBestChord,
    secondBestScore,
    confidence,
    scoreMargin,
    uncertain,
    rootClass,
    bassRootClass,
    bassConfidence,
    keyEstimate,
  }));
  return {
    rawFrames,
    windows,
    regions: cleaned.regions,
    settings,
    diagnostics,
    decoderAlternatives: {
      beatLevelWinner,
      viterbi: viterbiRegions,
      postHysteresis: postHysteresisRegions,
      production: cleaned.regions,
      reducedLatency: reducedLatencyRegions,
      reducedLatencyDecisions: reducedStages.reduced.decisions,
    },
  };
}
