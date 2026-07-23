import { readFile, writeFile } from "node:fs/promises";
import { analyzeChordProgression, type BeatGrid } from "../src/chordAnalysis";
import type { ChordAnalysisResult, ChordSmoothingSettings } from "../src/types";
import {
  evaluateChordReference,
  normalizeChordSymbol,
} from "../server/evaluation.mjs";

type DecodedWav = {
  samples: Float32Array;
  sampleRate: number;
};

type TempoCandidate = {
  bpm: number;
  lag: number;
  rawScore: number;
  rawRelative: number;
  weightedRelative: number;
};

function decodePcmWav(input: Buffer): DecodedWav {
  if (input.toString("ascii", 0, 4) !== "RIFF"
    || input.toString("ascii", 8, 12) !== "WAVE") {
    throw new Error("Expected a RIFF/WAVE input.");
  }
  let offset = 12;
  let format = 0;
  let channels = 0;
  let sampleRate = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataLength = 0;
  while (offset + 8 <= input.length) {
    const chunk = input.toString("ascii", offset, offset + 4);
    const length = input.readUInt32LE(offset + 4);
    const body = offset + 8;
    if (chunk === "fmt ") {
      format = input.readUInt16LE(body);
      channels = input.readUInt16LE(body + 2);
      sampleRate = input.readUInt32LE(body + 4);
      bitsPerSample = input.readUInt16LE(body + 14);
    } else if (chunk === "data") {
      dataOffset = body;
      dataLength = Math.min(length, input.length - body);
      break;
    }
    offset = body + length + (length % 2);
  }
  if (format !== 1 || bitsPerSample !== 16 || !channels || dataOffset < 0) {
    throw new Error("The section ablation tool currently supports PCM16 WAV files.");
  }
  const frames = Math.floor(dataLength / (channels * 2));
  const samples = new Float32Array(frames);
  for (let frame = 0; frame < frames; frame += 1) {
    let sum = 0;
    for (let channel = 0; channel < channels; channel += 1) {
      sum += input.readInt16LE(dataOffset + (frame * channels + channel) * 2) / 32768;
    }
    samples[frame] = sum / channels;
  }
  return { samples, sampleRate };
}

function resample(
  samples: Float32Array,
  inputRate: number,
  outputRate = 22050,
): Float32Array {
  if (inputRate === outputRate) return samples;
  const length = Math.max(1, Math.floor(samples.length * outputRate / inputRate));
  const result = new Float32Array(length);
  for (let index = 0; index < length; index += 1) {
    const source = index * inputRate / outputRate;
    const low = Math.floor(source);
    const high = Math.min(samples.length - 1, low + 1);
    const fraction = source - low;
    result[index] = samples[low] * (1 - fraction) + samples[high] * fraction;
  }
  return result;
}

function tempoCandidates(samples: Float32Array, sampleRate: number): TempoCandidate[] {
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
  const candidates = [];
  for (let bpm = 60; bpm <= 190; bpm += 1) {
    const lag = Math.round((60 * sampleRate) / (bpm * hop));
    let rawScore = 0;
    for (let index = lag; index < envelope.length; index += 1) {
      rawScore += envelope[index] * envelope[index - lag];
    }
    const midTempoPreference = 1 + 0.08 * (1 - Math.abs(bpm - 120) / 130);
    candidates.push({
      bpm,
      lag,
      rawScore,
      weightedScore: rawScore * midTempoPreference,
    });
  }
  const maximumRaw = Math.max(...candidates.map((candidate) => candidate.rawScore), 1e-12);
  const maximumWeighted = Math.max(...candidates.map((candidate) => candidate.weightedScore), 1e-12);
  const ranked = candidates
    .map((candidate) => ({
      bpm: candidate.bpm,
      lag: candidate.lag,
      rawScore: candidate.rawScore,
      rawRelative: candidate.rawScore / maximumRaw,
      weightedRelative: candidate.weightedScore / maximumWeighted,
    }))
    .sort((left, right) => right.weightedRelative - left.weightedRelative);
  const uniqueLags: TempoCandidate[] = [];
  for (const candidate of ranked) {
    if (!uniqueLags.some((existing) => existing.lag === candidate.lag)) {
      uniqueLags.push(candidate);
    }
    if (uniqueLags.length >= 8) break;
  }
  for (const target of [73, 103]) {
    const candidate = ranked.find((entry) => entry.bpm === target);
    if (candidate && !uniqueLags.some((entry) => entry.bpm === target)) {
      uniqueLags.push(candidate);
    }
  }
  return uniqueLags.map((candidate) => ({
    ...candidate,
    rawScore: Number(candidate.rawScore.toFixed(8)),
    rawRelative: Number(candidate.rawRelative.toFixed(6)),
    weightedRelative: Number(candidate.weightedRelative.toFixed(6)),
  }));
}

function predictionFromAnalysis(
  duration: number,
  analysis: ChordAnalysisResult,
  bpm?: number,
  chords = analysis.regions,
): Record<string, unknown> {
  return {
    duration,
    bpm,
    chords,
    chordAnalysis: analysis,
  };
}

function rawWinnerRegions(analysis: ChordAnalysisResult) {
  const regions: Array<{
    start: number;
    end: number;
    name: string;
    confidence: number;
  }> = [];
  for (const frame of analysis.rawFrames) {
    const previous = regions.at(-1);
    if (previous?.name === frame.bestChord) {
      previous.end = frame.end;
      previous.confidence = (previous.confidence + frame.confidence) / 2;
    } else {
      regions.push({
        start: frame.start,
        end: frame.end,
        name: frame.bestChord,
        confidence: frame.confidence,
      });
    }
  }
  return regions;
}

function chordAt(
  regions: Array<{ start: number; end: number; chord: string }>,
  time: number,
): { start: number; end: number; chord: string } | null {
  return regions.find((region) => time >= region.start && time < region.end) ?? null;
}

function bassEvidenceSummary(reference: any, analysis: ChordAnalysisResult) {
  let duration = 0;
  let matchingDuration = 0;
  let confidentDuration = 0;
  let confidentMatchingDuration = 0;
  let conflictingDuration = 0;
  for (const window of analysis.windows) {
    const midpoint = (window.start + window.end) / 2;
    const expectedRegion = chordAt(reference.alignment.regions, midpoint);
    if (!expectedRegion) continue;
    const overlap = Math.max(
      0,
      Math.min(window.end, expectedRegion.end) - Math.max(window.start, expectedRegion.start),
    );
    if (!overlap) continue;
    const expectedRoot = normalizeChordSymbol(expectedRegion.chord).root;
    duration += overlap;
    if (window.bassRootEstimate === expectedRoot) matchingDuration += overlap;
    if (window.bassRootConfidence >= 0.35) {
      confidentDuration += overlap;
      if (window.bassRootEstimate === expectedRoot) {
        confidentMatchingDuration += overlap;
      } else {
        conflictingDuration += overlap;
      }
    }
  }
  const ratio = (numerator: number, denominator: number) =>
    denominator ? Number((numerator / denominator).toFixed(4)) : 0;
  return {
    evaluatedWindowDurationSeconds: Number(duration.toFixed(3)),
    bassRootAgreement: ratio(matchingDuration, duration),
    confidentBassCoverage: ratio(confidentDuration, duration),
    confidentBassAgreement: ratio(confidentMatchingDuration, confidentDuration),
    confidentBassConflict: ratio(conflictingDuration, confidentDuration),
  };
}

function summarizeRun(
  id: string,
  reference: any,
  duration: number,
  analysis: ChordAnalysisResult,
  bpm: number,
) {
  const evaluation = evaluateChordReference(
    reference,
    predictionFromAnalysis(duration, analysis, bpm),
  );
  const start = Math.min(...reference.alignment.regions.map((region: any) => region.start));
  const end = Math.max(...reference.alignment.regions.map((region: any) => region.end));
  return {
    id,
    settings: {
      effectiveBeatDurationSeconds: analysis.settings.minimumChordDurationBeats
        ? Number((analysis.settings.minimumChordDurationSeconds
          / analysis.settings.minimumChordDurationBeats).toFixed(6))
        : null,
      rootSupportWeight: analysis.settings.rootSupportWeight,
      bassAgreementWeight: analysis.settings.bassAgreementWeight,
      keyCompatibilityWeight: analysis.settings.keyCompatibilityWeight,
      preserveSeventhThreshold: analysis.settings.preserveSeventhThreshold,
    },
    keyEstimate: analysis.diagnostics.keyEstimate,
    diagnostics: analysis.diagnostics,
    metrics: evaluation.metrics,
    sequence: evaluation.sequence,
    bassEvidence: bassEvidenceSummary(reference, analysis),
    sectionRegions: analysis.regions.filter((region) => region.end > start && region.start < end),
    mismatches: evaluation.mismatches,
    boundaryStages: evaluation.boundaryStages,
    sectionWindows: analysis.windows.filter(
      (window) => window.end > start && window.start < end,
    ),
    decoderComparisons: Object.fromEntries([
      ["raw-frame-winner", rawWinnerRegions(analysis)],
      ["beat-level-winner", analysis.decoderAlternatives?.beatLevelWinner ?? []],
      ["current-production-decoder", analysis.regions],
      ["reduced-latency-decoder", analysis.decoderAlternatives?.reducedLatency ?? []],
    ].map(([decoderId, chords]) => {
      const decoderEvaluation = evaluateChordReference(
        reference,
        predictionFromAnalysis(duration, analysis, bpm, chords as any),
      );
      return [decoderId, {
        metrics: decoderEvaluation.metrics,
        sequence: decoderEvaluation.sequence,
        sectionRegions: (chords as any[]).filter(
          (region) => region.end > start && region.start < end,
        ),
      }];
    })),
  };
}

const [
  cachePath,
  harmonyPath,
  bassPath,
  referencePath,
  outputPath,
  mode,
] = process.argv.slice(2);
if (!cachePath || !harmonyPath || !bassPath || !referencePath || !outputPath) {
  throw new Error(
    "Usage: chord-section-ablation.ts CACHE.json HARMONY.wav BASS.wav REFERENCE.json OUTPUT.json",
  );
}

const [cacheSource, harmonySource, bassSource, referenceSource] = await Promise.all([
  readFile(cachePath, "utf8"),
  readFile(harmonyPath),
  readFile(bassPath),
  readFile(referencePath, "utf8"),
]);
const cache = JSON.parse(cacheSource);
const reference = JSON.parse(referenceSource);
const harmonyDecoded = decodePcmWav(harmonySource);
const bassDecoded = decodePcmWav(bassSource);
const sampleRate = 22050;
const harmonySamples = resample(harmonyDecoded.samples, harmonyDecoded.sampleRate, sampleRate);
const bassSamples = resample(bassDecoded.samples, bassDecoded.sampleRate, sampleRate);
const duration = harmonySamples.length / sampleRate;
const settings = cache.settings.chordAnalysisSettings as Partial<ChordSmoothingSettings>;

const baselineEvaluation = evaluateChordReference(reference, {
  duration: cache.result.duration,
  bpm: cache.result.bpm,
  chords: cache.result.chords,
  chordAnalysis: cache.result.chordAnalysis,
});
const baseline = {
  id: "cached-harmonic-context-v2",
  cacheMetadata: cache.metadata,
  bpm: cache.result.bpm,
  keyEstimate: cache.result.chordAnalysis.diagnostics.keyEstimate,
  diagnostics: cache.result.chordAnalysis.diagnostics,
  metrics: baselineEvaluation.metrics,
  sequence: baselineEvaluation.sequence,
  bassEvidence: bassEvidenceSummary(reference, cache.result.chordAnalysis),
  sectionRegions: cache.result.chords.filter(
    (region: any) => region.end > 130 && region.start < 165,
  ),
  mismatches: baselineEvaluation.mismatches,
};

const runs: Record<string, ReturnType<typeof summarizeRun>> = {};
const configurations: Array<{
  id: string;
  beatGrid: BeatGrid;
  overrides: Partial<ChordSmoothingSettings>;
}> = [
  {
    id: "current-103-bpm-grid",
    beatGrid: { bpm: 103, beatDuration: 60 / 103, phase: 0 },
    overrides: {},
  },
  {
    id: "corrected-73-bpm-grid",
    beatGrid: {
      bpm: 73,
      beatDuration: 60 / 73,
      // The first manually verified bar attack is 129.614 s. Its modulo
      // places every analysis boundary on the same audible pulse.
      phase: 129.614 % (60 / 73),
    },
    overrides: {},
  },
  {
    id: "without-bass-root-prior",
    beatGrid: { bpm: 103, beatDuration: 60 / 103, phase: 0 },
    overrides: { rootSupportWeight: 0, bassAgreementWeight: 0 },
  },
  {
    id: "without-key-prior",
    beatGrid: { bpm: 103, beatDuration: 60 / 103, phase: 0 },
    overrides: { keyCompatibilityWeight: 0 },
  },
];
if (mode === "--decoder-only") {
  configurations.splice(1);
}

for (const configuration of configurations) {
  const analysis = analyzeChordProgression(
    harmonySamples,
    sampleRate,
    configuration.beatGrid,
    { ...settings, ...configuration.overrides },
    { bassSamples, source: "separated-harmonic-mix" },
  );
  runs[configuration.id] = summarizeRun(
    configuration.id,
    reference,
    duration,
    analysis,
    configuration.beatGrid.bpm!,
  );
}

const report = {
  schemaVersion: 1,
  purpose: "Offline development-set diagnosis; production chord parameters were not changed.",
  reference: {
    id: reference.id,
    artist: reference.artist,
    title: reference.title,
    source: reference.source,
    referenceUrl: reference.referenceUrl,
    version: reference.version,
    ratingCount: reference.ratingCount,
    averageRating: reference.averageRating,
    selectionReason: reference.selectionReason,
    capo: reference.capo,
    tuning: reference.tuning,
    split: reference.split,
    alignment: reference.alignment,
  },
  tempoCandidates: {
    harmonicMix: tempoCandidates(harmonySamples, sampleRate),
    bassStem: tempoCandidates(bassSamples, sampleRate),
  },
  baseline,
  ablations: runs,
  createdAt: new Date().toISOString(),
};
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({
  outputPath,
  baseline: baseline.metrics,
  ablations: Object.fromEntries(
    Object.entries(runs).map(([id, run]) => [id, run.metrics]),
  ),
}, null, 2)}\n`);
