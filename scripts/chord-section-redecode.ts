import { readFile, writeFile } from "node:fs/promises";
import {
  BALANCED_CHORD_SMOOTHING_SETTINGS,
  smoothChordObservationsReducedLatency,
  type ChordObservation,
} from "../src/chordAnalysis";
import type { ChordWindowDiagnostics } from "../src/types";
import { evaluateChordReference } from "../server/evaluation.mjs";

const [diagnosticPath, referencePath, outputPath] = process.argv.slice(2);
if (!diagnosticPath || !referencePath || !outputPath) {
  throw new Error(
    "Usage: chord-section-redecode.ts DIAGNOSTIC.json REFERENCE.json OUTPUT.json",
  );
}

const [diagnosticSource, referenceSource] = await Promise.all([
  readFile(diagnosticPath, "utf8"),
  readFile(referencePath, "utf8"),
]);
const diagnostic = JSON.parse(diagnosticSource);
const reference = JSON.parse(referenceSource);
const run = diagnostic.ablations["current-103-bpm-grid"];
if (!run?.sectionWindows?.length) {
  throw new Error("Diagnostic does not contain reusable section windows.");
}

const observations: ChordObservation[] = run.sectionWindows.map(
  (window: ChordWindowDiagnostics) => ({
    start: window.start,
    end: window.end,
    bestChord: window.topCandidateChord,
    bestScore: window.topCandidateScore,
    secondBestChord: window.secondBestChord,
    secondBestScore: window.secondBestScore,
    confidence: window.observationConfidence ?? window.topCandidateScore,
    scoreMargin: window.scoreMargin,
    uncertain: Boolean(window.uncertain),
    candidateScores: window.candidateScores ?? {
      [window.topCandidateChord]: window.topCandidateScore,
      [window.secondBestChord]: window.secondBestScore,
    },
    noChordScore: window.noChordScore ?? -0.5,
    seventhEvidence: window.seventhEvidence ?? 0,
    seventhEvidenceByChord: window.seventhEvidenceByChord,
    chroma: window.chroma,
    rootClass: window.rootClass,
    bassRootClass: window.bassRootEstimate,
    bassConfidence: window.bassRootConfidence,
    bassAttackRootClass: window.bassAttackRootEstimate,
    bassAttackConfidence: window.bassAttackConfidence,
    bassSustainRootClass: window.bassSustainRootEstimate,
    bassSustainConfidence: window.bassSustainConfidence,
    bassIsPassingTone: window.bassIsPassingTone,
    keyEstimate: null,
    usedBassSupport: window.usedBassSupport,
    boundaryStrength: window.beatStrength,
  }),
);
const settings = {
  ...BALANCED_CHORD_SMOOTHING_SETTINGS,
  minimumChordDurationSeconds: BALANCED_CHORD_SMOOTHING_SETTINGS.minimumChordDurationBeats
    * (60 / 103),
};
const reduced = smoothChordObservationsReducedLatency(observations, settings);
const report = evaluateChordReference(reference, {
  duration: 165,
  bpm: 103,
  chords: reduced.regions,
  chordAnalysis: {
    windows: run.sectionWindows,
  },
});
const output = {
  sourceDiagnostic: diagnosticPath,
  settings,
  metrics: report.metrics,
  sequence: report.sequence,
  regions: reduced.regions,
  decisions: reduced.decisions,
  mismatches: report.mismatches,
  createdAt: new Date().toISOString(),
};
await writeFile(outputPath, `${JSON.stringify(output, null, 2)}\n`, "utf8");
process.stdout.write(`${JSON.stringify({
  outputPath,
  metrics: output.metrics,
  sequence: output.sequence,
  regions: output.regions.map((region) => ({
    start: region.start,
    end: region.end,
    name: region.name,
  })),
}, null, 2)}\n`);
