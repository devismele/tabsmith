import test from "node:test";
import assert from "node:assert/strict";
import {
  aggregateChordEvaluationReports,
  alignChordSequences,
  createManualReference,
  evaluateChordReference,
  evaluateTranscription,
  normalizeChordSymbol,
  parseChordReferenceText,
  selectReferenceVersion,
} from "../server/evaluation.mjs";
import {
  evaluationStorageInfo,
  importChordReference,
  loadChordReference,
  saveManualReferenceAlignment,
} from "../server/evaluation-store.mjs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { after, before } from "node:test";

let evaluationRoot;
let previousEvaluationRoot;

before(async () => {
  evaluationRoot = await mkdtemp(path.join(os.tmpdir(), "tabsmith-evaluation-test-"));
  previousEvaluationRoot = process.env.TABSMITH_EVALUATION_DIR;
  process.env.TABSMITH_EVALUATION_DIR = evaluationRoot;
});

after(async () => {
  if (previousEvaluationRoot === undefined) delete process.env.TABSMITH_EVALUATION_DIR;
  else process.env.TABSMITH_EVALUATION_DIR = previousEvaluationRoot;
  await rm(evaluationRoot, { recursive: true, force: true });
});

test("scores notes, timing, fingering, and chord duration", () => {
  const reference = {
    duration: 2,
    notes: [
      { start: 0, midi: 40, string: 6, fret: 0 },
      { start: 1, midi: 45, string: 5, fret: 0 },
    ],
    chords: [{ start: 0, end: 1, name: "E" }, { start: 1, end: 2, name: "A" }],
  };
  const prediction = {
    notes: [
      { start: 0.03, midi: 40, string: 6, fret: 0 },
      { start: 1.04, midi: 45, string: 6, fret: 5 },
      { start: 1.5, midi: 48, string: 5, fret: 3 },
    ],
    chords: [{ start: 0, end: 1, name: "E" }, { start: 1, end: 2, name: "Am" }],
  };
  const score = evaluateTranscription(reference, prediction);
  assert.equal(score.noteRecall, 1);
  assert.equal(score.notePrecision, 0.6667);
  assert.equal(score.fretPositionAccuracy, 0.5);
  assert.equal(score.chordTimeAccuracy, 0.5);
  assert.equal(score.onsetMaeMs, 35);
});

test("reference selection prioritizes rating count over average rating", () => {
  const selected = selectReferenceVersion([
    { version: 1, type: "Chords", ratingCount: 200, averageRating: 5 },
    { version: 2, type: "Chords", ratingCount: 400, averageRating: 4.7 },
  ]);
  assert.equal(selected.version, 2);
  assert.match(selected.selectionReason, /Largest rating count/);
});

test("reference selection excludes unrelated tab types", () => {
  const selected = selectReferenceVersion([
    { version: 5, type: "Tab", ratingCount: 50_000, averageRating: 5 },
    { version: 2, type: "Chords", ratingCount: 300, averageRating: 4.8 },
  ]);
  assert.equal(selected.type, "Chords");
  assert.equal(selected.version, 2);
});

test("capo transposition stores displayed and concert-pitch chords", () => {
  const chord = normalizeChordSymbol("G/B", { capo: 2 });
  assert.equal(chord.displayed, "G/B");
  assert.equal(chord.sounding, "A/C#");
  assert.equal(chord.root, "A");
  assert.equal(chord.bass, "C#");
});

test("enharmonic chord roots normalize to one canonical pitch class", () => {
  assert.equal(normalizeChordSymbol("Gb").root, normalizeChordSymbol("F#").root);
  assert.equal(normalizeChordSymbol("Dbm").detailed, "C#m");
});

function referenceWithAlignment(chords, duration = chords.at(-1).end) {
  return {
    id: "reference-test",
    source: "ultimate-guitar",
    artist: "Test",
    title: "Song",
    referenceUrl: "",
    version: 1,
    ratingCount: 10,
    averageRating: 4.5,
    selectionReason: "test",
    chords: chords.map((chord, index) => ({
      order: index,
      section: "Test",
      ...normalizeChordSymbol(chord.name),
    })),
    alignment: {
      method: "manual",
      verified: true,
      confidence: 1,
      regions: chords.map((chord, index) => ({
        start: chord.start,
        end: chord.end,
        chord: chord.name,
        chordIndex: index,
      })),
    },
    duration,
  };
}

function prediction(chords, duration = chords.at(-1).end) {
  return { duration, chords, chordAnalysis: { windows: [] } };
}

test("root-only accuracy ignores chord quality", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([{ start: 0, end: 4, name: "E" }]),
    prediction([{ start: 0, end: 4, name: "E7" }]),
  );
  assert.equal(report.metrics.rootAccuracy, 1);
  assert.equal(report.metrics.detailedAccuracy, 0);
});

test("major/minor comparison collapses major sevenths and dominant sevenths", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([
      { start: 0, end: 2, name: "Cmaj7" },
      { start: 2, end: 4, name: "Am7" },
    ]),
    prediction([
      { start: 0, end: 2, name: "C7" },
      { start: 2, end: 4, name: "Am" },
    ]),
  );
  assert.equal(report.metrics.majorMinorAccuracy, 1);
  assert.equal(report.metrics.detailedAccuracy, 0);
});

test("detailed comparison preserves seventh errors", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([{ start: 0, end: 4, name: "E7" }]),
    prediction([{ start: 0, end: 4, name: "E" }]),
  );
  assert.equal(report.metrics.rootAccuracy, 1);
  assert.equal(report.metrics.majorMinorAccuracy, 1);
  assert.equal(report.metrics.detailedAccuracy, 0);
  assert.equal(report.mismatches[0].errorType, "seventh omitted");
});

test("relative-major-minor diagnostics require opposite chord qualities", () => {
  const trueRelativePair = evaluateChordReference(
    referenceWithAlignment([{ start: 0, end: 4, name: "G" }]),
    prediction([{ start: 0, end: 4, name: "Em" }]),
  );
  assert.equal(trueRelativePair.mismatches[0].errorType, "relative major/minor confusion");

  const sameQualityPair = evaluateChordReference(
    referenceWithAlignment([{ start: 0, end: 4, name: "G" }]),
    prediction([{ start: 0, end: 4, name: "E" }]),
  );
  assert.notEqual(sameQualityPair.mismatches[0].errorType, "relative major/minor confusion");
});

test("slash-chord inversion accuracy is reported separately from harmonic quality", () => {
  const reference = referenceWithAlignment([{ start: 0, end: 4, name: "D/F#" }]);
  const supported = evaluateChordReference(
    reference,
    {
      duration: 4,
      chords: [{ start: 0, end: 4, name: "D" }],
      chordAnalysis: {
        windows: [{
          start: 0,
          end: 4,
          bassRootEstimate: "F#",
          bassRootConfidence: 0.9,
        }],
      },
    },
  );
  assert.equal(supported.metrics.detailedAccuracy, 1);
  assert.equal(supported.metrics.bassInversionAccuracy, 1);

  const collapsed = evaluateChordReference(
    reference,
    prediction([{ start: 0, end: 4, name: "D" }]),
  );
  assert.equal(collapsed.metrics.detailedAccuracy, 1);
  assert.equal(collapsed.metrics.bassInversionAccuracy, 0);
  assert.equal(collapsed.mismatches[0].errorType, "inversion collapsed");
});

test("untimed sequence alignment uses dynamic programming instead of array index", () => {
  const alignment = alignChordSequences(
    [{ sounding: "C" }, { sounding: "G" }],
    [{ name: "D" }, { name: "C" }, { name: "G" }],
  );
  assert.deepEqual(
    alignment.pairs.map((pair) => [pair.referenceIndex, pair.detectedIndex]),
    [[null, 0], [0, 1], [1, 2]],
  );
  assert.equal(alignment.verified, false);
});

test("chord scoring is weighted by duration", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([
      { start: 0, end: 9, name: "C" },
      { start: 9, end: 10, name: "G" },
    ]),
    prediction([
      { start: 0, end: 9, name: "C" },
      { start: 9, end: 10, name: "D" },
    ]),
  );
  assert.equal(report.metrics.rootAccuracy, 0.9);
});

test("section evaluation excludes song-wide chord regions outside the aligned interval", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([{ start: 10, end: 14, name: "C" }]),
    prediction([
      { start: 0, end: 10, name: "G" },
      { start: 10, end: 14, name: "C" },
      { start: 14, end: 20, name: "D" },
    ], 20),
  );
  assert.equal(report.metrics.evaluatedDurationSeconds, 4);
  assert.equal(report.metrics.rootAccuracy, 1);
  assert.equal(report.sequence.extraDetectedChords, 0);
  assert.equal(report.sequence.missingReferenceChords, 0);
});

test("signed boundary metrics report late decisions in milliseconds and beats", () => {
  const reference = referenceWithAlignment([
    { start: 0, end: 2, name: "C" },
    { start: 2, end: 4, name: "G" },
    { start: 4, end: 6, name: "D" },
  ]);
  const production = [
    { start: 0, end: 2.6, name: "C" },
    { start: 2.6, end: 5.3, name: "G" },
    { start: 5.3, end: 6, name: "D" },
  ];
  const report = evaluateChordReference(reference, {
    duration: 6,
    bpm: 120,
    chords: production,
    chordAnalysis: {
      rawFrames: [
        { start: 0, end: 2.1, bestChord: "C" },
        { start: 2.1, end: 4.1, bestChord: "G" },
        { start: 4.1, end: 6, bestChord: "D" },
      ],
      windows: [],
      decoderAlternatives: {
        beatLevelWinner: [
          { start: 0, end: 2, name: "C" },
          { start: 2, end: 4, name: "G" },
          { start: 4, end: 6, name: "D" },
        ],
        viterbi: production,
        postHysteresis: production,
        production,
        reducedLatency: [
          { start: 0, end: 2.1, name: "C" },
          { start: 2.1, end: 4.1, name: "G" },
          { start: 4.1, end: 6, name: "D" },
        ],
        reducedLatencyDecisions: [],
      },
    },
  });
  assert.equal(report.metrics.meanSignedBoundaryErrorMs, 950);
  assert.equal(report.metrics.earlyBoundaryMeanMs, null);
  assert.equal(report.metrics.lateBoundaryMeanMs, 950);
  assert.equal(report.metrics.meanSignedBoundaryErrorBeats, 1.9);
  assert.equal(report.metrics.percentageBoundariesMoreThanOneBeatLate, 1);
  assert.equal(report.metrics.percentageBoundariesMoreThanTwoBeatsLate, 0.5);
  assert.deepEqual(report.boundaryStages[0], {
    referenceBoundary: 2,
    rawBestCandidateBoundary: 2.1,
    beatLevelWinnerBoundary: 2,
    viterbiBoundary: 2.6,
    hysteresisBoundary: 2.6,
    postProcessingBoundary: 2.6,
    finalRenderedBoundary: 2.6,
    reducedLatencyBoundary: 2.1,
  });
});

test("false N duration and no-chord precision are measured separately", () => {
  const report = evaluateChordReference(
    referenceWithAlignment([{ start: 0, end: 4, name: "D" }]),
    prediction([
      { start: 0, end: 3, name: "D" },
      { start: 3, end: 4, name: "N" },
    ]),
  );
  assert.equal(report.metrics.falseNoChordDurationPercent, 0.25);
  assert.equal(report.metrics.noChordPrecision, 0);
  assert.equal(report.sequence.falseNoChord, 1);
});

test("manual timestamp corrections persist in the separate evaluation store", async () => {
  const imported = await importChordReference({
    artist: "Test Artist",
    title: "Test Song",
    chordSequence: "Verse: C | G | Am | F",
  });
  await saveManualReferenceAlignment(imported.id, [
    { chordIndex: 0, start: 0 },
    { chordIndex: 1, start: 2 },
    { chordIndex: 2, start: 4 },
    { chordIndex: 3, start: 6 },
  ], 8);
  const loaded = await loadChordReference(imported.id);
  assert.equal(loaded.alignment.verified, true);
  assert.deepEqual(loaded.alignment.regions.map((region) => region.start), [0, 2, 4, 6]);
});

test("lyrics, HTML, and arbitrary page bodies are rejected instead of stored", () => {
  assert.throws(() => parseChordReferenceText("C G\nThese are song lyrics"), /chord symbols only|Unsupported chord/);
  assert.throws(() => parseChordReferenceText("<html><body>C G</body></html>"), /HTML\/page content/);
});

test("stored reference contains normalized chords but not pasted source text", async () => {
  const imported = await importChordReference({
    artist: "Minimal",
    title: "Derived data",
    chordSequence: "Intro: Gb | Dbm",
  });
  const storedText = await readFile(path.join(evaluationRoot, "references", `${imported.id}.json`), "utf8");
  const stored = JSON.parse(storedText);
  assert.equal(stored.chords[0].sounding, "F#");
  assert.equal(stored.chords[1].sounding, "C#m");
  assert.equal("chordSequence" in stored, false);
  assert.equal(storedText.includes("page body"), false);
});

test("evaluation storage is separate and has no automated network importer", () => {
  const info = evaluationStorageInfo();
  assert.equal(info.directory, evaluationRoot);
  assert.equal(info.productionProjectDataSeparated, true);
  assert.equal(info.networkImportEnabled, false);
});

test("aggregate findings only call an error recurring after three songs", () => {
  const reports = Array.from({ length: 3 }, (_value, index) => {
    const report = evaluateChordReference(
      {
        ...referenceWithAlignment([{ start: 0, end: 4, name: "Am" }]),
        artist: `Artist ${index}`,
      },
      prediction([{ start: 0, end: 4, name: "C" }]),
    );
    return report;
  });
  const aggregate = aggregateChordEvaluationReports(reports);
  assert.equal(aggregate.songCount, 3);
  assert.equal(aggregate.recurringErrors[0].errorType, "relative major/minor confusion");
  assert.equal(aggregate.recurringErrors[0].songsAffected, 3);
});

test("timed chord imports create verified monotonic regions", () => {
  const reference = createManualReference({
    artist: "Timed",
    title: "Reference",
    chordSequence: "0:00 C\n0:02 G\n0:04 Am\n0:06 F",
    duration: 8,
  });
  assert.equal(reference.alignment.verified, true);
  assert.deepEqual(reference.alignment.regions.map((region) => region.start), [0, 2, 4, 6]);
  assert.equal(reference.alignment.regions.at(-1).end, 8);
});
