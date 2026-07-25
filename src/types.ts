export type NoteEvent = {
  start: number;
  end: number;
  midi: number;
  name: string;
  confidence: number;
  string: number;
  fret: number;
};

export type CachedTranscriptionMetadata = {
  cacheSchemaVersion: number;
  transcriptionPipelineVersion: string;
  noteCleanupVersion: number;
  arrangementVersion: number;
  fretboardMapperVersion: number;
  chordAnalysisVersion: number;
  createdAt: string;
};

export type RawNoteEvent = {
  start: number;
  end: number;
  midi: number;
  name: string;
  confidence: number;
  /** Basic Pitch contour offsets retained for diagnostics; tab uses semitone pitches. */
  pitchBends?: number[];
};

export type ArrangementMode = "automatic" | "lead" | "rhythm" | "mixed" | "raw";
export type QualityPreset = "clean" | "balanced" | "detailed" | "raw";

export type NoteCleanupSettings = {
  /** Default minimum duration; confident notes in coherent fast phrases may survive. */
  minimumNoteDurationSeconds: number;
  /** Low-confidence notes must meet this stronger duration threshold. */
  lowConfidenceMinimumDurationSeconds: number;
  /** Consecutive same-pitch events separated by at most this gap are merged. */
  samePitchMergeGapSeconds: number;
  /** Onsets inside this tolerance are treated as one musical attack. */
  onsetClusterToleranceSeconds: number;
  /** Base confidence floor before local density and context adjustments. */
  minimumConfidence: number;
  /** Confidence that qualifies a short note for the coherent-phrase exception. */
  highConfidenceThreshold: number;
  /** Target local density above which confidence filtering becomes stricter. */
  maximumNotesPerSecond: number;
  /** Low-confidence octave duplicates below this confidence ratio are suppressed. */
  octaveGhostConfidenceRatio: number;
  /** Maximum voices selected for the melodic arrangement. */
  leadMaximumPolyphony: number;
  /** Maximum notes retained in a rhythm-chord attack. */
  rhythmMaximumPolyphony: number;
  /** Active-fret span for ordinary generated chord shapes. */
  maximumChordStretchFrets: number;
  preferredFretRange: [number, number];
  maximumFret: number;
  /** Low-confidence detector durations are capped conservatively. */
  lowConfidenceMaximumSustainSeconds: number;
};

export type NoteDiagnostics = {
  rawNoteCount: number;
  cleanedNoteCount: number;
  arrangedNoteCount: number;
  rawNotesPerSecond: number;
  cleanedNotesPerSecond: number;
  maximumPolyphonyRaw: number;
  maximumPolyphonyCleaned: number;
  medianNoteDurationRaw: number;
  medianNoteDurationCleaned: number;
  notesBelow100msRaw: number;
  notesBelow100msCleaned: number;
  duplicateOrSplitNotes: number;
  duplicatesMerged: number;
  octaveGhostsRemoved: number;
  densityFilteredNotes: number;
  averageFretMovementRaw: number;
  averageFretMovementFinal: number;
  maximumFretJumpRaw: number;
  maximumFretJumpFinal: number;
  sameStringOverlapsResolved: number;
};

export type TranscriptionWarning = {
  code:
    | "high-note-density"
    | "multiple-parts"
    | "excessive-polyphony"
    | "large-fret-jumps"
    | "unstable-chords"
    | "low-tempo-confidence";
  message: string;
  suggestion: string;
};

export type NoteAnalysisResult = {
  rawNotes: RawNoteEvent[];
  cleanedNotes: RawNoteEvent[];
  arrangedNotes: NoteEvent[];
  requestedMode: ArrangementMode;
  resolvedMode: Exclude<ArrangementMode, "automatic">;
  preset: QualityPreset;
  settings: NoteCleanupSettings;
  diagnostics: NoteDiagnostics;
  warnings: TranscriptionWarning[];
};

export type ChordEvent = {
  start: number;
  end: number;
  name: string;
  confidence: number;
  /** User corrections survive ordinary re-rendering and can be carried into reprocessing. */
  locked?: boolean;
  diagnostics?: ChordRegionDiagnostics;
};

export type ChordDisplayMode = "simple" | "detailed";
export type ChordConfidenceLevel = "high" | "medium" | "low";

export type ChordRegionDiagnostics = {
  finalLabel: string;
  confidence: number;
  confidenceLevel: ChordConfidenceLevel;
  bestScore: number;
  secondBestScore: number;
  scoreMargin: number;
  rootClass: string | null;
  keyEstimate: string | null;
  usedBassSupport: boolean;
  usedSmoothingOverride: boolean;
  simplifiedFromExtendedChord: boolean;
  mergedFromShortRegions: boolean;
  /** First beat where evidence for the accepted transition began. */
  candidateEvidenceStart?: number | null;
  /** End of the window that made accumulated change evidence decisive. */
  confirmationTime?: number | null;
  /** Beat selected after comparing preceding, nearest, and following candidates. */
  finalBoundaryTime?: number | null;
};

export type RawChordFrame = {
  start: number;
  end: number;
  bestChord: string;
  bestScore: number;
  secondBestChord: string;
  secondBestScore: number;
  confidence: number;
  scoreMargin: number;
  /** True when the winning template is not separated enough to justify a change. */
  uncertain: boolean;
  rootClass?: string | null;
  bassRootClass?: string | null;
  bassConfidence?: number;
  keyEstimate?: string | null;
};

export type ChordWindowDiagnostics = {
  start: number;
  end: number;
  topCandidateChord: string;
  topCandidateScore: number;
  secondBestChord: string;
  secondBestScore: number;
  scoreMargin: number;
  bassRootEstimate: string | null;
  bassRootConfidence: number;
  keyEstimate: string | null;
  beatStrength: number;
  finalLabel: string;
  usedBassSupport: boolean;
  usedSmoothingOverride: boolean;
  mergedFromNearbyWindows: boolean;
  bassAttackRootEstimate?: string | null;
  bassAttackConfidence?: number;
  bassSustainRootEstimate?: string | null;
  bassSustainConfidence?: number;
  bassIsPassingTone?: boolean;
  reducedLatencyLabel?: string;
  /** Offline decoder input retained to make parameter studies avoid new FFTs. */
  candidateScores?: Record<string, number>;
  noChordScore?: number;
  chroma?: number[];
  observationConfidence?: number;
  uncertain?: boolean;
  seventhEvidence?: number;
  seventhEvidenceByChord?: Record<string, number>;
  rootClass?: string | null;
};

export type ChordBoundaryDecision = {
  fromChord: string;
  toChord: string;
  candidateEvidenceStart: number;
  confirmationTime: number;
  finalBoundaryTime: number;
  accumulatedEvidence: number;
  supportingWindows: number;
};

export type ChordDecoderAlternatives = {
  beatLevelWinner: ChordEvent[];
  viterbi: ChordEvent[];
  postHysteresis: ChordEvent[];
  production: ChordEvent[];
  reducedLatency: ChordEvent[];
  reducedLatencyDecisions: ChordBoundaryDecision[];
};

export type KeyEstimate = {
  name: string;
  tonic: number;
  mode: "major" | "minor";
  confidence: number;
};

export type ChordSmoothingSettings = {
  /** Shorter chord regions are reassigned to the most plausible neighbour. */
  minimumChordDurationSeconds: number;
  /** A new label must keep winning for this many aggregate windows. */
  requiredConsecutiveWindows: number;
  /** Required score improvement over the currently stable chord. */
  changeMargin: number;
  /** Viterbi cost paid whenever the decoded chord label changes. */
  chordChangePenalty: number;
  /** Extra Viterbi cost when a state is left before its minimum duration. */
  shortRegionPenalty: number;
  /** Extra cost for a change whose observation evidence is only marginally better. */
  weakTransitionPenalty: number;
  /** Extra change cost away from inferred beat boundaries. */
  offBeatChangePenalty: number;
  /** Combined absolute template-fit confidence needed before a change is trusted. */
  minimumChordConfidence: number;
  /** Minimum separation between the best and second-best templates. */
  minimumScoreMargin: number;
  /** Minimum pitch-class energy required to retain a dominant seventh. */
  preserveSeventhThreshold: number;
  /** `N` gaps shorter than this are normally absorbed by neighbouring harmony. */
  shortNoChordMergeSeconds: number;
  /** Aggregate-window duration used when no reliable tempo is available. */
  fallbackWindowSeconds: number;
  /** Hop between raw chroma frames; raw frames are retained for diagnostics. */
  rawFrameHopSeconds: number;
  /** Confidence required for the beat-aligned short-chord exception. */
  highConfidenceShortChord: number;
  /** Beat-relative minimum used when a reliable tempo is available. */
  minimumChordDurationBeats: number;
  /** UI/output vocabulary. Detailed mode still requires sustained extension evidence. */
  chordDisplayMode: ChordDisplayMode;
  /** Weight for low-frequency root support extracted from harmonic audio. */
  rootSupportWeight: number;
  /** Additional weight for a dedicated Demucs bass stem when available. */
  bassAgreementWeight: number;
  /** Soft song/section-key compatibility prior; never forbids chromatic harmony. */
  keyCompatibilityWeight: number;
  /** Mild reward for common functional or fifth-related chord movement. */
  commonTransitionBonus: number;
  /** Cost for choosing a complex chord when its added tones are weak. */
  weakExtensionPenalty: number;
  /** Cost for `N` while harmonic or bass evidence remains present. */
  unnecessaryNoChordPenalty: number;
  /** Evidence needed for a seventh to survive the final simplification pass. */
  strongSeventhThreshold: number;
  /** Small second-pass reward for matching a genuinely similar repeated window. */
  repeatedSectionConsistencyWeight: number;
  /** Accumulated score/confidence advantage needed to accept a new chord. */
  changeEvidenceThreshold: number;
  /** Hard ceiling on repeated-section evidence in the reduced-latency decoder. */
  maximumRepeatedSectionBonus: number;
  /** Fraction of a beat window used to estimate the initial bass attack. */
  bassAttackWindowFraction: number;
  /** Score separation needed for a one-beat region without strong bass support. */
  validShortChordScoreMargin: number;
};

export type ChordDiagnostics = {
  rawChordChanges: number;
  finalChordRegions: number;
  averageRegionDuration: number;
  shortRegionsRemoved: number;
  regionsMerged: number;
  lowConfidenceRegions: number;
  analysisWindows: number;
  beatAlignedBoundaries: number;
  noChordRegions: number;
  keyEstimate: KeyEstimate | null;
  harmonyEvidenceSource: "separated-harmonic-mix" | "full-mix" | "guitar-only";
  bassSupportedWindows: number;
  smoothingOverrides: number;
};

export type ChordAnalysisResult = {
  rawFrames: RawChordFrame[];
  windows: ChordWindowDiagnostics[];
  regions: ChordEvent[];
  settings: ChordSmoothingSettings;
  diagnostics: ChordDiagnostics;
  /** Offline comparison outputs; normal UI playback continues to use `regions`. */
  decoderAlternatives?: ChordDecoderAlternatives;
};

export type AnalysisResult = {
  duration: number;
  bpm: number | null;
  tuning: string[];
  capo: number;
  /** Cleaned, playback-ready regions retained for API compatibility. */
  chords: ChordEvent[];
  /** Cleaned, arranged, and mapped notes retained for API compatibility. */
  notes: NoteEvent[];
  /** Raw detections and each note-processing stage for diagnostics/export. */
  noteAnalysis: NoteAnalysisResult;
  /** Raw evidence and smoothing diagnostics for tuning and debugging. */
  chordAnalysis: ChordAnalysisResult;
  engine: "basic-pitch" | "dsp";
  separation: "guitar" | "none";
  /** Present when the experimental learned chord engine ran; reports whether it
   *  was actually used or fell back to the rule-based regions. */
  learnedEngine?: { usedLearned: boolean; fallbackReason?: string };
};

export type HarmonyResult = Omit<AnalysisResult, "notes" | "noteAnalysis" | "engine" | "separation">;
