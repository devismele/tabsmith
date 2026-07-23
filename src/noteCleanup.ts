import {
  clipNoteSustains,
  estimateGuitarSetup,
  fretMovementDiagnostics,
  mapPolyphonicNotesToFretboard,
} from "./fretboard";
import type {
  ArrangementMode,
  ChordEvent,
  NoteAnalysisResult,
  NoteCleanupSettings,
  NoteDiagnostics,
  NoteEvent,
  QualityPreset,
  RawNoteEvent,
  TranscriptionWarning,
} from "./types";

export const NOTE_CLEANUP_PRESETS: Record<QualityPreset, NoteCleanupSettings> = {
  clean: {
    minimumNoteDurationSeconds: 0.1,
    lowConfidenceMinimumDurationSeconds: 0.13,
    samePitchMergeGapSeconds: 0.08,
    onsetClusterToleranceSeconds: 0.045,
    minimumConfidence: 0.52,
    highConfidenceThreshold: 0.82,
    maximumNotesPerSecond: 7,
    octaveGhostConfidenceRatio: 0.58,
    leadMaximumPolyphony: 2,
    rhythmMaximumPolyphony: 4,
    maximumChordStretchFrets: 4,
    preferredFretRange: [0, 12],
    maximumFret: 20,
    lowConfidenceMaximumSustainSeconds: 0.4,
  },
  balanced: {
    minimumNoteDurationSeconds: 0.08,
    lowConfidenceMinimumDurationSeconds: 0.11,
    samePitchMergeGapSeconds: 0.08,
    onsetClusterToleranceSeconds: 0.04,
    minimumConfidence: 0.42,
    highConfidenceThreshold: 0.78,
    maximumNotesPerSecond: 9,
    octaveGhostConfidenceRatio: 0.52,
    leadMaximumPolyphony: 2,
    rhythmMaximumPolyphony: 5,
    maximumChordStretchFrets: 4,
    preferredFretRange: [0, 15],
    maximumFret: 20,
    lowConfidenceMaximumSustainSeconds: 0.5,
  },
  detailed: {
    minimumNoteDurationSeconds: 0.055,
    lowConfidenceMinimumDurationSeconds: 0.08,
    samePitchMergeGapSeconds: 0.065,
    onsetClusterToleranceSeconds: 0.035,
    minimumConfidence: 0.33,
    highConfidenceThreshold: 0.74,
    maximumNotesPerSecond: 14,
    octaveGhostConfidenceRatio: 0.45,
    leadMaximumPolyphony: 2,
    rhythmMaximumPolyphony: 6,
    maximumChordStretchFrets: 5,
    preferredFretRange: [0, 17],
    maximumFret: 22,
    lowConfidenceMaximumSustainSeconds: 0.65,
  },
  raw: {
    minimumNoteDurationSeconds: 0.03,
    lowConfidenceMinimumDurationSeconds: 0.03,
    samePitchMergeGapSeconds: 0,
    onsetClusterToleranceSeconds: 0.03,
    minimumConfidence: 0,
    highConfidenceThreshold: 0.7,
    maximumNotesPerSecond: 1000,
    octaveGhostConfidenceRatio: 0,
    leadMaximumPolyphony: 2,
    rhythmMaximumPolyphony: 6,
    maximumChordStretchFrets: 7,
    preferredFretRange: [0, 20],
    maximumFret: 24,
    lowConfidenceMaximumSustainSeconds: 1,
  },
};

type CleanupCounts = {
  duplicateOrSplitNotes: number;
  duplicatesMerged: number;
  octaveGhostsRemoved: number;
  densityFilteredNotes: number;
};

export type NoteProcessingOptions = {
  mode?: ArrangementMode;
  preset?: QualityPreset;
  settings?: Partial<NoteCleanupSettings>;
  separation?: "guitar" | "none";
};

function clamp(value: number, low = 0, high = 1): number {
  return Math.max(low, Math.min(high, value));
}

function duration(note: Pick<RawNoteEvent, "start" | "end">): number {
  return Math.max(0, note.end - note.start);
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function maximumPolyphony(
  notes: Pick<RawNoteEvent, "start" | "end">[],
): number {
  const edges = notes.flatMap((note) => [
    { time: note.start, delta: 1 },
    { time: note.end, delta: -1 },
  ]).sort((a, b) => a.time - b.time || a.delta - b.delta);
  let active = 0;
  let maximum = 0;
  for (const edge of edges) {
    active += edge.delta;
    maximum = Math.max(maximum, active);
  }
  return maximum;
}

export function clusterNoteOnsets<T extends Pick<RawNoteEvent, "start">>(
  notes: T[],
  tolerance: number,
): T[][] {
  const groups: T[][] = [];
  for (const note of [...notes].sort((a, b) => a.start - b.start)) {
    const group = groups.at(-1);
    if (!group || note.start - group[0].start > tolerance) groups.push([note]);
    else group.push(note);
  }
  return groups;
}

function coherentFastNote(
  note: RawNoteEvent,
  notes: RawNoteEvent[],
  settings: NoteCleanupSettings,
): boolean {
  if (note.confidence < settings.highConfidenceThreshold) return false;
  const nearby = notes.filter((candidate) => candidate !== note
    && Math.abs(candidate.start - note.start) <= 0.22
    && Math.abs(candidate.midi - note.midi) <= 7
    && candidate.confidence >= settings.highConfidenceThreshold * 0.9);
  return nearby.length >= 1;
}

function mergeSamePitch(
  notes: RawNoteEvent[],
  settings: NoteCleanupSettings,
  counts: CleanupCounts,
): RawNoteEvent[] {
  if (settings.samePitchMergeGapSeconds <= 0) return notes.map((note) => ({ ...note }));
  const byPitch = new Map<number, RawNoteEvent[]>();
  for (const note of notes) {
    const pitch = byPitch.get(note.midi) ?? [];
    pitch.push({ ...note });
    byPitch.set(note.midi, pitch);
  }
  const merged: RawNoteEvent[] = [];
  for (const pitchNotes of byPitch.values()) {
    pitchNotes.sort((a, b) => a.start - b.start || a.end - b.end);
    for (const note of pitchNotes) {
      const prior = merged.at(-1);
      if (prior?.midi === note.midi
        && note.start - prior.end <= settings.samePitchMergeGapSeconds) {
        const firstDuration = duration(prior);
        const secondDuration = duration(note);
        const total = Math.max(0.001, firstDuration + secondDuration);
        prior.end = Math.max(prior.end, note.end);
        prior.confidence = (
          prior.confidence * firstDuration + note.confidence * secondDuration
        ) / total;
        if (note.pitchBends?.length) {
          prior.pitchBends = [...(prior.pitchBends ?? []), ...note.pitchBends];
        }
        counts.duplicateOrSplitNotes += 1;
        counts.duplicatesMerged += 1;
      } else {
        merged.push({ ...note });
      }
    }
  }
  return merged.sort((a, b) => a.start - b.start || a.midi - b.midi);
}

function suppressOctaveGhosts(
  notes: RawNoteEvent[],
  settings: NoteCleanupSettings,
  counts: CleanupCounts,
): RawNoteEvent[] {
  if (settings.octaveGhostConfidenceRatio <= 0) return notes;
  const kept: RawNoteEvent[] = [];
  for (const group of clusterNoteOnsets(notes, settings.onsetClusterToleranceSeconds)) {
    const rejected = new Set<RawNoteEvent>();
    for (let first = 0; first < group.length; first += 1) {
      for (let second = first + 1; second < group.length; second += 1) {
        const a = group[first];
        const b = group[second];
        const distance = Math.abs(a.midi - b.midi);
        if ((distance !== 12 && distance !== 24)
          || Math.abs(duration(a) - duration(b)) > 0.15) continue;
        const weak = a.confidence <= b.confidence ? a : b;
        const strong = weak === a ? b : a;
        if (weak.confidence < strong.confidence * settings.octaveGhostConfidenceRatio
          && weak.confidence < settings.highConfidenceThreshold) {
          rejected.add(weak);
        }
      }
    }
    counts.octaveGhostsRemoved += rejected.size;
    kept.push(...group.filter((note) => !rejected.has(note)));
  }
  return kept.sort((a, b) => a.start - b.start || a.midi - b.midi);
}

function localDensity(notes: RawNoteEvent[], time: number): number {
  return notes.filter((note) => Math.abs(note.start - time) <= 0.5).length;
}

/**
 * Converts detector events into defensible musical attacks. No song-wide cap
 * is used: rejection is based on duration, confidence, local density, and
 * neighbouring musical support.
 */
export function cleanupDetectedNotes(
  rawNotes: RawNoteEvent[],
  partialSettings: Partial<NoteCleanupSettings> = {},
  preset: QualityPreset = "balanced",
): { notes: RawNoteEvent[]; counts: CleanupCounts; settings: NoteCleanupSettings } {
  const settings: NoteCleanupSettings = {
    ...NOTE_CLEANUP_PRESETS[preset],
    ...partialSettings,
  };
  const counts: CleanupCounts = {
    duplicateOrSplitNotes: 0,
    duplicatesMerged: 0,
    octaveGhostsRemoved: 0,
    densityFilteredNotes: 0,
  };
  const normalized = rawNotes
    .filter((note) => Number.isFinite(note.start)
      && Number.isFinite(note.end)
      && note.midi >= 40
      && note.midi <= 88
      && note.end > note.start)
    .map((note) => ({
      ...note,
      start: Math.max(0, note.start),
      confidence: clamp(note.confidence),
    }))
    .sort((a, b) => a.start - b.start || a.midi - b.midi);
  const merged = mergeSamePitch(normalized, settings, counts);
  const durationFiltered = merged.filter((note) => {
    const noteDuration = duration(note);
    const required = note.confidence < settings.highConfidenceThreshold
      ? settings.lowConfidenceMinimumDurationSeconds
      : settings.minimumNoteDurationSeconds;
    return noteDuration >= required || coherentFastNote(note, merged, settings);
  });
  const ghostFiltered = suppressOctaveGhosts(durationFiltered, settings, counts);
  const confidenceFiltered = ghostFiltered.filter((note) => {
    const density = localDensity(durationFiltered, note.start);
    const excess = Math.max(0, density - settings.maximumNotesPerSecond);
    const adaptiveFloor = Math.min(
      settings.highConfidenceThreshold,
      settings.minimumConfidence + excess * 0.025,
    );
    const keep = note.confidence >= adaptiveFloor
      || (duration(note) >= 0.35 && note.confidence >= settings.minimumConfidence * 0.85);
    if (!keep && excess > 0) counts.densityFilteredNotes += 1;
    return keep;
  });
  // Give one chord attack one timestamp so quantization cannot turn 20–40 ms
  // detector jitter into separate printed events.
  const clustered = clusterNoteOnsets(
    confidenceFiltered,
    settings.onsetClusterToleranceSeconds,
  ).flatMap((group) => {
    const start = Math.min(...group.map((note) => note.start));
    return group
      .sort((a, b) => b.confidence - a.confidence)
      .map((note) => ({ ...note, start, end: Math.max(start + 0.02, note.end) }));
  });
  return {
    notes: clustered.sort((a, b) => a.start - b.start || a.midi - b.midi),
    counts,
    settings,
  };
}

function chordPitchClasses(chord: ChordEvent | undefined): Set<number> {
  if (!chord || chord.name === "N") return new Set();
  const roots: Record<string, number> = {
    C: 0, "C#": 1, "D♭": 1, D: 2, "D#": 3, "E♭": 3,
    E: 4, F: 5, "F#": 6, "G♭": 6, G: 7, "G#": 8, "A♭": 8,
    A: 9, "A#": 10, "B♭": 10, B: 11,
  };
  const match = /^([A-G](?:[#♯b♭])?)(m|7)?$/.exec(chord.name);
  if (!match) return new Set();
  const rootName = match[1].replace("♯", "#");
  const root = roots[rootName];
  if (root === undefined) return new Set();
  const third = match[2] === "m" ? 3 : 4;
  const tones = [root, (root + third) % 12, (root + 7) % 12];
  if (match[2] === "7") tones.push((root + 10) % 12);
  return new Set(tones);
}

function chordAt(chords: ChordEvent[], time: number): ChordEvent | undefined {
  return chords.find((chord) => time >= chord.start && time < chord.end);
}

function clipAtNextAttack(
  notes: RawNoteEvent[],
  onsetTolerance: number,
): RawNoteEvent[] {
  const groups = clusterNoteOnsets(notes, onsetTolerance);
  return groups.flatMap((group, index) => {
    const nextStart = groups[index + 1]?.[0].start ?? Infinity;
    return group.map((note) => ({
      ...note,
      end: Math.max(note.start + 0.02, Math.min(note.end, nextStart)),
    }));
  }).sort((a, b) => a.start - b.start || a.midi - b.midi);
}

function selectLeadVoice(
  notes: RawNoteEvent[],
  settings: NoteCleanupSettings,
): RawNoteEvent[] {
  const groups = clusterNoteOnsets(notes, settings.onsetClusterToleranceSeconds)
    .filter((group) => group.length);
  if (!groups.length) return [];
  // The state is the last emitted MIDI pitch, plus one "no established
  // register" state. A group may be skipped while retaining the previous
  // register. This is crucial for multi-guitar stems: a single bass/noise
  // onset must not force the melodic line to jump registers.
  const minimumMidi = 40;
  const maximumMidi = 88;
  const emptyState = maximumMidi - minimumMidi + 1;
  const stateCount = emptyState + 1;
  let priorScores = new Float64Array(stateCount);
  priorScores.fill(-Infinity);
  priorScores[emptyState] = 0;
  const backPointers = groups.map(() => new Int16Array(stateCount).fill(-1));
  const emittedIndexes = groups.map(() => new Int16Array(stateCount).fill(-1));

  for (let groupIndex = 0; groupIndex < groups.length; groupIndex += 1) {
    if (groupIndex > 0
      && groups[groupIndex][0].start - groups[groupIndex - 1][0].end > 1.25) {
      // A real rest permits a deliberate register reset.
      priorScores[emptyState] = Math.max(...priorScores);
    }
    const nextScores = new Float64Array(stateCount);
    nextScores.fill(-Infinity);

    // Skip this detector onset and keep the established voice/register.
    for (let state = 0; state < stateCount; state += 1) {
      if (!Number.isFinite(priorScores[state])) continue;
      nextScores[state] = priorScores[state] - 0.18;
      backPointers[groupIndex][state] = state;
    }
    // If several consecutive attacks support a genuinely different register,
    // paying this one-time reset becomes cheaper than skipping the passage.
    // An isolated outlier still loses to the inexpensive ordinary skip above.
    let resetFrom = 0;
    for (let state = 1; state < stateCount; state += 1) {
      if (priorScores[state] > priorScores[resetFrom]) resetFrom = state;
    }
    const resetScore = priorScores[resetFrom] - 1.6;
    if (resetScore > nextScores[emptyState]) {
      nextScores[emptyState] = resetScore;
      backPointers[groupIndex][emptyState] = resetFrom;
      emittedIndexes[groupIndex][emptyState] = -1;
    }

    groups[groupIndex].forEach((note, noteIndex) => {
      const nextState = note.midi - minimumMidi;
      if (nextState < 0 || nextState >= emptyState) return;
      const observationReward = Math.max(
        -0.2,
        (note.confidence - settings.minimumConfidence) * 2.2,
      ) + Math.min(0.35, duration(note));
      for (let state = 0; state < stateCount; state += 1) {
        if (!Number.isFinite(priorScores[state])) continue;
        const priorMidi = state === emptyState ? note.midi : state + minimumMidi;
        const leap = Math.abs(note.midi - priorMidi);
        const leapPenalty = leap * 0.035 + Math.max(0, leap - 12) * 0.12;
        const candidate = priorScores[state] + observationReward - leapPenalty;
        if (candidate > nextScores[nextState]) {
          nextScores[nextState] = candidate;
          backPointers[groupIndex][nextState] = state;
          emittedIndexes[groupIndex][nextState] = noteIndex;
        }
      }
    });
    priorScores = nextScores;
  }

  let cursor = priorScores.indexOf(Math.max(...priorScores));
  const selected: Array<{ note: RawNoteEvent; groupIndex: number }> = [];
  for (let groupIndex = groups.length - 1; groupIndex >= 0; groupIndex -= 1) {
    const emittedIndex = emittedIndexes[groupIndex][cursor];
    if (emittedIndex >= 0) {
      selected.push({
        note: groups[groupIndex][emittedIndex],
        groupIndex,
      });
    }
    cursor = backPointers[groupIndex][cursor];
    if (groupIndex > 0 && cursor < 0) cursor = emptyState;
  }
  selected.reverse();

  const voiced = settings.leadMaximumPolyphony < 2
    ? selected.map(({ note }) => note)
    : selected.flatMap(({ note: lead, groupIndex }) => {
    const companion = groups[groupIndex]
      .filter((candidate) => candidate !== lead
        && Math.abs(candidate.midi - lead.midi) <= 12
        && candidate.confidence >= Math.max(
          settings.highConfidenceThreshold,
          lead.confidence * 0.9,
        ))
      .sort((a, b) => b.confidence - a.confidence)[0];
    return companion ? [lead, companion] : [lead];
  });
  return clipAtNextAttack(voiced, settings.onsetClusterToleranceSeconds);
}

function selectRhythmVoice(
  notes: RawNoteEvent[],
  chords: ChordEvent[],
  bpm: number | null,
  settings: NoteCleanupSettings,
): RawNoteEvent[] {
  const beatDuration = bpm ? 60 / bpm : 0.5;
  const selected = clusterNoteOnsets(notes, settings.onsetClusterToleranceSeconds).flatMap((group) => {
    const onset = group[0].start;
    const beatDistance = Math.abs(onset / beatDuration - Math.round(onset / beatDuration));
    const aligned = beatDistance <= 0.14;
    const tones = chordPitchClasses(chordAt(chords, onset));
    const ranked = [...group].sort((a, b) => {
      const aSupport = tones.has(((a.midi % 12) + 12) % 12) ? 0.24 : 0;
      const bSupport = tones.has(((b.midi % 12) + 12) % 12) ? 0.24 : 0;
      return (b.confidence + bSupport + Math.min(0.2, duration(b)))
        - (a.confidence + aSupport + Math.min(0.2, duration(a)));
    });
    if (ranked.length < 2) {
      const single = ranked[0];
      return aligned
        && single
        && single.confidence >= settings.highConfidenceThreshold
        && duration(single) >= settings.minimumNoteDurationSeconds * 1.5
        ? [single]
        : [];
    }
    const supported = ranked.filter((note) => tones.size === 0
      || tones.has(((note.midi % 12) + 12) % 12)
      || note.confidence >= settings.highConfidenceThreshold);
    return (supported.length >= 2 ? supported : ranked)
      .slice(0, settings.rhythmMaximumPolyphony);
  }).sort((a, b) => a.start - b.start || a.midi - b.midi);
  return clipAtNextAttack(selected, settings.onsetClusterToleranceSeconds);
}

function combineVoices(
  lead: RawNoteEvent[],
  rhythm: RawNoteEvent[],
  settings: NoteCleanupSettings,
): RawNoteEvent[] {
  const unique = new Map<string, RawNoteEvent>();
  for (const note of [...rhythm, ...lead]) {
    const key = `${note.start.toFixed(4)}:${note.midi}`;
    const previous = unique.get(key);
    if (!previous || note.confidence > previous.confidence) unique.set(key, note);
  }
  const combined = clusterNoteOnsets(
    [...unique.values()],
    settings.onsetClusterToleranceSeconds,
  ).flatMap((group) => group
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, Math.min(4, settings.rhythmMaximumPolyphony)))
    .sort((a, b) => a.start - b.start || a.midi - b.midi);
  return clipAtNextAttack(combined, settings.onsetClusterToleranceSeconds);
}

export function arrangeDetectedNotes(
  cleaned: RawNoteEvent[],
  chords: ChordEvent[],
  bpm: number | null,
  requestedMode: ArrangementMode,
  settings: NoteCleanupSettings,
): { notes: RawNoteEvent[]; resolvedMode: Exclude<ArrangementMode, "automatic"> } {
  const rawPolyphony = maximumPolyphony(cleaned);
  const density = cleaned.length / Math.max(1, cleaned.at(-1)?.end ?? 1);
  const resolvedMode: Exclude<ArrangementMode, "automatic"> = requestedMode === "automatic"
    ? (rawPolyphony > 6 || density > settings.maximumNotesPerSecond * 0.8 ? "lead" : "mixed")
    : requestedMode;
  if (resolvedMode === "raw") return { notes: cleaned, resolvedMode };
  const lead = selectLeadVoice(cleaned, settings);
  if (resolvedMode === "lead") return { notes: lead, resolvedMode };
  const rhythm = selectRhythmVoice(cleaned, chords, bpm, settings);
  if (resolvedMode === "rhythm") return { notes: rhythm, resolvedMode };
  return { notes: combineVoices(lead, rhythm, settings), resolvedMode };
}

function warningList(
  diagnostics: NoteDiagnostics,
  chords: ChordEvent[],
  rawChordChanges: number,
  bpm: number | null,
  separation: "guitar" | "none",
): TranscriptionWarning[] {
  const warnings: TranscriptionWarning[] = [];
  if (diagnostics.rawNotesPerSecond > 12) warnings.push({
    code: "high-note-density",
    message: "Very high raw note density was detected.",
    suggestion: "Use the Clean preset or Lead mode for a sparser part.",
  });
  if (diagnostics.maximumPolyphonyRaw > 6 || (separation === "guitar"
    && diagnostics.rawNotesPerSecond > 9)) warnings.push({
    code: "multiple-parts",
    message: "The source appears to contain more than one guitar part.",
    suggestion: "Try Lead or Rhythm mode instead of combining every part.",
  });
  if (diagnostics.maximumPolyphonyRaw > 6) warnings.push({
    code: "excessive-polyphony",
    message: `Raw detections reached ${diagnostics.maximumPolyphonyRaw} simultaneous notes.`,
    suggestion: "The arranged tab has applied a playable polyphony limit.",
  });
  if (diagnostics.maximumFretJumpFinal > 10) warnings.push({
    code: "large-fret-jumps",
    message: "Some large hand-position changes remain in the arrangement.",
    suggestion: "Try the Clean preset or edit the affected phrase.",
  });
  if (rawChordChanges > Math.max(8, chords.length * 4)) warnings.push({
    code: "unstable-chords",
    message: "Raw harmonic labels were unusually unstable.",
    suggestion: "Final labels were sequence-smoothed; verify solo sections by ear.",
  });
  if (!bpm) warnings.push({
    code: "low-tempo-confidence",
    message: "Tempo confidence was too low for reliable beat alignment.",
    suggestion: "Analyze a section with clearer rhythm or adjust timing manually.",
  });
  return warnings;
}

export function processDetectedNotes(
  rawNotes: RawNoteEvent[],
  songDuration: number,
  bpm: number | null,
  chords: ChordEvent[],
  rawChordChanges = 0,
  options: NoteProcessingOptions = {},
): { notes: NoteEvent[]; noteAnalysis: NoteAnalysisResult; tuning: string[]; capo: number } {
  const configuredPreset = options.preset ?? "balanced";
  const requestedMode = options.mode ?? "automatic";
  const preset: QualityPreset = requestedMode === "raw" || configuredPreset === "raw"
    ? "raw"
    : configuredPreset;
  const cleaned = cleanupDetectedNotes(rawNotes, options.settings, preset);
  const arranged = requestedMode === "raw" || preset === "raw"
    ? { notes: cleaned.notes, resolvedMode: "raw" as const }
    : arrangeDetectedNotes(cleaned.notes, chords, bpm, requestedMode, cleaned.settings);
  const setup = estimateGuitarSetup(arranged.notes);
  const mapped = mapPolyphonicNotesToFretboard(
    arranged.notes,
    cleaned.settings.onsetClusterToleranceSeconds,
    setup,
    {
      maximumPolyphony: arranged.resolvedMode === "lead"
        ? cleaned.settings.leadMaximumPolyphony
        : arranged.resolvedMode === "rhythm"
          ? cleaned.settings.rhythmMaximumPolyphony
          : 6,
      maximumChordStretchFrets: cleaned.settings.maximumChordStretchFrets,
      preferredFretRange: cleaned.settings.preferredFretRange,
      maximumFret: cleaned.settings.maximumFret,
    },
  );
  const clipped = clipNoteSustains(
    mapped,
    bpm,
    cleaned.settings.lowConfidenceMaximumSustainSeconds,
  );
  const rawMapped = mapPolyphonicNotesToFretboard(
    rawNotes,
    cleaned.settings.onsetClusterToleranceSeconds,
    setup,
    { maximumPolyphony: 6, maximumChordStretchFrets: 7 },
  );
  const rawMovement = fretMovementDiagnostics(rawMapped);
  const finalMovement = fretMovementDiagnostics(clipped.notes);
  const finalDuration = Math.max(
    0.001,
    songDuration,
    rawNotes.reduce((maximum, note) => Math.max(maximum, note.end), 0),
  );
  const diagnostics: NoteDiagnostics = {
    rawNoteCount: rawNotes.length,
    cleanedNoteCount: cleaned.notes.length,
    arrangedNoteCount: clipped.notes.length,
    rawNotesPerSecond: rawNotes.length / finalDuration,
    cleanedNotesPerSecond: clipped.notes.length / finalDuration,
    maximumPolyphonyRaw: maximumPolyphony(rawNotes),
    maximumPolyphonyCleaned: maximumPolyphony(clipped.notes),
    medianNoteDurationRaw: median(rawNotes.map(duration)),
    medianNoteDurationCleaned: median(clipped.notes.map(duration)),
    notesBelow100msRaw: rawNotes.filter((note) => duration(note) < 0.1).length,
    notesBelow100msCleaned: clipped.notes.filter((note) => duration(note) < 0.1).length,
    duplicateOrSplitNotes: cleaned.counts.duplicateOrSplitNotes,
    duplicatesMerged: cleaned.counts.duplicatesMerged,
    octaveGhostsRemoved: cleaned.counts.octaveGhostsRemoved,
    densityFilteredNotes: cleaned.counts.densityFilteredNotes,
    averageFretMovementRaw: rawMovement.average,
    averageFretMovementFinal: finalMovement.average,
    maximumFretJumpRaw: rawMovement.maximum,
    maximumFretJumpFinal: finalMovement.maximum,
    sameStringOverlapsResolved: clipped.overlapsResolved,
  };
  const warnings = warningList(
    diagnostics,
    chords,
    rawChordChanges,
    bpm,
    options.separation ?? "none",
  );
  const noteAnalysis: NoteAnalysisResult = {
    rawNotes: rawNotes.map((note) => ({ ...note })),
    cleanedNotes: cleaned.notes,
    arrangedNotes: clipped.notes,
    requestedMode,
    resolvedMode: arranged.resolvedMode,
    preset,
    settings: cleaned.settings,
    diagnostics,
    warnings,
  };
  return {
    notes: clipped.notes,
    noteAnalysis,
    tuning: setup.tuningNames,
    capo: setup.capo,
  };
}
