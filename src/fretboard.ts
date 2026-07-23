import type { NoteEvent } from "./types";

// MIDI pitches for strings 6 to 1 in standard tuning: E2 A2 D3 G3 B3 E4.
export const STANDARD_TUNING = [40, 45, 50, 55, 59, 64];
export const TUNING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"];

export type Position = { string: number; fret: number };
export type FretboardSetup = { tuning: number[]; tuningNames: string[]; capo: number; name: string };
export type FretboardMappingOptions = {
  maximumPolyphony: number;
  maximumChordStretchFrets: number;
  preferredFretRange: [number, number];
  maximumFret: number;
  largePositionShiftPenalty: number;
  openStringBonus: number;
  stringChangePenalty: number;
};

const DEFAULT_MAPPING_OPTIONS: FretboardMappingOptions = {
  maximumPolyphony: 6,
  maximumChordStretchFrets: 4,
  preferredFretRange: [0, 15],
  maximumFret: 20,
  largePositionShiftPenalty: 1.8,
  openStringBonus: 0.35,
  stringChangePenalty: 0.35,
};

export function positionsForMidi(midi: number, maxFret = 20, tuning = STANDARD_TUNING, capo = 0): Position[] {
  return tuning.flatMap((openMidi, index) => {
    const fret = midi - (openMidi + capo);
    return fret >= 0 && fret <= maxFret ? [{ string: 6 - index, fret }] : [];
  });
}

function transitionCost(a: Position, b: Position): number {
  const fretTravel = Math.abs(a.fret - b.fret);
  const stringTravel = Math.abs(a.string - b.string);
  const highFretPenalty = Math.max(0, b.fret - 12) * 0.08;
  const openBonus = b.fret === 0 ? -0.35 : 0;
  return fretTravel * 0.7 + stringTravel * 0.45 + highFretPenalty + openBonus;
}

/** Chooses a playable path globally instead of assigning each note in isolation. */
export function mapNotesToFretboard(
  events: Omit<NoteEvent, "string" | "fret">[],
  setup: Pick<FretboardSetup, "tuning" | "capo"> = { tuning: STANDARD_TUNING, capo: 0 },
): NoteEvent[] {
  const valid = events.filter((event) => positionsForMidi(event.midi, 20, setup.tuning, setup.capo).length > 0);
  if (!valid.length) return [];

  const choices = valid.map((event) => positionsForMidi(event.midi, 20, setup.tuning, setup.capo));
  const costs: number[][] = choices.map((positions) => positions.map(() => Infinity));
  const previous: number[][] = choices.map((positions) => positions.map(() => -1));

  choices[0].forEach((position, index) => {
    costs[0][index] = position.fret * 0.06;
  });

  for (let i = 1; i < choices.length; i += 1) {
    choices[i].forEach((current, currentIndex) => {
      choices[i - 1].forEach((prior, priorIndex) => {
        const gap = valid[i].start - valid[i - 1].end;
        const movementWeight = gap > 1 ? 0.55 : 1;
        const candidate =
          costs[i - 1][priorIndex] + transitionCost(prior, current) * movementWeight;
        if (candidate < costs[i][currentIndex]) {
          costs[i][currentIndex] = candidate;
          previous[i][currentIndex] = priorIndex;
        }
      });
    });
  }

  let cursor = costs.at(-1)!.indexOf(Math.min(...costs.at(-1)!));
  const path = Array<number>(valid.length);
  for (let i = valid.length - 1; i >= 0; i -= 1) {
    path[i] = cursor;
    cursor = previous[i][cursor];
  }

  return valid.map((event, index) => ({ ...event, ...choices[index][path[index]] }));
}

/**
 * Maps polyphonic onsets as chord shapes. Notes beginning together must use
 * distinct strings, and ascending pitches must move from lower to higher
 * strings. Between shapes, the chosen hand position is kept reasonably close.
 */
type ShapeCandidate = {
  positions: Position[];
  center: number;
  stringCenter: number;
  localCost: number;
};

const KNOWN_VOICINGS = new Set([
  "1:0|2:0|3:1|4:2|5:2|6:0", // E
  "1:0|2:0|3:0|4:2|5:2|6:0", // Em
  "1:0|2:1|3:0|4:2|5:3", // C
  "1:0|2:2|3:2|4:2|5:0", // A
  "1:0|2:1|3:2|4:2|5:0", // Am
  "1:3|2:0|3:0|4:0|5:2|6:3", // G
  "1:2|2:3|3:2|4:0", // D
  "1:1|2:3|3:2|4:0", // Dm
]);

function shapeKey(positions: Position[]): string {
  return [...positions]
    .sort((a, b) => a.string - b.string)
    .map((position) => `${position.string}:${position.fret}`)
    .join("|");
}

function shapeCandidates(
  group: Omit<NoteEvent, "string" | "fret">[],
  setup: Pick<FretboardSetup, "tuning" | "capo">,
  options: FretboardMappingOptions,
): ShapeCandidate[] {
  const candidates = group.map((event) => positionsForMidi(
    event.midi,
    options.maximumFret,
    setup.tuning,
    setup.capo,
  ));
  const shapes: ShapeCandidate[] = [];
  const search = (index: number, previousString: number, shape: Position[]) => {
    if (index === candidates.length) {
      const activeFrets = shape.map((position) => position.fret).filter((fret) => fret > 0);
      const spread = activeFrets.length
        ? Math.max(...activeFrets) - Math.min(...activeFrets)
        : 0;
      if (spread > options.maximumChordStretchFrets) return;
      const center = activeFrets.length
        ? activeFrets.reduce((sum, fret) => sum + fret, 0) / activeFrets.length
        : 0;
      const stringCenter = shape.reduce((sum, position) => sum + position.string, 0) / shape.length;
      const outsidePreferred = activeFrets.reduce((penalty, fret) => (
        penalty
        + Math.max(0, options.preferredFretRange[0] - fret)
        + Math.max(0, fret - options.preferredFretRange[1])
      ), 0);
      const openStrings = shape.filter((position) => position.fret === 0).length;
      const knownVoicingBonus = KNOWN_VOICINGS.has(shapeKey(shape)) ? 3.5 : 0;
      shapes.push({
        positions: shape.map((position) => ({ ...position })),
        center,
        stringCenter,
        localCost: spread * 1.5
          + outsidePreferred * 2
          + Math.max(0, center - 12) * 0.15
          - openStrings * options.openStringBonus
          - knownVoicingBonus,
      });
      return;
    }
    for (const position of candidates[index]) {
      if (position.string >= previousString) continue;
      search(index + 1, position.string, [...shape, position]);
    }
  };
  search(0, 7, []);
  return shapes.sort((a, b) => a.localCost - b.localCost).slice(0, 48);
}

function shapeTransitionCost(
  prior: ShapeCandidate,
  current: ShapeCandidate,
  gap: number,
  options: FretboardMappingOptions,
): number {
  const movementWeight = gap > 1 ? 0.45 : 1;
  const positionShift = Math.abs(current.center - prior.center);
  const largeShift = Math.max(0, positionShift - 4);
  return movementWeight * (
    positionShift * 0.75
    + largeShift * options.largePositionShiftPenalty
    + Math.abs(current.stringCenter - prior.stringCenter) * options.stringChangePenalty
  );
}

/**
 * Plans onset-group shapes across the complete phrase. Every group uses
 * distinct strings and an ordinary active-fret stretch of at most four frets.
 */
export function mapPolyphonicNotesToFretboard(
  events: Omit<NoteEvent, "string" | "fret">[],
  onsetTolerance = 0.045,
  setup: Pick<FretboardSetup, "tuning" | "capo"> = { tuning: STANDARD_TUNING, capo: 0 },
  partialOptions: Partial<FretboardMappingOptions> = {},
): NoteEvent[] {
  const options = { ...DEFAULT_MAPPING_OPTIONS, ...partialOptions };
  const sorted = events
    .filter((event) => positionsForMidi(
      event.midi,
      options.maximumFret,
      setup.tuning,
      setup.capo,
    ).length > 0)
    .sort((a, b) => a.start - b.start || a.midi - b.midi);
  const groups: typeof sorted[] = [];
  for (const event of sorted) {
    const group = groups.at(-1);
    if (!group || event.start - group[0].start > onsetTolerance) groups.push([event]);
    else group.push(event);
  }

  const plannedGroups: typeof sorted[] = [];
  const allCandidates: ShapeCandidate[][] = [];
  for (const originalGroup of groups) {
    let group = [...originalGroup]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, options.maximumPolyphony)
      .sort((a, b) => a.midi - b.midi);
    let candidates = shapeCandidates(group, setup, options);
    // If every detected pitch cannot form one physical shape, discard the
    // weakest candidate until the attack is playable instead of mapping it as
    // overlapping independent notes.
    while (!candidates.length && group.length > 1) {
      const weakest = group.reduce(
        (best, event, index) => event.confidence < group[best].confidence ? index : best,
        0,
      );
      group = group.filter((_, index) => index !== weakest);
      candidates = shapeCandidates(group, setup, options);
    }
    if (!candidates.length) continue;
    plannedGroups.push(group);
    allCandidates.push(candidates);
  }
  if (!plannedGroups.length) return [];

  const costs = allCandidates.map((candidates) => candidates.map(() => Infinity));
  const previous = allCandidates.map((candidates) => candidates.map(() => -1));
  allCandidates[0].forEach((candidate, index) => { costs[0][index] = candidate.localCost; });
  for (let groupIndex = 1; groupIndex < allCandidates.length; groupIndex += 1) {
    const gap = plannedGroups[groupIndex][0].start - plannedGroups[groupIndex - 1][0].end;
    allCandidates[groupIndex].forEach((candidate, candidateIndex) => {
      allCandidates[groupIndex - 1].forEach((prior, priorIndex) => {
        const cost = costs[groupIndex - 1][priorIndex]
          + candidate.localCost
          + shapeTransitionCost(prior, candidate, gap, options);
        if (cost < costs[groupIndex][candidateIndex]) {
          costs[groupIndex][candidateIndex] = cost;
          previous[groupIndex][candidateIndex] = priorIndex;
        }
      });
    });
  }

  let cursor = costs.at(-1)!.indexOf(Math.min(...costs.at(-1)!));
  const path = Array<number>(plannedGroups.length);
  for (let groupIndex = plannedGroups.length - 1; groupIndex >= 0; groupIndex -= 1) {
    path[groupIndex] = cursor;
    cursor = previous[groupIndex][cursor];
  }
  return plannedGroups.flatMap((group, groupIndex) => {
    const positions = allCandidates[groupIndex][path[groupIndex]].positions;
    return group.map((event, eventIndex) => ({ ...event, ...positions[eventIndex] }));
  }).sort((a, b) => a.start - b.start || a.midi - b.midi);
}

export function fretMovementDiagnostics(
  notes: Pick<NoteEvent, "start" | "fret">[],
  onsetTolerance = 0.045,
): { average: number; maximum: number } {
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  const groups: typeof sorted[] = [];
  for (const note of sorted) {
    const group = groups.at(-1);
    if (!group || note.start - group[0].start > onsetTolerance) groups.push([note]);
    else group.push(note);
  }
  const centers = groups.map((group) => {
    const active = group.map((note) => note.fret).filter((fret) => fret > 0);
    return active.length ? active.reduce((sum, fret) => sum + fret, 0) / active.length : 0;
  });
  const movements = centers.slice(1).map((center, index) => Math.abs(center - centers[index]));
  return {
    average: movements.length
      ? movements.reduce((sum, movement) => sum + movement, 0) / movements.length
      : 0,
    maximum: movements.length ? Math.max(...movements) : 0,
  };
}

export function clipNoteSustains(
  notes: NoteEvent[],
  bpm: number | null,
  lowConfidenceMaximumSeconds = 0.5,
): { notes: NoteEvent[]; overlapsResolved: number } {
  const result = notes.map((note) => ({ ...note }));
  const safeBpm = bpm && bpm >= 40 && bpm <= 240 ? bpm : 120;
  const barDuration = (60 / safeBpm) * 4;
  let overlapsResolved = 0;
  for (let string = 1; string <= 6; string += 1) {
    const stringNotes = result
      .filter((note) => note.string === string)
      .sort((a, b) => a.start - b.start || b.confidence - a.confidence);
    for (let index = 0; index < stringNotes.length; index += 1) {
      const note = stringNotes[index];
      const next = stringNotes[index + 1];
      const barEnd = (Math.floor(note.start / barDuration) + 1) * barDuration;
      let end = Math.min(note.end, barEnd);
      if (note.confidence < 0.4) {
        end = Math.min(end, note.start + lowConfidenceMaximumSeconds);
      }
      if (next && end > next.start) {
        end = next.start;
        overlapsResolved += 1;
      }
      note.end = Math.max(note.start + 0.02, end);
    }
  }
  return {
    notes: result.sort((a, b) => a.start - b.start || a.midi - b.midi),
    overlapsResolved,
  };
}

const SETUP_CANDIDATES: FretboardSetup[] = [
  { name: "Standard", tuning: STANDARD_TUNING, tuningNames: TUNING_NAMES, capo: 0 },
  { name: "Drop D", tuning: [38, 45, 50, 55, 59, 64], tuningNames: ["D2", "A2", "D3", "G3", "B3", "E4"], capo: 0 },
  { name: "E♭ standard", tuning: [39, 44, 49, 54, 58, 63], tuningNames: ["E♭2", "A♭2", "D♭3", "G♭3", "B♭3", "E♭4"], capo: 0 },
  ...[1, 2, 3, 4, 5].map((capo) => ({ name: `Standard · capo ${capo}`, tuning: STANDARD_TUNING, tuningNames: TUNING_NAMES, capo })),
];

/** Conservative setup estimate: only leaves standard when at least two candidate open strings are heard repeatedly. */
export function estimateGuitarSetup(events: Pick<NoteEvent, "midi" | "confidence" | "start" | "end">[]): FretboardSetup {
  if (!events.length) return SETUP_CANDIDATES[0];
  const score = (setup: FretboardSetup) => {
    const soundingOpen = setup.tuning.map((midi) => midi + setup.capo);
    const openEvidence = new Set<number>();
    let value = -setup.capo * 0.2;
    for (const event of events) {
      const weight = Math.max(0.05, event.end - event.start) * Math.max(0.1, event.confidence);
      const positions = positionsForMidi(event.midi, 20, setup.tuning, setup.capo);
      if (!positions.length) { value -= weight * 8; continue; }
      const minimumFret = Math.min(...positions.map((position) => position.fret));
      value += weight * (2 - minimumFret * 0.06);
      if (soundingOpen.includes(event.midi)) {
        value += weight * 2.4;
        openEvidence.add(event.midi);
      }
    }
    return { value, openStrings: openEvidence.size };
  };
  const standard = score(SETUP_CANDIDATES[0]);
  const ranked = SETUP_CANDIDATES.slice(1).map((setup) => ({ setup, ...score(setup) })).sort((a, b) => b.value - a.value);
  const best = ranked[0];
  return best && best.openStrings >= 2 && best.value > standard.value * 1.12 ? best.setup : SETUP_CANDIDATES[0];
}
