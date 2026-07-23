const PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const PITCH_CLASS_INDEX = new Map([
  ["C", 0], ["B#", 0],
  ["C#", 1], ["DB", 1],
  ["D", 2],
  ["D#", 3], ["EB", 3],
  ["E", 4], ["FB", 4],
  ["E#", 5], ["F", 5],
  ["F#", 6], ["GB", 6],
  ["G", 7],
  ["G#", 8], ["AB", 8],
  ["A", 9],
  ["A#", 10], ["BB", 10],
  ["B", 11], ["CB", 11],
]);

const CHORD_PATTERN = /^(N|([A-Ga-g])([#b♯♭]?)(maj7|min7|m7|maj|minor|min|m|7|sus2|sus4|add9|dim|°|aug|\+|5)?(?:\/([A-Ga-g])([#b♯♭]?))?)$/;
const SECTION_PATTERN = /^[A-Za-z][A-Za-z0-9 '&()/-]{0,48}$/;
const TIME_PATTERN = /^(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s+(.+)$/;

function safeDivide(numerator, denominator) {
  return denominator ? numerator / denominator : 0;
}

function round(value, places = 4) {
  const factor = 10 ** places;
  return Math.round(value * factor) / factor;
}

function cleanAccidental(value = "") {
  return value.replace("♯", "#").replace("♭", "b");
}

function canonicalPitchClass(letter, accidental = "", semitones = 0) {
  const token = `${String(letter).toUpperCase()}${cleanAccidental(accidental).toUpperCase()}`;
  const index = PITCH_CLASS_INDEX.get(token);
  if (index === undefined) throw new Error(`Unsupported pitch class: ${letter}${accidental}`);
  return PITCH_CLASSES[(index + semitones % 12 + 12) % 12];
}

function normalizeQuality(raw = "") {
  const quality = raw.toLowerCase();
  if (!quality || quality === "maj") return "maj";
  if (quality === "minor" || quality === "min" || quality === "m") return "min";
  if (quality === "m7" || quality === "min7") return "min7";
  if (quality === "maj7") return "maj7";
  if (quality === "°") return "dim";
  if (quality === "+") return "aug";
  return quality;
}

function detailedSuffix(quality) {
  return {
    maj: "",
    min: "m",
    min7: "m7",
    maj7: "maj7",
    7: "7",
    sus2: "sus2",
    sus4: "sus4",
    add9: "add9",
    dim: "dim",
    aug: "aug",
    5: "5",
  }[quality] ?? quality;
}

function majorMinorFamily(quality) {
  if (quality === "min" || quality === "min7") return "min";
  if (quality === "dim") return "dim";
  if (quality === "N") return "N";
  return "maj";
}

export function normalizeChordSymbol(symbol, { capo = 0 } = {}) {
  const compact = String(symbol || "").trim().replace(/\s+/g, "");
  const match = compact.match(CHORD_PATTERN);
  if (!match) throw new Error(`Unsupported chord symbol "${symbol}". Enter chord symbols only; lyrics are not accepted.`);
  if (match[1].toUpperCase() === "N") {
    return {
      displayed: "N",
      sounding: "N",
      root: null,
      quality: "N",
      bass: null,
      rootOnly: "N",
      majorMinor: "N",
      detailed: "N",
    };
  }
  const transposition = Number.isInteger(Number(capo)) ? Number(capo) : 0;
  const displayedRoot = canonicalPitchClass(match[2], match[3]);
  const root = canonicalPitchClass(match[2], match[3], transposition);
  const quality = normalizeQuality(match[4]);
  const displayedBass = match[5] ? canonicalPitchClass(match[5], match[6]) : null;
  const bass = match[5] ? canonicalPitchClass(match[5], match[6], transposition) : null;
  const suffix = detailedSuffix(quality);
  const displayed = `${displayedRoot}${suffix}${displayedBass ? `/${displayedBass}` : ""}`;
  const sounding = `${root}${suffix}${bass ? `/${bass}` : ""}`;
  return {
    displayed,
    sounding,
    root,
    quality,
    bass,
    rootOnly: root,
    majorMinor: `${root}:${majorMinorFamily(quality)}`,
    detailed: `${root}${suffix}`,
  };
}

function parseTimestamp(hours, minutes, seconds) {
  return Number(hours || 0) * 3600 + Number(minutes) * 60 + Number(seconds);
}

function splitChordTokens(value) {
  const normalized = value
    .replace(/[|,]/g, " ")
    .replace(/\s+x(\d+)\s*$/i, " x$1")
    .trim();
  if (!normalized) return { tokens: [], repeats: 1 };
  const parts = normalized.split(/\s+/);
  let repeats = 1;
  const repeat = parts.at(-1)?.match(/^x(\d+)$/i);
  if (repeat) {
    repeats = Math.max(1, Math.min(16, Number(repeat[1])));
    parts.pop();
  }
  return { tokens: parts, repeats };
}

/**
 * Parse a deliberately narrow chord-only format. Any unrecognized text rejects
 * the whole import so the original page body or lyrics can never be persisted.
 */
export function parseChordReferenceText(text, { capo = 0, duration = null } = {}) {
  if (typeof text !== "string" || !text.trim()) throw new Error("Enter at least one chord.");
  if (text.length > 50_000) throw new Error("Chord reference is too large for the manual importer.");
  if (/<\/?[a-z][\s\S]*>/i.test(text)) throw new Error("HTML/page content is not accepted. Paste chord symbols only.");

  const events = [];
  let currentSection = "Unsectioned";
  for (const [lineIndex, sourceLine] of text.split(/\r?\n/).entries()) {
    let line = sourceLine.trim();
    if (!line) continue;
    const bracketSection = line.match(/^\[([^\]]+)]$/);
    if (bracketSection) {
      if (!SECTION_PATTERN.test(bracketSection[1])) throw new Error(`Invalid section label on line ${lineIndex + 1}.`);
      currentSection = bracketSection[1].trim();
      continue;
    }

    let timestamp = null;
    const timed = line.match(TIME_PATTERN);
    if (timed) {
      timestamp = parseTimestamp(timed[1], timed[2], timed[3]);
      line = timed[4].trim();
    }

    const colon = line.indexOf(":");
    if (colon >= 0 && timestamp === null) {
      const possibleSection = line.slice(0, colon).trim();
      if (!SECTION_PATTERN.test(possibleSection)) throw new Error(`Invalid section label on line ${lineIndex + 1}.`);
      currentSection = possibleSection;
      line = line.slice(colon + 1).trim();
    }

    const { tokens, repeats } = splitChordTokens(line);
    if (!tokens.length) continue;
    const parsed = tokens.map((token) => normalizeChordSymbol(token, { capo }));
    for (let repeat = 0; repeat < repeats; repeat += 1) {
      for (const chord of parsed) {
        events.push({
          order: events.length,
          section: currentSection,
          start: timestamp !== null && repeat === 0 && parsed.indexOf(chord) === 0 ? timestamp : null,
          ...chord,
        });
      }
    }
  }
  if (!events.length) throw new Error("No valid chord symbols were found.");

  const timedEvents = events.filter((event) => Number.isFinite(event.start));
  let alignment = null;
  if (timedEvents.length === events.length) {
    const starts = events.map((event) => event.start);
    if (starts.some((start, index) => index > 0 && start <= starts[index - 1])) {
      throw new Error("Timed chord starts must increase monotonically.");
    }
    alignment = {
      method: "manual",
      verified: true,
      confidence: 1,
      regions: events.map((event, index) => ({
        start: event.start,
        end: index + 1 < events.length
          ? events[index + 1].start
          : Math.max(event.start + 0.1, Number(duration) || event.start + 4),
        chord: event.sounding,
        chordIndex: index,
      })),
    };
  } else if (timedEvents.length > 0) {
    throw new Error("Either timestamp every chord or use an untimed chord sequence.");
  }
  return { chords: events, alignment };
}

function normalizeReferenceUrl(value, source) {
  if (!value) return "";
  let url;
  try {
    url = new URL(value);
  } catch {
    throw new Error("Reference URL must be a valid http(s) URL.");
  }
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("Reference URL must use http or https.");
  if (source === "ultimate-guitar") {
    const host = url.hostname.toLowerCase();
    if (host !== "ultimate-guitar.com" && !host.endsWith(".ultimate-guitar.com")) {
      throw new Error("An Ultimate Guitar reference must use an ultimate-guitar.com URL.");
    }
  }
  url.hash = "";
  return url.toString();
}

export function createManualReference(input) {
  const source = input.source === "timed-research" ? "timed-research" : "ultimate-guitar";
  const artist = String(input.artist || "").trim();
  const title = String(input.title || "").trim();
  if (!artist || !title) throw new Error("Artist and song title are required.");
  const capo = Math.max(0, Math.min(24, Number(input.capo) || 0));
  const parsed = parseChordReferenceText(input.chordSequence, {
    capo,
    duration: input.duration,
  });
  const now = new Date().toISOString();
  return {
    schemaVersion: 1,
    source,
    artist: artist.slice(0, 160),
    title: title.slice(0, 160),
    referenceUrl: normalizeReferenceUrl(input.referenceUrl, source),
    version: Math.max(0, Number(input.version) || 0),
    type: "Chords",
    ratingCount: Math.max(0, Number(input.ratingCount) || 0),
    averageRating: Math.max(0, Math.min(5, Number(input.averageRating) || 0)),
    selectionReason: String(input.selectionReason || "Manually selected chord reference").slice(0, 400),
    capo,
    tuning: String(input.tuning || "Standard").slice(0, 80),
    key: String(input.key || "").slice(0, 32),
    genre: String(input.genre || "Unspecified").slice(0, 80),
    recordingNotes: String(input.recordingNotes || "").slice(0, 500),
    split: ["development", "validation", "test"].includes(input.split) ? input.split : "development",
    chords: parsed.chords,
    alignment: parsed.alignment,
    createdAt: now,
    updatedAt: now,
  };
}

export function selectReferenceVersion(candidates) {
  const eligible = (candidates || []).filter((candidate) =>
    String(candidate.type || "").toLowerCase() === "chords");
  if (!eligible.length) return null;
  const ranked = [...eligible].sort((a, b) =>
    (Number(b.ratingCount) || 0) - (Number(a.ratingCount) || 0)
    || (Number(b.averageRating) || 0) - (Number(a.averageRating) || 0)
    || (Number(b.completeness) || 0) - (Number(a.completeness) || 0)
    || (Number(a.version) || 0) - (Number(b.version) || 0));
  return {
    ...ranked[0],
    selectionReason: "Largest rating count among Chords versions; average rating and completeness used as tie-breakers.",
  };
}

function chordDistance(referenceSymbol, detectedSymbol) {
  const reference = normalizeChordSymbol(referenceSymbol);
  const detected = normalizeChordSymbol(detectedSymbol);
  if (reference.detailed === detected.detailed) return 0;
  if (reference.root === detected.root) return reference.majorMinor === detected.majorMinor ? 0.18 : 0.34;
  if (reference.root === null || detected.root === null) return 0.9;
  const referenceIndex = PITCH_CLASSES.indexOf(reference.root);
  const detectedIndex = PITCH_CLASSES.indexOf(detected.root);
  const distance = (detectedIndex - referenceIndex + 12) % 12;
  if (
    (majorMinorFamily(reference.quality) === "min" && distance === 3 && majorMinorFamily(detected.quality) === "maj")
    || (majorMinorFamily(reference.quality) === "maj" && distance === 9 && majorMinorFamily(detected.quality) === "min")
  ) return 0.48;
  if (distance === 5 || distance === 7) return 0.65;
  return 1;
}

export function alignChordSequences(referenceChords, detectedRegions) {
  const reference = referenceChords.map((chord) => chord.sounding || chord.chord || chord.name);
  const detected = detectedRegions.map((chord) => chord.name || chord.chord);
  const insertionCost = 0.72;
  const rows = reference.length + 1;
  const columns = detected.length + 1;
  const costs = Array.from({ length: rows }, () => Array(columns).fill(0));
  const moves = Array.from({ length: rows }, () => Array(columns).fill(null));
  for (let i = 1; i < rows; i += 1) { costs[i][0] = i * insertionCost; moves[i][0] = "missing"; }
  for (let j = 1; j < columns; j += 1) { costs[0][j] = j * insertionCost; moves[0][j] = "extra"; }
  for (let i = 1; i < rows; i += 1) {
    for (let j = 1; j < columns; j += 1) {
      const options = [
        { cost: costs[i - 1][j - 1] + chordDistance(reference[i - 1], detected[j - 1]), move: "match" },
        { cost: costs[i - 1][j] + insertionCost, move: "missing" },
        { cost: costs[i][j - 1] + insertionCost, move: "extra" },
      ];
      options.sort((a, b) => a.cost - b.cost);
      costs[i][j] = options[0].cost;
      moves[i][j] = options[0].move;
    }
  }
  const pairs = [];
  let i = reference.length;
  let j = detected.length;
  while (i > 0 || j > 0) {
    const move = moves[i][j];
    if (move === "match") {
      pairs.unshift({
        referenceIndex: i - 1,
        detectedIndex: j - 1,
        cost: round(chordDistance(reference[i - 1], detected[j - 1])),
      });
      i -= 1;
      j -= 1;
    } else if (move === "missing") {
      pairs.unshift({ referenceIndex: i - 1, detectedIndex: null, cost: insertionCost });
      i -= 1;
    } else {
      pairs.unshift({ referenceIndex: null, detectedIndex: j - 1, cost: insertionCost });
      j -= 1;
    }
  }
  const maximumCost = Math.max(1, Math.max(reference.length, detected.length) * insertionCost);
  return {
    method: "sequence",
    verified: false,
    confidence: round(Math.max(0, 1 - costs.at(-1).at(-1) / maximumCost)),
    cost: round(costs.at(-1).at(-1)),
    pairs,
  };
}

export function sequenceAlignmentToRegions(reference, detectedRegions, duration) {
  const alignment = alignChordSequences(reference.chords, detectedRegions);
  const matched = new Map(alignment.pairs
    .filter((pair) => pair.referenceIndex !== null && pair.detectedIndex !== null)
    .map((pair) => [pair.referenceIndex, pair.detectedIndex]));
  const fallbackStep = Math.max(0.1, Number(duration) / Math.max(1, reference.chords.length));
  const starts = reference.chords.map((_chord, index) => {
    const detectedIndex = matched.get(index);
    return detectedIndex === undefined ? null : detectedRegions[detectedIndex].start;
  });
  for (let index = 0; index < starts.length; index += 1) {
    if (starts[index] !== null) continue;
    let previous = index - 1;
    while (previous >= 0 && starts[previous] === null) previous -= 1;
    let next = index + 1;
    while (next < starts.length && starts[next] === null) next += 1;
    if (previous >= 0 && next < starts.length) {
      starts[index] = starts[previous] + (starts[next] - starts[previous]) * ((index - previous) / (next - previous));
    } else if (previous >= 0) {
      starts[index] = starts[previous] + fallbackStep * (index - previous);
    } else if (next < starts.length) {
      starts[index] = Math.max(0, starts[next] - fallbackStep * (next - index));
    } else {
      starts[index] = fallbackStep * index;
    }
  }
  const regions = reference.chords.map((chord, index) => ({
    start: Math.max(0, starts[index]),
    end: index + 1 < starts.length
      ? Math.max(starts[index] + 0.05, starts[index + 1])
      : Math.max(starts[index] + 0.05, Number(duration) || detectedRegions.at(-1)?.end || starts[index] + fallbackStep),
    chord: chord.sounding,
    chordIndex: index,
  }));
  return { ...alignment, regions };
}

export function alignReferenceToBeatGrid(reference, beatTimes, {
  duration = null,
  beatsPerChord = 4,
} = {}) {
  const beats = [...new Set((beatTimes || []).filter(Number.isFinite))].sort((a, b) => a - b);
  if (!beats.length) throw new Error("Beat alignment requires a beat grid.");
  const step = Math.max(1, Math.round(beatsPerChord));
  const regions = reference.chords.map((chord, index) => {
    const beatIndex = Math.min(index * step, beats.length - 1);
    const nextIndex = Math.min((index + 1) * step, beats.length - 1);
    const start = beats[beatIndex];
    const end = index + 1 < reference.chords.length
      ? Math.max(start + 0.05, beats[nextIndex])
      : Math.max(start + 0.05, Number(duration) || beats.at(-1));
    return { start, end, chord: chord.sounding, chordIndex: index };
  });
  return { method: "beat-section", verified: false, confidence: 0.45, regions };
}

export function createManualAlignment(reference, marks, duration) {
  const ordered = [...(marks || [])].sort((a, b) => a.chordIndex - b.chordIndex);
  if (ordered.length !== reference.chords.length) {
    throw new Error(`Mark all ${reference.chords.length} reference chords before saving.`);
  }
  if (ordered.some((mark, index) =>
    mark.chordIndex !== index
    || !Number.isFinite(mark.start)
    || (index > 0 && mark.start <= ordered[index - 1].start))) {
    throw new Error("Manual chord marks must be complete, ordered, and monotonic.");
  }
  return {
    method: "manual",
    verified: true,
    confidence: 1,
    regions: ordered.map((mark, index) => ({
      start: mark.start,
      end: index + 1 < ordered.length
        ? ordered[index + 1].start
        : Math.max(mark.start + 0.05, Number(duration) || mark.start + 4),
      chord: reference.chords[index].sounding,
      chordIndex: index,
    })),
  };
}

function chordAt(regions, time) {
  return regions.find((region) => time >= region.start && time < region.end) ?? null;
}

function intervalLabel(symbol) {
  return normalizeChordSymbol(symbol || "N");
}

function extensionType(quality) {
  return ["7", "maj7", "min7"].includes(quality) ? quality : null;
}

function windowAt(prediction, time) {
  return prediction.chordAnalysis?.windows?.find((window) => time >= window.start && time < window.end)
    ?? prediction.chordAnalysis?.windows?.reduce((closest, window) => {
      const distance = Math.abs((window.start + window.end) / 2 - time);
      return !closest || distance < closest.distance ? { window, distance } : closest;
    }, null)?.window
    ?? null;
}

function classifyMismatch(reference, detected, diagnostic, context = {}) {
  const evidence = {
    topCandidateChord: diagnostic?.topCandidateChord ?? null,
    topCandidateScore: diagnostic?.topCandidateScore ?? null,
    secondBestChord: diagnostic?.secondBestChord ?? null,
    secondBestScore: diagnostic?.secondBestScore ?? null,
    bassRoot: diagnostic?.bassRootEstimate ?? null,
    bassRootConfidence: diagnostic?.bassRootConfidence ?? null,
    detectedRoot: detected.root,
    scoreMargin: diagnostic?.scoreMargin ?? null,
    confidence: diagnostic?.topCandidateScore ?? null,
    keyEstimate: diagnostic?.keyEstimate ?? null,
    beatStrength: diagnostic?.beatStrength ?? null,
    usedBassSupport: diagnostic?.usedBassSupport ?? false,
    usedSmoothingOverride: diagnostic?.usedSmoothingOverride ?? false,
  };
  if (context.inversionMismatch) {
    return { errorType: "inversion collapsed", likelyFix: "Compare the harmonic chord and bass inversion as separate targets.", evidence };
  }
  if (context.capoMismatch) {
    return { errorType: "wrong capo normalization", likelyFix: "Correct the reference capo before evaluating concert-pitch chords.", evidence };
  }
  if (context.tuningMismatch) {
    return { errorType: "wrong tuning", likelyFix: "Verify the reference and recording tuning before comparison.", evidence };
  }
  if (reference.root === null && detected.root !== null) {
    return { errorType: "missed N", likelyFix: "Raise the no-harmony evidence requirement for this passage.", evidence };
  }
  if (reference.root !== null && detected.root === null) {
    return {
      errorType: diagnostic?.bassRootEstimate === reference.root ? "bass root ignored" : "false N",
      likelyFix: diagnostic?.bassRootEstimate === reference.root
        ? "Increase bass-root continuity when local chord templates are ambiguous."
        : "Reduce unnecessary N decisions when neighbouring harmony is supported.",
      evidence,
    };
  }
  if (context.boundaryNearby) {
    return { errorType: "boundary snapped to wrong beat", likelyFix: "Review beat/downbeat snapping at this transition.", evidence };
  }
  if (reference.root === detected.root) {
    if (!extensionType(reference.quality) && extensionType(detected.quality)) {
      return { errorType: "transient seventh detected", likelyFix: "Require sustained extension evidence.", evidence };
    }
    if (extensionType(reference.quality) && !extensionType(detected.quality)) {
      return { errorType: "seventh omitted", likelyFix: "Retain consistently supported sevenths in Detailed mode.", evidence };
    }
    return {
      errorType: (diagnostic?.scoreMargin ?? 1) < 0.12 ? "weak third caused major/minor ambiguity" : "wrong chord quality",
      likelyFix: "Use broader harmonic and third evidence before choosing chord quality.",
      evidence,
    };
  }
  const referenceIndex = PITCH_CLASSES.indexOf(reference.root);
  const detectedIndex = PITCH_CLASSES.indexOf(detected.root);
  const distance = (detectedIndex - referenceIndex + 12) % 12;
  const referenceFamily = majorMinorFamily(reference.quality);
  const detectedFamily = majorMinorFamily(detected.quality);
  if (
    (referenceFamily === "min" && detectedFamily === "maj" && distance === 3)
    || (referenceFamily === "maj" && detectedFamily === "min" && distance === 9)
  ) {
    return { errorType: "relative major/minor confusion", likelyFix: "Use bass-root and third evidence to resolve the relative pair.", evidence };
  }
  if (diagnostic?.bassRootEstimate === reference.root && detected.root !== reference.root) {
    return { errorType: "bass root ignored", likelyFix: "Increase bass-root prior when chord-template scores are close.", evidence };
  }
  if (context.recordingMismatch) {
    return { errorType: "different recording/version", likelyFix: "Verify the reference against the exact audio recording.", evidence };
  }
  if (context.referenceSimplified) {
    return { errorType: "chord sheet simplified the recording", likelyFix: "Review this mismatch manually before changing the detector.", evidence };
  }
  return {
    errorType: diagnostic?.usedBassSupport ? "lead melody contaminated harmony" : "wrong root",
    likelyFix: "Inspect stem leakage, bass evidence, and local harmony candidates.",
    evidence,
  };
}

function nearestBoundaryError(boundary, predictions) {
  if (!predictions.length) return null;
  return predictions.reduce((best, candidate) =>
    Math.abs(candidate - boundary) < Math.abs(best - boundary) ? candidate : best) - boundary;
}

function nearestBoundary(boundary, predictions) {
  const error = nearestBoundaryError(boundary, predictions);
  return Number.isFinite(error) ? boundary + error : null;
}

function regionBoundaries(regions, start, end) {
  return (regions || [])
    .map((region) => region.start)
    .filter((boundary) => boundary > start + 1e-6 && boundary < end - 1e-6);
}

function frameWinnerBoundaries(frames, start, end) {
  const boundaries = [];
  let previous = null;
  for (const frame of frames || []) {
    if (frame.end <= start || frame.start >= end) continue;
    if (previous !== null && frame.bestChord !== previous && frame.start > start) {
      boundaries.push(frame.start);
    }
    previous = frame.bestChord;
  }
  return boundaries;
}

function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function evaluateChordReference(reference, prediction, {
  alignment = reference.alignment,
  context = {},
} = {}) {
  if (!alignment?.regions?.length) {
    alignment = sequenceAlignmentToRegions(reference, prediction.chords || [], prediction.duration);
  }
  const referenceRegions = alignment.regions;
  const evaluationStart = Math.min(...referenceRegions.map((region) => region.start));
  const evaluationEnd = Math.max(...referenceRegions.map((region) => region.end));
  const predictedRegions = (prediction.chords || []).filter((region) =>
    region.end > evaluationStart && region.start < evaluationEnd);
  const boundaries = new Set();
  for (const region of [...referenceRegions, ...predictedRegions]) {
    boundaries.add(region.start);
    boundaries.add(region.end);
  }
  const points = [...boundaries].filter(Number.isFinite).sort((a, b) => a - b);
  let totalDuration = 0;
  let rootCorrect = 0;
  let familyCorrect = 0;
  let detailCorrect = 0;
  let referenceNoChordDuration = 0;
  let predictedNoChordDuration = 0;
  let noChordTruePositiveDuration = 0;
  let inversionDuration = 0;
  let inversionCorrectDuration = 0;
  const mismatches = [];
  const confusionDuration = new Map();

  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (end <= start) continue;
    const midpoint = (start + end) / 2;
    const expectedRegion = chordAt(referenceRegions, midpoint);
    if (!expectedRegion) continue;
    const detectedRegion = chordAt(predictedRegions, midpoint);
    const expected = intervalLabel(expectedRegion.chord || expectedRegion.name);
    const detected = intervalLabel(detectedRegion?.name || "N");
    const diagnostic = windowAt(prediction, midpoint);
    const duration = end - start;
    totalDuration += duration;
    if (expected.rootOnly === detected.rootOnly) rootCorrect += duration;
    if (expected.majorMinor === detected.majorMinor) familyCorrect += duration;
    if (expected.detailed === detected.detailed) detailCorrect += duration;
    if (expected.root === null) referenceNoChordDuration += duration;
    if (detected.root === null) predictedNoChordDuration += duration;
    if (expected.root === null && detected.root === null) noChordTruePositiveDuration += duration;
    const detectedBass = detected.bass
      ?? (diagnostic?.bassRootConfidence >= 0.35 ? diagnostic.bassRootEstimate : null);
    const inversionMismatch = expected.bass !== null && detectedBass !== expected.bass;
    if (expected.bass !== null) {
      inversionDuration += duration;
      if (!inversionMismatch) inversionCorrectDuration += duration;
    }
    if (expected.detailed !== detected.detailed || inversionMismatch) {
      const boundaryNearby = Math.min(
        Math.abs(midpoint - expectedRegion.start),
        Math.abs(expectedRegion.end - midpoint),
      ) <= 0.6;
      const expectedIndex = referenceRegions.indexOf(expectedRegion);
      const adjacentReferenceLabels = [
        referenceRegions[expectedIndex - 1],
        referenceRegions[expectedIndex + 1],
      ].filter(Boolean).map((region) => intervalLabel(region.chord || region.name).detailed);
      const classified = classifyMismatch(expected, detected, diagnostic, {
        ...context,
        inversionMismatch: expected.detailed === detected.detailed && inversionMismatch,
        boundaryNearby: expected.detailed !== detected.detailed
          && boundaryNearby
          && adjacentReferenceLabels.includes(detected.detailed),
      });
      const key = `${expected.detailed} → ${detected.detailed}`;
      confusionDuration.set(key, (confusionDuration.get(key) || 0) + duration);
      const previous = mismatches.at(-1);
      if (
        previous
        && previous.reference === expected.detailed
        && previous.detected === detected.detailed
        && previous.errorType === classified.errorType
        && Math.abs(previous.end - start) < 0.001
      ) {
        previous.end = end;
        previous.duration = round(previous.end - previous.start, 3);
      } else {
        mismatches.push({
          start,
          end,
          duration: round(duration, 3),
          reference: expected.detailed,
          detected: detected.detailed,
          ...classified,
        });
      }
    }
  }

  const referenceBoundaries = referenceRegions.slice(1).map((region) => region.start);
  const predictedBoundaries = predictedRegions.slice(1).map((region) => region.start);
  const boundaryErrors = referenceBoundaries
    .map((boundary) => nearestBoundaryError(boundary, predictedBoundaries))
    .filter(Number.isFinite);
  const earlyBoundaryErrors = boundaryErrors.filter((error) => error < 0);
  const lateBoundaryErrors = boundaryErrors.filter((error) => error > 0);
  const bpm = Number(prediction.bpm ?? context.bpm);
  const beatDuration = Number.isFinite(bpm) && bpm > 0 ? 60 / bpm : null;
  const alternatives = prediction.chordAnalysis?.decoderAlternatives ?? {};
  const rawWinnerBoundaries = frameWinnerBoundaries(
    prediction.chordAnalysis?.rawFrames,
    evaluationStart,
    evaluationEnd,
  );
  const beatWinnerBoundaries = regionBoundaries(
    alternatives.beatLevelWinner,
    evaluationStart,
    evaluationEnd,
  );
  const viterbiBoundaries = regionBoundaries(
    alternatives.viterbi,
    evaluationStart,
    evaluationEnd,
  );
  const postHysteresisBoundaries = regionBoundaries(
    alternatives.postHysteresis,
    evaluationStart,
    evaluationEnd,
  );
  const postProcessingBoundaries = regionBoundaries(
    alternatives.production,
    evaluationStart,
    evaluationEnd,
  );
  const reducedLatencyBoundaries = regionBoundaries(
    alternatives.reducedLatency,
    evaluationStart,
    evaluationEnd,
  );
  const boundaryStages = referenceBoundaries.map((boundary) => ({
    referenceBoundary: round(boundary, 3),
    rawBestCandidateBoundary: nearestBoundary(boundary, rawWinnerBoundaries),
    beatLevelWinnerBoundary: nearestBoundary(boundary, beatWinnerBoundaries),
    viterbiBoundary: nearestBoundary(boundary, viterbiBoundaries),
    hysteresisBoundary: nearestBoundary(boundary, postHysteresisBoundaries),
    postProcessingBoundary: nearestBoundary(boundary, postProcessingBoundaries),
    finalRenderedBoundary: nearestBoundary(boundary, predictedBoundaries),
    reducedLatencyBoundary: nearestBoundary(boundary, reducedLatencyBoundaries),
  }));
  const referenceSymbols = new Set(referenceRegions.map((region) => intervalLabel(region.chord || region.name).detailed));
  const detectedSymbols = new Set(predictedRegions.map((region) => intervalLabel(region.name).detailed));
  const recalledSymbols = [...referenceSymbols].filter((symbol) => detectedSymbols.has(symbol)).length;
  const sequence = alignChordSequences(
    referenceRegions.map((region) => ({ sounding: region.chord || region.name })),
    predictedRegions,
  );

  const sequenceCounts = {
    missingReferenceChords: sequence.pairs.filter((pair) => pair.referenceIndex !== null && pair.detectedIndex === null).length,
    extraDetectedChords: sequence.pairs.filter((pair) => pair.referenceIndex === null && pair.detectedIndex !== null).length,
    wrongChordQuality: mismatches.filter((mismatch) =>
      mismatch.errorType.includes("quality")
      || mismatch.errorType.includes("third")
      || mismatch.errorType.includes("seventh")).length,
    wrongRoot: mismatches.filter((mismatch) =>
      !["wrong chord quality", "weak third caused major/minor ambiguity", "transient seventh detected", "seventh omitted", "false N", "missed N"].includes(mismatch.errorType)).length,
    incorrectExtension: mismatches.filter((mismatch) =>
      mismatch.errorType === "transient seventh detected"
      || mismatch.errorType === "seventh omitted").length,
    falseNoChord: mismatches.filter((mismatch) => mismatch.errorType === "false N" || mismatch.errorType === "bass root ignored" && mismatch.detected === "N").length,
    missedNoChord: mismatches.filter((mismatch) => mismatch.errorType === "missed N").length,
    earlyBoundary: boundaryErrors.filter((error) => error < -0.08).length,
    lateBoundary: boundaryErrors.filter((error) => error > 0.08).length,
    fragmentedChord: referenceRegions.filter((referenceRegion) =>
      predictedRegions.filter((detectedRegion) =>
        detectedRegion.start < referenceRegion.end && detectedRegion.end > referenceRegion.start).length > 1).length,
    mergedReferenceChords: predictedRegions.filter((detectedRegion) =>
      referenceRegions.filter((referenceRegion) =>
        referenceRegion.start < detectedRegion.end && referenceRegion.end > detectedRegion.start).length > 1).length,
  };

  return {
    schemaVersion: 1,
    song: { artist: reference.artist, title: reference.title, genre: reference.genre || "Unspecified" },
    referenceId: reference.id ?? null,
    referenceSelection: {
      source: reference.source,
      referenceUrl: reference.referenceUrl,
      version: reference.version,
      ratingCount: reference.ratingCount,
      averageRating: reference.averageRating,
      selectionReason: reference.selectionReason,
    },
    alignment: {
      method: alignment.method,
      verified: alignment.verified,
      confidence: alignment.confidence,
    },
    metrics: {
      evaluatedDurationSeconds: round(totalDuration, 3),
      rootAccuracy: round(safeDivide(rootCorrect, totalDuration)),
      majorMinorAccuracy: round(safeDivide(familyCorrect, totalDuration)),
      detailedAccuracy: round(safeDivide(detailCorrect, totalDuration)),
      bassInversionAccuracy: inversionDuration
        ? round(safeDivide(inversionCorrectDuration, inversionDuration))
        : null,
      chordSymbolRecall: round(safeDivide(recalledSymbols, referenceSymbols.size)),
      weightedChordSymbolRecall: round(safeDivide(detailCorrect, totalDuration)),
      noChordPrecision: round(safeDivide(noChordTruePositiveDuration, predictedNoChordDuration)),
      noChordRecall: round(safeDivide(noChordTruePositiveDuration, referenceNoChordDuration)),
      falseNoChordDurationPercent: round(safeDivide(
        Math.max(0, predictedNoChordDuration - noChordTruePositiveDuration),
        totalDuration,
      )),
      meanBoundaryErrorMs: boundaryErrors.length
        ? round(boundaryErrors.reduce((sum, error) => sum + Math.abs(error), 0) / boundaryErrors.length * 1000, 1)
        : null,
      medianBoundaryErrorMs: boundaryErrors.length
        ? round(median(boundaryErrors.map(Math.abs)) * 1000, 1)
        : null,
      meanSignedBoundaryErrorMs: boundaryErrors.length
        ? round(boundaryErrors.reduce((sum, error) => sum + error, 0) / boundaryErrors.length * 1000, 1)
        : null,
      earlyBoundaryMeanMs: earlyBoundaryErrors.length
        ? round(earlyBoundaryErrors.reduce((sum, error) => sum + error, 0)
          / earlyBoundaryErrors.length * 1000, 1)
        : null,
      lateBoundaryMeanMs: lateBoundaryErrors.length
        ? round(lateBoundaryErrors.reduce((sum, error) => sum + error, 0)
          / lateBoundaryErrors.length * 1000, 1)
        : null,
      meanAbsoluteBoundaryErrorBeats: beatDuration && boundaryErrors.length
        ? round(boundaryErrors.reduce((sum, error) => sum + Math.abs(error), 0)
          / boundaryErrors.length / beatDuration)
        : null,
      medianAbsoluteBoundaryErrorBeats: beatDuration && boundaryErrors.length
        ? round(median(boundaryErrors.map(Math.abs)) / beatDuration)
        : null,
      meanSignedBoundaryErrorBeats: beatDuration && boundaryErrors.length
        ? round(boundaryErrors.reduce((sum, error) => sum + error, 0)
          / boundaryErrors.length / beatDuration)
        : null,
      percentageBoundariesMoreThanOneBeatLate: beatDuration && boundaryErrors.length
        ? round(safeDivide(
          boundaryErrors.filter((error) => error > beatDuration).length,
          boundaryErrors.length,
        ))
        : null,
      percentageBoundariesMoreThanTwoBeatsLate: beatDuration && boundaryErrors.length
        ? round(safeDivide(
          boundaryErrors.filter((error) => error > beatDuration * 2).length,
          boundaryErrors.length,
        ))
        : null,
      fragmentationRate: round(safeDivide(
        sequenceCounts.fragmentedChord,
        referenceRegions.length,
      )),
    },
    sequence: sequenceCounts,
    boundaryStages,
    mismatches,
    rootConfusions: [...confusionDuration.entries()]
      .map(([pair, duration]) => ({ pair, durationSeconds: round(duration, 3) }))
      .sort((a, b) => b.durationSeconds - a.durationSeconds),
    createdAt: new Date().toISOString(),
  };
}

export const CHORD_EVALUATION_ABLATIONS = Object.freeze([
  { id: "simple-harmonic-context-v3", label: "Simple chords + harmonic-context-v3 reduced latency", overrides: { chordDisplayMode: "simple" } },
  { id: "detailed-harmonic-context-v3", label: "Detailed chords + harmonic-context-v3 reduced latency", overrides: { chordDisplayMode: "detailed" } },
  { id: "without-bass-root-prior", label: "Without bass-root prior", overrides: { rootSupportWeight: 0, bassAgreementWeight: 0 } },
  { id: "without-key-prior", label: "Without key prior", overrides: { keyCompatibilityWeight: 0 } },
  { id: "without-transition-prior", label: "Without transition prior", overrides: { commonTransitionBonus: 0 } },
  { id: "without-repeated-section-consistency", label: "Without repeated-section consistency", overrides: { repeatedSectionConsistencyWeight: 0 } },
]);

const RECOMMENDATIONS = {
  "bass root ignored": {
    recommendation: "Increase or repair bass/root evidence when local candidates are ambiguous.",
    expectedImprovement: "high",
    regressionRisk: "medium",
    implementationEffort: "medium",
  },
  "relative major/minor confusion": {
    recommendation: "Combine bass-root continuity with stronger third evidence.",
    expectedImprovement: "high",
    regressionRisk: "medium",
    implementationEffort: "medium",
  },
  "transient seventh detected": {
    recommendation: "Raise sustained-extension evidence requirements.",
    expectedImprovement: "medium",
    regressionRisk: "low",
    implementationEffort: "low",
  },
  "seventh omitted": {
    recommendation: "Use repeated extension evidence in Detailed mode.",
    expectedImprovement: "medium",
    regressionRisk: "medium",
    implementationEffort: "low",
  },
  "false N": {
    recommendation: "Penalize N when neighbours and bass imply harmonic continuity.",
    expectedImprovement: "high",
    regressionRisk: "medium",
    implementationEffort: "low",
  },
  "lead melody contaminated harmony": {
    recommendation: "Reduce lead-guitar evidence relative to bass and accompaniment stems.",
    expectedImprovement: "high",
    regressionRisk: "medium",
    implementationEffort: "high",
  },
};

export function aggregateChordEvaluationReports(reports) {
  const valid = (reports || []).filter((report) => report?.metrics);
  const causeMap = new Map();
  const rootConfusions = new Map();
  for (const report of valid) {
    const causesInSong = new Set();
    for (const mismatch of report.mismatches || []) {
      causesInSong.add(mismatch.errorType);
      const current = causeMap.get(mismatch.errorType) || { songs: new Set(), durationSeconds: 0, occurrences: 0 };
      current.songs.add(`${report.song.artist} — ${report.song.title}`);
      current.durationSeconds += mismatch.duration || 0;
      current.occurrences += 1;
      causeMap.set(mismatch.errorType, current);
    }
    for (const confusion of report.rootConfusions || []) {
      rootConfusions.set(confusion.pair, (rootConfusions.get(confusion.pair) || 0) + confusion.durationSeconds);
    }
  }
  const metricNames = [
    "rootAccuracy", "majorMinorAccuracy", "detailedAccuracy", "bassInversionAccuracy", "chordSymbolRecall",
    "weightedChordSymbolRecall", "noChordPrecision", "noChordRecall",
    "falseNoChordDurationPercent", "meanBoundaryErrorMs", "medianBoundaryErrorMs",
  ];
  const averages = Object.fromEntries(metricNames.map((name) => {
    const values = valid.map((report) => report.metrics[name]).filter(Number.isFinite);
    return [name, values.length ? round(values.reduce((sum, value) => sum + value, 0) / values.length) : null];
  }));
  const genres = {};
  for (const report of valid) {
    const genre = report.song.genre || "Unspecified";
    const current = genres[genre] || { songCount: 0, rootAccuracy: [], majorMinorAccuracy: [], detailedAccuracy: [] };
    current.songCount += 1;
    for (const metric of ["rootAccuracy", "majorMinorAccuracy", "detailedAccuracy"]) {
      if (Number.isFinite(report.metrics[metric])) current[metric].push(report.metrics[metric]);
    }
    genres[genre] = current;
  }
  const genreResults = Object.entries(genres).map(([genre, values]) => ({
    genre,
    songCount: values.songCount,
    rootAccuracy: round(safeDivide(values.rootAccuracy.reduce((sum, value) => sum + value, 0), values.rootAccuracy.length)),
    majorMinorAccuracy: round(safeDivide(values.majorMinorAccuracy.reduce((sum, value) => sum + value, 0), values.majorMinorAccuracy.length)),
    detailedAccuracy: round(safeDivide(values.detailedAccuracy.reduce((sum, value) => sum + value, 0), values.detailedAccuracy.length)),
  })).sort((a, b) => a.rootAccuracy - b.rootAccuracy);
  const causes = [...causeMap.entries()].map(([errorType, value]) => ({
    errorType,
    songsAffected: value.songs.size,
    songs: [...value.songs],
    occurrences: value.occurrences,
    durationSeconds: round(value.durationSeconds, 3),
  })).sort((a, b) => b.songsAffected - a.songsAffected || b.durationSeconds - a.durationSeconds);
  const rankedFixes = causes.map((cause) => ({
    ...cause,
    ...(RECOMMENDATIONS[cause.errorType] || {
      recommendation: "Review diagnostics across affected songs before changing the detector.",
      expectedImprovement: "unknown",
      regressionRisk: "unknown",
      implementationEffort: "unknown",
    }),
  }));
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    songCount: valid.length,
    averages,
    rootConfusions: [...rootConfusions.entries()]
      .map(([pair, durationSeconds]) => ({ pair, durationSeconds: round(durationSeconds, 3) }))
      .sort((a, b) => b.durationSeconds - a.durationSeconds),
    genreResults,
    recurringErrors: causes.filter((cause) => cause.songsAffected >= 3),
    rankedFixes,
    ablations: CHORD_EVALUATION_ABLATIONS,
    songs: valid,
  };
}

export function chordEvaluationReportMarkdown(aggregate) {
  const percentage = (value) => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "n/a";
  const lines = [
    "# Tabsmith chord evaluation",
    "",
    `Generated: ${aggregate.createdAt}`,
    `Songs evaluated: ${aggregate.songCount}`,
    "",
    "## Aggregate metrics",
    "",
    `- Root accuracy: ${percentage(aggregate.averages.rootAccuracy)}`,
    `- Major/minor accuracy: ${percentage(aggregate.averages.majorMinorAccuracy)}`,
    `- Detailed accuracy: ${percentage(aggregate.averages.detailedAccuracy)}`,
    `- Weighted chord-symbol recall: ${percentage(aggregate.averages.weightedChordSymbolRecall)}`,
    `- False N duration: ${percentage(aggregate.averages.falseNoChordDurationPercent)}`,
    `- Median boundary error: ${Number.isFinite(aggregate.averages.medianBoundaryErrorMs) ? `${aggregate.averages.medianBoundaryErrorMs.toFixed(1)} ms` : "n/a"}`,
    "",
    "## Genre summary",
    "",
    ...(aggregate.genreResults.length
      ? aggregate.genreResults.map((genre) =>
        `- ${genre.genre}: ${genre.songCount} song(s), ${percentage(genre.rootAccuracy)} root accuracy`)
      : ["- No genre metadata available."]),
    "",
    "## Recurring errors (three or more songs)",
    "",
    ...(aggregate.recurringErrors.length
      ? aggregate.recurringErrors.map((cause) =>
        `- ${cause.errorType}: ${cause.songsAffected} songs, ${cause.durationSeconds.toFixed(1)} seconds`)
      : ["- Not enough repeated evidence yet."]),
    ...(aggregate.ablationSummary?.length ? [
      "",
      "## Ablation study",
      "",
      ...aggregate.ablationSummary.map((configuration) =>
        `- ${configuration.label}: ${configuration.songCount} song(s), ${percentage(configuration.weightedChordSymbolRecall)} weighted chord-symbol recall`),
    ] : []),
    "",
    "## Ranked candidate fixes",
    "",
    ...aggregate.rankedFixes.map((fix, index) =>
      `${index + 1}. ${fix.recommendation} (${fix.songsAffected} songs; ${fix.durationSeconds.toFixed(1)} s; risk ${fix.regressionRisk}; effort ${fix.implementationEffort})`),
    "",
    "## Per-song results",
    "",
    ...aggregate.songs.flatMap((report) => [
      `### ${report.song.artist} — ${report.song.title}`,
      "",
      `- Alignment: ${report.alignment.method}${report.alignment.verified ? " (manually verified)" : " (automatic, unverified)"}`,
      `- Root accuracy: ${percentage(report.metrics.rootAccuracy)}`,
      `- Major/minor accuracy: ${percentage(report.metrics.majorMinorAccuracy)}`,
      `- Detailed accuracy: ${percentage(report.metrics.detailedAccuracy)}`,
      `- Median boundary error: ${report.metrics.medianBoundaryErrorMs ?? "n/a"} ms`,
      `- False N duration: ${percentage(report.metrics.falseNoChordDurationPercent)}`,
      `- Extra detected regions: ${report.sequence.extraDetectedChords}`,
      "",
    ]),
  ];
  return `${lines.join("\n")}\n`;
}

export function evaluateAblations(reference, predictions) {
  return CHORD_EVALUATION_ABLATIONS.map((configuration) => {
    const prediction = predictions?.[configuration.id];
    return prediction
      ? { configuration, report: evaluateChordReference(reference, prediction) }
      : { configuration, report: null, status: "prediction-not-supplied" };
  });
}

export function summarizeAblationResults(songResults) {
  const byConfiguration = new Map();
  for (const song of songResults || []) {
    for (const entry of song.configurations || []) {
      if (!entry.report?.metrics) continue;
      const current = byConfiguration.get(entry.configuration.id) || {
        id: entry.configuration.id,
        label: entry.configuration.label,
        songs: 0,
        rootAccuracy: [],
        majorMinorAccuracy: [],
        detailedAccuracy: [],
        weightedChordSymbolRecall: [],
      };
      current.songs += 1;
      for (const metric of ["rootAccuracy", "majorMinorAccuracy", "detailedAccuracy", "weightedChordSymbolRecall"]) {
        const value = entry.report.metrics[metric];
        if (Number.isFinite(value)) current[metric].push(value);
      }
      byConfiguration.set(entry.configuration.id, current);
    }
  }
  return [...byConfiguration.values()].map((configuration) => ({
    id: configuration.id,
    label: configuration.label,
    songCount: configuration.songs,
    rootAccuracy: round(safeDivide(
      configuration.rootAccuracy.reduce((sum, value) => sum + value, 0),
      configuration.rootAccuracy.length,
    )),
    majorMinorAccuracy: round(safeDivide(
      configuration.majorMinorAccuracy.reduce((sum, value) => sum + value, 0),
      configuration.majorMinorAccuracy.length,
    )),
    detailedAccuracy: round(safeDivide(
      configuration.detailedAccuracy.reduce((sum, value) => sum + value, 0),
      configuration.detailedAccuracy.length,
    )),
    weightedChordSymbolRecall: round(safeDivide(
      configuration.weightedChordSymbolRecall.reduce((sum, value) => sum + value, 0),
      configuration.weightedChordSymbolRecall.length,
    )),
  })).sort((a, b) => b.weightedChordSymbolRecall - a.weightedChordSymbolRecall);
}

export function evaluateTranscription(reference, prediction, onsetTolerance = 0.08) {
  const expectedNotes = [...(reference.notes || [])].sort((a, b) => a.start - b.start || a.midi - b.midi);
  const predictedNotes = [...(prediction.notes || [])].sort((a, b) => a.start - b.start || a.midi - b.midi);
  const used = new Set();
  const matches = [];
  for (const expected of expectedNotes) {
    let bestIndex = -1;
    let bestDistance = Infinity;
    predictedNotes.forEach((candidate, index) => {
      const distance = Math.abs(candidate.start - expected.start);
      if (!used.has(index) && candidate.midi === expected.midi && distance <= onsetTolerance && distance < bestDistance) {
        bestIndex = index;
        bestDistance = distance;
      }
    });
    if (bestIndex >= 0) {
      used.add(bestIndex);
      matches.push({ expected, predicted: predictedNotes[bestIndex], onsetError: bestDistance });
    }
  }
  const precision = safeDivide(matches.length, predictedNotes.length);
  const recall = safeDivide(matches.length, expectedNotes.length);
  const f1 = safeDivide(2 * precision * recall, precision + recall);
  const positioned = matches.filter(({ expected, predicted }) =>
    Number.isFinite(expected.string) && Number.isFinite(expected.fret) && Number.isFinite(predicted.string) && Number.isFinite(predicted.fret));
  const exactPositions = positioned.filter(({ expected, predicted }) => expected.string === predicted.string && expected.fret === predicted.fret).length;

  const boundaries = new Set([0, Number(reference.duration) || 0]);
  for (const chord of [...(reference.chords || []), ...(prediction.chords || [])]) {
    boundaries.add(chord.start);
    boundaries.add(chord.end);
  }
  const points = [...boundaries].filter(Number.isFinite).sort((a, b) => a - b);
  let chordCorrect = 0;
  let chordTotal = 0;
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (end <= start) continue;
    const midpoint = (start + end) / 2;
    const expected = reference.chords?.find((chord) => midpoint >= chord.start && midpoint < chord.end)?.name ?? "N";
    const predicted = prediction.chords?.find((chord) => midpoint >= chord.start && midpoint < chord.end)?.name ?? "N";
    chordTotal += end - start;
    if (expected === predicted) chordCorrect += end - start;
  }

  return {
    notePrecision: round(precision),
    noteRecall: round(recall),
    noteF1: round(f1),
    matchedNotes: matches.length,
    referenceNotes: expectedNotes.length,
    predictedNotes: predictedNotes.length,
    onsetMaeMs: matches.length ? round(matches.reduce((sum, match) => sum + match.onsetError, 0) / matches.length * 1000) : null,
    fretPositionAccuracy: positioned.length ? round(exactPositions / positioned.length) : null,
    chordTimeAccuracy: chordTotal ? round(chordCorrect / chordTotal) : null,
  };
}
