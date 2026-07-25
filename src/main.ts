import "./styles.css";
import {
  BASIC_PITCH_THRESHOLDS,
  transcribeWithBasicPitch,
} from "./mlTranscription";
import { getPlaybackState } from "./playback";
import {
  NOTE_CLEANUP_PRESETS,
  processDetectedNotes,
} from "./noteCleanup";
import type { NoteProcessingOptions } from "./noteCleanup";
import {
  noteEditorWindowKey,
  selectNoteEditorRows,
  type NoteEditorView,
} from "./noteEditor";
import { buildTablature, tablatureToText, type TabMeasure } from "./tablature";
import { TabRenderer } from "./tabRenderer";
import {
  BALANCED_CHORD_SMOOTHING_SETTINGS,
  type HarmonyEvidenceOptions,
} from "./chordAnalysis";
import { TUNING_NAMES } from "./fretboard";
import {
  ARRANGEMENT_VERSION,
  analysisCountLabels,
  BASIC_PITCH_MODEL_VERSION,
  buildTranscriptionCacheKey,
  cacheReadDecision,
  cacheOutcomeLabels,
  CHORD_ANALYSIS_VERSION,
  createCacheMetadata,
  FRETBOARD_MAPPER_VERSION,
  NOTE_CLEANUP_VERSION,
  pipelineLabel,
  reprocessPreservingPrevious,
  sha256Hex,
  SOURCE_SEPARATION_MODEL_VERSION,
  TEMPO_QUANTIZATION_VERSION,
  TRANSCRIPTION_PIPELINE_VERSION,
  type CachedTranscriptionEntry,
  type ProcessingCacheOutcome,
  type TranscriptionCacheKeyInput,
} from "./transcriptionCache";
import type {
  AnalysisResult,
  ArrangementMode,
  ChordDisplayMode,
  HarmonyResult,
  ChordEvent,
  QualityPreset,
} from "./types";

const app = document.querySelector<HTMLDivElement>("#app")!;
let result: AnalysisResult | null = null;
let currentFile: File | null = null;
let currentSourceUrl: string | null = null;
let lastActiveChordIndex = -2;
let tabMeasures: TabMeasure[] = [];
let tabRenderer: TabRenderer | null = null;
let cursorRaf = 0;
let activeJobId: string | null = null;
let activeController: AbortController | null = null;
let analysisGeneration = 0;
let lastNoteViewKey = "";
let lastCacheOutcome: ProcessingCacheOutcome | null = null;
let selectedChordIndex = -1;
let chordDebugVisible = false;
let evaluationReferences: EvaluationReference[] = [];
let evaluationMarks: Array<{ chordIndex: number; start: number }> = [];
let latestEvaluationAggregate: unknown = null;

type EvaluationReference = {
  id: string;
  artist: string;
  title: string;
  referenceUrl: string;
  version: number;
  ratingCount: number;
  averageRating: number;
  selectionReason: string;
  capo: number;
  tuning: string;
  key: string;
  genre: string;
  chords: Array<{
    order: number;
    section: string;
    displayed: string;
    sounding: string;
  }>;
  alignment: {
    method: string;
    verified: boolean;
    confidence: number;
    regions: Array<{ start: number; end: number; chord: string; chordIndex: number }>;
  } | null;
};

type ProcessingJob = {
  id: string;
  name: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  stage: string;
  progress: number;
  cached: boolean;
  error: string | null;
  createdAt: string;
};

app.innerHTML = `
  <main class="shell">
    <nav class="topbar">
      <div class="brand"><span class="mark">⌁</span> tabsmith</div>
      <span class="local-pill">private · runs locally</span>
    </nav>
    <section class="hero">
      <div>
        <div class="eyebrow">Audio → playable guitar tab</div>
        <h1>Hear the song.<br>Find the fretboard.</h1>
        <p class="lede">Tabsmith listens to the audio itself—no tab databases, no copied transcriptions. Drop in a song and get an editable first-pass map of its chords and notes.</p>
      </div>
      <div class="process">
        <div class="process-row"><span>01</span> Isolate the guitar from the mix</div>
        <div class="process-row"><span>02</span> Detect its pitch and harmony</div>
        <div class="process-row"><span>03</span> Optimize string and fret positions</div>
      </div>
    </section>

    <section class="input-card">
      <div class="tabs">
        <button class="tab active" data-mode="file">Audio file</button>
        <button class="tab" data-mode="link">YouTube link</button>
      </div>
      <div id="file-panel">
        <div class="dropzone" id="dropzone">
          <strong>Drop a song here</strong>
          <p>MP3, WAV, M4A, OGG, or any format your browser can decode</p>
          <div class="input-actions"><label class="choose">Choose audio<input class="hidden" id="file-input" type="file" accept="audio/*" /></label><button class="demo-button" id="demo-song" type="button">Try demo song</button></div>
        </div>
        <p class="privacy">Audio stays on this device. For the cleanest result, start with an isolated or prominent guitar track.</p>
      </div>
      <div id="link-panel" class="hidden">
        <form class="url-form" id="url-form">
          <div class="url-main"><input id="source-url" type="url" placeholder="Paste a YouTube video URL" required /><button id="analyze-link">Import audio</button></div>
          <label class="url-consent"><input id="rights-confirm" type="checkbox" /> I have permission to process this audio.</label>
        </form>
        <p class="url-note" id="url-note">Direct videos and Shorts up to 15 minutes are supported. Playlists and live streams are rejected.</p>
      </div>
      <div class="engine-row">
        <div><strong>Transcription engine</strong><span>Choose accuracy or speed</span></div>
        <select id="engine-select" aria-label="Transcription engine">
          <option value="ml">High accuracy · Basic Pitch</option>
          <option value="dsp">Fast · lightweight detector</option>
        </select>
      </div>
      <div class="engine-row">
        <div><strong>Guitar arrangement</strong><span>Select one playable part from the detected audio</span></div>
        <select id="arrangement-select" aria-label="Guitar arrangement">
          <option value="automatic">Automatic</option>
          <option value="lead">Lead</option>
          <option value="rhythm">Rhythm</option>
          <option value="mixed">Mixed</option>
          <option value="raw">Raw transcription</option>
        </select>
      </div>
      <div class="engine-row">
        <div><strong>Transcription quality</strong><span>Balanced preserves detail without printing every detection</span></div>
        <select id="quality-select" aria-label="Transcription quality">
          <option value="clean">Clean</option>
          <option value="balanced" selected>Balanced</option>
          <option value="detailed">Detailed</option>
          <option value="raw">Raw</option>
        </select>
      </div>
      <div class="engine-row">
        <div><strong>Chord labels</strong><span>Simple favors reliable triads and sustained dominant sevenths</span></div>
        <select id="chord-mode-select" aria-label="Chord label detail">
          <option value="simple" selected>Simple chords</option>
          <option value="detailed">Detailed chords</option>
        </select>
      </div>
      <div class="engine-row">
        <div><strong>Output</strong><span>Show the chord progression, the tab, or both</span></div>
        <select id="output-mode-select" aria-label="Output mode">
          <option value="both" selected>Chords + tab</option>
          <option value="chords">Chords only</option>
          <option value="tab">Tab only</option>
        </select>
      </div>
      <div class="engine-row">
        <div><strong>Chord engine</strong><span>Rule-based is the default; others are experimental</span></div>
        <select id="chord-engine-select" aria-label="Chord detection engine">
          <option value="rule" selected>Rule-based (default)</option>
          <option value="learned">Learned model (experimental)</option>
          <option value="onehotchord">OneHotChord (experimental)</option>
        </select>
      </div>
      <label class="separation-row">
        <input id="isolate-guitar" type="checkbox" checked />
        <span><strong>Isolate guitar before transcription</strong><small>Demucs separates vocals, drums, bass, piano, and other sounds so only its guitar stem is analyzed.</small></span>
        <em>recommended</em>
      </label>
      <div class="status hidden" id="status"><div class="status-top"><span id="status-label">Listening…</span><button class="cancel-job hidden" id="cancel-job" type="button">Cancel</button></div><div class="bar"><div id="analysis-progress"></div></div></div>
      <div class="job-history hidden" id="job-history"><div class="history-title"><span>RECENT PROCESSING</span><span>cached locally</span></div><div id="job-list"></div></div>
      <details class="cache-management" id="cache-management">
        <summary>Advanced cache management</summary>
        <div class="cache-usage" id="cache-usage">Calculating disk usage…</div>
        <div class="cache-actions">
          <button id="clear-transcription-cache" type="button">Clear transcription cache</button>
          <button id="clear-separation-cache" type="button">Clear separated-stem cache</button>
          <button id="clear-all-cache" type="button">Clear all processing data</button>
          <button id="open-cache-location" type="button">Open cache location</button>
        </div>
        <div class="cache-message" id="cache-message"></div>
      </details>
    </section>

    <section class="results hidden" id="results">
      <div class="result-head">
        <div><h2 id="song-title">Untitled track</h2><div class="metadata" id="metadata"></div></div>
        <div class="actions"><button id="reprocess" type="button">Reprocess transcription</button><button id="export">Export JSON</button><button id="new-song">New song</button></div>
      </div>
      <div class="pipeline-status" id="pipeline-status"></div>
      <div class="quality-gate hidden" id="quality-gate"></div>
      <audio id="player" controls></audio>
      <div class="playback-readout">
        <div class="now-chord"><span>NOW</span><strong id="now-chord">—</strong><small id="chord-window">Press play to follow the progression</small></div>
        <div class="next-chord"><span>NEXT</span><strong id="next-chord">—</strong><small id="next-chord-time">No upcoming chord</small></div>
        <div class="song-clock">
          <div><span id="current-time">0:00</span><span id="total-time">0:00</span></div>
          <input id="song-scrubber" type="range" min="0" max="1" step="0.001" value="0" aria-label="Song position" />
        </div>
      </div>
      <div class="timeline-caption"><span>CHORD PROGRESSION</span><span>click a chord to review or correct</span></div>
      <div class="chord-tools"><button id="next-uncertain-chord" type="button">Jump to next uncertain chord</button><button id="toggle-chord-debug" type="button">Chord diagnostics</button></div>
      <div class="chord-strip" id="chords"></div>
      <div class="chord-editor hidden" id="chord-editor"></div>
      <div class="chord-debug hidden" id="chord-debug"></div>
      <details class="evaluation-tools" id="evaluation-tools">
        <summary>Internal chord-reference evaluation</summary>
        <div class="evaluation-notice">
          <strong>Manual, chord-only import</strong>
          <span>Review the reference in your own authorized browser session. Do not paste lyrics, complete tabs, or page HTML. Tabsmith never fetches Ultimate Guitar.</span>
        </div>
        <form id="evaluation-import-form">
          <div class="evaluation-fields">
            <label>Artist<input id="evaluation-artist" required maxlength="160"></label>
            <label>Song<input id="evaluation-title" required maxlength="160"></label>
            <label class="wide">Reference URL<input id="evaluation-url" type="url" placeholder="https://tabs.ultimate-guitar.com/..."></label>
            <label>Version<input id="evaluation-version" type="number" min="0" value="0"></label>
            <label>Rating count<input id="evaluation-rating-count" type="number" min="0" value="0"></label>
            <label>Average rating<input id="evaluation-average-rating" type="number" min="0" max="5" step="0.1" value="0"></label>
            <label>Capo<input id="evaluation-capo" type="number" min="0" max="24" value="0"></label>
            <label>Tuning<input id="evaluation-tuning" value="Standard"></label>
            <label>Key<input id="evaluation-key" placeholder="optional"></label>
            <label>Genre<input id="evaluation-genre" placeholder="e.g. rock"></label>
            <label>Dataset split<select id="evaluation-split"><option value="development">Development</option><option value="validation">Validation</option><option value="test">Final test</option></select></label>
            <label class="wide">Selection reason<input id="evaluation-selection-reason" value="Largest rating count among manually reviewed Chords versions"></label>
          </div>
          <label class="evaluation-sequence">Chord sequence or timed chords
            <textarea id="evaluation-sequence" required placeholder="Intro: Bm | F#7 | A | E&#10;Verse: G | D | Em | F#7"></textarea>
          </label>
          <div class="evaluation-actions"><button type="submit">Import chord reference</button><span>Accepted: chord symbols, section labels, and optional timestamps only.</span></div>
        </form>
        <div class="evaluation-workbench">
          <label>Imported reference
            <select id="evaluation-reference-select"><option value="">No references imported</option></select>
          </label>
          <div class="evaluation-reference-summary" id="evaluation-reference-summary">Import a chord-only reference to begin.</div>
          <div class="evaluation-actions">
            <button id="evaluation-mark" type="button">Mark next chord at playback (M)</button>
            <button id="evaluation-undo-mark" type="button">Undo mark</button>
            <button id="evaluation-save-alignment" type="button">Save verified alignment</button>
            <button id="evaluation-compare" type="button">Compare current detection</button>
            <button id="evaluation-download-report" type="button" disabled>Download aggregate JSON</button>
          </div>
          <div class="evaluation-markers" id="evaluation-markers"></div>
          <div class="evaluation-report" id="evaluation-report"></div>
        </div>
        <div class="evaluation-message" id="evaluation-message"></div>
      </details>
      <div class="tab-card tab-card--score">
        <div class="tab-title">
          <span>PLAYABLE TAB</span>
          <div class="tab-tools">
            <div class="zoom-group"><button id="zoom-out" type="button" aria-label="Zoom out">−</button><button id="zoom-in" type="button" aria-label="Zoom in">+</button></div>
            <label class="count-toggle"><input id="count-overlay" type="checkbox" /> Count</label>
            <button id="export-tab">Export .txt</button>
          </div>
        </div>
        <div class="tab-score" id="tab-score"></div>
      </div>
      <div class="tab-card">
        <div class="tab-title"><span>ARRANGED NOTES</span><span>playback-aware · editable positions</span></div>
        <div class="note-view-controls"><label>View
          <select id="note-view-select" aria-label="Note list view">
            <option value="follow">Follow playback</option>
            <option value="measure">Current measure</option>
            <option value="section">Current section</option>
            <option value="all">All final notes</option>
            <option value="low-confidence">Low-confidence notes</option>
            <option value="raw">Raw detections · diagnostic</option>
          </select>
        </label></div>
        <div class="note-grid" id="notes"></div>
      </div>
    </section>
  </main>`;

const byId = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const dropzone = byId<HTMLDivElement>("dropzone");
const fileInput = byId<HTMLInputElement>("file-input");
const status = byId<HTMLDivElement>("status");
const player = byId<HTMLAudioElement>("player");

byId<HTMLButtonElement>("cancel-job").addEventListener("click", () => {
  const jobId = activeJobId;
  analysisGeneration += 1;
  activeController?.abort();
  if (jobId) void fetch(`/api/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" }).finally(refreshJobs);
  updateStatus("Cancelling processing…");
  byId<HTMLButtonElement>("cancel-job").classList.add("hidden");
});

byId<HTMLButtonElement>("demo-song").addEventListener("click", async () => {
  const response = await fetch("/tabsmith-demo.wav");
  if (!response.ok) return showError("The bundled demo song could not be loaded.");
  byId<HTMLInputElement>("isolate-guitar").checked = false;
  const blob = await response.blob();
  await loadFile(new File([blob], "Tabsmith demo · E-A-B7-E.wav", { type: "audio/wav" }));
});

void refreshJobs();

document.querySelectorAll<HTMLButtonElement>(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    byId("file-panel").classList.toggle("hidden", tab.dataset.mode !== "file");
    byId("link-panel").classList.toggle("hidden", tab.dataset.mode !== "link");
  });
});

byId<HTMLFormElement>("url-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const url = byId<HTMLInputElement>("source-url").value;
  const note = byId("url-note");
  const button = byId<HTMLButtonElement>("analyze-link");
  if (!byId<HTMLInputElement>("rights-confirm").checked) {
    note.textContent = "Confirm that you have permission to process this audio first.";
    note.classList.add("error-text");
    return;
  }
  note.classList.remove("error-text");
  button.disabled = true;
  status.classList.remove("hidden");
  updateStatus("Checking the YouTube video…");
  try {
    const info = await postJson<{ title: string; duration: number }>("/api/youtube/info", { url });
    note.textContent = `Found “${info.title}” · ${formatTime(info.duration)}. Importing its audio locally…`;
    updateStatus("Importing YouTube audio…");
    const response = await fetch("/api/youtube/audio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) throw new Error(await responseError(response));
    const blob = await response.blob();
    const extension = response.headers.get("X-Audio-Extension")?.replace(/[^a-z0-9]/gi, "") || "webm";
    const safeTitle = info.title.replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_").slice(0, 120) || "YouTube audio";
    const file = new File([blob], `${safeTitle}.${extension}`, { type: response.headers.get("Content-Type") || blob.type });
    await loadFile(file, url);
  } catch (error) {
    status.classList.add("hidden");
    note.textContent = error instanceof Error ? error.message : "YouTube import failed.";
    note.classList.add("error-text");
  } finally {
    button.disabled = false;
  }
});

for (const eventName of ["dragenter", "dragover"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); });
for (const eventName of ["dragleave", "drop"]) dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); });
dropzone.addEventListener("drop", (event) => void loadFile(event.dataTransfer?.files[0]));
fileInput.addEventListener("change", () => void loadFile(fileInput.files?.[0]));
byId<HTMLSelectElement>("note-view-select").addEventListener("change", () => {
  lastNoteViewKey = "";
  renderNoteList(true);
});
// Output mode (chords / tab / both) is a live display toggle — no reprocessing.
function applyOutputMode(mode: string): void {
  byId("results").dataset.outputMode = mode;
  try { localStorage.setItem("tabsmith.outputMode", mode); } catch { /* storage unavailable */ }
}
const outputModeSelect = byId<HTMLSelectElement>("output-mode-select");
try {
  const saved = localStorage.getItem("tabsmith.outputMode");
  if (saved && ["both", "chords", "tab"].includes(saved)) outputModeSelect.value = saved;
} catch { /* storage unavailable */ }
outputModeSelect.addEventListener("change", () => applyOutputMode(outputModeSelect.value));
applyOutputMode(outputModeSelect.value);
byId("notes").addEventListener("change", (event) => {
  const input = event.target as HTMLInputElement;
  if (!result || !input.matches("input[data-index]")) return;
  const note = result.notes[Number(input.dataset.index)];
  if (!note) return;
  if (input.dataset.field === "string") note.string = Number(input.value);
  else note.fret = Number(input.value);
  result.noteAnalysis.arrangedNotes = result.notes;
  renderTabScore(result);
});
byId<HTMLButtonElement>("toggle-chord-debug").addEventListener("click", () => {
  chordDebugVisible = !chordDebugVisible;
  byId("chord-debug").classList.toggle("hidden", !chordDebugVisible);
  if (result) renderChordDiagnostics(result);
});
byId<HTMLButtonElement>("next-uncertain-chord").addEventListener("click", () => {
  if (!result?.chords.length) return;
  const uncertain = result.chords.findIndex((chord) => (
    chord.start > player.currentTime + 0.01
      && (chord.diagnostics?.confidenceLevel === "low" || chord.confidence < 0.42)
  ));
  const fallback = result.chords.findIndex(
    (chord) => chord.diagnostics?.confidenceLevel === "low" || chord.confidence < 0.42,
  );
  const index = uncertain >= 0 ? uncertain : fallback;
  if (index < 0) return;
  selectedChordIndex = index;
  player.currentTime = result.chords[index].start;
  renderChordTrack(result);
  renderChordEditor(result);
  renderChordDiagnostics(result);
});
byId("chord-editor").addEventListener("click", (event) => {
  const button = (event.target as HTMLElement).closest<HTMLButtonElement>("button[data-chord-action]");
  if (!button || !result || selectedChordIndex < 0) return;
  editSelectedChord(button.dataset.chordAction ?? "");
});
byId<HTMLButtonElement>("reprocess").addEventListener("click", async () => {
  if (!currentFile) return;
  const previous = result;
  if (!previous) return;
  const locked = previous.chords.filter((chord) => chord.locked).map((chord) => ({ ...chord }));
  const preserved = await reprocessPreservingPrevious(previous, async () => {
    const succeeded = await loadFile(currentFile!, currentSourceUrl, {
      forceTranscriptionCache: true,
      reprocessing: true,
    });
    if (!succeeded || !result) throw new Error("Reprocessing did not produce a replacement.");
    applyLockedChordCorrections(result, locked);
    renderResult(currentFile!, result);
    return result;
  });
  result = preserved.result;
});
byId("cache-management").addEventListener("toggle", () => {
  if ((byId("cache-management") as HTMLDetailsElement).open) void refreshCacheUsage();
});
byId<HTMLButtonElement>("clear-transcription-cache").addEventListener("click", () => {
  void clearCache("transcription");
});
byId<HTMLButtonElement>("clear-separation-cache").addEventListener("click", () => {
  void clearCache("separation");
});
byId<HTMLButtonElement>("clear-all-cache").addEventListener("click", () => {
  if (!window.confirm("Clear all cached transcriptions and separated stems? Expensive separation will need to run again.")) return;
  void clearCache("all");
});
byId<HTMLButtonElement>("open-cache-location").addEventListener("click", async () => {
  const response = await fetch("/api/cache/open", { method: "POST" });
  const body = await response.json() as { location?: string; error?: string };
  byId("cache-message").textContent = response.ok
    ? `Opened ${body.location ?? "processing cache"}.`
    : body.error ?? "Could not open the cache location.";
});

byId("evaluation-tools").addEventListener("toggle", () => {
  if ((byId("evaluation-tools") as HTMLDetailsElement).open) void refreshEvaluationReferences();
});
byId<HTMLFormElement>("evaluation-import-form").addEventListener("submit", (event) => {
  event.preventDefault();
  void importEvaluationReference();
});
byId<HTMLSelectElement>("evaluation-reference-select").addEventListener("change", () => {
  const reference = selectedEvaluationReference();
  evaluationMarks = reference?.alignment?.method === "manual"
    ? reference.alignment.regions.map((region) => ({ chordIndex: region.chordIndex, start: region.start }))
    : [];
  renderEvaluationWorkbench();
});
byId<HTMLButtonElement>("evaluation-mark").addEventListener("click", markNextEvaluationChord);
byId<HTMLButtonElement>("evaluation-undo-mark").addEventListener("click", () => {
  evaluationMarks.pop();
  renderEvaluationWorkbench();
});
byId<HTMLButtonElement>("evaluation-save-alignment").addEventListener("click", () => {
  void saveEvaluationAlignment();
});
byId<HTMLButtonElement>("evaluation-compare").addEventListener("click", () => {
  void compareEvaluationReference();
});
byId<HTMLButtonElement>("evaluation-download-report").addEventListener("click", () => {
  if (!latestEvaluationAggregate) return;
  downloadJson(latestEvaluationAggregate, "chord-evaluation-report.json");
});
document.addEventListener("keydown", (event) => {
  if (
    event.key.toLowerCase() !== "m"
    || !(byId("evaluation-tools") as HTMLDetailsElement).open
    || (event.target as HTMLElement).matches("input, textarea, select")
  ) return;
  event.preventDefault();
  markNextEvaluationChord();
});

type LoadFileOptions = {
  forceTranscriptionCache?: boolean;
  reprocessing?: boolean;
};

async function loadFile(
  file?: File,
  sourceUrl: string | null = null,
  options: LoadFileOptions = {},
): Promise<boolean> {
  if (!file) return false;
  activeController?.abort();
  const generation = ++analysisGeneration;
  const controller = new AbortController();
  activeController = controller;
  currentFile = file;
  currentSourceUrl = sourceUrl;
  status.classList.remove("hidden");
  status.classList.remove("error-status");
  byId<HTMLButtonElement>("cancel-job").classList.remove("hidden");
  byId("status-label").textContent = "Decoding audio…";
  try {
    const [originalBuffer, audioContentHash] = await Promise.all([
      decodeAudio(file),
      hashBlob(file),
    ]);
    const isolateGuitar = byId<HTMLInputElement>("isolate-guitar").checked;
    const arrangementMode = byId<HTMLSelectElement>("arrangement-select").value as ArrangementMode;
    const qualityPreset = byId<HTMLSelectElement>("quality-select").value as QualityPreset;
    const noteOptions: NoteProcessingOptions = {
      mode: arrangementMode,
      preset: qualityPreset,
      separation: isolateGuitar ? "guitar" : "none",
    };
    let buffer = originalBuffer;
    let harmonyBuffer = originalBuffer;
    let bassBuffer: AudioBuffer | null = null;
    let analysisAudioHash = audioContentHash;
    let harmonyAudioHash = audioContentHash;
    let bassAudioHash: string | null = null;
    let separatedStemReused = false;
    if (isolateGuitar) {
      updateStatus("Separating guitar and harmonic evidence with Demucs… first run downloads the model");
      const separated = await requestGuitarStem(file, controller.signal);
      separatedStemReused = separated.reused;
      [analysisAudioHash, harmonyAudioHash, bassAudioHash] = await Promise.all([
        hashBlob(separated.guitarBlob),
        hashBlob(separated.harmonyBlob),
        hashBlob(separated.bassBlob),
      ]);
      updateStatus("Decoding separated harmonic evidence…", 0.02);
      [buffer, harmonyBuffer, bassBuffer] = await Promise.all([
        decodeAudio(separated.guitarBlob),
        decodeAudio(separated.harmonyBlob),
        decodeAudio(separated.bassBlob),
      ]);
    }
    updateStatus("Preparing 22.05 kHz mono audio…", 0.03);
    const mono = await resampleToMono(buffer);
    const harmonyMono = isolateGuitar ? await resampleToMono(harmonyBuffer) : mono;
    const bassMono = bassBuffer ? await resampleToMono(bassBuffer) : undefined;
    const harmonyEvidenceSource = isolateGuitar
      ? "separated-harmonic-mix" as const
      : "full-mix" as const;
    const chordDisplayMode = byId<HTMLSelectElement>("chord-mode-select").value as ChordDisplayMode;
    const useMachineLearning = byId<HTMLSelectElement>("engine-select").value === "ml";
    const detectorPreset = arrangementMode === "raw" ? "raw" : qualityPreset;
    const cleanupPreset = arrangementMode === "raw" || qualityPreset === "raw"
      ? "raw"
      : qualityPreset;
    let cacheSettings: TranscriptionCacheKeyInput = {
      audioContentHash,
      analysisAudioHash,
      harmonyAudioHash,
      bassAudioHash,
      harmonyEvidenceSource,
      engine: useMachineLearning ? "basic-pitch" : "dsp",
      basicPitchModelVersion: BASIC_PITCH_MODEL_VERSION,
      basicPitchThresholds: { ...BASIC_PITCH_THRESHOLDS[detectorPreset] },
      transcriptionPipelineVersion: TRANSCRIPTION_PIPELINE_VERSION,
      noteCleanupVersion: NOTE_CLEANUP_VERSION,
      noteCleanupSettings: { ...NOTE_CLEANUP_PRESETS[cleanupPreset] },
      arrangementVersion: ARRANGEMENT_VERSION,
      arrangementMode,
      qualityPreset,
      tuningMode: "automatic",
      selectedTuning: [...TUNING_NAMES],
      capoMode: "automatic",
      capoPosition: 0,
      fretboardMapperVersion: FRETBOARD_MAPPER_VERSION,
      chordAnalysisVersion: CHORD_ANALYSIS_VERSION,
      chordAnalysisSettings: {
        ...BALANCED_CHORD_SMOOTHING_SETTINGS,
        chordDisplayMode,
      },
      chordDisplayMode,
      tempoQuantizationVersion: TEMPO_QUANTIZATION_VERSION,
      processingRange: { startSeconds: 0, endSeconds: null },
      guitarIsolationEnabled: isolateGuitar,
      sourceSeparationModelVersion: SOURCE_SEPARATION_MODEL_VERSION,
    };
    let transcriptionCacheKey = await buildTranscriptionCacheKey(cacheSettings);
    let legacyInvalidatedReason: string | null = null;

    if (options.forceTranscriptionCache) {
      updateStatus("Reprocessing with updated pipeline…", 0.05);
    } else {
      const cached = await lookupTranscriptionCache(
        transcriptionCacheKey,
        controller.signal,
      );
      const decision = cacheReadDecision(cached.entry);
      if (decision.reuse && cached.entry) {
        if (controller.signal.aborted || generation !== analysisGeneration) return false;
        result = cached.entry.result;
        lastCacheOutcome = {
          transcription: "loaded-current",
          separatedStemReused,
        };
        updateStatus("Loaded current transcription cache", 1);
        status.classList.add("hidden");
        renderResult(file, result);
        return true;
      }
      if (cached.status === "stale" || decision.status === "stale") {
        legacyInvalidatedReason = cached.reason ?? decision.reason;
        console.warn(`Ignoring stale transcription cache: ${legacyInvalidatedReason}`);
        updateStatus("Legacy cache invalidated", 0.05);
      }
    }

    let analysis: AnalysisResult;
    if (useMachineLearning) {
      try {
        updateStatus("Loading the local Basic Pitch model…", 0.06);
        const harmonyPromise = runAnalysisWorker<HarmonyResult>(
          harmonyMono,
          "harmony",
          controller.signal,
          undefined,
          {
            bassSamples: bassMono,
            source: harmonyEvidenceSource,
            chordSettings: { chordDisplayMode },
          },
        );
        const notesPromise = transcribeWithBasicPitch(mono, (progress) => {
          updateStatus(`Transcribing polyphonic notes… ${Math.round(progress * 100)}%`, (isolateGuitar ? 0.48 : 0.08) + progress * (isolateGuitar ? 0.48 : 0.84));
        }, controller.signal, detectorPreset);
        const [harmony, rawNotes] = await Promise.all([harmonyPromise, notesPromise]);
        const noteProcessing = processDetectedNotes(
          rawNotes,
          harmony.duration,
          harmony.bpm,
          harmony.chords,
          harmony.chordAnalysis.diagnostics.rawChordChanges,
          noteOptions,
        );
        analysis = {
          ...harmony,
          ...noteProcessing,
          engine: "basic-pitch",
          separation: isolateGuitar ? "guitar" : "none",
        };
      } catch (modelError) {
        if (controller.signal.aborted) throw modelError;
        console.warn("Basic Pitch unavailable; using DSP fallback.", modelError);
        updateStatus("Neural model unavailable—using the local fallback…");
        analysis = await runDspAnalysis(
          mono,
          harmonyMono,
          bassMono,
          harmonyEvidenceSource,
          chordDisplayMode,
          isolateGuitar,
          noteOptions,
          controller.signal,
        );
      }
    } else {
      updateStatus("Finding notes and chords with the fast engine…");
      analysis = await runDspAnalysis(
        mono,
        harmonyMono,
        bassMono,
        harmonyEvidenceSource,
        chordDisplayMode,
        isolateGuitar,
        noteOptions,
        controller.signal,
      );
    }
    if (controller.signal.aborted || generation !== analysisGeneration) return false;
    analysis.separation = isolateGuitar ? "guitar" : "none";
    if (analysis.engine !== cacheSettings.engine) {
      cacheSettings = { ...cacheSettings, engine: analysis.engine };
      transcriptionCacheKey = await buildTranscriptionCacheKey(cacheSettings);
    }
    const entry: CachedTranscriptionEntry = {
      metadata: createCacheMetadata(),
      cacheKey: transcriptionCacheKey,
      settings: cacheSettings,
      result: analysis,
    };
    await storeTranscriptionCache(entry, controller.signal).catch((cacheError) => {
      console.warn("The fresh transcription could not be cached.", cacheError);
    });
    result = analysis;
    lastCacheOutcome = {
      transcription: options.reprocessing
        ? "reprocessed"
        : legacyInvalidatedReason
          ? "legacy-invalidated"
          : "fresh",
      separatedStemReused,
      detail: legacyInvalidatedReason ?? undefined,
    };
    updateStatus(
      options.reprocessing
        ? "Reprocessing with updated pipeline completed"
        : "Fresh transcription completed",
      1,
    );
    status.classList.add("hidden");
    renderResult(file, analysis);
    return true;
  } catch (error) {
    if (controller.signal.aborted || generation !== analysisGeneration) {
      status.classList.add("hidden");
    } else showError(error instanceof Error ? error.message : "This audio format could not be decoded.");
    return false;
  } finally {
    if (generation === analysisGeneration) {
      activeController = null;
      activeJobId = null;
      byId<HTMLButtonElement>("cancel-job").classList.add("hidden");
    }
    void refreshJobs();
  }
}

async function requestGuitarStem(
  file: File,
  signal: AbortSignal,
): Promise<{
  guitarBlob: Blob;
  bassBlob: Blob;
  harmonyBlob: Blob;
  reused: boolean;
}> {
  const extension = file.name.split(".").at(-1)?.toLowerCase() || "audio";
  const response = await fetch("/api/jobs/separation", {
    method: "POST",
    headers: {
      "Content-Type": file.type || "application/octet-stream",
      "X-Input-Extension": extension,
      "X-Input-Name": encodeURIComponent(file.name),
    },
    body: file,
    signal,
  });
  if (!response.ok) throw new Error(await responseError(response));
  const created = await response.json() as ProcessingJob;
  activeJobId = created.id;
  await refreshJobs();
  while (true) {
    await abortableDelay(350, signal);
    const jobResponse = await fetch(`/api/jobs/${encodeURIComponent(created.id)}`, { signal });
    if (!jobResponse.ok) throw new Error(await responseError(jobResponse));
    const job = await jobResponse.json() as ProcessingJob;
    updateStatus(`${job.stage}… ${Math.round(job.progress * 100)}%`, job.progress * 0.44);
    void refreshJobs();
    if (job.status === "failed") throw new Error(job.error || "Guitar separation failed.");
    if (job.status === "cancelled") throw new DOMException("Processing was cancelled.", "AbortError");
    if (job.status === "completed") {
      const [guitarResponse, bassResponse, harmonyResponse] = await Promise.all(
        ["guitar", "bass", "harmony"].map((stem) => fetch(
          `/api/jobs/${encodeURIComponent(created.id)}/result?stem=${stem}`,
          { signal },
        )),
      );
      for (const stemResponse of [guitarResponse, bassResponse, harmonyResponse]) {
        if (!stemResponse.ok) throw new Error(await responseError(stemResponse));
      }
      return {
        guitarBlob: await guitarResponse.blob(),
        bassBlob: await bassResponse.blob(),
        harmonyBlob: await harmonyResponse.blob(),
        reused: job.cached,
      };
    }
  }
}

function abortableDelay(milliseconds: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const abort = () => {
      window.clearTimeout(timer);
      reject(new DOMException("Processing was cancelled.", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    signal.addEventListener("abort", abort, { once: true });
  });
}

async function decodeAudio(file: Blob): Promise<AudioBuffer> {
  const context = new AudioContext();
  try {
    return await context.decodeAudioData(await file.arrayBuffer());
  } finally {
    await context.close();
  }
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await responseError(response));
  return response.json() as Promise<T>;
}

async function responseError(response: Response): Promise<string> {
  try {
    const body = await response.json() as { error?: string };
    return body.error || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

function evaluationInput(id: string): string {
  return byId<HTMLInputElement>(id).value.trim();
}

function selectedEvaluationReference(): EvaluationReference | null {
  const id = byId<HTMLSelectElement>("evaluation-reference-select").value;
  return evaluationReferences.find((reference) => reference.id === id) ?? null;
}

async function refreshEvaluationReferences(preferredId?: string): Promise<void> {
  const message = byId("evaluation-message");
  try {
    const response = await fetch("/api/evaluation/references");
    if (!response.ok) throw new Error(await responseError(response));
    const body = await response.json() as { references: EvaluationReference[] };
    evaluationReferences = body.references;
    const select = byId<HTMLSelectElement>("evaluation-reference-select");
    const current = preferredId || select.value;
    select.innerHTML = evaluationReferences.length
      ? evaluationReferences.map((reference) =>
        `<option value="${escapeHtml(reference.id)}">${escapeHtml(reference.artist)} — ${escapeHtml(reference.title)} · v${reference.version || "?"}</option>`,
      ).join("")
      : `<option value="">No references imported</option>`;
    if (evaluationReferences.some((reference) => reference.id === current)) select.value = current;
    const reference = selectedEvaluationReference();
    evaluationMarks = reference?.alignment?.method === "manual"
      ? reference.alignment.regions.map((region) => ({ chordIndex: region.chordIndex, start: region.start }))
      : [];
    message.textContent = evaluationReferences.length
      ? `${evaluationReferences.length} local chord-only reference${evaluationReferences.length === 1 ? "" : "s"} available.`
      : "No local references yet.";
    renderEvaluationWorkbench();
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "Could not load evaluation references.";
  }
}

async function importEvaluationReference(): Promise<void> {
  const message = byId("evaluation-message");
  message.textContent = "Validating chord-only reference…";
  try {
    const imported = await postJson<{ reference: EvaluationReference }>(
      "/api/evaluation/references/import",
      {
        source: "ultimate-guitar",
        artist: evaluationInput("evaluation-artist"),
        title: evaluationInput("evaluation-title"),
        referenceUrl: evaluationInput("evaluation-url"),
        version: Number(evaluationInput("evaluation-version")),
        ratingCount: Number(evaluationInput("evaluation-rating-count")),
        averageRating: Number(evaluationInput("evaluation-average-rating")),
        capo: Number(evaluationInput("evaluation-capo")),
        tuning: evaluationInput("evaluation-tuning"),
        key: evaluationInput("evaluation-key"),
        genre: evaluationInput("evaluation-genre"),
        split: byId<HTMLSelectElement>("evaluation-split").value,
        selectionReason: evaluationInput("evaluation-selection-reason"),
        chordSequence: byId<HTMLTextAreaElement>("evaluation-sequence").value,
        duration: result?.duration ?? null,
        recordingNotes: currentFile?.name
          ? `Compared with locally supplied audio: ${currentFile.name}`
          : "",
      },
    );
    byId<HTMLTextAreaElement>("evaluation-sequence").value = "";
    message.textContent = `Imported ${imported.reference.chords.length} normalized chord events. The pasted text was not stored.`;
    await refreshEvaluationReferences(imported.reference.id);
  } catch (error) {
    message.textContent = error instanceof Error ? error.message : "The chord reference could not be imported.";
  }
}

function markNextEvaluationChord(): void {
  const reference = selectedEvaluationReference();
  if (!reference) {
    byId("evaluation-message").textContent = "Select an imported reference first.";
    return;
  }
  if (evaluationMarks.length >= reference.chords.length) {
    byId("evaluation-message").textContent = "Every reference chord is already marked. Undo a mark to change it.";
    return;
  }
  const previous = evaluationMarks.at(-1);
  if (previous && player.currentTime <= previous.start) {
    byId("evaluation-message").textContent = "Move playback after the previous mark before marking the next chord.";
    return;
  }
  evaluationMarks.push({ chordIndex: evaluationMarks.length, start: player.currentTime });
  byId("evaluation-message").textContent = `Marked chord ${evaluationMarks.length} of ${reference.chords.length} at ${formatPreciseTime(player.currentTime)}.`;
  renderEvaluationWorkbench();
}

async function saveEvaluationAlignment(): Promise<void> {
  const reference = selectedEvaluationReference();
  if (!reference || !result) {
    byId("evaluation-message").textContent = "Load a transcription and select a reference first.";
    return;
  }
  try {
    const response = await fetch(`/api/evaluation/references/${encodeURIComponent(reference.id)}/alignment`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ marks: evaluationMarks, duration: result.duration }),
    });
    if (!response.ok) throw new Error(await responseError(response));
    const body = await response.json() as { reference: EvaluationReference };
    byId("evaluation-message").textContent = "Saved verified manual timestamp alignment.";
    await refreshEvaluationReferences(body.reference.id);
  } catch (error) {
    byId("evaluation-message").textContent = error instanceof Error ? error.message : "Could not save the alignment.";
  }
}

async function compareEvaluationReference(): Promise<void> {
  const reference = selectedEvaluationReference();
  if (!reference || !result) {
    byId("evaluation-message").textContent = "Load a transcription and select a reference first.";
    return;
  }
  byId("evaluation-message").textContent = reference.alignment?.verified
    ? "Comparing against the verified alignment…"
    : "Comparing with experimental sequence alignment; results will be labelled unverified…";
  try {
    const response = await postJson<{
      report: {
        alignment: { method: string; verified: boolean; confidence: number };
        metrics: Record<string, number | null>;
        sequence: Record<string, number>;
        mismatches: Array<{
          start: number;
          end: number;
          reference: string;
          detected: string;
          errorType: string;
          likelyFix: string;
        }>;
      };
      aggregate: unknown;
    }>(`/api/evaluation/references/${encodeURIComponent(reference.id)}/compare`, {
      prediction: {
        duration: result.duration,
        chords: result.chords,
        chordAnalysis: result.chordAnalysis,
      },
    });
    latestEvaluationAggregate = response.aggregate;
    byId<HTMLButtonElement>("evaluation-download-report").disabled = false;
    renderEvaluationReport(response.report);
    byId("evaluation-message").textContent = response.report.alignment.verified
      ? "Evaluation complete. JSON and Markdown reports were saved in the separate local evaluation directory."
      : "Provisional evaluation complete. Verify the automatic sequence alignment before using it as ground truth.";
  } catch (error) {
    byId("evaluation-message").textContent = error instanceof Error ? error.message : "Could not compare chord sequences.";
  }
}

function renderEvaluationWorkbench(): void {
  const reference = selectedEvaluationReference();
  const summary = byId("evaluation-reference-summary");
  const markerPanel = byId("evaluation-markers");
  if (!reference) {
    summary.textContent = "Import a chord-only reference to begin.";
    markerPanel.innerHTML = "";
    return;
  }
  const selection = reference.ratingCount
    ? `${reference.ratingCount.toLocaleString()} ratings · ${reference.averageRating.toFixed(1)}/5`
    : "rating metadata not entered";
  summary.innerHTML = `
    <strong>${escapeHtml(reference.artist)} — ${escapeHtml(reference.title)}</strong>
    <span>${reference.chords.length} normalized chords · capo ${reference.capo} · ${escapeHtml(reference.tuning)} · ${selection}</span>
    <span>${escapeHtml(reference.selectionReason)}</span>
    <span>Alignment: ${reference.alignment
      ? `${escapeHtml(reference.alignment.method)} · ${reference.alignment.verified ? "verified" : "unverified"}`
      : "not timestamped"}</span>`;
  const next = reference.chords[evaluationMarks.length];
  markerPanel.innerHTML = `
    <div><strong>${evaluationMarks.length}/${reference.chords.length} marked</strong>${next ? ` · next: ${escapeHtml(next.sounding)} (${escapeHtml(next.section)})` : " · alignment complete"}</div>
    <div class="marker-list">${evaluationMarks.slice(-10).map((mark) => {
      const chord = reference.chords[mark.chordIndex];
      return `<span>${formatPreciseTime(mark.start)} ${escapeHtml(chord?.sounding ?? "?")}</span>`;
    }).join("")}</div>`;
}

function renderEvaluationReport(report: {
  alignment: { method: string; verified: boolean; confidence: number };
  metrics: Record<string, number | null>;
  sequence: Record<string, number>;
  mismatches: Array<{
    start: number;
    end: number;
    reference: string;
    detected: string;
    errorType: string;
    likelyFix: string;
  }>;
}): void {
  const percent = (value: number | null | undefined) =>
    Number.isFinite(value) ? `${((value as number) * 100).toFixed(1)}%` : "n/a";
  byId("evaluation-report").innerHTML = `
    <div class="evaluation-metrics">
      <span><strong>${percent(report.metrics.rootAccuracy)}</strong> root</span>
      <span><strong>${percent(report.metrics.majorMinorAccuracy)}</strong> major/minor</span>
      <span><strong>${percent(report.metrics.detailedAccuracy)}</strong> detailed</span>
      <span><strong>${report.metrics.medianBoundaryErrorMs ?? "n/a"} ms</strong> median boundary</span>
      <span><strong>${percent(report.metrics.falseNoChordDurationPercent)}</strong> false N duration</span>
      <span><strong>${report.sequence.extraDetectedChords ?? 0}</strong> extra regions</span>
    </div>
    <div class="evaluation-confidence">${report.alignment.verified
      ? "Manually verified alignment"
      : `Automatic ${escapeHtml(report.alignment.method)} alignment · unverified · ${Math.round(report.alignment.confidence * 100)}% alignment confidence`}</div>
    <div class="evaluation-mismatches">
      ${report.mismatches.length
        ? report.mismatches.slice(0, 30).map((mismatch) => `<button type="button" data-evaluation-time="${mismatch.start}">
          <span>${formatTime(mismatch.start)}–${formatTime(mismatch.end)}</span>
          <strong>${escapeHtml(mismatch.reference)} → ${escapeHtml(mismatch.detected)}</strong>
          <span>${escapeHtml(mismatch.errorType)}</span>
          <small>${escapeHtml(mismatch.likelyFix)}</small>
        </button>`).join("")
        : `<div class="empty">No detailed chord mismatches in the evaluated interval.</div>`}
    </div>`;
  byId("evaluation-report").querySelectorAll<HTMLButtonElement>("button[data-evaluation-time]").forEach((button) => {
    button.addEventListener("click", () => {
      player.currentTime = Number(button.dataset.evaluationTime);
      syncPlayback();
    });
  });
}

function downloadJson(value: unknown, name: string): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function resampleToMono(buffer: AudioBuffer): Promise<Float32Array> {
  const sampleRate = 22050;
  const frameCount = Math.max(1, Math.ceil(buffer.duration * sampleRate));
  const context = new OfflineAudioContext(1, frameCount, sampleRate);
  const source = context.createBufferSource();
  source.buffer = buffer;
  source.connect(context.destination);
  source.start();
  const rendered = await context.startRendering();
  return rendered.getChannelData(0).slice();
}

async function hashBlob(blob: Blob): Promise<string> {
  return sha256Hex(await blob.arrayBuffer());
}

async function lookupTranscriptionCache(
  key: string,
  signal?: AbortSignal,
): Promise<{
  status: "hit" | "miss" | "stale";
  entry: CachedTranscriptionEntry | null;
  reason: string | null;
}> {
  const response = await fetch("/api/cache/transcriptions/lookup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ key }),
    signal,
  });
  const body = await response.json() as {
    status?: "hit" | "miss" | "stale";
    entry?: CachedTranscriptionEntry | null;
    reason?: string | null;
    error?: string;
  };
  if (response.status !== 404 && !response.ok) {
    throw new Error(body.error ?? "Transcription cache lookup failed.");
  }
  return {
    status: body.status ?? "miss",
    entry: body.entry ?? null,
    reason: body.reason ?? null,
  };
}

async function storeTranscriptionCache(
  entry: CachedTranscriptionEntry,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(
    `/api/cache/transcriptions/${encodeURIComponent(entry.cacheKey)}`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(entry),
      signal,
    },
  );
  if (!response.ok) throw new Error(await responseError(response));
}

type CacheUsage = {
  separation: { bytes: number; files: number };
  transcription: { bytes: number; files: number };
  total: { bytes: number; files: number };
  location: string;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

async function refreshCacheUsage(): Promise<void> {
  const response = await fetch("/api/cache/status");
  const body = await response.json() as CacheUsage & { error?: string };
  if (!response.ok) {
    byId("cache-usage").textContent = body.error ?? "Could not calculate cache usage.";
    return;
  }
  byId("cache-usage").textContent = [
    `Transcriptions: ${formatBytes(body.transcription.bytes)} (${body.transcription.files})`,
    `Separated stems: ${formatBytes(body.separation.bytes)} (${body.separation.files})`,
    `Total: ${formatBytes(body.total.bytes)}`,
  ].join(" · ");
}

async function clearCache(
  kind: "transcription" | "separation" | "all",
): Promise<void> {
  const response = await fetch(`/api/cache/${kind}`, { method: "DELETE" });
  const body = await response.json() as CacheUsage & { error?: string };
  byId("cache-message").textContent = response.ok
    ? kind === "transcription"
      ? "Transcription cache cleared. Compatible separated stems were kept."
      : kind === "separation"
        ? "Separated-stem cache cleared. Transcriptions were kept."
        : "All transcription and separation processing data was cleared."
    : body.error ?? "Could not clear processing cache.";
  if (response.ok) await refreshCacheUsage();
}

async function runDspAnalysis(
  noteSamples: Float32Array,
  harmonySamples: Float32Array,
  bassSamples: Float32Array | undefined,
  harmonyEvidenceSource: HarmonyEvidenceOptions["source"],
  chordDisplayMode: ChordDisplayMode,
  isolated: boolean,
  noteOptions: NoteProcessingOptions,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  if (!isolated) {
    return runAnalysisWorker<AnalysisResult>(
      noteSamples,
      "full",
      signal,
      noteOptions,
      {
        source: harmonyEvidenceSource,
        chordSettings: { chordDisplayMode },
      },
    );
  }
  const [harmony, noteResult] = await Promise.all([
    runAnalysisWorker<HarmonyResult>(
      harmonySamples,
      "harmony",
      signal,
      undefined,
      {
        bassSamples,
        source: harmonyEvidenceSource,
        chordSettings: { chordDisplayMode },
      },
    ),
    runAnalysisWorker<AnalysisResult>(noteSamples, "full", signal, noteOptions),
  ]);
  return {
    ...harmony,
    notes: noteResult.notes,
    noteAnalysis: noteResult.noteAnalysis,
    tuning: noteResult.tuning,
    capo: noteResult.capo,
    engine: "dsp",
    separation: "guitar",
  };
}

function runAnalysisWorker<T>(
  samples: Float32Array,
  mode: "full" | "harmony",
  signal?: AbortSignal,
  noteOptions?: NoteProcessingOptions,
  harmonyOptions?: HarmonyEvidenceOptions & {
    chordSettings?: Partial<typeof BALANCED_CHORD_SMOOTHING_SETTINGS>;
  },
): Promise<T> {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./analysis.worker.ts", import.meta.url), { type: "module" });
    const abort = () => { worker.terminate(); reject(new DOMException("Processing was cancelled.", "AbortError")); };
    if (signal?.aborted) return abort();
    signal?.addEventListener("abort", abort, { once: true });
    const copy = samples.slice();
    const bassCopy = harmonyOptions?.bassSamples?.slice();
    const transfer = [copy.buffer];
    if (bassCopy) transfer.push(bassCopy.buffer);
    worker.postMessage({
      samples: copy,
      sampleRate: 22050,
      mode,
      noteOptions,
      bassSamples: bassCopy,
      harmonySource: harmonyOptions?.source,
      chordSettings: harmonyOptions?.chordSettings,
    }, transfer);
    worker.onmessage = (event: MessageEvent<{ type: string; result?: T; message?: string }>) => {
      signal?.removeEventListener("abort", abort);
      worker.terminate();
      if (event.data.type === "error" || !event.data.result) reject(new Error(event.data.message ?? "Analysis failed"));
      else resolve(event.data.result);
    };
    worker.onerror = () => { signal?.removeEventListener("abort", abort); worker.terminate(); reject(new Error("The analysis worker stopped unexpectedly.")); };
  });
}

async function refreshJobs(): Promise<void> {
  try {
    const response = await fetch("/api/jobs");
    if (!response.ok) return;
    const { jobs } = await response.json() as { jobs: ProcessingJob[] };
    const history = byId("job-history");
    history.classList.toggle("hidden", jobs.length === 0);
    byId("job-list").innerHTML = jobs.slice(0, 6).map((job) => `
      <div class="history-job">
        <span><strong>${escapeHtml(job.name)}</strong><small>${new Date(job.createdAt).toLocaleString()}</small></span>
        <span class="job-state ${job.status}">${job.cached ? "reused stem · " : ""}${escapeHtml(job.status)}</span>
      </div>`).join("");
  } catch { /* History is non-critical. */ }
}

function updateStatus(label: string, progress?: number): void {
  status.classList.remove("error-status");
  byId("status-label").textContent = label;
  status.classList.toggle("determinate", progress !== undefined);
  if (progress !== undefined) byId("analysis-progress").style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
}

function showError(message: string): void {
  status.classList.remove("hidden");
  status.classList.add("error-status");
  byId("status-label").textContent = `Could not analyze audio · ${message}`;
}

function renderResult(file: File, analysis: AnalysisResult): void {
  lastActiveChordIndex = -2;
  const resultsElement = byId("results");
  resultsElement.classList.remove("hidden");
  resultsElement.dataset.diagnostics = JSON.stringify({
    notes: analysis.noteAnalysis.diagnostics,
    chords: analysis.chordAnalysis.diagnostics,
  });
  byId("song-title").textContent = file.name.replace(/\.[^.]+$/, "");
  if (!byId<HTMLInputElement>("evaluation-title").value) {
    byId<HTMLInputElement>("evaluation-title").value = file.name.replace(/\.[^.]+$/, "");
  }
  const minutes = Math.floor(analysis.duration / 60);
  const seconds = Math.round(analysis.duration % 60).toString().padStart(2, "0");
  const engineName = analysis.engine === "basic-pitch" ? "Basic Pitch ML" : "fast DSP";
  const sourceName = analysis.separation === "guitar" ? "isolated guitar · Demucs 6-stem" : "full mix";
  const setupName = `${analysis.tuning.join(" ")}${analysis.capo ? ` · capo ${analysis.capo}` : ""}`;
  const countLabels = analysisCountLabels(analysis);
  byId("metadata").innerHTML = [
    `<strong>${countLabels.main}</strong>`,
    countLabels.raw,
    countLabels.cleaned,
    countLabels.chords,
    `${minutes}:${seconds}`,
    `${analysis.bpm ?? "—"} BPM`,
    escapeHtml(setupName),
    escapeHtml(sourceName),
    escapeHtml(engineName),
  ].join(" · ");
  const requestedMode = analysis.noteAnalysis.requestedMode === "automatic"
    ? `Automatic → ${analysis.noteAnalysis.resolvedMode}`
    : analysis.noteAnalysis.requestedMode;
  const processingStatuses = cacheOutcomeLabels(lastCacheOutcome ?? {
    transcription: "fresh",
    separatedStemReused: false,
  });
  byId("pipeline-status").textContent = [
    `Arrangement: ${capitalize(requestedMode)}`,
    `Quality: ${capitalize(analysis.noteAnalysis.preset)}`,
    `Chords: ${capitalize(analysis.chordAnalysis.settings.chordDisplayMode)}`,
    `Harmony: ${analysis.chordAnalysis.diagnostics.harmonyEvidenceSource.replaceAll("-", " ")}`,
    `Key: ${analysis.chordAnalysis.diagnostics.keyEstimate?.name ?? "uncertain"}`,
    `Pipeline: ${pipelineLabel()}`,
    ...processingStatuses,
  ].join(" · ");
  const qualityGate = byId("quality-gate");
  qualityGate.classList.toggle("hidden", analysis.noteAnalysis.warnings.length === 0);
  qualityGate.innerHTML = analysis.noteAnalysis.warnings.length ? `
    <div class="quality-summary"><strong>Transcription quality check</strong><span>${escapeHtml(analysis.noteAnalysis.resolvedMode)} arrangement · ${escapeHtml(analysis.noteAnalysis.preset)} preset</span></div>
    ${analysis.noteAnalysis.warnings.map((warning) => `<div class="quality-warning"><span>${escapeHtml(warning.message)}</span><small>${escapeHtml(warning.suggestion)}</small></div>`).join("")}
  ` : "";
  player.src = URL.createObjectURL(file);
  byId("total-time").textContent = formatTime(analysis.duration);
  byId<HTMLInputElement>("song-scrubber").max = String(analysis.duration);
  selectedChordIndex = -1;
  renderChordTrack(analysis);
  renderChordEditor(analysis);
  renderChordDiagnostics(analysis);
  renderEvaluationWorkbench();
  lastNoteViewKey = "";
  renderNoteList(true);
  renderTabScore(analysis);
  syncPlayback();
  byId("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function chordConfidenceLabel(chord: ChordEvent): string {
  const level = chord.diagnostics?.confidenceLevel
    ?? (chord.confidence >= 0.72 ? "high" : chord.confidence >= 0.42 ? "medium" : "low");
  return `${capitalize(level)} confidence`;
}

function renderChordTrack(analysis: AnalysisResult): void {
  byId("chords").innerHTML = analysis.chords.length
    ? analysis.chords.map((chord, index) => {
      const width = Math.max(104, Math.min(250, (chord.end - chord.start) * 34));
      const level = chord.diagnostics?.confidenceLevel
        ?? (chord.confidence >= 0.72 ? "high" : chord.confidence >= 0.42 ? "medium" : "low");
      return `<button class="chord confidence-${level}${index === selectedChordIndex ? " selected" : ""}${chord.locked ? " locked" : ""}" data-index="${index}" data-time="${chord.start}" style="width:${width}px;--played:0%">
        <span class="chord-fill"></span>
        <span class="chord-content"><strong>${escapeHtml(chord.name)}${chord.locked ? " 🔒" : ""}</strong><small>${formatTime(chord.start)}–${formatTime(chord.end)} · ${escapeHtml(chordConfidenceLabel(chord))}</small></span>
      </button>`;
    }).join("")
    : `<div class="empty">No stable chords found.</div>`;
  document.querySelectorAll<HTMLButtonElement>(".chord").forEach((item) => {
    item.addEventListener("click", () => {
      if (!result) return;
      selectedChordIndex = Number(item.dataset.index);
      player.currentTime = Number(item.dataset.time);
      renderChordTrack(result);
      renderChordEditor(result);
      renderChordDiagnostics(result);
      syncPlayback();
    });
  });
}

function renderChordEditor(analysis: AnalysisResult): void {
  const editor = byId("chord-editor");
  const chord = analysis.chords[selectedChordIndex];
  editor.classList.toggle("hidden", !chord);
  if (!chord) {
    editor.innerHTML = "";
    return;
  }
  editor.innerHTML = `
    <div class="chord-editor-title"><strong>Chord correction · ${formatTime(chord.start)}–${formatTime(chord.end)}</strong><button data-chord-action="close" type="button">Close</button></div>
    <div class="chord-editor-fields">
      <label>Label <input id="manual-chord-label" value="${escapeHtml(chord.name)}" aria-label="Corrected chord label"></label>
      <label><input id="manual-chord-lock" type="checkbox"${chord.locked ? " checked" : ""}> Lock correction</label>
      <button data-chord-action="apply" type="button">Apply</button>
      <button data-chord-action="no-chord" type="button">Mark N</button>
      <button data-chord-action="split" type="button">Split at playback</button>
      <button data-chord-action="merge-previous" type="button"${selectedChordIndex === 0 ? " disabled" : ""}>Merge previous</button>
      <button data-chord-action="merge-next" type="button"${selectedChordIndex >= analysis.chords.length - 1 ? " disabled" : ""}>Merge next</button>
    </div>`;
}

function renderChordDiagnostics(analysis: AnalysisResult): void {
  const panel = byId("chord-debug");
  if (!chordDebugVisible) return;
  const diagnostics = analysis.chordAnalysis.diagnostics;
  const focusTime = analysis.chords[selectedChordIndex]?.start ?? player.currentTime;
  const nearby = analysis.chordAnalysis.windows
    .filter((window) => Math.abs((window.start + window.end) / 2 - focusTime) <= 4)
    .slice(0, 12);
  panel.innerHTML = `
    <div class="chord-debug-summary">
      <strong>Harmony diagnostics</strong>
      <span>Evidence: ${escapeHtml(diagnostics.harmonyEvidenceSource)}</span>
      <span>Key: ${escapeHtml(diagnostics.keyEstimate?.name ?? "uncertain")} (${Math.round((diagnostics.keyEstimate?.confidence ?? 0) * 100)}%)</span>
      <span>Bass-supported windows: ${diagnostics.bassSupportedWindows}/${diagnostics.analysisWindows}</span>
      <span>Smoothing overrides: ${diagnostics.smoothingOverrides}</span>
      <span>N regions: ${diagnostics.noChordRegions}</span>
    </div>
    <div class="chord-debug-table">
      <div class="debug-row debug-head"><span>Time</span><span>Candidates</span><span>Bass/root</span><span>Key</span><span>Beat</span><span>Final</span><span>Reasoning</span></div>
      ${nearby.map((window) => `<div class="debug-row">
        <span>${formatTime(window.start)}</span>
        <span>${escapeHtml(window.topCandidateChord)} ${window.topCandidateScore.toFixed(2)} / ${escapeHtml(window.secondBestChord)} ${window.secondBestScore.toFixed(2)} · Δ${window.scoreMargin.toFixed(2)}</span>
        <span>${escapeHtml(window.bassRootEstimate ?? "—")} ${Math.round(window.bassRootConfidence * 100)}%</span>
        <span>${escapeHtml(window.keyEstimate ?? "—")}</span>
        <span>${Math.round(window.beatStrength * 100)}%</span>
        <span>${escapeHtml(window.finalLabel)}</span>
        <span>${[
          window.usedBassSupport ? "bass" : "",
          window.usedSmoothingOverride ? "smoothed" : "",
          window.mergedFromNearbyWindows ? "merged" : "",
        ].filter(Boolean).join(", ") || "observation"}</span>
      </div>`).join("")}
    </div>`;
}

function normalizeManualChord(value: string): string | null {
  const normalized = value.trim().replaceAll("♯", "#").replaceAll("♭", "b");
  if (normalized.toUpperCase() === "N") return "N";
  return /^[A-G](?:#|b)?(?:maj7|m7|sus2|sus4|add9|dim|aug|m|7)?$/.test(normalized)
    ? normalized
    : null;
}

function syncEditedChords(analysis: AnalysisResult): void {
  analysis.chordAnalysis.regions = analysis.chords;
  analysis.chordAnalysis.diagnostics.finalChordRegions = analysis.chords.length;
  analysis.chordAnalysis.diagnostics.noChordRegions = analysis.chords.filter(
    (chord) => chord.name === "N",
  ).length;
  renderChordTrack(analysis);
  renderChordEditor(analysis);
  renderChordDiagnostics(analysis);
  syncPlayback();
}

function editSelectedChord(action: string): void {
  if (!result) return;
  const chord = result.chords[selectedChordIndex];
  if (!chord) return;
  if (action === "close") {
    selectedChordIndex = -1;
    renderChordTrack(result);
    renderChordEditor(result);
    return;
  }
  if (action === "apply" || action === "no-chord") {
    const requested = action === "no-chord"
      ? "N"
      : normalizeManualChord(byId<HTMLInputElement>("manual-chord-label").value);
    if (!requested) {
      byId<HTMLInputElement>("manual-chord-label").setCustomValidity("Enter a chord such as A, Am, E7, F# or N.");
      byId<HTMLInputElement>("manual-chord-label").reportValidity();
      return;
    }
    chord.name = requested;
    chord.locked = byId<HTMLInputElement>("manual-chord-lock").checked || action === "no-chord";
    if (chord.diagnostics) {
      chord.diagnostics.finalLabel = requested;
      chord.diagnostics.usedSmoothingOverride = true;
    }
  } else if (action === "split") {
    const split = player.currentTime;
    if (split <= chord.start + 0.1 || split >= chord.end - 0.1) return;
    const right = {
      ...chord,
      start: split,
      diagnostics: chord.diagnostics ? { ...chord.diagnostics } : undefined,
    };
    chord.end = split;
    chord.locked = true;
    right.locked = true;
    result.chords.splice(selectedChordIndex + 1, 0, right);
  } else if (action === "merge-previous" && selectedChordIndex > 0) {
    const previous = result.chords[selectedChordIndex - 1];
    chord.start = previous.start;
    chord.locked ||= previous.locked;
    if (chord.diagnostics) chord.diagnostics.mergedFromShortRegions = true;
    result.chords.splice(selectedChordIndex - 1, 1);
    selectedChordIndex -= 1;
  } else if (action === "merge-next" && selectedChordIndex + 1 < result.chords.length) {
    const next = result.chords[selectedChordIndex + 1];
    chord.end = next.end;
    chord.locked ||= next.locked;
    if (chord.diagnostics) chord.diagnostics.mergedFromShortRegions = true;
    result.chords.splice(selectedChordIndex + 1, 1);
  }
  syncEditedChords(result);
}

function applyLockedChordCorrections(
  analysis: AnalysisResult,
  corrections: ChordEvent[],
): void {
  for (const correction of corrections) {
    const midpoint = (correction.start + correction.end) / 2;
    const target = analysis.chords.find(
      (chord) => midpoint >= chord.start && midpoint < chord.end,
    );
    if (!target) continue;
    target.name = correction.name;
    target.locked = true;
    if (target.diagnostics) {
      target.diagnostics.finalLabel = correction.name;
      target.diagnostics.usedSmoothingOverride = true;
    }
  }
  analysis.chordAnalysis.regions = analysis.chords;
}

function renderNoteList(force = false): void {
  if (!result) return;
  const view = byId<HTMLSelectElement>("note-view-select").value as NoteEditorView;
  const key = noteEditorWindowKey(result, player.currentTime, view);
  if (!force && key === lastNoteViewKey) return;
  lastNoteViewKey = key;
  const visible = selectNoteEditorRows(result, player.currentTime, view);

  byId("notes").innerHTML = visible.length ? visible.map((row) => {
    if (row.kind === "raw") {
      const { note, index } = row;
      const live = player.currentTime >= note.start && player.currentTime < note.end;
      return `<div class="note raw-note${live ? " live" : ""}" data-raw-index="${index}" data-start="${note.start}" data-end="${note.end}">
        <span class="note-name">${escapeHtml(note.name)}</span>
        <span>${formatPreciseTime(note.start)} – ${formatPreciseTime(note.end)}</span>
        <span class="raw-label">RAW</span><span>diagnostic</span>
        <span class="confidence">${Math.round(note.confidence * 100)}% detector</span>
      </div>`;
    }
    const { note, index } = row;
    const live = player.currentTime >= note.start && player.currentTime < note.end;
    return `<div class="note${live ? " live" : ""}" data-index="${index}" data-start="${note.start}" data-end="${note.end}">
        <span class="note-name">${escapeHtml(note.name)}</span>
        <span>${formatPreciseTime(note.start)} – ${formatPreciseTime(note.end)}</span>
        <label>S <input data-field="string" data-index="${index}" type="number" min="1" max="6" value="${note.string}"></label>
        <label>F <input data-field="fret" data-index="${index}" type="number" min="0" max="24" value="${note.fret}"></label>
        <span class="confidence">${Math.round(note.confidence * 100)}% sure</span>
      </div>`;
  }).join("") : `<div class="empty">No ${view === "raw" ? "raw detections" : "final arranged notes"} in this view.</div>`;
}

function renderTabScore(analysis: AnalysisResult): void {
  // Keep the ASCII measures for the unchanged .txt export; the on-screen view is
  // now the graphical SVG renderer driven from the same canonical note data.
  tabMeasures = buildTablature(analysis.notes, analysis.duration, analysis.bpm);
  if (!tabRenderer) {
    tabRenderer = new TabRenderer(byId("tab-score"), {
      onSeek: (time) => { player.currentTime = time; void player.play(); },
      onSelectNote: highlightNoteRow,
    });
  }
  tabRenderer.setScore(analysis);
  tabRenderer.setTime(player.currentTime, false);
}

function highlightNoteRow(noteIndex: number | null): void {
  document.querySelectorAll<HTMLElement>(".note.selected").forEach((el) => el.classList.remove("selected"));
  if (noteIndex === null) return;
  const row = document.querySelector<HTMLElement>(`.note[data-index="${noteIndex}"]`);
  row?.classList.add("selected");
  row?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function syncPlayback(): void {
  if (!result) return;
  const state = getPlaybackState(result.chords, player.currentTime, result.duration);
  byId("current-time").textContent = formatTime(player.currentTime);
  byId<HTMLInputElement>("song-scrubber").value = String(player.currentTime);
  const current = state.currentIndex >= 0 ? result.chords[state.currentIndex] : null;
  const next = state.nextIndex >= 0 ? result.chords[state.nextIndex] : null;
  byId("now-chord").textContent = current?.name ?? "—";
  byId("chord-window").textContent = current
    ? `${formatTime(current.start)}–${formatTime(current.end)} · ${Math.max(0, current.end - player.currentTime).toFixed(1)}s left`
    : "No chord detected here";
  byId("next-chord").textContent = next?.name ?? "—";
  byId("next-chord-time").textContent = next ? `at ${formatTime(next.start)}` : "End of progression";

  document.querySelectorAll<HTMLElement>(".chord").forEach((element) => {
    const index = Number(element.dataset.index);
    element.classList.toggle("live", index === state.currentIndex);
    element.classList.toggle("past", index < state.currentIndex || (state.currentIndex < 0 && result!.chords[index].end <= player.currentTime));
    element.setAttribute("aria-current", index === state.currentIndex ? "true" : "false");
    const played = index < state.currentIndex ? 100 : index === state.currentIndex ? state.chordProgress * 100 : 0;
    element.style.setProperty("--played", `${played}%`);
  });
  if (state.currentIndex !== lastActiveChordIndex) {
    lastActiveChordIndex = state.currentIndex;
    if (state.currentIndex >= 0) {
      const active = document.querySelector<HTMLElement>(`.chord[data-index="${state.currentIndex}"]`);
      active?.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }
  // While playing, a requestAnimationFrame loop drives the cursor for smoothness;
  // this handles paused seeks/scrubbing where no frame loop is running.
  if (cursorRaf === 0) tabRenderer?.setTime(player.currentTime, true);
  renderNoteList();
  document.querySelectorAll<HTMLElement>(".note").forEach((element) => element.classList.toggle("live", player.currentTime >= Number(element.dataset.start) && player.currentTime < Number(element.dataset.end)));
}

function startCursorLoop(): void {
  if (cursorRaf !== 0) return;
  const tick = () => {
    tabRenderer?.setTime(player.currentTime, true);
    cursorRaf = requestAnimationFrame(tick);
  };
  cursorRaf = requestAnimationFrame(tick);
}

function stopCursorLoop(): void {
  if (cursorRaf === 0) return;
  cancelAnimationFrame(cursorRaf);
  cursorRaf = 0;
  tabRenderer?.setTime(player.currentTime, true);
}

player.addEventListener("timeupdate", syncPlayback);
player.addEventListener("seeked", syncPlayback);
player.addEventListener("loadedmetadata", syncPlayback);
player.addEventListener("play", startCursorLoop);
player.addEventListener("pause", stopCursorLoop);
player.addEventListener("ended", stopCursorLoop);

byId<HTMLButtonElement>("zoom-in").addEventListener("click", () => tabRenderer?.setZoom(tabRenderer.getZoom() + 0.2));
byId<HTMLButtonElement>("zoom-out").addEventListener("click", () => tabRenderer?.setZoom(tabRenderer.getZoom() - 0.2));
byId<HTMLInputElement>("count-overlay").addEventListener("change", (event) => tabRenderer?.setCountOverlay((event.target as HTMLInputElement).checked));
byId<HTMLInputElement>("song-scrubber").addEventListener("input", (event) => {
  player.currentTime = Number((event.target as HTMLInputElement).value);
  syncPlayback();
});

byId("export").addEventListener("click", () => {
  if (!result) return;
  const blob = new Blob([JSON.stringify({ source: currentFile?.name, sourceUrl: currentSourceUrl, createdAt: new Date().toISOString(), ...result }, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${currentFile?.name.replace(/\.[^.]+$/, "") ?? "tab"}.tabsmith.json`;
  link.click();
  URL.revokeObjectURL(link.href);
});
byId("export-tab").addEventListener("click", () => {
  if (!result) return;
  const title = currentFile?.name.replace(/\.[^.]+$/, "") ?? "tab";
  const blob = new Blob([tablatureToText(tabMeasures, title, result.bpm, result.tuning, result.capo)], { type: "text/plain" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${title}.txt`;
  link.click();
  URL.revokeObjectURL(link.href);
});
byId("new-song").addEventListener("click", () => location.reload());

function formatTime(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${Math.floor(seconds % 60).toString().padStart(2, "0")}`;
}
function formatPreciseTime(seconds: number): string {
  return `${formatTime(seconds)}.${Math.floor((seconds % 1) * 10)}`;
}
function capitalize(value: string): string {
  return value ? `${value[0].toUpperCase()}${value.slice(1)}` : value;
}
function escapeHtml(value: string): string {
  const node = document.createElement("div"); node.textContent = value; return node.innerHTML;
}
