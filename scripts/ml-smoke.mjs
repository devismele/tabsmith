import { spawn } from "node:child_process";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const profile = await mkdtemp(path.join(os.tmpdir(), "tabsmith-ml-smoke-"));
const port = 9337;
const appPortArgument = process.argv.find((argument) => argument.startsWith("--app-port="));
const appPort = Number(appPortArgument?.slice("--app-port=".length) || 5173);
const useYoutube = process.argv.includes("--youtube");
const useSeparation = process.argv.includes("--separation");
const useDemo = process.argv.includes("--demo");
const inputArgument = process.argv.find((argument) => argument.startsWith("--input="));
const inputPath = inputArgument ? path.resolve(inputArgument.slice("--input=".length)) : null;
const inspectHotelSection = process.argv.includes("--hotel");
const inspectChordUi = process.argv.includes("--chord-ui");
const inspectEvaluationUi = process.argv.includes("--evaluation-ui");
const screenshotArgument = process.argv.find((argument) => argument.startsWith("--screenshot="));
const screenshotPath = screenshotArgument
  ? path.resolve(screenshotArgument.slice("--screenshot=".length))
  : null;
const edge = spawn(edgePath, [
  "--headless=new",
  "--enable-unsafe-swiftshader",
  `--remote-debugging-port=${port}`,
  `--user-data-dir=${profile}`,
  "--no-first-run",
  "--disable-default-apps",
  `http://127.0.0.1:${appPort}`,
], { stdio: "ignore", windowsHide: true });

const delay = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function findPage() {
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      const pages = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json());
      const page = pages.find((candidate) => candidate.type === "page" && candidate.url.includes(`127.0.0.1:${appPort}`));
      if (page) return page;
    } catch { /* Browser is still starting. */ }
    await delay(200);
  }
  throw new Error("Timed out connecting to the headless browser.");
}

let socket;
try {
  const page = await findPage();
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });

  let requestId = 0;
  const pending = new Map();
  socket.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(message.error.message));
    else resolve(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++requestId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const response = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (response.exceptionDetails) throw new Error(response.exceptionDetails.exception?.description || response.exceptionDetails.text);
    return response.result.value;
  };

  await send("Runtime.enable");
  await send("DOM.enable");
  await send("Page.enable");
  for (let attempt = 0; attempt < 50; attempt += 1) {
    if (await evaluate("Boolean(document.querySelector('#file-input') && document.querySelector('#isolate-guitar'))")) break;
    await delay(100);
  }

  if (inputPath) {
    await evaluate(`(() => {
      document.querySelector('#isolate-guitar').checked = false;
      document.querySelector('#arrangement-select').value = 'lead';
      document.querySelector('#quality-select').value = 'balanced';
      return true;
    })()`);
    const documentNode = await send("DOM.getDocument");
    const inputNode = await send("DOM.querySelector", {
      nodeId: documentNode.root.nodeId,
      selector: "#file-input",
    });
    await send("DOM.setFileInputFiles", {
      nodeId: inputNode.nodeId,
      files: [inputPath],
    });
    await evaluate(`(() => {
      document.querySelector('#file-input').dispatchEvent(new Event('change', { bubbles: true }));
      return true;
    })()`);
  } else if (useYoutube) {
    await evaluate(`(() => {
      document.querySelector('#isolate-guitar').checked = ${useSeparation};
      document.querySelector('[data-mode="link"]').click();
      document.querySelector('#source-url').value = 'https://www.youtube.com/watch?v=jNQXAC9IVRw';
      document.querySelector('#rights-confirm').checked = true;
      document.querySelector('#url-form').requestSubmit();
      return true;
    })()`);
  } else if (useDemo) {
    await evaluate("document.querySelector('#demo-song').click(); true");
  } else await evaluate(`(() => {
    document.querySelector('#isolate-guitar').checked = ${useSeparation};
    const sampleRate = 22050;
    const seconds = 3;
    const samples = new Int16Array(sampleRate * seconds);
    const frequencies = [82.41, 110, 164.81];
    for (let i = 0; i < samples.length; i += 1) {
      const segment = Math.min(2, Math.floor(i / sampleRate));
      const fade = Math.min(1, (i % sampleRate) / 500, (sampleRate - (i % sampleRate)) / 500);
      samples[i] = Math.round(Math.sin(2 * Math.PI * frequencies[segment] * i / sampleRate) * fade * 18000);
    }
    const wav = new ArrayBuffer(44 + samples.byteLength);
    const view = new DataView(wav);
    const text = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
    text(0, 'RIFF'); view.setUint32(4, 36 + samples.byteLength, true); text(8, 'WAVE'); text(12, 'fmt ');
    view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
    view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, samples.byteLength, true);
    new Int16Array(wav, 44).set(samples);
    const file = new File([wav], 'synthetic-guitar.wav', { type: 'audio/wav' });
    const transfer = new DataTransfer(); transfer.items.add(file);
    const input = document.querySelector('#file-input'); input.files = transfer.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
    return true;
  })()`);

  let outcome;
  for (let attempt = 0; attempt < (inputPath ? 7200 : useSeparation ? 3600 : useYoutube ? 480 : 240); attempt += 1) {
    outcome = await evaluate(`(() => ({
      done: !document.querySelector('#results').classList.contains('hidden'),
      failed: document.querySelector('#status').textContent.includes('Could not analyze'),
      status: document.querySelector('#status-label')?.textContent || '',
      metadata: document.querySelector('#metadata')?.textContent || '',
      pipelineStatus: document.querySelector('#pipeline-status')?.textContent || '',
      title: document.querySelector('#song-title')?.textContent || '',
      notes: document.querySelectorAll('.note').length,
      chords: document.querySelectorAll('.chord').length,
      tabMeasures: document.querySelectorAll('.tabx-measure-number').length,
      diagnostics: JSON.parse(document.querySelector('#results')?.dataset.diagnostics || '{}')
    }))()`);
    if (outcome.done || outcome.failed) break;
    await delay(250);
  }
  if (!outcome?.done) throw new Error(`ML smoke test did not finish: ${outcome?.status ?? "unknown"}`);
  if (!outcome.metadata.includes("Basic Pitch ML")) throw new Error(`ML fallback was used unexpectedly: ${outcome.metadata}`);
  if (useSeparation && !outcome.metadata.includes("isolated guitar · Demucs 6-stem")) throw new Error(`Guitar separation was not applied: ${outcome.metadata}`);
  if ((inputPath ? outcome.diagnostics?.notes?.arrangedNoteCount : outcome.notes) < 1) {
    throw new Error("Basic Pitch completed but returned no arranged notes.");
  }
  if (outcome.tabMeasures < 1) throw new Error("Transcription completed but no tab measures were rendered.");
  if (useYoutube && outcome.title !== "Me at the zoo") throw new Error(`Unexpected YouTube title: ${outcome.title}`);
  if (useDemo && !outcome.title.startsWith("Tabsmith demo")) throw new Error(`Unexpected demo title: ${outcome.title}`);
  if (inspectChordUi) {
    const chordUi = await evaluate(`(() => {
      const firstChord = document.querySelector('#chords .chord');
      firstChord?.click();
      document.querySelector('#toggle-chord-debug')?.click();
      const input = document.querySelector('#manual-chord-label');
      const lock = document.querySelector('#manual-chord-lock');
      if (input) input.value = 'Am';
      if (lock) lock.checked = true;
      document.querySelector('[data-chord-action="apply"]')?.click();
      return {
        editorVisible: !document.querySelector('#chord-editor')?.classList.contains('hidden'),
        debugVisible: !document.querySelector('#chord-debug')?.classList.contains('hidden'),
        correctedLabel: document.querySelector('#chords .chord strong')?.textContent || '',
        locked: document.querySelector('#chords .chord')?.classList.contains('locked') || false,
        diagnosticsText: document.querySelector('#chord-debug')?.textContent || '',
        confidenceText: document.querySelector('#chords .chord small')?.textContent || '',
      };
    })()`);
    outcome.chordUi = chordUi;
    if (!chordUi.editorVisible || !chordUi.debugVisible) {
      throw new Error("Chord review controls did not open.");
    }
    if (!chordUi.correctedLabel.includes("Am") || !chordUi.locked) {
      throw new Error(`Manual locked chord correction failed: ${JSON.stringify(chordUi)}`);
    }
    if (!chordUi.diagnosticsText.includes("Harmony diagnostics")
      || !chordUi.diagnosticsText.includes("Candidates")) {
      throw new Error("Chord diagnostic reasoning table was not rendered.");
    }
    if (!/confidence/i.test(chordUi.confidenceText)) {
      throw new Error("Chord confidence indicator was not rendered.");
    }
  }
  if (inspectHotelSection) {
    const section = await evaluate(`(() => {
      const player = document.querySelector('#player');
      player.currentTime = 98;
      player.dispatchEvent(new Event('seeked'));
      const rows = [...document.querySelectorAll('#notes .note')];
      return {
        playbackTime: player.currentTime,
        noteView: document.querySelector('#note-view-select').value,
        firstVisibleNoteTime: rows.length ? Number(rows[0].dataset.start) : null,
        visibleNoteTimes: rows.slice(0, 12).map((row) => Number(row.dataset.start)),
        chordsFrom132To145: [...document.querySelectorAll('#chords .chord')]
          .filter((chord) => Number(chord.dataset.time) >= 90 && Number(chord.dataset.time) <= 106)
          .map((chord) => ({
            start: Number(chord.dataset.time),
            label: chord.querySelector('strong')?.textContent || '',
            window: chord.querySelector('small')?.textContent || ''
          })),
        pipelineStatus: document.querySelector('#pipeline-status')?.textContent || '',
        metadata: document.querySelector('#metadata')?.textContent || ''
      };
    })()`);
    outcome.sectionAt138 = section;
    if (section.noteView !== "follow") throw new Error(`Note editor did not default to Follow playback: ${section.noteView}`);
    if (section.firstVisibleNoteTime !== null
      && Math.abs(section.firstVisibleNoteTime - 98) > 3) {
      throw new Error(`Note editor did not follow 1:38: ${section.firstVisibleNoteTime}`);
    }
    if (!section.pipelineStatus.includes("Arrangement: Lead")
      || !section.pipelineStatus.includes("Quality: Balanced")) {
      throw new Error(`Unexpected Hotel regression settings: ${section.pipelineStatus}`);
    }
  }
  if (inspectEvaluationUi) {
    await evaluate(`(() => {
      const details = document.querySelector('#evaluation-tools');
      details.open = true;
      details.dispatchEvent(new Event('toggle'));
      document.querySelector('#evaluation-artist').value = 'Smoke Artist';
      document.querySelector('#evaluation-title').value = 'Smoke Progression';
      document.querySelector('#evaluation-version').value = '2';
      document.querySelector('#evaluation-rating-count').value = '500';
      document.querySelector('#evaluation-average-rating').value = '4.8';
      document.querySelector('#evaluation-capo').value = '2';
      document.querySelector('#evaluation-genre').value = 'synthetic';
      document.querySelector('#evaluation-selection-reason').value = 'Largest rating count among manually reviewed Chords versions';
      document.querySelector('#evaluation-sequence').value = 'Progression: G | C | D7 | G';
      document.querySelector('#evaluation-import-form').requestSubmit();
      return true;
    })()`);
    let imported = false;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      imported = await evaluate("document.querySelectorAll('#evaluation-reference-select option').length > 0 && Boolean(document.querySelector('#evaluation-reference-select').value)");
      if (imported) break;
      await delay(50);
    }
    if (!imported) {
      const message = await evaluate("document.querySelector('#evaluation-message')?.textContent || ''");
      throw new Error(`Manual chord reference did not import: ${message}`);
    }
    for (const time of [0, 0.75, 1.5, 2.25]) {
      await evaluate(`(() => {
        const player = document.querySelector('#player');
        player.currentTime = ${time};
        document.querySelector('#evaluation-mark').click();
        return true;
      })()`);
    }
    await evaluate("document.querySelector('#evaluation-save-alignment').click(); true");
    let aligned = false;
    for (let attempt = 0; attempt < 80; attempt += 1) {
      aligned = await evaluate("document.querySelector('#evaluation-reference-summary')?.textContent.includes('verified') || false");
      if (aligned) break;
      await delay(50);
    }
    if (!aligned) {
      const message = await evaluate("document.querySelector('#evaluation-message')?.textContent || ''");
      throw new Error(`Manual playback alignment did not persist: ${message}`);
    }
    await evaluate("document.querySelector('#evaluation-compare').click(); true");
    let evaluated = false;
    for (let attempt = 0; attempt < 100; attempt += 1) {
      evaluated = await evaluate("document.querySelectorAll('.evaluation-metrics span').length === 6");
      if (evaluated) break;
      await delay(50);
    }
    if (!evaluated) {
      const message = await evaluate("document.querySelector('#evaluation-message')?.textContent || ''");
      throw new Error(`Chord reference comparison did not render: ${message}`);
    }
    outcome.evaluationUi = await evaluate(`(() => ({
      referenceSummary: document.querySelector('#evaluation-reference-summary')?.textContent || '',
      marks: document.querySelector('#evaluation-markers')?.textContent || '',
      metrics: document.querySelector('#evaluation-report')?.textContent || '',
      message: document.querySelector('#evaluation-message')?.textContent || '',
      reportDownloadEnabled: !document.querySelector('#evaluation-download-report')?.disabled,
      rawTextRetained: document.querySelector('#evaluation-sequence')?.value || ''
    }))()`);
    if (!outcome.evaluationUi.referenceSummary.includes("Smoke Artist")
      || !outcome.evaluationUi.referenceSummary.includes("capo 2")
      || !outcome.evaluationUi.referenceSummary.includes("verified")) {
      throw new Error(`Evaluation reference summary is incomplete: ${JSON.stringify(outcome.evaluationUi)}`);
    }
    if (!outcome.evaluationUi.marks.includes("4/4 marked")
      || !outcome.evaluationUi.metrics.includes("root")
      || !outcome.evaluationUi.reportDownloadEnabled
      || outcome.evaluationUi.rawTextRetained) {
      throw new Error(`Evaluation UI validation failed: ${JSON.stringify(outcome.evaluationUi)}`);
    }
  }
  if (screenshotPath) {
    await evaluate(`(() => {
      document.querySelector('${inspectEvaluationUi ? "#evaluation-tools" : "#results"}')?.scrollIntoView({ block: 'start' });
      return true;
    })()`);
    await delay(250);
    const screenshot = await send("Page.captureScreenshot", {
      format: "png",
      captureBeyondViewport: true,
      fromSurface: true,
    });
    await mkdir(path.dirname(screenshotPath), { recursive: true });
    await writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));
    outcome.screenshot = screenshotPath;
  }
  process.stdout.write(`${JSON.stringify(outcome, null, 2)}\n`);
} finally {
  socket?.close();
  edge.kill();
  await delay(300);
  if (path.resolve(profile).startsWith(path.resolve(os.tmpdir()))) {
    await rm(profile, { recursive: true, force: true, maxRetries: 3 }).catch(() => {});
  }
}
