import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export const MAX_INPUT_BYTES = 150 * 1024 * 1024;
const MAX_OUTPUT_BYTES = 200 * 1024 * 1024;
const MODEL_NAME = "htdemucs_6s-harmonic-v1";
const SAFE_EXTENSIONS = new Set(["aac", "flac", "m4a", "mp3", "mp4", "oga", "ogg", "opus", "wav", "webm"]);

export class SeparatorError extends Error {
  constructor(message, status = 500) {
    super(message);
    this.status = status;
  }
}

function pythonCommand() {
  const local = path.resolve(".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  if (existsSync(local)) return local;
  return process.env.TABSMITH_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function cleanError(stderr) {
  if (/No module named ['\"]?(audio_separator|static_ffmpeg)/i.test(stderr)) {
    return "Guitar separation is not installed. Run the Python setup command from README.md.";
  }
  if (/out of memory|cannot allocate memory|memoryerror/i.test(stderr)) {
    return "The separator ran out of memory. Try a shorter audio file.";
  }
  const useful = stderr.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).at(-1);
  return useful?.replace(/^(ERROR|RuntimeError):\s*/i, "") || "The guitar stem could not be separated.";
}

function runPython(args, {
  timeoutMs = 20 * 60_000,
  maxOutputBytes = 5_000_000,
  signal,
  onProgress = () => {},
} = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonCommand(), args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    const stdout = [];
    const stderr = [];
    let outputBytes = 0;
    let settled = false;
    let timer;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      callback();
    };
    const abort = () => {
      child.kill();
      finish(() => reject(new SeparatorError("Guitar separation was cancelled.", 499)));
    };
    if (signal?.aborted) return abort();
    signal?.addEventListener("abort", abort, { once: true });
    timer = setTimeout(() => {
      child.kill();
      finish(() => reject(new SeparatorError("Guitar separation timed out.", 504)));
    }, timeoutMs);
    child.on("error", (error) => finish(() => reject(new SeparatorError(
      error.code === "ENOENT" ? "Python is required for guitar separation." : error.message,
      503,
    ))));
    const collect = (target, diagnostic = false) => (chunk) => {
      outputBytes += chunk.length;
      if (outputBytes > maxOutputBytes) {
        child.kill();
        finish(() => reject(new SeparatorError("The separator produced too much diagnostic output.", 502)));
      } else {
        target.push(chunk);
        if (diagnostic) {
          const text = chunk.toString("utf8");
          const percentages = [...text.matchAll(/(\d{1,3})(?:\.\d+)?%/g)];
          if (percentages.length) {
            const percent = Math.min(100, Number(percentages.at(-1)[1]));
            onProgress(0.12 + percent / 100 * 0.8, "Separating guitar stem");
          } else if (/download/i.test(text)) onProgress(0.06, "Downloading separation model");
          else if (/load.*model|loading.*model/i.test(text)) onProgress(0.1, "Loading separation model");
        }
      }
    };
    child.stdout.on("data", collect(stdout));
    child.stderr.on("data", collect(stderr, true));
    child.on("close", (code) => finish(() => {
      const stdoutText = Buffer.concat(stdout).toString("utf8");
      const stderrText = Buffer.concat(stderr).toString("utf8");
      if (code === 0) resolve(stdoutText);
      else reject(new SeparatorError(cleanError(stderrText), 502));
    }));
  });
}

export function normalizeAudioExtension(value) {
  const extension = String(value || "").toLowerCase().replace(/^\./, "");
  return SAFE_EXTENSIONS.has(extension) ? extension : "audio";
}

export function validateAudioInput(input, extension) {
  if (!Buffer.isBuffer(input) || input.length === 0) throw new SeparatorError("Send a non-empty audio file.", 400);
  if (input.length > MAX_INPUT_BYTES) throw new SeparatorError("Audio for separation is limited to 150 MB.", 413);
  const kind = normalizeAudioExtension(extension);
  if (kind === "audio") return;
  const ascii = (start, end) => input.subarray(start, end).toString("ascii");
  const signatures = {
    wav: () => ascii(0, 4) === "RIFF" && ascii(8, 12) === "WAVE",
    flac: () => ascii(0, 4) === "fLaC",
    ogg: () => ascii(0, 4) === "OggS",
    oga: () => ascii(0, 4) === "OggS",
    opus: () => ascii(0, 4) === "OggS",
    mp4: () => ascii(4, 8) === "ftyp",
    m4a: () => ascii(4, 8) === "ftyp",
    webm: () => input.length >= 4 && input.readUInt32BE(0) === 0x1a45dfa3,
    mp3: () => ascii(0, 3) === "ID3" || (input[0] === 0xff && (input[1] & 0xe0) === 0xe0),
    aac: () => input[0] === 0xff && (input[1] & 0xf6) === 0xf0,
  };
  if (!signatures[kind]?.()) throw new SeparatorError(`The ${kind.toUpperCase()} file is corrupted or has an unexpected format.`, 415);
}

export async function separateGuitar(input, extension, { signal, onProgress = () => {} } = {}) {
  validateAudioInput(input, extension);

  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), "tabsmith-separator-"));
  const outputDirectory = path.join(tempDirectory, "output");
  const modelDirectory = process.env.TABSMITH_MODEL_DIR || path.resolve(".models", "audio-separator");
  const inputPath = path.join(tempDirectory, `input.${normalizeAudioExtension(extension)}`);
  try {
    if (signal?.aborted) throw new SeparatorError("Guitar separation was cancelled.", 499);
    onProgress(0.02, "Preparing audio");
    await mkdir(outputDirectory);
    await mkdir(modelDirectory, { recursive: true });
    await writeFile(inputPath, input);
    onProgress(0.04, "Starting guitar separator");
    const script = process.env.TABSMITH_SEPARATOR_SCRIPT || path.resolve("server", "separate_guitar.py");
    const output = await runPython([script, inputPath, outputDirectory, modelDirectory], { signal, onProgress });
    const jsonLine = output.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).at(-1);
    let separated;
    try {
      separated = JSON.parse(jsonLine);
    } catch {
      throw new SeparatorError("The separator returned an unreadable result.", 502);
    }
    const stemPaths = {
      guitar: path.resolve(separated.guitarPath || ""),
      bass: path.resolve(separated.bassPath || ""),
      harmony: path.resolve(separated.harmonyPath || ""),
    };
    const stemSizes = {};
    for (const [stem, outputPath] of Object.entries(stemPaths)) {
      if (path.dirname(outputPath) !== path.resolve(outputDirectory)) {
        throw new SeparatorError(`The separator returned an unexpected ${stem} path.`, 502);
      }
      const details = await stat(outputPath);
      if (!details.isFile() || details.size === 0) {
        throw new SeparatorError(`The separated ${stem} stem was empty.`, 502);
      }
      if (details.size > MAX_OUTPUT_BYTES) {
        throw new SeparatorError(`The separated ${stem} stem exceeds 200 MB.`, 413);
      }
      stemSizes[stem] = details.size;
    }
    onProgress(1, "Guitar stem ready");
    return {
      paths: stemPaths,
      sizes: stemSizes,
      model: MODEL_NAME,
      cleanup: () => rm(tempDirectory, { recursive: true, force: true }),
    };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

export async function separatorDependencyVersion() {
  const output = await runPython([
    "-c",
    "from importlib.metadata import version; print(version('audio-separator'))",
  ], { timeoutMs: 10_000, maxOutputBytes: 10_000 });
  return output.trim();
}
