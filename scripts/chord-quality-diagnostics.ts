import { readFile } from "node:fs/promises";
import { analyzeHarmony } from "../src/analysis";

function decodePcmWav(input: Buffer): { samples: Float32Array; sampleRate: number } {
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
    throw new Error("Diagnostic script currently supports PCM16 WAV files.");
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

const inputPath = process.argv[2];
const sectionStart = Number(process.argv[3] ?? 92);
const sectionEnd = Number(process.argv[4] ?? 105);
if (!inputPath) throw new Error("Usage: chord-quality-diagnostics.ts INPUT.wav [START] [END]");

const decoded = decodePcmWav(await readFile(inputPath));
const harmony = analyzeHarmony(
  resample(decoded.samples, decoded.sampleRate),
  22050,
  { chordDisplayMode: "simple" },
  { source: "guitar-only" },
);
process.stdout.write(`${JSON.stringify({
  duration: harmony.duration,
  bpm: harmony.bpm,
  diagnostics: harmony.chordAnalysis.diagnostics,
  regions: harmony.chords.filter(
    (region) => region.end > sectionStart && region.start < sectionEnd,
  ),
  windows: harmony.chordAnalysis.windows.filter(
    (window) => window.end > sectionStart && window.start < sectionEnd,
  ),
}, null, 2)}\n`);
