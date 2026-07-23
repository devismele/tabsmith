import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const sampleRate = 44100;
const chordSeconds = 2;
const chords = [
  [40, 47, 52, 56, 59, 64],
  [45, 52, 57, 61, 64],
  [47, 54, 57, 63, 66],
  [40, 47, 52, 56, 59, 64],
];
const frameCount = sampleRate * chordSeconds * chords.length;
const pcm = Buffer.alloc(frameCount * 2);
for (let frame = 0; frame < frameCount; frame += 1) {
  const time = frame / sampleRate;
  const chordIndex = Math.min(chords.length - 1, Math.floor(time / chordSeconds));
  const chordTime = time - chordIndex * chordSeconds;
  let sample = 0;
  chords[chordIndex].forEach((midi, stringIndex) => {
    const arpeggioDelay = stringIndex * 0.035;
    if (chordTime < arpeggioDelay) return;
    const afterDelay = chordTime - arpeggioDelay;
    const pluckTime = afterDelay % 0.5;
    const frequency = 440 * 2 ** ((midi - 69) / 12);
    const envelope = Math.exp(-5.2 * pluckTime) * Math.min(1, pluckTime * 280);
    const phase = 2 * Math.PI * frequency * afterDelay;
    sample += envelope * (Math.sin(phase) + 0.36 * Math.sin(phase * 2) + 0.17 * Math.sin(phase * 3)) / chords[chordIndex].length;
  });
  const intro = Math.min(1, time * 12);
  const outro = Math.min(1, (chords.length * chordSeconds - time) * 8);
  pcm.writeInt16LE(Math.max(-32767, Math.min(32767, Math.round(sample * intro * outro * 23000))), frame * 2);
}

const wav = Buffer.alloc(44 + pcm.length);
wav.write("RIFF", 0, "ascii");
wav.writeUInt32LE(wav.length - 8, 4);
wav.write("WAVEfmt ", 8, "ascii");
wav.writeUInt32LE(16, 16);
wav.writeUInt16LE(1, 20);
wav.writeUInt16LE(1, 22);
wav.writeUInt32LE(sampleRate, 24);
wav.writeUInt32LE(sampleRate * 2, 28);
wav.writeUInt16LE(2, 32);
wav.writeUInt16LE(16, 34);
wav.write("data", 36, "ascii");
wav.writeUInt32LE(pcm.length, 40);
pcm.copy(wav, 44);
await mkdir(path.resolve("public"), { recursive: true });
await writeFile(path.resolve("public", "tabsmith-demo.wav"), wav);
console.log(`Generated public/tabsmith-demo.wav (${(wav.length / 1024).toFixed(1)} KB)`);
