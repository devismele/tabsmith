const sampleRate = 22050;
const seconds = 2;
const samples = sampleRate * seconds;
const wav = Buffer.alloc(44 + samples * 2);
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
wav.writeUInt32LE(samples * 2, 40);
const uniqueFrequency = 120 + (Date.now() % 80);
for (let index = 0; index < samples; index += 1) {
  wav.writeInt16LE(Math.round(Math.sin(2 * Math.PI * uniqueFrequency * index / sampleRate) * 15000), 44 + index * 2);
}

const createdResponse = await fetch("http://127.0.0.1:5173/api/jobs/separation", {
  method: "POST",
  headers: { "Content-Type": "audio/wav", "X-Input-Extension": "wav", "X-Input-Name": "cancel-smoke.wav" },
  body: wav,
});
if (!createdResponse.ok) throw new Error(`Could not create cancellation job: ${createdResponse.status}`);
const created = await createdResponse.json();
await fetch(`http://127.0.0.1:5173/api/jobs/${created.id}`, { method: "DELETE" });
for (let attempt = 0; attempt < 100; attempt += 1) {
  const job = await fetch(`http://127.0.0.1:5173/api/jobs/${created.id}`).then((response) => response.json());
  if (job.status === "cancelled") {
    process.stdout.write(`${JSON.stringify(job, null, 2)}\n`);
    process.exit(0);
  }
  await new Promise((resolve) => setTimeout(resolve, 50));
}
throw new Error("Cancellation job did not reach the cancelled state.");
