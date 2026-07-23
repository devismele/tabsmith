import test from "node:test";
import assert from "node:assert/strict";
import { cancelJob, createSeparationJob, getJob } from "../server/jobs.mjs";

test("a queued separation job can be cancelled before Python starts", async () => {
  const wav = Buffer.alloc(12);
  wav.write("RIFF", 0, "ascii");
  wav.write("WAVE", 8, "ascii");
  const created = createSeparationJob(wav, "wav", "C:\\Music\\demo.wav");
  const cancelled = cancelJob(created.id);
  assert.equal(cancelled.status, "cancelled");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(getJob(created.id).status, "cancelled");
});
