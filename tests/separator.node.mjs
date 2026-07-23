import test from "node:test";
import assert from "node:assert/strict";
import { MAX_INPUT_BYTES, normalizeAudioExtension, separateGuitar, SeparatorError, validateAudioInput } from "../server/separator.mjs";

test("allows common audio extensions and rejects path-like values", () => {
  assert.equal(normalizeAudioExtension(".WAV"), "wav");
  assert.equal(normalizeAudioExtension("webm"), "webm");
  assert.equal(normalizeAudioExtension("../../exe"), "audio");
});

test("rejects an empty separation request before starting Python", async () => {
  await assert.rejects(() => separateGuitar(Buffer.alloc(0), "wav"), SeparatorError);
});

test("rejects corrupted audio before starting the model", () => {
  assert.throws(() => validateAudioInput(Buffer.from("not a wave file"), "wav"), /corrupted/i);
  assert.throws(() => validateAudioInput(Buffer.from("not an mp4 file"), "m4a"), /corrupted/i);
});

test("recognizes a minimal WAV signature and enforces the large-file limit", () => {
  const header = Buffer.alloc(12);
  header.write("RIFF", 0, "ascii");
  header.write("WAVE", 8, "ascii");
  assert.doesNotThrow(() => validateAudioInput(header, "wav"));
  assert.throws(() => validateAudioInput(Buffer.allocUnsafe(MAX_INPUT_BYTES + 1), "wav"), (error) => error.status === 413);
});
