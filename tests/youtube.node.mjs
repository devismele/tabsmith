import assert from "node:assert/strict";
import test from "node:test";
import { validateYoutubeUrl, YoutubeError } from "../server/youtube.mjs";

test("accepts direct YouTube video URLs", () => {
  assert.match(validateYoutubeUrl("https://www.youtube.com/watch?v=BaW_jenozKc"), /youtube\.com/);
  assert.match(validateYoutubeUrl("https://youtu.be/BaW_jenozKc"), /youtu\.be/);
  assert.match(validateYoutubeUrl("https://www.youtube.com/shorts/BaW_jenozKc"), /shorts/);
});

test("rejects non-video and non-YouTube URLs", () => {
  assert.throws(() => validateYoutubeUrl("https://example.com/watch?v=x"), YoutubeError);
  assert.throws(() => validateYoutubeUrl("https://www.youtube.com/playlist?list=x"), YoutubeError);
  assert.throws(() => validateYoutubeUrl("not a URL"), YoutubeError);
});
