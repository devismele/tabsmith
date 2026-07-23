import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdtemp, readdir, rm, stat } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

const ALLOWED_HOSTS = new Set([
  "youtube.com",
  "www.youtube.com",
  "m.youtube.com",
  "music.youtube.com",
  "youtu.be",
  "youtube-nocookie.com",
  "www.youtube-nocookie.com",
]);
const MAX_DURATION_SECONDS = 15 * 60;
const MAX_AUDIO_BYTES = 100 * 1024 * 1024;

export class YoutubeError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

export function validateYoutubeUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    throw new YoutubeError("Enter a valid YouTube video URL.");
  }
  const host = parsed.hostname.toLowerCase();
  if (!["https:", "http:"].includes(parsed.protocol) || !ALLOWED_HOSTS.has(host)) {
    throw new YoutubeError("Only YouTube video links are supported.");
  }
  const isShortLink = host === "youtu.be" && parsed.pathname.length > 1;
  const isWatchLink = parsed.pathname === "/watch" && Boolean(parsed.searchParams.get("v"));
  const isVideoPath = /^\/(shorts|embed|live)\/[^/]+/.test(parsed.pathname);
  if (!isShortLink && !isWatchLink && !isVideoPath) {
    throw new YoutubeError("Use a direct YouTube video, Short, or live-video URL—not a channel or playlist.");
  }
  parsed.hash = "";
  return parsed.toString();
}

function pythonCommand() {
  const local = path.resolve(".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");
  if (existsSync(local)) return local;
  return process.env.TABSMITH_PYTHON || (process.platform === "win32" ? "python" : "python3");
}

function cleanError(stderr) {
  const useful = stderr.split(/\r?\n/).map((line) => line.trim()).filter(Boolean).at(-1);
  return useful?.replace(/^ERROR:\s*/i, "") ?? "YouTube audio could not be retrieved.";
}

function runYtDlp(args, { timeoutMs = 45_000, maxOutputBytes = 2_000_000 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonCommand(), ["-m", "yt_dlp", ...args], {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout = [];
    const stderr = [];
    let outputBytes = 0;
    let settled = false;
    const finish = (callback) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      callback();
    };
    const timer = setTimeout(() => {
      child.kill();
      finish(() => reject(new YoutubeError("YouTube processing timed out.", 504)));
    }, timeoutMs);
    child.on("error", (error) => finish(() => reject(new YoutubeError(
      error.code === "ENOENT" ? "Python is required for YouTube links." : error.message,
      503,
    ))));
    child.stdout.on("data", (chunk) => {
      outputBytes += chunk.length;
      if (outputBytes > maxOutputBytes) {
        child.kill();
        finish(() => reject(new YoutubeError("YouTube metadata response was unexpectedly large.", 502)));
      } else stdout.push(chunk);
    });
    child.stderr.on("data", (chunk) => stderr.push(chunk));
    child.on("close", (code) => finish(() => {
      const errorText = Buffer.concat(stderr).toString("utf8");
      if (code === 0) resolve(Buffer.concat(stdout).toString("utf8"));
      else if (/No module named yt_dlp/i.test(errorText)) reject(new YoutubeError("yt-dlp is not installed. Run the project setup command first.", 503));
      else reject(new YoutubeError(cleanError(errorText), 502));
    }));
  });
}

export async function getYoutubeInfo(rawUrl) {
  const url = validateYoutubeUrl(rawUrl);
  const output = await runYtDlp([
    "--dump-single-json",
    "--skip-download",
    "--no-playlist",
    "--no-warnings",
    url,
  ]);
  let info;
  try {
    info = JSON.parse(output);
  } catch {
    throw new YoutubeError("YouTube returned unreadable metadata.", 502);
  }
  if (info.is_live || info.live_status === "is_live") throw new YoutubeError("Currently-live streams are not supported.");
  if (!Number.isFinite(info.duration) || info.duration <= 0) throw new YoutubeError("The video duration could not be determined.");
  if (info.duration > MAX_DURATION_SECONDS) throw new YoutubeError("Videos are limited to 15 minutes for local transcription.", 413);
  return {
    id: info.id,
    title: info.title || "YouTube audio",
    duration: info.duration,
    channel: info.channel || info.uploader || null,
    thumbnail: info.thumbnail || null,
    webpageUrl: info.webpage_url || url,
  };
}

export async function downloadYoutubeAudio(rawUrl) {
  const url = validateYoutubeUrl(rawUrl);
  const info = await getYoutubeInfo(url);
  const tempDirectory = await mkdtemp(path.join(os.tmpdir(), "tabsmith-youtube-"));
  try {
    await runYtDlp([
      "--no-playlist",
      "--no-warnings",
      "--no-progress",
      "--quiet",
      "--format",
      "bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best",
      "--max-filesize",
      "100M",
      "--paths",
      tempDirectory,
      "--output",
      "source.%(ext)s",
      url,
    ], { timeoutMs: 120_000 });
    const files = await readdir(tempDirectory, { withFileTypes: true });
    const audioEntry = files.find((entry) => entry.isFile() && !entry.name.endsWith(".part"));
    if (!audioEntry) throw new YoutubeError("No compatible audio stream was available.", 502);
    const audioPath = path.join(tempDirectory, audioEntry.name);
    const details = await stat(audioPath);
    if (details.size > MAX_AUDIO_BYTES) throw new YoutubeError("The audio exceeds the 100 MB local limit.", 413);
    return {
      path: audioPath,
      extension: path.extname(audioEntry.name).slice(1).toLowerCase() || "audio",
      size: details.size,
      info,
      cleanup: () => rm(tempDirectory, { recursive: true, force: true }),
    };
  } catch (error) {
    await rm(tempDirectory, { recursive: true, force: true }).catch(() => {});
    throw error;
  }
}

export async function youtubeDependencyVersion() {
  return (await runYtDlp(["--version"], { timeoutMs: 10_000, maxOutputBytes: 10_000 })).trim();
}
