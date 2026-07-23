import { ZipArchive } from "archiver";
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { access, mkdir, readFile, readdir, rename, rm, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import extractZip from "extract-zip";

if (process.platform !== "win32") {
  throw new Error("The current runtime builder is pinned to Windows. Add a verified Python embed URL/checksum for this target first.");
}

const runtimeRoot = path.resolve(".runtime");
const staging = path.join(runtimeRoot, "python-staging");
const embedArchive = path.join(runtimeRoot, "python-3.13.14-embed-amd64.zip");
const output = path.join(runtimeRoot, "tabsmith-runtime.zip");
const partial = `${output}.partial`;
const embedUrl = "https://www.python.org/ftp/python/3.13.14/python-3.13.14-embed-amd64.zip";
const embedSha256 = "90b4e5b9898b72d744650524bff92377c367f44bd5fbd09e3148656c080ad907";

const outputDetails = await stat(output).catch(() => null);
if (outputDetails && !process.argv.includes("--force")) {
  const modelFiles = await readdir(path.resolve(".models", "audio-separator")).catch(() => []);
  const tracked = [
    fileURLToPath(import.meta.url),
    path.resolve("requirements.txt"),
    path.resolve("server", "separate_guitar.py"),
    ...modelFiles.map((name) => path.resolve(".models", "audio-separator", name)),
  ];
  const sourceDetails = await Promise.all(tracked.map((file) => stat(file).catch(() => null)));
  if (sourceDetails.every((details) => details && details.mtimeMs <= outputDetails.mtimeMs)) {
    console.log(`Using cached portable runtime: ${output} (${(outputDetails.size / 1024 / 1024).toFixed(1)} MB)`);
    process.exit(0);
  }
}

await mkdir(runtimeRoot, { recursive: true });
let download = true;
try {
  const existing = await readFile(embedArchive);
  download = createHash("sha256").update(existing).digest("hex") !== embedSha256;
} catch { /* Download below. */ }
if (download) {
  const response = await fetch(embedUrl);
  if (!response.ok) throw new Error(`Python embed download failed (${response.status}).`);
  const bytes = Buffer.from(await response.arrayBuffer());
  const checksum = createHash("sha256").update(bytes).digest("hex");
  if (checksum !== embedSha256) throw new Error(`Python embed checksum mismatch: ${checksum}`);
  await writeFile(embedArchive, bytes);
}

if (path.dirname(staging) !== runtimeRoot) throw new Error("Unsafe runtime staging path.");
await rm(staging, { recursive: true, force: true });
await mkdir(staging);
await extractZip(embedArchive, { dir: staging });
await writeFile(path.join(staging, "python313._pth"), "python313.zip\n.\nLib/site-packages\nimport site\n");
await rm(partial, { force: true });

const stream = createWriteStream(partial);
const archive = new ZipArchive({ zlib: { level: 9 } });
const finished = new Promise((resolve, reject) => {
  stream.on("close", resolve);
  stream.on("error", reject);
  archive.on("error", reject);
  archive.on("warning", (error) => error.code === "ENOENT" ? undefined : reject(error));
});
archive.pipe(stream);
archive.directory(staging, false);
archive.glob("**/*", {
  cwd: path.resolve(".venv", "Lib", "site-packages"),
  dot: true,
  ignore: ["**/__pycache__/**", "**/*.pyc", "**/tests/**", "**/test/**"],
}, { prefix: "Lib/site-packages" });
archive.directory(path.resolve(".models", "audio-separator"), "models/audio-separator");
archive.file(path.resolve("server", "separate_guitar.py"), { name: "separate_guitar.py" });
await archive.finalize();
await finished;
await access(partial);
await rm(output, { force: true });
await rename(partial, output);
const details = await stat(output);
console.log(`Prepared portable Python runtime: ${output} (${(details.size / 1024 / 1024).toFixed(1)} MB)`);
