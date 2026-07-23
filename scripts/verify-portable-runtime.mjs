import { spawn } from "node:child_process";
import { mkdir, rm } from "node:fs/promises";
import path from "node:path";
import extractZip from "extract-zip";

const runtimeRoot = path.resolve(".runtime");
const target = path.join(runtimeRoot, "verify");
if (path.dirname(target) !== runtimeRoot) throw new Error("Unsafe runtime verification path.");
await rm(target, { recursive: true, force: true });
await mkdir(target);
await extractZip(path.join(runtimeRoot, "tabsmith-runtime.zip"), { dir: target });
const code = "import torch, audio_separator, static_ffmpeg, subprocess; static_ffmpeg.add_paths(); print('torch', torch.__version__); subprocess.run(['ffmpeg','-version'], check=True, stdout=subprocess.DEVNULL); print('audio-separator and ffmpeg ok')";
const exitCode = await new Promise((resolve, reject) => {
  const child = spawn(path.join(target, "python.exe"), ["-c", code], { stdio: "inherit", windowsHide: true });
  child.on("error", reject);
  child.on("exit", resolve);
});
if (exitCode !== 0) throw new Error(`Portable runtime verification failed with exit code ${exitCode}.`);
console.log("Portable runtime verification passed.");
