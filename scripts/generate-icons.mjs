import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import pngToIco from "png-to-ico";
import sharp from "sharp";

const source = path.resolve("assets/icon.svg");
const directory = path.resolve("assets/generated-icons");
await mkdir(directory, { recursive: true });
const sizes = [16, 24, 32, 48, 64, 128, 256];
const pngPaths = [];
for (const size of sizes) {
  const output = path.join(directory, `icon-${size}.png`);
  await sharp(source).resize(size, size).png().toFile(output);
  pngPaths.push(output);
}
await sharp(source).resize(512, 512).png().toFile(path.resolve("assets/icon.png"));
await writeFile(path.resolve("assets/icon.ico"), await pngToIco(pngPaths));
console.log("Generated assets/icon.png and assets/icon.ico");
