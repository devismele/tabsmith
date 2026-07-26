import fs from "node:fs";
import { runTcn, type TcnWeights } from "../src/learnedHarmony/onnxProvider";

function readArgument(name: string): string | null {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : null;
}

function maxAbsoluteError(actual: number[][] | number[], expected: number[][] | number[]): number {
  const actualFlat = actual.flat(Number.POSITIVE_INFINITY) as number[];
  const expectedFlat = expected.flat(Number.POSITIVE_INFINITY) as number[];
  if (actualFlat.length !== expectedFlat.length) {
    throw new Error(`parity shape mismatch: ${actualFlat.length} != ${expectedFlat.length}`);
  }
  return actualFlat.reduce(
    (maximum, value, index) => Math.max(maximum, Math.abs(value - expectedFlat[index])),
    0,
  );
}

const weightsPath = readArgument("--weights");
if (!weightsPath) {
  throw new Error("usage: vite-node evaluation/verify-temporal-v2-parity.ts --weights <export.json>");
}

const weights = JSON.parse(fs.readFileSync(weightsPath, "utf8")) as TcnWeights;
if (!weights.paritySample) {
  throw new Error("export has no parity sample");
}
const actual = runTcn(weights, weights.paritySample.input);
const errors = {
  root: maxAbsoluteError(actual.root, weights.paritySample.root),
  quality: maxAbsoluteError(actual.quality, weights.paritySample.quality),
  nochord: maxAbsoluteError(actual.nochord, weights.paritySample.nochord),
  boundary: maxAbsoluteError(actual.boundary, weights.paritySample.boundary),
};
const maximumError = Math.max(...Object.values(errors));
const tolerance = 1e-4;
const result = {
  modelVersion: weights.modelVersion,
  modelChecksum: weights.modelChecksum,
  maximumAbsoluteError: maximumError,
  headErrors: errors,
  tolerance,
  passed: maximumError <= tolerance,
};
console.log(JSON.stringify(result));
if (!result.passed) {
  process.exitCode = 1;
}
