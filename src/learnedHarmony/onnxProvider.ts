// Dev-only learned-harmony provider that runs the exported temporal-baseline TCN
// with a hand-written forward pass in pure TypeScript — no ONNX runtime, no native
// binaries, no new dependency. It fulfils the "OnnxLearnedHarmonyProvider" slot the
// integration contract was frozen waiting for. It is reached only by the
// development worker's gated dynamic import.
//
// The forward pass is verified numerically identical to PyTorch via the exporter's
// parity sample (ml/exports/export_tcn_json.py). Weights load from a JSON produced
// there; the model was trained on harmony-features-v1 (app-identical chroma).
import { CHROMA_BINS, CONTRACT_VERSION, FEATURE_VERSION, validateRequest } from "./contract";
import type {
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "./types";

interface ConvWeights { weight: number[][]; bias: number[] }               // 1x1: (out, in)
interface Conv3Weights { weight: number[][][]; bias: number[] }            // k3: (out, in, 3)
interface BnWeights { gamma: number[]; beta: number[]; mean: number[]; var: number[]; eps: number }
interface BlockWeights { dilation: number; conv1: Conv3Weights; conv2: Conv3Weights; norm1: BnWeights; norm2: BnWeights }

export interface TcnWeights {
  modelVersion: string;
  modelChecksum: string;
  featureVersion: string;
  config: { inputDim: number; channels: number; kernelSize: number; dilations: number[]; qualities: string[] };
  inputProj: ConvWeights;
  blocks: BlockWeights[];
  heads: { root: ConvWeights; quality: ConvWeights; nochord: ConvWeights; boundary: ConvWeights };
  paritySample?: { input: number[][]; root: number[][]; quality: number[][]; nochord: number[]; boundary: number[] };
}

// h is channel-major [channel][time], matching PyTorch (C, T).
function conv1x1(input: number[][], weights: ConvWeights): number[][] {
  const [outC, inC] = [weights.weight.length, weights.weight[0].length];
  const T = input[0].length;
  const out: number[][] = Array.from({ length: outC }, () => new Array(T).fill(0));
  for (let o = 0; o < outC; o += 1) {
    const w = weights.weight[o];
    const b = weights.bias[o];
    for (let t = 0; t < T; t += 1) {
      let sum = b;
      for (let i = 0; i < inC; i += 1) sum += w[i] * input[i][t];
      out[o][t] = sum;
    }
  }
  return out;
}

// k=3 dilated conv, padding=dilation (length-preserving). PyTorch:
// out[t] = sum_j W[:,:,j] * in[t - d + j*d], j in {0,1,2}; out-of-range = 0.
function conv3(input: number[][], weights: Conv3Weights, dilation: number): number[][] {
  const outC = weights.weight.length;
  const inC = weights.weight[0].length;
  const T = input[0].length;
  const out: number[][] = Array.from({ length: outC }, () => new Array(T).fill(0));
  const offsets = [-dilation, 0, dilation];
  for (let o = 0; o < outC; o += 1) {
    const W = weights.weight[o];
    const b = weights.bias[o];
    for (let t = 0; t < T; t += 1) {
      let sum = b;
      for (let j = 0; j < 3; j += 1) {
        const tt = t + offsets[j];
        if (tt < 0 || tt >= T) continue;
        for (let i = 0; i < inC; i += 1) sum += W[i][j] * input[i][tt];
      }
      out[o][t] = sum;
    }
  }
  return out;
}

function batchNormReluInplace(h: number[][], bn: BnWeights, relu: boolean): void {
  for (let c = 0; c < h.length; c += 1) {
    const scale = bn.gamma[c] / Math.sqrt(bn.var[c] + bn.eps);
    const shift = bn.beta[c] - bn.mean[c] * scale;
    const row = h[c];
    for (let t = 0; t < row.length; t += 1) {
      let v = row[t] * scale + shift;
      if (relu && v < 0) v = 0;
      row[t] = v;
    }
  }
}

function softmaxRows(logits: number[][]): number[][] {
  return logits.map((row) => {
    const max = Math.max(...row);
    const exps = row.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0) || 1;
    return exps.map((v) => v / sum);
  });
}

function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }

/** Raw forward pass -> per-frame head logits. Exported for the parity test. */
export function runTcn(weights: TcnWeights, input: number[][]): {
  root: number[][]; quality: number[][]; nochord: number[]; boundary: number[];
} {
  const T = input.length;
  // To channel-major (inputDim, T).
  const x: number[][] = Array.from({ length: weights.config.inputDim }, (_v, c) => input.map((row) => row[c]));
  let h = conv1x1(x, weights.inputProj);
  for (const block of weights.blocks) {
    const residual = h;
    let y = conv3(h, block.conv1, block.dilation);
    batchNormReluInplace(y, block.norm1, true);
    y = conv3(y, block.conv2, block.dilation);
    batchNormReluInplace(y, block.norm2, true);
    h = y.map((row, c) => row.map((v, t) => v + residual[c][t]));
  }
  // Heads (1x1) -> time-major.
  const rootCM = conv1x1(h, weights.heads.root);
  const qualCM = conv1x1(h, weights.heads.quality);
  const ncCM = conv1x1(h, weights.heads.nochord);
  const bndCM = conv1x1(h, weights.heads.boundary);
  const toTimeMajor = (cm: number[][]) => Array.from({ length: T }, (_v, t) => cm.map((row) => row[t]));
  return {
    root: toTimeMajor(rootCM),
    quality: toTimeMajor(qualCM),
    nochord: ncCM[0].slice(),
    boundary: bndCM[0].slice(),
  };
}

export class LearnedTcnProvider implements LearnedHarmonyProvider {
  readonly id = "onnx";
  constructor(private readonly weights: TcnWeights) {}

  async isAvailable(): Promise<boolean> { return true; }

  async getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return {
      modelVersion: this.weights.modelVersion,
      modelChecksum: this.weights.modelChecksum,
      contractVersion: CONTRACT_VERSION,
      featureVersion: this.weights.featureVersion,
      format: "onnx",
      vocabulary: {
        roots: CHROMA_BINS,
        qualities: this.weights.config.qualities,
        hasNoChordHead: true,
        hasBoundaryHead: true,
      },
      inputShapes: { features: ["batch", "frames", this.weights.config.inputDim] },
      disclaimer: "Pure-TS forward pass of the exported temporal-baseline TCN (not onnxruntime); experimental.",
    };
  }

  async predict(request: LearnedHarmonyRequest): Promise<LearnedHarmonyResponse> {
    const validation = validateRequest(request);
    if (!validation.ok) throw new Error(`LearnedTcnProvider received invalid request: ${validation.errors.join("; ")}`);

    const started = Date.now();
    const T = request.frameTimes.length;
    // Build the 25-dim input the model expects: harmonicChroma | bassChroma | energy.
    // The request carries onsetStrength in the energy slot (documented approximation).
    const input: number[][] = new Array(T);
    for (let t = 0; t < T; t += 1) {
      input[t] = [...request.harmonicChroma[t], ...request.bassChroma[t], request.onsetStrength[t]];
    }
    const out = runTcn(this.weights, input);

    return {
      contractVersion: CONTRACT_VERSION,
      requestId: request.requestId,
      modelVersion: this.weights.modelVersion,
      modelChecksum: this.weights.modelChecksum,
      featureVersion: this.weights.featureVersion,
      frameTimes: request.frameTimes.slice(),
      rootProbabilities: softmaxRows(out.root),
      qualityProbabilities: softmaxRows(out.quality),
      noChordProbabilities: out.nochord.map(sigmoid),
      boundaryProbabilities: out.boundary.map(sigmoid),
      diagnostics: {
        inferenceMilliseconds: Date.now() - started,
        backend: "onnx",
        warnings: ["Pure-TS forward pass of exported temporal-baseline TCN; energy slot fed from onsetStrength."],
      },
    };
  }
}
