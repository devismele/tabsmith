import { createHash } from "node:crypto";
import {
  mkdir,
  readFile,
  rename,
  unlink,
  writeFile,
} from "node:fs/promises";
import path from "node:path";
import { validateResponse } from "../src/learnedHarmony/contract";
import type {
  LearnedHarmonyModelMetadata,
  LearnedHarmonyProvider,
  LearnedHarmonyRequest,
  LearnedHarmonyResponse,
} from "../src/learnedHarmony/types";

const CACHE_SCHEMA_VERSION = 1;

interface CacheEnvelope {
  schemaVersion: typeof CACHE_SCHEMA_VERSION;
  cacheKey: string;
  captureType: string;
  modelVersion: string;
  modelChecksum: string;
  featureVersion: string;
  response: LearnedHarmonyResponse;
}

export interface LearnedInferenceCacheStats {
  memoryHits: number;
  diskHits: number;
  misses: number;
  invalidEntries: number;
}

function sha256(value: string): string {
  return createHash("sha256").update(value).digest("hex");
}

export function learnedInferenceCacheKey(
  request: LearnedHarmonyRequest,
  captureType: string,
): string {
  return sha256(JSON.stringify({
    schemaVersion: CACHE_SCHEMA_VERSION,
    captureType,
    audioHash: request.audioHash,
    sectionStartSeconds: request.sectionStartSeconds,
    sectionEndSeconds: request.sectionEndSeconds,
    frameTimes: request.frameTimes,
    contractVersion: request.contractVersion,
    modelVersion: request.modelMetadata.modelVersion,
    modelChecksum: request.modelMetadata.modelChecksum,
    featureVersion: request.featureVersion,
  }));
}

function responseMatchesRequest(
  response: LearnedHarmonyResponse,
  request: LearnedHarmonyRequest,
): boolean {
  return validateResponse(response, request).ok
    && response.modelVersion === request.modelMetadata.modelVersion
    && response.modelChecksum === request.modelMetadata.modelChecksum
    && response.featureVersion === request.featureVersion;
}

/**
 * Evaluation-only provider cache. Cache files contain learned probabilities,
 * never audio, and are keyed by audio/model/feature/capture identity.
 */
export class CachedLearnedHarmonyProvider implements LearnedHarmonyProvider {
  readonly id: string;
  readonly stats: LearnedInferenceCacheStats = {
    memoryHits: 0,
    diskHits: 0,
    misses: 0,
    invalidEntries: 0,
  };

  private readonly memory = new Map<string, LearnedHarmonyResponse>();

  constructor(
    private readonly inner: LearnedHarmonyProvider,
    private readonly cacheDirectory: string,
    private readonly captureType: string,
  ) {
    this.id = `${inner.id}-cached`;
  }

  isAvailable(): Promise<boolean> {
    return this.inner.isAvailable();
  }

  getMetadata(): Promise<LearnedHarmonyModelMetadata> {
    return this.inner.getMetadata();
  }

  private cachePath(key: string): string {
    return path.join(this.cacheDirectory, `${key}.json`);
  }

  async predict(
    request: LearnedHarmonyRequest,
    signal?: AbortSignal,
  ): Promise<LearnedHarmonyResponse> {
    if (signal?.aborted) throw new DOMException("Aborted", "AbortError");
    const key = learnedInferenceCacheKey(request, this.captureType);
    const inMemory = this.memory.get(key);
    if (inMemory) {
      this.stats.memoryHits += 1;
      return inMemory;
    }

    const target = this.cachePath(key);
    try {
      const envelope = JSON.parse(await readFile(target, "utf8")) as CacheEnvelope;
      if (envelope.schemaVersion === CACHE_SCHEMA_VERSION
        && envelope.cacheKey === key
        && envelope.captureType === this.captureType
        && envelope.modelVersion === request.modelMetadata.modelVersion
        && envelope.modelChecksum === request.modelMetadata.modelChecksum
        && envelope.featureVersion === request.featureVersion
        && responseMatchesRequest(envelope.response, request)) {
        this.memory.set(key, envelope.response);
        this.stats.diskHits += 1;
        return envelope.response;
      }
      this.stats.invalidEntries += 1;
      await unlink(target).catch(() => undefined);
    } catch (error) {
      const code = error && typeof error === "object" && "code" in error
        ? String(error.code)
        : "";
      if (code !== "ENOENT") {
        this.stats.invalidEntries += 1;
        await unlink(target).catch(() => undefined);
      }
    }

    this.stats.misses += 1;
    const response = await this.inner.predict(request, signal);
    if (!responseMatchesRequest(response, request)) {
      throw new Error("Learned provider returned a response that failed cache validation");
    }
    await mkdir(this.cacheDirectory, { recursive: true });
    const envelope: CacheEnvelope = {
      schemaVersion: CACHE_SCHEMA_VERSION,
      cacheKey: key,
      captureType: this.captureType,
      modelVersion: response.modelVersion,
      modelChecksum: response.modelChecksum,
      featureVersion: response.featureVersion,
      response,
    };
    const temporary = `${target}.${process.pid}.${Date.now()}.tmp`;
    await writeFile(temporary, `${JSON.stringify(envelope)}\n`, "utf8");
    await unlink(target).catch(() => undefined);
    await rename(temporary, target);
    this.memory.set(key, response);
    return response;
  }
}
