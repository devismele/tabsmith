import path from "node:path";

export interface LocalDatasetPathConfig {
  datasetIdentifier: string;
  manifestPath: string;
  annotationDirectory: string;
  audioRoot?: string;
  captureType?: string;
}

export interface EvaluationEnvironment {
  TABSMITH_GUITARSET_MANIFEST?: string;
  TABSMITH_GUITARSET_ANNOTATIONS?: string;
  TABSMITH_GUITARSET_AUDIO?: string;
  TABSMITH_HYBRID_EVAL_CAPTURE?: string;
  TABSMITH_HYBRID_EVAL_OUTPUT_TAG?: string;
}

export interface ResolvedDatasetPaths {
  manifestPath: string;
  annotationDirectory: string;
  audioRoot: string | null;
  captureType: string | null;
}

export interface HybridReportFileNames {
  json: string;
  markdown: string;
  perTrackCsv: string;
  disagreements: string;
}

function nonEmpty(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

function resolveFromConfig(configPath: string, target: string): string {
  return path.resolve(path.dirname(configPath), target);
}

export function validatePortableLabel(
  value: string | undefined,
  fieldName: string,
): string | undefined {
  const normalized = nonEmpty(value);
  if (!normalized) return undefined;
  if (!/^[a-z0-9][a-z0-9_-]*$/i.test(normalized)) {
    throw new Error(
      `${fieldName} must be a portable label containing only letters,`
        + " digits, hyphens, and underscores",
    );
  }
  return normalized;
}

/**
 * Local GuitarSet paths are deliberately supplied at execution time. The
 * committed JSON remains portable and no absolute user path reaches a report.
 */
export function resolveDatasetPaths(
  dataset: LocalDatasetPathConfig,
  configPath: string,
  environment: EvaluationEnvironment,
): ResolvedDatasetPaths {
  const guitarSet = dataset.datasetIdentifier.startsWith("guitarset-");
  const manifestOverride = guitarSet
    ? nonEmpty(environment.TABSMITH_GUITARSET_MANIFEST)
    : undefined;
  const annotationOverride = guitarSet
    ? nonEmpty(environment.TABSMITH_GUITARSET_ANNOTATIONS)
    : undefined;
  const audioOverride = guitarSet
    ? nonEmpty(environment.TABSMITH_GUITARSET_AUDIO)
    : undefined;
  const captureType = validatePortableLabel(
    guitarSet
      ? nonEmpty(environment.TABSMITH_HYBRID_EVAL_CAPTURE)
        ?? dataset.captureType
      : dataset.captureType,
    "capture type",
  ) ?? null;
  return {
    manifestPath: manifestOverride
      ? path.resolve(manifestOverride)
      : resolveFromConfig(configPath, dataset.manifestPath),
    annotationDirectory: annotationOverride
      ? path.resolve(annotationOverride)
      : resolveFromConfig(configPath, dataset.annotationDirectory),
    audioRoot: audioOverride
      ? path.resolve(audioOverride)
      : dataset.audioRoot
        ? resolveFromConfig(configPath, dataset.audioRoot)
        : null,
    captureType,
  };
}

export function audioBasenameForCapture(
  referencedAudioPath: string,
  captureType: string | null,
): string {
  const basename = path.basename(referencedAudioPath);
  if (captureType === "audio_mono-pickup_mix") {
    return basename.replace(/_mic\.wav$/i, "_mix.wav");
  }
  return basename;
}

export function resolveOutputTag(
  configuredTag: string | undefined,
  environment: EvaluationEnvironment,
): string | undefined {
  return validatePortableLabel(
    nonEmpty(environment.TABSMITH_HYBRID_EVAL_OUTPUT_TAG) ?? configuredTag,
    "output tag",
  );
}

export function reportFileNames(outputTag?: string): HybridReportFileNames {
  const tag = validatePortableLabel(outputTag, "output tag");
  if (!tag) {
    return {
      json: "hybrid-accuracy-comparison.json",
      markdown: "hybrid-accuracy-comparison.md",
      perTrackCsv: "hybrid-accuracy-per-track.csv",
      disagreements: "hybrid-disagreements.json",
    };
  }
  const base = tag.startsWith("calibrated-")
    ? `hybrid-${tag}`
    : `hybrid-accuracy-${tag}`;
  return {
    json: `${base}.json`,
    markdown: `${base}.md`,
    perTrackCsv: `${base}-per-track.csv`,
    disagreements: `${base}-disagreements.json`,
  };
}
