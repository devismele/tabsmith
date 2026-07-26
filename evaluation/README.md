# Chord-reference evaluation

This directory is for local, manually authorized chord evaluation. It is
separate from Tabsmith projects, processing caches, and release assets.

The application and offline harness store only:

- song/reference metadata;
- normalized chord labels and optional section names;
- manual or explicitly unverified automatic alignment data; and
- comparison statistics and diagnostic summaries.

Do not place lyrics, complete tabs, copied page bodies, or downloaded
recordings here. Tabsmith has no Ultimate Guitar fetcher or scraper. Open and
review a reference in your own authorized browser session, then enter only the
minimum chord symbols needed for evaluation.

The committed song list contains metadata-only candidates. `audioPath` and
`referenceUrl` are blank until the user supplies or authorizes them. Local
imports are written below `evaluation/local` in development and the Electron
user-data `evaluation` directory in packaged builds.

Use development songs for parameter choices. Validation songs measure changes
after tuning. Do not inspect or tune against the final test set until the
candidate algorithm is frozen.

For an offline comparison, add local-only `referencePath` and `predictionPath`
fields to a private manifest and run:

```text
npm run evaluate:chords -- path/to/private-manifest.json
```

Optional `ablationPredictionPaths` can map the six configuration IDs reported
by the harness to separately generated Tabsmith JSON files. The normal
application never runs these expensive variants automatically.

## TypeScript hybrid parity evaluation

`npm run evaluate:hybrid` is the raw-audio comparison for:

- production `harmonic-context-v3-reduced-latency`;
- evaluation-only ML-only Viterbi;
- the actual observation-level TypeScript hybrid decoder; and
- an optional evaluation-only reconstruction of the old region replacement.

The harness creates rule observations once, captures one learned response, gives
that exact response object to both ML-only and hybrid, and calls production
`runHybridHarmony()` / `decodeHarmonyObservations()` for the hybrid result. It
does not use `ml/evaluation/adapters.py` as a hybrid result.

The committed configuration keeps the historical internal identifier
`guitarset-zenodo-1492449`, but records the actual file distribution separately
as Zenodo record `3371780`, GuitarSet `1.1.0`. Local paths are supplied only at
runtime. After importing the normalized annotation JSON outside the repository,
run the microphone capture with:

```powershell
$env:TABSMITH_GUITARSET_MANIFEST = "<local prepared dataset>/dataset_manifest.json"
$env:TABSMITH_GUITARSET_ANNOTATIONS = "<local prepared dataset>/guitarset/annotations"
$env:TABSMITH_GUITARSET_AUDIO = "<local GuitarSet>/audio_mono-mic"
$env:TABSMITH_HYBRID_EVAL_CAPTURE = "audio_mono-mic"
$env:TABSMITH_HYBRID_EVAL_OUTPUT_TAG = "mono-mic"
npm run evaluate:hybrid
```

For the paired pickup capture, keep the same manifest, annotations, split,
model, and settings; point `TABSMITH_GUITARSET_AUDIO` to
`audio_mono-pickup_mix`, set the capture to `audio_mono-pickup_mix`, and use
the output tag `pickup-mix`. Output tags must be portable labels and create
capture-specific filenames without embedding any local path.

Reports are written under `evaluation/reports/` only after at least one track
completes real learned inference and the production TypeScript temporal decoder.
Only completed, validated real-data reports are eligible for commit. A missing
dataset, all-fallback run, synthetic-only run, or failed run does not create an
accuracy report.

To use a private full-band raw-audio corpus, copy the configuration outside
Git, add a `full-band` dataset entry with timed annotations and a frozen split,
then pass it explicitly:

```powershell
npm run evaluate:hybrid -- --config evaluation/private/hybrid-config.json
```

Billboard precomputed features are intentionally unsupported here because they
are not equivalent to raw-audio application input.
