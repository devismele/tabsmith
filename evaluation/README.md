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
