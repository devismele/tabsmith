# Tabsmith

Tabsmith is a local-first audio-to-guitar-tab prototype. It separates and analyzes the audio itself; it does not query tab or chord databases or send the song to an external service.

## Run it

```powershell
npm install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm run dev
```

Open `http://127.0.0.1:5173`, choose an audio file, and wait for analysis. Build and test with:

```powershell
npm test
npm run build
npm run test:separation-smoke
npm run test:cancel-smoke
npm run desktop:smoke
npm run desktop:package
```

The separation smoke test starts a real model inference and assumes `npm run dev` is already running. Source-mode setup caches FFmpeg and Demucs locally. The desktop release instead includes a checksum-verified portable Python runtime and model archive, which it expands into versioned application data on first launch. Uncheck **Isolate guitar before transcription** when you intentionally want to analyze a solo stem or the full mix without separation.

## What the MVP does

- decodes browser-supported audio locally;
- imports authorized audio from direct YouTube videos and Shorts through a localhost-only service;
- includes an original eight-second E–A–B7–E synthetic guitar demo for first-time use;
- isolates a dedicated guitar stem with Audio Separator and Demucs `htdemucs_6s` before transcription;
- uses Spotify Basic Pitch as the default polyphonic note-transcription engine;
- estimates monophonic guitar-range notes with a YIN-style pitch detector;
- estimates major, minor, and evidence-gated dominant-seventh regions from spectral chroma;
- aggregates chroma by beat and uses duration-aware Viterbi decoding, hysteresis, and region cleanup to produce a stable progression;
- maps pitches to standard-tuned strings and frets with movement optimization and distinct-string chord shapes;
- advances a current/next chord display and scrollable timeline with playback;
- queues separation work with progress, cancellation, recent-job history, and content-addressed local caching;
- runs Basic Pitch with cooperative cancellation between inference frames;
- conservatively estimates standard, Drop D, E-flat, or capo setups from recurring open-string evidence;
- lets you correct string/fret choices and export JSON.
- renders a graphical six-string tablature view (SVG) with measure wrapping, rhythm stems, a smooth green playback cursor, active-note/measure highlighting, zoom, click-to-seek, and viewport virtualization for long songs; the optional "Count" overlay restores the `1 e & a` guide, and the tab still exports as `.txt`.

Automatic transcription remains approximate. The six-stem model can leak or remove sound when instruments overlap, and it combines multiple guitars into one stem. Distortion, bends, harmonics, and alternate tunings can still require correction. Link input records provenance and asks for user-authorized audio.

See [RESEARCH.md](./RESEARCH.md) for the open-source chord and tablature projects evaluated for future accuracy modes.

The Basic Pitch model is packaged and served locally by the development/production build. The original DSP detector remains available under **Transcription engine** and is used automatically if ML initialization fails. Guitar separation uses the project-local Python environment and keeps downloaded model files under `.models/`. YouTube import uses the project-local `yt-dlp` installation, accepts only direct YouTube video hosts, rejects live streams and playlists, and limits input to 15 minutes / 100 MB.

## Desktop and evaluation

Electron packaging is configured for Windows ARM64 and bundles an x64 Python runtime because the current PyTorch wheels run through Windows emulation. `npm run desktop:package` creates an unpacked application under `out/`; `npm run desktop:make` creates the ZIP and Squirrel installer. The package includes Python, FFmpeg, yt-dlp, Audio Separator, and the Demucs model, so it is intentionally large. It now has a generated multi-size application icon and first-run setup screen. Public distribution still needs an Authenticode certificate and an update publishing location.

The benchmark harness reads a user-created `benchmarks/manifest.json` and compares known references against exported Tabsmith predictions. Run `npm run benchmark` to measure note precision/recall/F1, onset error, chord-time accuracy, and exact string/fret accuracy. See [benchmarks/README.md](./benchmarks/README.md) and [THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
