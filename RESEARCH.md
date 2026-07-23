# Open-source transcription research

This file records projects evaluated for Tabsmith. Their code was not copied into the MVP; the links document architectural options and their licenses.

## Strong candidates

- [Spotify Basic Pitch for TypeScript](https://github.com/spotify/basic-pitch-ts) — Apache-2.0, browser-compatible polyphonic audio-to-note transcription with onset, contour, and pitch-bend output. Best candidate for an optional high-accuracy note engine.
- [Spotify Basic Pitch](https://github.com/spotify/basic-pitch) — Apache-2.0 Python reference implementation and model. Useful if Tabsmith adds a local Python service.
- [AMT Tools](https://github.com/cwitkowitz/amt-tools) — MIT framework with guitar-transcription research implementations. Useful for evaluating guitar-specific models and datasets.
- [TabCNN paper/code](https://github.com/andywiggins/tab-cnn) — direct audio-to-string/fret estimation for isolated guitar. A future guitar-specific model should follow this joint-prediction direction.
- [Chordino / NNLS Chroma](https://isophonics.net/nnls-chroma) — established chord-recognition pipeline. It motivates chroma post-processing and stable time-labelled chord sequences.

## Consider with care

- [Essentia.js](https://github.com/MTG/essentia.js) includes browser-side music analysis and chord descriptors, but its AGPL-3.0 licensing has product implications.
- [music-transcription](https://github.com/trimplexx/music-transcription) demonstrates a newer CRNN guitar-tab pipeline and reports strong GuitarSet metrics, but needs GPU-oriented Python dependencies and further model/reproducibility validation.

## Applied now

Tabsmith now uses Spotify Basic Pitch as its default local polyphonic note engine under the Apache-2.0 license. It also applies an independently implemented small mode filter to frame-level chroma chord labels. This suppresses isolated chord flicker, and playback uses the timestamped sequence to drive the current/next chord UI and timeline. The original DSP pitch detector remains as a fast fallback.

For source separation, Tabsmith uses [Audio Separator](https://github.com/nomadkaraoke/python-audio-separator) with the six-source `htdemucs_6s` model from Demucs. The model produces a dedicated guitar stem in addition to vocals, drums, bass, piano, and other. Audio Separator is MIT-licensed and maintained, whereas the original Demucs repository is archived. Spleeter was not selected because its standard five-stem model has no dedicated guitar output.
