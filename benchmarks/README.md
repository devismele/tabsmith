# Accuracy benchmarks

Create `manifest.json` in this directory using `manifest.example.json` as the template. Each reference and prediction is a Tabsmith-style JSON file containing `duration`, `notes`, and `chords`.

Use short, legally redistributable isolated-guitar recordings and cover clean acoustic, clean electric, distorted rhythm, lead, chords, and alternate tunings. Keep the audio outside Git unless its license permits redistribution.

Run `npm run benchmark`. The report measures note precision/recall/F1 with an 80 ms onset window, onset mean absolute error, exact string/fret accuracy, and duration-weighted chord accuracy.
