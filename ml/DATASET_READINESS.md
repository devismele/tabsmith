# Dataset readiness — gate before real learned-harmony training

ML development is **paused**. The limiting factor is trustworthy, licensed real
data — not code. The next ML milestone is **not** another architecture component;
it is a completed readiness report plus a cleared data gate. Do not begin real
training until licensing and audio/feature availability are unambiguous.

## 1. Readiness report (fill one row per candidate source)

| Field | Notes |
|---|---|
| Dataset / source | e.g. Isophonics (Beatles/Queen/Carole King), McGill Billboard, licensed library |
| License & permitted uses | Must be documented and allow this use. No scraping, no auto-download. |
| Audio availability | `audio` / `features` / `annotations` — only `audio` is raw-audio-trainable |
| Timed chord annotations | present? format? tolerance/quality? |
| Beat / downbeat annotations | present? |
| Number of songs | |
| Total duration (hours) | |
| Genre distribution | |
| Artist distribution | for artist-level splitting |
| Chord-class distribution | maj / min / 7 / N balance |
| Train / validation / test eligibility | which split, honoring artist separation |

The importers already exist (`ml/preprocessing/import_isophonics.py`,
`import_billboard.py`) and mark audio availability honestly. Run
`python -m ml.preprocessing.prepare_dataset --isophonics-dir <dir> --billboard-dir <dir>`
to produce a manifest with class distribution + leakage warnings.

## 2. Minimum real-data gate (all must hold)

- [ ] Several hours of usable **real** audio, or legally usable precomputed features
- [ ] Multiple artists
- [ ] Multiple genres and production styles
- [ ] Timed chord boundaries of documented reliability
- [ ] **Artist-separated** validation set (no artist across splits)
- [ ] An **untouched** final test set (frozen until model + thresholds are frozen)
- [ ] Enough major, minor, seventh, and no-chord examples

Diversity, annotation reliability, and leakage-avoidance matter more than a raw
hour count.

## 3. Keep collecting Tabsmith evaluation examples (valuable now)

The manually aligned reference system (`evaluation/chord-reference-songs.json`,
imported by `ml/preprocessing/import_tabsmith_ref.py`) is useful even before
training. Keep building, per the existing artist-level splits:

- development examples for diagnosing errors,
- validation songs never used for tuning,
- final test songs left untouched,
- short difficult sections with verified boundaries,
- examples of bass ambiguity, passing melody, arpeggiation, distortion, and
  multiple guitars.

These will later reveal whether the learned or hybrid model actually beats the
rule-based engine.

## 4. Promotion criteria (later, on held-out real data)

Better held-out **root** and **chord-quality** accuracy, improved or preserved
**boundary timing**, and no serious **fragmentation** regression versus
`harmonic-context-v3-reduced-latency`. The final test split stays untouched until
the model and thresholds are frozen.

## 5. Next implementation step (once the gate is cleared)

Replace `MockLearnedHarmonyProvider` with a dev-only `OnnxLearnedHarmonyProvider`
behind `TABSMITH_EXPERIMENTAL_LEARNED_HARMONY` (excluded from packaging), then run
the three-way offline comparison (rule / ML-only / hybrid) on the held-out real
data. Until then, the production app stays exactly as it is.
