# Dataset readiness report — learned-harmony real training

Status: **candidate assessment complete; data not yet acquired.** This is the
report `DATASET_READINESS.md` asks for before real training may begin. It does
not itself clear the gate — acquisition + a manifest run do.

Compiled 2026-07-25. License findings are documented terms as of that date and
are **not legal advice**; re-verify each source's current license before use.
No scraping, no auto-download (workspace policy).

## 1. Candidate sources (per `DATASET_READINESS.md` §1)

| Field | GuitarSet | McGill Billboard | Isophonics |
|---|---|---|---|
| Source | Zenodo record 1492449 (NYU MARL) | DDMAL, McGill | Centre for Digital Music, QMUL |
| **License & permitted uses** | **CC-BY 4.0 — commercial use OK with attribution** | Derived data (annotations + features) **CC0**; cite ISMIR paper | **Not clearly documented** on the reference page; research-oriented, ambiguous for a shipped product |
| **Audio availability** | **`audio`** — real recorded audio included (raw-audio-trainable) | **`features`** — NNLS-chroma + tuning (Chordino) distributed; **audio NOT distributed** | **`annotations`** only; you must legally own the CDs |
| Timed chord annotations | Yes — chords + string/fret, time-aligned | Yes — expert harmony + form, high reliability | Yes — high reliability (Beatles best-checked) |
| Beat / downbeat | Yes (beats + downbeats) | Yes (via annotations) | Yes |
| Number of songs | 360 excerpts (6 players × 2 versions × 5 styles × 3 progressions) | ~740 distinct songs (890 slots) | 225 tracks (Beatles/Queen/Carole King/Zweieck) |
| Total duration | ~3 hours (~30s/excerpt) | N/A as audio (features only) | N/A as audio |
| Genre distribution | Rock, Singer-Songwriter, Bossa Nova, Jazz, Funk — **narrow, solo acoustic guitar only** | Broad — Billboard chart pop/rock/soul/country across decades | Rock/pop (Beatles/Queen catalog) |
| Artist distribution | 6 players (performer-level split possible) | Many chart artists | 4 artists (weak for artist-level split) |
| Chord-class dist. | **VERIFY on download** — run `prepare_dataset` | **VERIFY on download** | **VERIFY on download** |
| Split eligibility | Player-separated train/val/test | Artist-separated splits feasible | Better as eval-only reference |

Exact chord-class counts are intentionally left as **VERIFY** — do not fabricate;
they come from `python -m ml.preprocessing.prepare_dataset` after acquisition.

## 2. Gate assessment (`DATASET_READINESS.md` §2)

Against **GuitarSet alone** (the one legally-clean *audio* source):

- [~] Several hours real audio — ~3h, borderline "several"
- [~] Multiple artists — 6 performers, not commercial-artist diversity
- [✗] **Multiple genres AND production styles** — all solo acoustic guitar, one rig → the weakest box
- [✓] Timed chord boundaries of documented reliability
- [✓] Artist-separated validation set — split by player
- [✓] Untouched final test set — hold out players/progressions
- [~] Enough maj/min/7/N — VERIFY; jazz/bossa give 7ths, but N (no-chord) may be sparse

**Conclusion: GuitarSet clears the *licensing* gate outright but not the
*diversity* gate.** Production-style diversity is the real deficiency for a model
meant to beat `harmonic-context-v3` on full-band mixes.

## Decision: multi-genre from the start (2026-07-25)

Chosen over a guitar-only first proof. Implication: the model must see more than
solo acoustic guitar, so both a clean audio source **and** the Billboard feature
path are in scope. Note GuitarSet already spans **five musical genres** (Rock,
Singer-Songwriter, Bossa Nova, Jazz, Funk) — it broadens *genre* immediately; the
remaining gap it does **not** close is full-band *production/instrumentation*
diversity, which is exactly what the Billboard features add.

**Progress (2026-07-25):**
- GuitarSet importer built + wired + tested, and **format verified against the real
  Zenodo 3371780 release** (360 files, CC-BY-4.0): namespace `chord`, lead-sheet
  labels first (clean maj/min/7), beats via `position==1`. All 360 import; 6 players
  × 60, 5 genres × 72, **3.05h**, family dist maj .73 / min .22 / dim .05.
- **GuitarSet acquired** locally (annotation.zip 39 MB + audio_mono-mic.zip 657 MB)
  and audio confirmed loadable (librosa, 44.1k→22050 mono, ~0.15s/clip features).
- Billboard feature path built (schema `feature_path` + SCHEMA_VERSION 2,
  `billboard-bothchroma-v1` loader, `feature_source` dispatch, `make_samples_from_tracks`).
- Real-data trainer added: `ml/training/train_real.py` (artist/player-separated splits).
- **First real training run executed on GuitarSet** (experimental; not a shipping model).

**Billboard integrated + verified against real files (2026-07-25):**
- Acquired DDMAL release (index.csv + LAB + chordino features, ~264 MB xz). 889 songs,
  **420 distinct artists**, **53.4 h** of features, family maj .73 / min .22 / **N .048**
  (the no-chord frames GuitarSet lacked).
- Real `bothchroma.csv` has a **leading filename column** (26 cols: name,time,24 chroma) —
  loader fixed to skip it. Bin order/origin **calibrated on real files**: `BASS_FIRST=True,
  origin=9 (A)` gave the best bass-root alignment (0.515) of all 24 combos → defaults confirmed.
- `import_billboard` now parses `billboard-2.0-index.csv` → per-artist splits (no longer one
  "Billboard" bucket).

## First real result (2026-07-25, held-out validation player p00, 60 clips)

Three-way offline comparison, duration-weighted, on a performer the model never saw:

Argmax decode (initial):

| engine | root % | majmin % | detail % | frag % | mean boundary err |
|---|---|---|---|---|---|
| rule (chroma-template-v0 floor) | 17.0 | 15.6 | 15.0 | 37.5 | 4058 ms |
| ml (argmax)                     | 52.0 | 40.5 | 33.0 | 81.1 | 557 ms |
| hybrid (argmax)                 | 51.2 | 40.2 | 32.9 | 81.7 | 500 ms |

Viterbi decode (self-transition penalty λ=4, `ml/evaluation/decode.py`, now default):

| engine | root % | majmin % | detail % | frag % | mean boundary err |
|---|---|---|---|---|---|
| ml (viterbi)     | 53.2 | 42.3 | 34.9 | 73.4 | 670 ms |
| hybrid (viterbi) | 52.6 | 41.2 | 33.8 | 72.5 | 664 ms |

The Viterbi decode is a Pareto win over argmax — higher root/majmin/detail *and*
lower fragmentation (81→73) — at a modest boundary-lag cost (stickier regions).
Higher λ cuts fragmentation further but trades accuracy and boundary timing; λ=4
is the sweet spot on this held-out player. Fragmentation is still high in absolute
terms: this small model on solo guitar is inherently unstable, so pushing it lower
needs a stronger model or the in-app smoothing decoder, not more decode tuning.

### Promotion gate: vs production harmonic-context-v3 (held-out GuitarSet p00)

Real v3 run offline on the same audio (node harness → `analyzeChordProgression`),
scored with the identical `evaluate_regions`:

| engine | root % | majmin % | detail % | frag % | boundary |
|---|---|---|---|---|---|
| **harmonic-context-v3 (production)** | 35.0 | 31.5 | 29.5 | 53.2 | 1728 ms |
| ml solo-guitarset (viterbi) | 53.2 | 42.3 | 34.9 | 73.4 | 670 ms |

ml beats v3 here (root 53 vs 35) — **but this is domain-favorable to ml and does NOT
clear the real gate.** v3's home is full-band mixes; it is tested here out-of-distribution
on solo acoustic guitar, while ml was trained on GuitarSet. v3 is also less fragmented.
This shows the pipeline learns real signal, not that ml should replace v3 in the app.

### Combined GuitarSet + Billboard model (temporal-baseline-real-combined-v0)

Trained on 360 GuitarSet audio + 200 Billboard feature tracks (382 train / 91 dev /
87 val by artist hash, 30 min CPU). Two held-out domains:

| test set | engine | root % | majmin % | detail % | frag % | no-chord recall |
|---|---|---|---|---|---|---|
| GuitarSet p00 (audio) | ml combined | 49.8 | 41.7 | 18.7 | 63.7 | — |
| Billboard artists (features) | ml combined | 76.2 | 60.3 | 39.2 | 64.1 | **55.1** |
| Billboard artists (features) | ml solo-guitarset | 35.8 | 24.9 | 18.9 | 54.8 | 2.7 |

Reading it honestly:
- **No-chord now works** (recall 55% vs 2.7% for the guitar-only model) — Billboard's N
  frames filled the gap, as intended.
- **Adding Billboard helped Billboard, slightly hurt guitar** (detail 34.9→18.7 on p00):
  the combined model generalizes across domains but is less specialized than the guitar-only one.
- **The GuitarSet and Billboard columns are NOT comparable**: different feature pipelines
  (`numpy-chroma-v1` audio vs `billboard-bothchroma-v1`), different domains. Billboard's
  train/test share the same precomputed-feature pipeline, which is easier than raw audio.
- **There is NO v3 baseline on Billboard** (v3 needs audio; Billboard ships none). So the 76%
  has nothing to beat — it is not a promotion result, just evidence the model learns full-band harmony.

**Bottom line:** the pipeline runs end-to-end on real, licensed, multi-genre data and the model
learns genuine harmony signal across two domains with working no-chord. None of it clears the
production gate, which needs v3-comparable evaluation on representative full-band **audio** —
still the missing ingredient (licensed full-band audio, or v3 tested in its own domain).

### Original first-result notes

**Reading it honestly:** the model learned real signal — root 53% on an unseen
player is ~3× the non-learned chroma floor, with far better boundary timing. But:
- the "rule" column is the weak offline **floor**, NOT production `harmonic-context-v3`
  (which lives in JS and is far stronger). Beating the floor is necessary, not
  sufficient — the real promotion gate is the JS three-way comparison vs v3.
- **fragmentation is high (81%)**: raw frame-argmax decode flips regions constantly.
  The production decoder smooths; the plan's whole point is to feed ML probabilities
  into that decoder rather than argmax-decode standalone.
- hybrid ≈ ml here because on solo guitar bass and treble chroma nearly coincide;
  the bass blend should matter more on full-band Billboard material.

## 3. Recommended path

1. **GuitarSet — importer DONE** (`ml/preprocessing/import_guitarset.py`, wired into
   `prepare_dataset` via `--guitarset-dir`, tests in `ml/tests/test_import_guitarset.py`).
   CC-BY 4.0, real audio through the existing `numpy-chroma-v1` audio path (no frozen-
   contract change), player-separated splits, genre tagged in notes. **Next:** acquire
   Zenodo 1492449 locally → `prepare_dataset --guitarset-dir <path>` → confirm class
   distribution → train + three-way offline compare (rule / ML-only / hybrid) on a
   held-out player split.
2. **Add McGill Billboard *features* (CC0)** — the remaining production-diversity work.
   `import_billboard.py` already marks feature-backed tracks `features`, but nothing
   *loads* those features yet. Required (each a real change, so done as a separate
   increment): (a) add optional `feature_path` to the `Track` schema (bump
   SCHEMA_VERSION 1→2, backward-compatible); (b) new `billboard-bothchroma-v1` loader
   parsing `bothchroma.csv` (Chordino treble+bass 12+12) into the same 25-dim layout
   as `numpy-chroma-v1`, energy flagged as unavailable, pitch-class bins rolled to the
   C-based order + L1-normalized — **verify bin order/hop against a real file on
   acquisition**; (c) teach the training assembler to consume `features` tracks with
   per-source provenance so cross-source leakage/normalization is explicit.
3. **Keep Isophonics as evaluation-only** reference material (like the existing
   Hotel California ref in `evaluation/chord-reference-songs.json`), not shipped-model
   training — audio ownership + license ambiguity make it unfit for a product model.

Attribution obligations to record if used: GuitarSet → CC-BY credit; Billboard →
cite Burgoyne et al. ISMIR 2011.

## 4. Open decisions for the user

- Is the first milestone acceptable as a **guitar-only** proof (GuitarSet), or must
  the first real model already be multi-genre (needs the Billboard features work up front)?
- Commercial licensing matters only if the learned engine ships to users. If it stays
  an **offline evaluation** engine indefinitely, research-only sources (incl. Isophonics
  with owned audio) reopen — but that contradicts the "selectable engine" goal.

## 5. Next concrete step once a source is chosen

Acquire it manually into a local path, then:
`python -m ml.preprocessing.prepare_dataset --<source>-dir <path>` → review manifest
(class dist + leakage warnings) → only then train. Until then production is untouched.
