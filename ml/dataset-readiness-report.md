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

**Known gaps still to close:**
- GuitarSet has **no no-chord (N) frames** — the no-chord head gets no negatives from it;
  Billboard supplies those. Add Billboard before trusting the no-chord output.
- Billboard `import_billboard` sets `artist="Billboard"` for every track → they cannot be
  artist-separated internally. Parse `billboard-2.0-index` (id→artist) before using Billboard
  for held-out splits, or all Billboard lands in one split.
- `billboard-bothchroma-v1` bin order (`BASS_FIRST`) and origin (`NNLS_BIN_ORIGIN_PC`) are
  assumptions — run `calibrate_against_annotations` on a real `bothchroma.csv` at acquisition.

## First real result (2026-07-25, held-out validation player p00, 60 clips)

Three-way offline comparison, duration-weighted, on a performer the model never saw:

| engine | root % | majmin % | detail % | frag % | mean boundary err |
|---|---|---|---|---|---|
| rule (chroma-template-v0 floor) | 17.0 | 15.6 | 15.0 | 37.5 | 4058 ms |
| ml (temporal-baseline-real-v0)  | 52.0 | 40.5 | 33.0 | 81.1 | 557 ms |
| hybrid                          | 51.2 | 40.2 | 32.9 | 81.7 | 500 ms |

**Reading it honestly:** the model learned real signal — root 52% on an unseen
player is ~3× the non-learned chroma floor, with ~7× better boundary timing. But:
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
