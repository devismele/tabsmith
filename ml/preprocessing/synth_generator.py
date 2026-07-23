"""Deterministic synthetic chord audio with perfect ground truth.

Synthetic data supplements (never replaces) licensed real recordings. Each track
renders a known progression to mono audio and emits exact chord boundaries,
beats, downbeats, key, and sections in the internal schema. Because the labels
are generated (not scraped), these examples carry a clean CC0-style license.

Rendering is pure numpy + the stdlib ``wave`` module, so it runs anywhere the
foundation pass runs — no SoundFonts, torch, or librosa required.
"""
from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from ..schema import ChordRegion, Section, Track

# Quality -> semitone intervals above the root.
_QUALITY_INTERVALS = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "5": (0, 7),
    "add9": (0, 4, 7, 14),
}

# Diatonic-ish progressions expressed as (scale-degree-semitone, quality).
_PROGRESSIONS = [
    [(0, "maj"), (9, "min"), (5, "maj"), (7, "7")],        # I vi IV V7
    [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")],      # I V vi IV
    [(9, "min"), (5, "maj"), (0, "maj"), (7, "maj")],      # vi IV I V
    [(0, "maj"), (5, "maj7"), (7, "7"), (0, "maj")],       # I IV7 V7 I
    [(9, "min"), (2, "min"), (7, "7"), (0, "maj")],        # vi ii V7 I
]

_KEY_ROOTS = [0, 2, 4, 5, 7, 9, 11]  # C D E F G A B tonics


def midi_to_freq(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _chord_midis(root_pc: int, quality: str, base_octave: int = 4) -> list[int]:
    root = 12 * base_octave + root_pc
    return [root + interval for interval in _QUALITY_INTERVALS.get(quality, (0, 4, 7))]


def _render_tone(freq: float, n: int, sr: int, envelope: np.ndarray, partials: int = 4) -> np.ndarray:
    t = np.arange(n) / sr
    wave_sum = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, partials + 1):
        wave_sum += (1.0 / harmonic) * np.sin(2.0 * np.pi * freq * harmonic * t)
    return wave_sum * envelope


def _adsr(n: int, sr: int, attack: float = 0.01, release: float = 0.08) -> np.ndarray:
    env = np.ones(n, dtype=np.float64)
    a = min(int(attack * sr), n)
    r = min(int(release * sr), n)
    if a:
        env[:a] = np.linspace(0.0, 1.0, a)
    if r:
        env[-r:] *= np.linspace(1.0, 0.0, r)
    return env


def generate_track(index: int, seed: int, duration: float, tempo_range: tuple[int, int]) -> tuple[Track, np.ndarray, int]:
    rng = np.random.default_rng(seed + index)
    sr = 22050
    tempo = int(rng.integers(tempo_range[0], tempo_range[1] + 1))
    beat = 60.0 / tempo
    bar = beat * 4
    key_root = int(rng.choice(_KEY_ROOTS))
    progression = _PROGRESSIONS[int(rng.integers(0, len(_PROGRESSIONS)))]

    audio = np.zeros(int(duration * sr), dtype=np.float64)
    chords: list[ChordRegion] = []
    beats: list[float] = []
    downbeats: list[float] = []

    bar_index = 0
    time = 0.0
    while time + bar <= duration + 1e-6:
        degree, quality = progression[bar_index % len(progression)]
        root_pc = (key_root + degree) % 12
        label = f"{['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root_pc]}:{quality}"
        start_sample = int(time * sr)
        end_sample = min(len(audio), int((time + bar) * sr))
        n = end_sample - start_sample
        if n <= 0:
            break

        env = _adsr(n, sr)
        # Chord voices (mid register).
        for midi in _chord_midis(root_pc, quality):
            audio[start_sample:end_sample] += 0.14 * _render_tone(midi_to_freq(midi), n, sr, env)
        # Bass root, an octave below.
        bass_midi = 12 * 2 + root_pc + 12
        audio[start_sample:end_sample] += 0.22 * _render_tone(midi_to_freq(bass_midi), n, sr, env, partials=2)
        # A short passing melody note that must NOT redefine the chord.
        if rng.random() < 0.6:
            mel_start = start_sample + int(2 * beat * sr)
            mel_len = min(int(0.4 * beat * sr), len(audio) - mel_start)
            if mel_len > 0:
                mel_midi = 12 * 5 + (root_pc + int(rng.choice([2, 4, 9, 11]))) % 12
                mel_env = _adsr(mel_len, sr, attack=0.005, release=0.05)
                audio[mel_start:mel_start + mel_len] += 0.10 * _render_tone(midi_to_freq(mel_midi), mel_len, sr, mel_env)

        chords.append(ChordRegion(round(time, 4), round(time + bar, 4), label))
        downbeats.append(round(time, 4))
        for b in range(4):
            beats.append(round(time + b * beat, 4))
        time += bar
        bar_index += 1

    total = time if time > 0 else duration
    # Light broadband noise so features are not perfectly clean.
    audio[: int(total * sr)] += 0.004 * rng.standard_normal(int(total * sr))
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak) * 0.89

    key_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_root]
    sections = [
        Section(0.0, round(total / 2, 4), "A"),
        Section(round(total / 2, 4), round(total, 4), "A'"),
    ]
    track = Track(
        track_id=f"synthetic-{index:04d}",
        artist="Synthetic",
        title=f"Progression {index} in {key_name} @ {tempo}bpm",
        duration=round(total, 4),
        source="synthetic",
        audio_availability="audio",
        split="training",
        key=f"{key_name}:maj",
        license="CC0-synthetic",
        source_url="",
        beats=beats,
        downbeats=downbeats,
        chords=chords,
        sections=sections,
        notes=f"Generated by synth_generator (seed={seed}, tempo={tempo}).",
    )
    # audio_path is assigned by render_dataset; validate happens there.
    return track, audio[: int(total * sr)].astype(np.float32), sr


_PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
# Split-separated seed namespaces so near-identical renders never cross splits.
SPLIT_SEEDS = {"training": 1000, "development": 2000, "synthetic_test": 3000}


def _simple_reverb(audio: np.ndarray, sr: int, decay: float = 0.3) -> np.ndarray:
    out = audio.copy()
    for delay_ms, gain in ((37, decay), (61, decay * 0.6), (89, decay * 0.35)):
        d = int(delay_ms / 1000 * sr)
        if d < len(out):
            out[d:] += gain * audio[:-d]
    return out


def generate_varied_track(index: int, seed: int, duration: float,
                          tempo_range: tuple[int, int]) -> tuple[Track, np.ndarray, int]:
    """Randomized track for pipeline-generalization testing (exact ground truth kept).

    Adds inversions, arpeggiation, 1-2 chords per bar, dominant sevenths, delayed
    bass, passing tones, missing thirds, weak fifths, noise, distortion, reverb,
    and instrument dropout. Labels always reflect the intended chord regardless of
    which chord tones were rendered.
    """
    rng = np.random.default_rng(seed * 99991 + index)
    sr = 22050
    tempo = int(rng.integers(tempo_range[0], tempo_range[1] + 1))
    beat = 60.0 / tempo
    bar = beat * 4
    chords_per_bar = int(rng.choice([1, 2]))
    slot = bar / chords_per_bar
    key_root = int(rng.choice(_KEY_ROOTS))
    progression = _PROGRESSIONS[int(rng.integers(0, len(_PROGRESSIONS)))]
    arpeggiate = rng.random() < 0.4
    add_reverb = rng.random() < 0.5
    distort = rng.random() < 0.35

    audio = np.zeros(int(duration * sr), dtype=np.float64)
    chords: list[ChordRegion] = []
    beats: list[float] = []
    downbeats: list[float] = []

    step = 0
    time = 0.0
    while time + slot <= duration + 1e-6:
        if abs((time / beat) - round(time / beat)) < 1e-3 and (round(time / beat)) % 4 == 0:
            downbeats.append(round(time, 4))
        degree, quality = progression[step % len(progression)]
        if quality == "maj" and rng.random() < 0.22:
            quality = "7"  # occasional dominant colouring
        root_pc = (key_root + degree) % 12
        label = f"{_PITCH_NAMES[root_pc]}:{quality}"

        start_sample = int(time * sr)
        end_sample = min(len(audio), int((time + slot) * sr))
        n = end_sample - start_sample
        if n <= 0:
            break

        missing_third = rng.random() < 0.25
        weak_fifth = rng.random() < 0.30
        drop_chord = rng.random() < 0.12
        drop_bass = rng.random() < 0.12

        midis = _chord_midis(root_pc, quality)
        if not drop_chord:
            for voice, midi in enumerate(midis):
                interval = midi - midis[0]
                if missing_third and interval in (3, 4):
                    continue
                amp = 0.13
                if weak_fifth and interval == 7:
                    amp *= 0.4
                if arpeggiate:
                    v_start = start_sample + int(voice * (n / max(1, len(midis))))
                    v_len = end_sample - v_start
                    if v_len > 0:
                        env = _adsr(v_len, sr)
                        audio[v_start:end_sample] += amp * _render_tone(midi_to_freq(midi), v_len, sr, env)
                else:
                    env = _adsr(n, sr)
                    audio[start_sample:end_sample] += amp * _render_tone(midi_to_freq(midi), n, sr, env)

        if not drop_bass:
            # Delayed bass attack, sometimes an inversion (chord tone) + a passing tone.
            bass_delay = int(rng.uniform(0.0, 0.12) * beat * sr)
            bass_start = start_sample + bass_delay
            bass_len = end_sample - bass_start
            if bass_len > 0:
                bass_interval = int(rng.choice([0, 0, 0, 4, 7]))  # mostly root, sometimes inversion
                bass_midi = 12 * 2 + (root_pc + bass_interval) % 12 + 12
                env = _adsr(bass_len, sr)
                audio[bass_start:end_sample] += 0.22 * _render_tone(midi_to_freq(bass_midi), bass_len, sr, env, partials=2)
                if rng.random() < 0.3 and bass_len > int(0.3 * beat * sr):
                    pass_len = int(0.2 * beat * sr)
                    pass_start = end_sample - pass_len
                    pass_midi = bass_midi + int(rng.choice([-2, 2, 5]))
                    audio[pass_start:end_sample] += 0.14 * _render_tone(
                        midi_to_freq(pass_midi), pass_len, sr, _adsr(pass_len, sr), partials=2)

        if rng.random() < 0.55:  # passing melody note that must not redefine the chord
            mel_start = start_sample + int(rng.uniform(0.3, 0.7) * slot * sr)
            mel_len = min(int(0.3 * beat * sr), len(audio) - mel_start)
            if mel_len > 0:
                mel_midi = 12 * 5 + (root_pc + int(rng.choice([2, 4, 9, 11]))) % 12
                audio[mel_start:mel_start + mel_len] += 0.09 * _render_tone(
                    midi_to_freq(mel_midi), mel_len, sr, _adsr(mel_len, sr, attack=0.005))

        chords.append(ChordRegion(round(time, 4), round(time + slot, 4), label))
        beats.append(round(time, 4))
        time += slot
        step += 1

    total = time if time > 0 else duration
    active = audio[: int(total * sr)]
    if distort:
        active = np.tanh(active * 2.2)
    if add_reverb:
        active = _simple_reverb(active, sr)
    active += 0.005 * rng.standard_normal(active.shape)
    peak = float(np.max(np.abs(active))) or 1.0
    active = (active / peak) * 0.89

    key_name = _PITCH_NAMES[key_root]
    track = Track(
        track_id=f"synthvar-{seed}-{index:04d}",
        artist="Synthetic",
        title=f"Varied {index} in {key_name} @ {tempo}bpm ({chords_per_bar}/bar)",
        duration=round(total, 4),
        source="synthetic",
        audio_availability="audio",
        split="training",
        key=f"{key_name}:maj",
        license="CC0-synthetic",
        beats=beats,
        downbeats=downbeats,
        chords=chords,
        notes=f"Varied synthetic (seed={seed}, arp={arpeggiate}, reverb={add_reverb}, distort={distort}).",
    )
    return track, active.astype(np.float32), sr


def build_split_tracks(split: str, count: int, duration: float,
                       tempo_range: tuple[int, int]) -> list[tuple[Track, np.ndarray, int]]:
    """In-memory varied tracks for a named split, using its dedicated seed namespace."""
    seed = SPLIT_SEEDS.get(split)
    if seed is None:
        raise ValueError(f"Unknown split '{split}'; expected one of {list(SPLIT_SEEDS)}")
    out = []
    for index in range(count):
        track, samples, sr = generate_varied_track(index, seed, duration, tempo_range)
        track.split = split if split != "synthetic_test" else "test"
        out.append((track, samples, sr))
    return out


def write_wav(path: str | Path, samples: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())
    return path


def render_dataset(out_dir: str | Path, count: int, seed: int, duration: float, tempo_range: tuple[int, int]) -> list[Track]:
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    ann_dir = out_dir / "annotations"
    tracks: list[Track] = []
    for index in range(count):
        track, samples, sr = generate_track(index, seed, duration, tempo_range)
        wav_path = write_wav(audio_dir / f"{track.track_id}.wav", samples, sr)
        track.audio_path = str(wav_path.resolve())
        track.validate()
        track.save(ann_dir / f"{track.track_id}.json")
        tracks.append(track)
    return tracks


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic chord audio with ground truth.")
    parser.add_argument("--out", default="ml/datasets/synthetic", help="output directory")
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--duration", type=float, default=24.0)
    args = parser.parse_args()
    tracks = render_dataset(args.out, args.count, args.seed, args.duration, (72, 132))
    print(f"Wrote {len(tracks)} synthetic tracks to {args.out}")


if __name__ == "__main__":
    _main()
