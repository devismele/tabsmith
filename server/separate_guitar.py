"""Run six-stem Demucs and emit guitar, bass, and harmonic-evidence WAVs."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import static_ffmpeg
from audio_separator.separator import Separator


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: separate_guitar.py INPUT OUTPUT_DIR MODEL_DIR")

    input_path = Path(sys.argv[1]).resolve(strict=True)
    output_dir = Path(sys.argv[2]).resolve()
    model_dir = Path(sys.argv[3]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # audio-separator requires the FFmpeg executable. static-ffmpeg keeps it
    # project-installable and adds its cache location only to this child process.
    static_ffmpeg.add_paths(weak=True)

    separator = Separator(
        log_level=logging.INFO,
        model_file_dir=str(model_dir),
        output_dir=str(output_dir),
        output_format="WAV",
        sample_rate=44100,
        use_soundfile=True,
        demucs_params={
            "segment_size": "Default",
            "shifts": 1,
            "overlap": 0.25,
            "segments_enabled": True,
        },
    )
    separator.load_model(model_filename="htdemucs_6s.yaml")
    output_files = separator.separate(
        str(input_path),
        custom_output_names={
            "Guitar": "guitar",
            "Bass": "bass",
            "Piano": "piano",
            "Other": "other",
            "Vocals": "vocals",
            "Drums": "drums",
        },
    )
    if not output_files:
        raise RuntimeError("The separator did not create a guitar stem.")

    resolved: dict[str, Path] = {}
    for item in output_files:
        output_path = Path(item)
        if not output_path.is_absolute():
            output_path = output_dir / output_path
        output_path = output_path.resolve(strict=True)
        if os.path.commonpath((str(output_path), str(output_dir))) != str(output_dir):
            raise RuntimeError("The separator returned an unexpected output path.")
        lowered = output_path.stem.lower()
        for stem_name in ("guitar", "bass", "piano", "other", "vocals", "drums"):
            if stem_name in lowered:
                resolved[stem_name] = output_path

    if "guitar" not in resolved or "bass" not in resolved:
        raise RuntimeError("Demucs did not return the required guitar and bass stems.")

    # Harmony deliberately excludes drums/vocals. Bass is strongest, piano and
    # other accompaniment reveal chord quality, while guitar is supportive so
    # a lead fill cannot dominate the harmonic label.
    weighted_stems = [
        ("bass", 1.0),
        ("piano", 0.8),
        ("other", 0.5),
        ("guitar", 0.35),
    ]
    loaded: list[tuple[np.ndarray, float]] = []
    sample_rate: int | None = None
    channels = 1
    maximum_frames = 0
    for stem_name, weight in weighted_stems:
        stem_path = resolved.get(stem_name)
        if stem_path is None:
            continue
        audio, current_rate = sf.read(stem_path, dtype="float32", always_2d=True)
        if sample_rate is None:
            sample_rate = current_rate
        elif sample_rate != current_rate:
            raise RuntimeError("Separated stems use inconsistent sample rates.")
        channels = max(channels, audio.shape[1])
        maximum_frames = max(maximum_frames, audio.shape[0])
        loaded.append((audio, weight))
    if not loaded or sample_rate is None:
        raise RuntimeError("No harmonic stems were available.")

    harmonic = np.zeros((maximum_frames, channels), dtype=np.float32)
    total_weight = 0.0
    for audio, weight in loaded:
        if audio.shape[1] == 1 and channels == 2:
            audio = np.repeat(audio, 2, axis=1)
        harmonic[: audio.shape[0], : audio.shape[1]] += audio * weight
        total_weight += weight
    harmonic /= max(total_weight, 1e-6)
    peak = float(np.max(np.abs(harmonic)))
    if peak > 0.98:
        harmonic *= 0.98 / peak

    harmony_path = (output_dir / "harmony.wav").resolve()
    sf.write(harmony_path, harmonic, sample_rate, subtype="PCM_16")
    print(json.dumps({
        "guitarPath": str(resolved["guitar"]),
        "bassPath": str(resolved["bass"]),
        "harmonyPath": str(harmony_path),
        "model": "htdemucs_6s-harmonic-v1",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
