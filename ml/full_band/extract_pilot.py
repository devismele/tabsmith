"""Extract audio for the frozen pilot subset only.

Bounded by construction: the track list comes from the committed pilot subset in
the preparation report, so this can never widen into a full 104 GB extraction.
Resumable -- members already present at the right size are skipped, so an
interrupted run continues instead of rewriting what it has.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .prepare import extract_audio_for
from .slakh import SLAKH_ZENODO

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "evaluation" / "reports" / "full-band-slakh-preparation.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pilot-subset audio.")
    parser.add_argument("--root", required=True, help="Dataset root outside the repository.")
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--limit", type=int, default=None,
                        help="Extract only the first N pilot tracks.")
    parser.add_argument("--split", default=None,
                        help="Select a deterministic subset from this official split "
                             "instead of the committed train-split pilot (e.g. validation).")
    parser.add_argument("--size", type=int, default=40,
                        help="Subset size when --split is used.")
    parser.add_argument("--tracks-manifest", default=None,
                        help="slakh-tracks.json; required with --split.")
    parser.add_argument("--stems", action="store_true", default=True,
                        help="Also extract per-instrument stems (needed for oracle views).")
    parser.add_argument("--no-stems", dest="stems", action="store_false")
    args = parser.parse_args()

    root = Path(args.root)
    if args.split:
        # Deterministic development subset: one rendering per composition,
        # ordered by aligned-MIDI SHA-256, exactly as the train-split pilot is
        # chosen. Selecting by content hash keeps it stable across machines.
        manifest_path = Path(args.tracks_manifest or (root / "manifests" / "slakh-tracks.json"))
        tracks = json.loads(manifest_path.read_text(encoding="utf-8"))
        seen: set[str] = set()
        chosen: list[str] = []
        for entry in sorted(tracks, key=lambda t: (t.get("midiSha256") or "", t["trackId"])):
            if entry.get("officialSplit") != args.split or entry.get("error"):
                continue
            digest = entry.get("midiSha256")
            if not digest or digest in seen:
                continue
            seen.add(digest)
            chosen.append(entry["trackId"])
            if len(chosen) >= args.size:
                break
        track_ids = chosen
        pilot = {"checksum": hashlib.sha256(
            "\n".join(sorted(track_ids)).encode("ascii")).hexdigest(),
            "trackIds": track_ids}
        print(f"selected {len(track_ids)} tracks from split '{args.split}'", flush=True)
    else:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        pilot = report.get("pilotSubset")
        if not pilot:
            parser.error("preparation report has no pilotSubset; run run_preparation first")
        track_ids = list(pilot["trackIds"])
    if args.limit:
        track_ids = track_ids[: args.limit]

    archive = root / "downloads" / SLAKH_ZENODO["archiveName"]
    if not archive.exists():
        parser.error(f"verified archive not found: {archive.name}")

    print(f"extracting {len(track_ids)} pilot tracks "
          f"(subset checksum {pilot['checksum'][:16]}...)", flush=True)
    summary = extract_audio_for(
        archive,
        set(track_ids),
        root / "extracted",
        members=("mix.flac", "metadata.yaml", "all_src.mid"),
        member_prefixes=("stems/",) if args.stems else (),
    )
    summary["pilotChecksum"] = pilot["checksum"]
    summary["requestedTrackIds"] = len(track_ids)
    (root / "manifests" / (("dev-" if args.split else "pilot-") + "extraction.json")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "requestedTrackIds"}, indent=2))


if __name__ == "__main__":
    main()
