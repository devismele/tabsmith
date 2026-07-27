"""Freeze a portable full-band split manifest after license/data approval."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import load_manifest, portable_manifest_checksum
from .splits import build_grouped_splits


def freeze_splits(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    seed: int,
) -> dict:
    manifest = load_manifest(manifest_path, require_approved=True)
    split = build_grouped_splits(manifest["tracks"], seed=seed)
    payload = {
        **split,
        "datasetId": manifest["datasetId"],
        "datasetVersion": manifest["datasetVersion"],
        "sourceManifestChecksum": portable_manifest_checksum(manifest),
        "portableIdsOnly": True,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()
    payload = freeze_splits(args.manifest, args.output, seed=args.seed)
    print(json.dumps({
        "datasetId": payload["datasetId"],
        "componentCount": payload["componentCount"],
        "trackCount": len(payload["assignments"]),
        "seed": payload["seed"],
    }, indent=2))


if __name__ == "__main__":
    main()
