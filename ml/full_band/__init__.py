"""Licensed full-band harmony dataset and evaluation foundations.

This package contains no audio and performs no downloads.  It validates portable
manifests, derives deterministic chord regions from symbolic note events, keeps
composition/artist groups together, and adapts approved audio views to the
existing learned-harmony training schema.
"""

from .manifest import (
    REQUIRED_AUDIO_VIEWS,
    FullBandManifestError,
    load_manifest,
    portable_manifest_checksum,
    validate_manifest,
)
from .splits import build_grouped_splits, validate_grouped_splits
from .symbolic import derive_chord_regions, normalize_chord_regions

__all__ = [
    "REQUIRED_AUDIO_VIEWS",
    "FullBandManifestError",
    "build_grouped_splits",
    "derive_chord_regions",
    "load_manifest",
    "normalize_chord_regions",
    "portable_manifest_checksum",
    "validate_grouped_splits",
    "validate_manifest",
]
