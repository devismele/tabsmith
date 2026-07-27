from __future__ import annotations

import unittest

from ml.schema import ChordRegion, Track
from ml.training.train_real import partition


def track(track_id: str, artist: str, source: str, split: str) -> Track:
    return Track(
        track_id=track_id,
        artist=artist,
        title=track_id,
        duration=2.0,
        source=source,
        audio_availability="audio",
        split=split,
        license="fixture",
        audio_path=f"{track_id}.wav",
        chords=[ChordRegion(0.0, 2.0, "C:maj")],
    )


class FullBandTrainingPartitionTests(unittest.TestCase):
    def test_frozen_full_band_split_is_preserved_while_guitarset_is_hashed(self):
        full_band = track(
            "full-band@full-mix",
            "licensed-artist",
            "full-band:licensed-fixture",
            "development",
        )
        guitarset = track(
            "guitarset-track",
            "guitarset-p01",
            "guitarset",
            "development",
        )
        train, development, assignment = partition(
            [full_band, guitarset], seed=20260727, cap=None
        )
        self.assertEqual(
            assignment["trackSplit"][full_band.track_id], "development"
        )
        self.assertIn(full_band, development)
        self.assertEqual(guitarset.split, assignment["trackSplit"][guitarset.track_id])
        self.assertEqual(len(train) + len(development), 2)


if __name__ == "__main__":
    unittest.main()
