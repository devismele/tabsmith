r"""The pilot feature cache: cheap restarts that cannot serve stale features.

Feature extraction costs ~13 minutes per pilot run, which an interrupted long
CPU job should not have to re-pay. The danger of caching in a study like this is
not a slow run but a wrong one: features built under a different pipeline,
boundary tolerance or track set must never be silently reused, because nothing
downstream would notice.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.run_pilot import cache_key, cached_samples


class Counter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.payload


class CacheKeyTests(unittest.TestCase):
    def test_same_inputs_give_the_same_key(self):
        self.assertEqual(cache_key(pipeline="numpy-chroma-v1", tolerance=0.12),
                         cache_key(tolerance=0.12, pipeline="numpy-chroma-v1"))

    def test_a_different_feature_pipeline_changes_the_key(self):
        self.assertNotEqual(cache_key(pipeline="numpy-chroma-v1", tolerance=0.12),
                            cache_key(pipeline="harmony-features-v1", tolerance=0.12))

    def test_a_different_boundary_tolerance_changes_the_key(self):
        self.assertNotEqual(cache_key(pipeline="numpy-chroma-v1", tolerance=0.12),
                            cache_key(pipeline="numpy-chroma-v1", tolerance=0.25))

    def test_a_different_track_set_changes_the_key(self):
        self.assertNotEqual(cache_key(trackIds=["Track00001"]),
                            cache_key(trackIds=["Track00001", "Track00002"]))

    def test_track_order_is_part_of_the_key(self):
        """Sample order drives the seeded epoch stream, so it is not incidental."""
        self.assertNotEqual(cache_key(trackIds=["a", "b"]), cache_key(trackIds=["b", "a"]))


class CachedSamplesTests(unittest.TestCase):
    def test_first_call_builds_and_second_call_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Counter(["s0", "s1", "s2"])
            first = cached_samples(Path(tmp), "guitarset", "key-a", build)
            second = cached_samples(Path(tmp), "guitarset", "key-a", build)
            self.assertEqual(build.calls, 1)
            self.assertEqual(first, second)

    def test_a_changed_key_recomputes_rather_than_serving_stale_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Counter(["old"])
            cached_samples(Path(tmp), "guitarset", "key-a", original)
            replacement = Counter(["new"])
            result = cached_samples(Path(tmp), "guitarset", "key-b", replacement)
            self.assertEqual(result, ["new"])
            self.assertEqual(replacement.calls, 1)

    def test_the_recomputed_cache_is_then_reused_under_the_new_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_samples(Path(tmp), "guitarset", "key-a", Counter(["old"]))
            cached_samples(Path(tmp), "guitarset", "key-b", Counter(["new"]))
            again = Counter(["should-not-be-used"])
            self.assertEqual(cached_samples(Path(tmp), "guitarset", "key-b", again), ["new"])
            self.assertEqual(again.calls, 0)

    def test_separate_names_do_not_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_samples(Path(tmp), "slakh-train", "k", Counter(["train"]))
            dev = cached_samples(Path(tmp), "slakh-dev", "k", Counter(["dev"]))
            self.assertEqual(dev, ["dev"])

    def test_disabling_the_cache_always_builds_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Counter(["s"])
            cached_samples(None, "guitarset", "k", build)
            cached_samples(None, "guitarset", "k", build)
            self.assertEqual(build.calls, 2)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_no_temporary_file_is_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_samples(Path(tmp), "guitarset", "k", Counter(["s"]))
            self.assertFalse((Path(tmp) / "guitarset.pkl.tmp").exists())
            self.assertTrue((Path(tmp) / "guitarset.pkl").exists())

    def test_sidecar_records_the_key_and_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_samples(Path(tmp), "guitarset", "key-a", Counter(["a", "b"]))
            sidecar = json.loads((Path(tmp) / "guitarset.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["key"], "key-a")
            self.assertEqual(sidecar["count"], 2)

    def test_a_missing_sidecar_forces_a_rebuild(self):
        """A blob without its recorded identity cannot be trusted."""
        with tempfile.TemporaryDirectory() as tmp:
            cached_samples(Path(tmp), "guitarset", "key-a", Counter(["old"]))
            (Path(tmp) / "guitarset.json").unlink()
            build = Counter(["rebuilt"])
            self.assertEqual(cached_samples(Path(tmp), "guitarset", "key-a", build), ["rebuilt"])
            self.assertEqual(build.calls, 1)

    def test_numpy_payloads_round_trip(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            payload = [np.arange(6, dtype=np.float32).reshape(3, 2)]
            cached_samples(Path(tmp), "guitarset", "k", Counter(payload))
            restored = cached_samples(Path(tmp), "guitarset", "k", Counter([]))
            np.testing.assert_allclose(restored[0], payload[0])


if __name__ == "__main__":
    unittest.main()
