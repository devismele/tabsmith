r"""Per-model evaluation caching: survive an interruption, never a stale score.

Scoring four models over 600 GuitarSet tracks and 40 full-band compositions in
three views takes longer than one uninterrupted window, and the results were
previously written only at the end, so a stop discarded all of it. Caching fixes
that, but a cache keyed loosely would be far worse than a slow evaluation: a
gate decision read from metrics belonging to different weights is wrong in a way
the report cannot show. The key therefore includes the checkpoint digest.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml.full_band.evaluate_pilot import cached_evaluation, file_digest


class Counter:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.payload


class FileDigestTests(unittest.TestCase):
    def test_same_bytes_give_the_same_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.pt", Path(tmp) / "b.pt"
            a.write_bytes(b"weights"), b.write_bytes(b"weights")
            self.assertEqual(file_digest(a), file_digest(b))

    def test_changed_weights_change_the_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"weights-v1")
            first = file_digest(path)
            path.write_bytes(b"weights-v2")
            self.assertNotEqual(first, file_digest(path))

    def test_it_reads_files_larger_than_one_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            path.write_bytes(b"x" * ((1 << 20) + 17))
            self.assertEqual(len(file_digest(path)), 64)


class CachedEvaluationTests(unittest.TestCase):
    def test_first_call_evaluates_and_second_reuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Counter({"fullBand": {"full-mix": {"rootAccuracy": 0.7}}})
            first = cached_evaluation(Path(tmp), "v1", "key-a", build)
            second = cached_evaluation(Path(tmp), "v1", "key-a", build)
            self.assertEqual(build.calls, 1)
            self.assertEqual(first, second)

    def test_different_weights_are_re_evaluated(self):
        """The failure this prevents: gates applied to another model's metrics."""
        with tempfile.TemporaryDirectory() as tmp:
            cached_evaluation(Path(tmp), "candidate", "digest-a", Counter({"root": 0.5}))
            rebuilt = Counter({"root": 0.9})
            value = cached_evaluation(Path(tmp), "candidate", "digest-b", rebuilt)
            self.assertEqual(value, {"root": 0.9})
            self.assertEqual(rebuilt.calls, 1)

    def test_each_model_is_cached_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_evaluation(Path(tmp), "v1", "k", Counter({"root": 0.1}))
            other = cached_evaluation(Path(tmp), "rehearsal-heavy", "k", Counter({"root": 0.2}))
            self.assertEqual(other, {"root": 0.2})

    def test_nested_metrics_round_trip_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"fullBand": {"full-mix": {"rootAccuracy": 0.7296,
                                                 "detailedAccuracy": 0.6444}},
                       "guitarset": {"audio_mono-mic": {"rootAccuracy": 0.4718}}}
            cached_evaluation(Path(tmp), "m", "k", Counter(payload))
            self.assertEqual(cached_evaluation(Path(tmp), "m", "k", Counter(None)), payload)

    def test_disabling_the_cache_always_evaluates(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Counter({"root": 0.5})
            cached_evaluation(None, "v1", "k", build)
            cached_evaluation(None, "v1", "k", build)
            self.assertEqual(build.calls, 2)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_no_temporary_file_survives_a_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_evaluation(Path(tmp), "v1", "k", Counter({"root": 0.5}))
            self.assertFalse((Path(tmp) / "v1.json.tmp").exists())
            self.assertTrue((Path(tmp) / "v1.json").exists())

    def test_the_key_is_recorded_alongside_the_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached_evaluation(Path(tmp), "v1", "key-a", Counter({"root": 0.5}))
            stored = json.loads((Path(tmp) / "v1.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["key"], "key-a")
            self.assertEqual(stored["value"], {"root": 0.5})


if __name__ == "__main__":
    unittest.main()
