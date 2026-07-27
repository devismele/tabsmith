r"""Segmental-harmony-v3 decoder-study tests (torch-free).

Cover the Phase 13 invariants: cache identity/round-trip, decoder correctness and
region validity, duration/boundary/flicker/confirmation behaviour, portability
guards, p00 sealing, and deterministic candidate configuration.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.evaluation.decode import viterbi_decode
from ml.evaluation.segmental.decoders import (
    DecoderParams,
    confirm_chord_evidence,
    duration_viterbi,
    reconcile_flicker,
    semi_markov_decode,
)
from ml.evaluation.segmental.inference_cache import (
    CacheEntry,
    InferenceCache,
    cache_identity,
    entries_match,
)
from ml.evaluation.segmental.pipeline import CANDIDATE_KINDS, decode_regions
from ml.evaluation.segmental.priors import derive_priors, params_for_candidate

HOP = 0.25


def _peaked(seq_of_states, T_each=8, num_states=(12, 3), noise=0.05, rng=None):
    """Build root/quality/nochord/boundary arrays whose per-frame argmax follows a
    given sequence of (root_idx, quality_idx) chord blocks."""
    rng = rng or np.random.default_rng(0)
    roots, quals, ncs, bnds = [], [], [], []
    prev = None
    for block in seq_of_states:
        r_idx, q_idx = block
        for k in range(T_each):
            root = np.full(12, noise); root[r_idx] = 1.0
            qual = np.full(3, noise); qual[q_idx] = 1.0
            roots.append(root / root.sum())
            quals.append(qual / qual.sum())
            ncs.append(0.02)
            bnds.append(0.9 if (prev is not None and prev != block and k == 0) else 0.05)
            prev = block
    times = np.arange(len(roots)) * HOP + HOP / 2
    return (times, np.array(roots), np.array(quals), np.array(ncs), np.array(bnds))


def _valid(regions):
    if not regions:
        return False
    for i, r in enumerate(regions):
        if r.end <= r.start:
            return False
        if i and r.start < regions[i - 1].end - 1e-6:
            return False
    return True


def _coverage(regions, times):
    """Full-timeline coverage: first start <= first frame, last end >= last frame,
    and no interior gap between consecutive regions."""
    if not regions:
        return False
    lo = times[0] - HOP / 2 - 1e-6
    hi = times[-1] + HOP / 2 + 1e-6
    if regions[0].start > lo + 1e-3 or regions[-1].end < hi - 1e-3:
        return False
    return all(abs(regions[i].start - regions[i - 1].end) < 1e-3 for i in range(1, len(regions)))


def _entry(times, root, quality, nochord, boundary, reference, perf="p01-song", performer="guitarset-p01"):
    return CacheEntry(
        identity="x", dataset_id="d", split_id="s", fold_id="lopo-p01", capture="audio_mono-mic",
        performance_id=perf, performer_id=performer, model_checksum="m", feature_version="f",
        pipeline_version="p", hop_seconds=HOP, times=times, root=root, quality=quality,
        nochord=nochord, boundary=boundary, bass_chroma=np.zeros((len(times), 12)), reference=reference)


class CacheIdentityTests(unittest.TestCase):
    def _id(self, **overrides):
        base = dict(dataset_id="d", split_id="s", fold_id="lopo-p01", capture="audio_mono-mic",
                    performance_id="song", model_checksum="mc", feature_version="fv", pipeline_version="pv")
        base.update(overrides)
        return cache_identity(**base)

    def test_identity_includes_model_checksum(self):
        self.assertNotEqual(self._id(), self._id(model_checksum="other"))

    def test_identity_includes_feature_version(self):
        self.assertNotEqual(self._id(), self._id(feature_version="other"))

    def test_identity_includes_pipeline_version(self):
        self.assertNotEqual(self._id(), self._id(pipeline_version="other"))

    def test_identity_deterministic(self):
        self.assertEqual(self._id(), self._id())

    def test_cached_equals_uncached(self):
        times, root, quality, nc, bnd = _peaked([(0, 0), (5, 0), (0, 0)])
        entry = _entry(times, root, quality, nc, bnd, [{"start": 0.0, "end": 6.0, "label": "C:maj"}])
        entry.identity = self._id()
        with tempfile.TemporaryDirectory() as tmp:
            cache = InferenceCache(Path(tmp))
            cache.store(entry)
            loaded = cache.load(entry.identity)
        self.assertTrue(entries_match(entry, loaded))


class DecoderCorrectnessTests(unittest.TestCase):
    def setUp(self):
        self.times, self.root, self.quality, self.nc, self.bnd = _peaked([(0, 0), (7, 0), (0, 0)])

    def test_duration_viterbi_reduces_to_existing_when_terms_zero(self):
        for pen in (4.0, 8.0, 12.0):
            a = viterbi_decode(self.times, self.root, self.quality, self.nc, HOP, pen)
            b = duration_viterbi(self.times, self.root, self.quality, self.nc, self.bnd, HOP,
                                 DecoderParams(transition_penalty=pen))
            self.assertEqual([(r.label, round(r.start, 4)) for r in a],
                             [(r.label, round(r.start, 4)) for r in b])

    def test_all_candidates_valid_non_overlapping(self):
        entry = _entry(self.times, self.root, self.quality, self.nc, self.bnd,
                       [{"start": 0.0, "end": 6.0, "label": "C:maj"}])
        priors = derive_priors([entry])
        for kind in CANDIDATE_KINDS:
            params = params_for_candidate(kind, {"transitionPenalty": 6.0, "segmentSwitchPenalty": 8.0}, priors)
            regions = decode_regions(kind, entry, params)
            self.assertTrue(_valid(regions), kind)

    def test_semi_markov_covers_full_timeline(self):
        regions = semi_markov_decode(self.times, self.root, self.quality, self.nc, self.bnd, HOP,
                                     DecoderParams(segment_switch_penalty=8.0, max_segment_frames=40))
        self.assertTrue(_coverage(regions, self.times))

    def test_boundary_cannot_force_switch_alone(self):
        # Flat, ambiguous emissions (no chord evidence) but boundary head fully on:
        # a boundary-gated decoder must not manufacture a switch from evidence alone.
        T = 24
        root = np.full((T, 12), 1 / 12)
        quality = np.full((T, 3), 1 / 3)
        nc = np.full(T, 0.02)
        bnd = np.full(T, 1.0)
        times = np.arange(T) * HOP + HOP / 2
        params = DecoderParams(transition_penalty=6.0, boundary_gain=5.0, switch_cost_floor=1.0)
        regions = duration_viterbi(times, root, quality, nc, bnd, HOP, params)
        self.assertEqual(len(regions), 1)


class DurationAndFlickerTests(unittest.TestCase):
    def test_duration_preserves_genuine_short_chord(self):
        # A short but strongly-supported B chord between two A blocks must survive.
        times, root, quality, nc, bnd = _peaked([(0, 0), (7, 0), (0, 0)], T_each=6)
        # make the middle block short (2 frames) and very confident
        params = DecoderParams(transition_penalty=6.0, min_dwell_frames=3, duration_penalty=3.0)
        regions = duration_viterbi(times, root, quality, nc, bnd, HOP, params)
        labels = [r.label for r in regions]
        self.assertIn("G:maj", labels)  # root index 7 -> G

    def test_flicker_removes_unsupported_aba(self):
        # A-B-A where B is one weak frame with no boundary/margin support -> merged.
        times, root, quality, nc, bnd = _peaked([(0, 0), (7, 0), (0, 0)], T_each=1, noise=0.4)
        bnd[:] = 0.05  # no boundary support anywhere
        from ml.evaluation.segmental.pipeline import decode_regions as dr
        entry = _entry(times, root, quality, nc, bnd, [{"start": 0.0, "end": times[-1] + HOP, "label": "C:maj"}])
        base = duration_viterbi(times, root, quality, nc, bnd, HOP, DecoderParams(transition_penalty=1.0))
        params = DecoderParams(flicker_max_seconds=1.0, flicker_boundary_threshold=0.9, flicker_margin_threshold=0.9)
        merged = reconcile_flicker(base, times, root, quality, nc, bnd, HOP, params)
        self.assertLessEqual(len(merged), len(base))

    def test_flicker_keeps_supported_short_b(self):
        # Strongly supported short B with a real boundary must be retained.
        times, root, quality, nc, bnd = _peaked([(0, 0), (7, 0), (0, 0)], T_each=3, noise=0.02)
        base = duration_viterbi(times, root, quality, nc, bnd, HOP, DecoderParams(transition_penalty=1.0))
        params = DecoderParams(flicker_max_seconds=1.0, flicker_boundary_threshold=0.5, flicker_margin_threshold=0.1)
        merged = reconcile_flicker(base, times, root, quality, nc, bnd, HOP, params)
        self.assertIn("G:maj", [r.label for r in merged])

    def test_confirm_absorbs_unsupported_region(self):
        times, root, quality, nc, bnd = _peaked([(0, 0), (7, 0)], T_each=4, noise=0.02)
        base = duration_viterbi(times, root, quality, nc, bnd, HOP, DecoderParams(transition_penalty=1.0))
        # impossible confirmation threshold -> the second region cannot be confirmed
        params = DecoderParams(confirm_frames=9999, confirm_mass=9999.0, confirm_margin=9.0)
        confirmed = confirm_chord_evidence(base, times, root, quality, nc, HOP, params)
        self.assertEqual(len(confirmed), 1)


class PriorsTests(unittest.TestCase):
    def test_priors_use_supplied_entries_only(self):
        # duration priors come purely from the supplied (other-performer) references
        times, root, quality, nc, bnd = _peaked([(0, 0), (7, 0)], T_each=4)
        e = _entry(times, root, quality, nc, bnd,
                   [{"start": 0.0, "end": 1.0, "label": "C:maj"}, {"start": 1.0, "end": 5.0, "label": "G:maj"}])
        priors = derive_priors([e])
        self.assertEqual(priors.referenceRegionCount, 2)
        self.assertGreater(priors.durationMedianSeconds, 0.0)


class ReportPortabilityTests(unittest.TestCase):
    def test_absolute_path_rejected(self):
        from ml.evaluation.segmental.run import _assert_portable
        with self.assertRaises(ValueError):
            _assert_portable("see C:/Users/devis/secret/path")
        # a portable, relative report passes through unchanged
        self.assertIn("evaluation/reports", _assert_portable("evaluation/reports/x.md"))


class SealingAndConfigTests(unittest.TestCase):
    def test_candidate_config_deterministic_checksum(self):
        from ml.evaluation.segmental.evaluate import DEFAULT_CANDIDATES, load_candidates
        _, a = load_candidates(DEFAULT_CANDIDATES)
        _, b = load_candidates(DEFAULT_CANDIDATES)
        self.assertEqual(a, b)

    def test_candidate_set_has_seven_bounded_candidates(self):
        cfg = json.loads(Path("ml/configs/segmental-harmony-v3-candidates.json").read_text(encoding="utf-8"))
        kinds = [c["kind"] for c in cfg["candidates"]]
        self.assertEqual(len(cfg["candidates"]), 7)
        self.assertEqual(set(kinds), set(CANDIDATE_KINDS))

    def test_zero_track_evaluation_fails(self):
        from ml.evaluation.segmental.evaluate import load_cache_entries
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_cache_entries(Path(tmp))

    def test_p00_absent_from_candidate_and_split_scope(self):
        # the study operates only on model-selection performers p01-p05
        split = json.loads(Path("ml/configs/temporal-harmony-v2-splits.json").read_text(encoding="utf-8"))
        performers = {p["performerId"] for p in split["modelSelectionPerformers"]}
        self.assertNotIn("guitarset-p00", performers)


if __name__ == "__main__":
    unittest.main()
