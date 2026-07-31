r"""The decoder study's two load-bearing claims, asserted rather than trusted.

1. The scalar dials are transferred from segmental-v3 *verbatim*. If they were
   quietly adjusted for this model, the study would be tuning on the set its
   gates score and the result would mean nothing.
2. Data-derived parameters never come from the performer being scored. That is
   the segmental-v3 policy applied to a study with no fold structure of its own.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from ml.full_band.decoder_study import Response, evaluate, guitarset_params

ML_ROOT = Path(__file__).resolve().parents[1]
STUDY = ML_ROOT / "configs" / "full-band-decoder-v1.json"
SEGMENTAL = ML_ROOT / "configs" / "segmental-harmony-v3-candidates.json"
GATES = ML_ROOT / "configs" / "full-band-pilot-gates.json"


def _response(performer: str, capture: str = "audio_mono-mic", frames: int = 40) -> Response:
    rng = np.random.default_rng(abs(hash(performer)) % 2**32)
    root = rng.random((frames, 12))
    root /= root.sum(axis=1, keepdims=True)
    quality = rng.random((frames, 3))
    quality /= quality.sum(axis=1, keepdims=True)
    times = np.arange(frames) * 0.25
    return Response(
        times=times, root=root, quality=quality,
        nochord=rng.random(frames) * 0.1, boundary=rng.random(frames) * 0.2,
        hop_seconds=0.25,
        reference=[{"start": 0.0, "end": 5.0, "label": "C:maj"},
                   {"start": 5.0, "end": 10.0, "label": "G:maj"}],
        performer_id=performer, capture=capture, group="guitarset")


class FrozenStudyConfigTests(unittest.TestCase):
    def setUp(self):
        self.study = json.loads(STUDY.read_text(encoding="utf-8"))
        self.segmental = json.loads(SEGMENTAL.read_text(encoding="utf-8"))

    def test_gate_set_is_reused_unchanged(self):
        gates = json.loads(GATES.read_text(encoding="utf-8"))
        self.assertEqual(self.study["gateSet"], gates["gateSetId"])

    def test_the_bound_that_failed_pilot_v2_did_not_move(self):
        gates = json.loads(GATES.read_text(encoding="utf-8"))
        preservation = gates["preservationGatesVsV1GuitarSetDevelopment"]
        self.assertEqual(preservation["fragmentationRateIncreaseMaximum"], 0.015)
        self.assertEqual(preservation["regionsPerMinuteIncreaseMaximum"], 0.75)

    def test_decoder_knobs_are_copied_verbatim_from_segmental_v3(self):
        """The 'not re-tuned' claim, enforced against the source of truth."""
        source = {c["id"]: c.get("knobs", {}) for c in self.segmental["candidates"]}
        checked = 0
        for candidate in self.study["candidates"]:
            if candidate["id"] in source:
                self.assertEqual(candidate["knobs"], source[candidate["id"]],
                                 f"{candidate['id']} knobs diverge from segmental-v3")
                checked += 1
        self.assertGreaterEqual(checked, 3, "expected the transferred candidates to be checked")

    def test_candidate_kinds_are_supported_by_the_pipeline(self):
        from ml.evaluation.segmental.pipeline import CANDIDATE_KINDS

        for candidate in self.study["candidates"]:
            self.assertIn(candidate["kind"], CANDIDATE_KINDS)

    def test_exactly_one_control_that_reproduces_pilot_v2(self):
        controls = [c for c in self.study["candidates"] if c["role"] == "control"]
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]["kind"], "existing")
        self.assertEqual(controls[0]["knobs"]["transitionPenalty"], 4.0)

    def test_weights_are_declared_untouched(self):
        self.assertIn("NOT retrained", self.study["baseModel"]["note"])

    def test_p00_and_test_split_limits_are_recorded(self):
        text = json.dumps(self.study)
        self.assertIn("p00 stays sealed", text)
        self.assertIn("never used for selection", text)


class LeaveOnePerformerOutTests(unittest.TestCase):
    def setUp(self):
        self.responses = [_response(f"guitarset-p0{i}") for i in range(1, 6)]

    def test_every_performer_gets_parameters(self):
        params = guitarset_params("duration-viterbi",
                                  {"transitionPenalty": 6.0, "useDuration": True,
                                   "durationPenalty": 4.0, "maxMinDwellSeconds": 1.0},
                                  self.responses)
        self.assertEqual(sorted(params), [f"guitarset-p0{i}" for i in range(1, 6)])

    def test_a_performers_own_references_do_not_shape_its_decoder(self):
        """Change only p01's references; p01's parameters must be unaffected."""
        knobs = {"transitionPenalty": 6.0, "useDuration": True,
                 "durationPenalty": 4.0, "maxMinDwellSeconds": 1.0}
        before = guitarset_params("duration-viterbi", knobs, self.responses)
        mutated = list(self.responses)
        mutated[0].reference = [{"start": 0.0, "end": 0.05, "label": "C:maj"}] * 50
        after = guitarset_params("duration-viterbi", knobs, mutated)
        self.assertEqual(before["guitarset-p01"], after["guitarset-p01"])

    def test_other_performers_parameters_do_respond_to_that_change(self):
        """The converse: the mutation is real, it just must not reach p01."""
        knobs = {"transitionPenalty": 6.0, "useDuration": True,
                 "durationPenalty": 4.0, "maxMinDwellSeconds": 1.0}
        before = guitarset_params("duration-viterbi", knobs, self.responses)
        mutated = list(self.responses)
        mutated[0].reference = [{"start": 0.0, "end": 0.05, "label": "C:maj"}] * 50
        after = guitarset_params("duration-viterbi", knobs, mutated)
        self.assertNotEqual(before["guitarset-p02"], after["guitarset-p02"])

    def test_a_single_performer_cannot_supply_its_own_priors(self):
        with self.assertRaises(ValueError):
            guitarset_params("duration-viterbi", {}, [_response("guitarset-p01")])


class EvaluateGroupingTests(unittest.TestCase):
    def test_results_are_grouped_by_capture(self):
        from ml.evaluation.segmental.decoders import DecoderParams

        responses = [_response("guitarset-p01", "audio_mono-mic"),
                     _response("guitarset-p02", "audio_mono-pickup_mix")]
        out = evaluate(responses, "existing", lambda r: DecoderParams())
        self.assertEqual(sorted(out), ["audio_mono-mic", "audio_mono-pickup_mix"])
        for summary in out.values():
            self.assertIn("fragmentationRate", summary)
            self.assertIn("regionsPerMinute", summary)


if __name__ == "__main__":
    unittest.main()
