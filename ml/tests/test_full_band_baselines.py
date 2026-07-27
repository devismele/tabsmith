r"""Full-band baseline harness tests (offline, synthetic audio, rule engine only).

Exercise the engine x view evaluation loop, guitar-presence splitting, the
no-harmony control handling and the oracle-vs-mixture separation gap without
needing checkpoints or the Slakh archive.

Run: .\.venv\Scripts\python.exe -m unittest discover -s ml/tests -t .
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ml.full_band.baselines import (
    EngineResult,
    aggregate,
    build_engines,
    evaluate_track_views,
    separation_gap,
    write_report,
)
from ml.schema import ChordRegion


def _write_track(root: Path, classes: dict[str, str], seconds=2.0, rate=22050) -> None:
    import soundfile as sf

    stem_dir = root / "stems"
    stem_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(rate * seconds)) / rate
    for i, stem_id in enumerate(classes):
        tone = 0.1 * np.sin(2 * np.pi * (130.81 * (i + 1)) * t)
        sf.write(str(stem_dir / f"{stem_id}.flac"), tone.astype(np.float32), rate)
    sf.write(str(root / "mix.flac"), (0.2 * np.sin(2 * np.pi * 130.81 * t)).astype(np.float32), rate)


def _metadata(classes: dict[str, str]) -> dict:
    return {"stems": {k: {"inst_class": v, "audio_rendered": True}
                      for k, v in classes.items()}}


def _result(engine, view, track, guitar, *, root=0.7, detailed=0.6, duration=60.0,
            regions=20, fell_back=False):
    metrics = {} if fell_back else {
        "rootAccuracy": root, "majorMinorAccuracy": root - 0.02,
        "detailedAccuracy": detailed, "noChordPrecision": 0.5,
        "noChordRecall": 0.4, "fragmentationRate": 0.6,
        "predictedRegions": regions, "evaluatedDurationSeconds": duration,
        "meanBoundaryErrorMs": 400.0,
    }
    return EngineResult(engine, view, track, guitar, metrics, 0.5, fell_back)


class EngineConstructionTests(unittest.TestCase):
    def test_rule_engine_is_always_available(self):
        engines = build_engines({})
        self.assertIn("rule-v3", engines)

    def test_missing_checkpoints_are_skipped_not_fatal(self):
        engines = build_engines({"v1-ml": Path("does/not/exist.pt")})
        self.assertEqual(list(engines), ["rule-v3"])


class TrackViewEvaluationTests(unittest.TestCase):
    def test_scores_every_available_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = {"S00": "Guitar", "S01": "Bass", "S02": "Drums"}
            _write_track(root, classes)
            reference = [ChordRegion(0.0, 2.0, "C:maj")]
            results = evaluate_track_views(
                root, _metadata(classes), reference, build_engines({}),
                track_id="Track00001", has_guitar=True)
            views = {r.view for r in results}
            self.assertIn("full-mix", views)
            self.assertIn("oracle-guitar", views)
            self.assertIn("percussion-only", views)
            self.assertTrue(all(r.engine_id == "rule-v3" for r in results))

    def test_absent_views_are_skipped_without_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = {"S00": "Piano", "S01": "Drums"}
            _write_track(root, classes)
            results = evaluate_track_views(
                root, _metadata(classes), [ChordRegion(0.0, 2.0, "C:maj")],
                build_engines({}), track_id="T", has_guitar=False)
            self.assertNotIn("oracle-guitar", {r.view for r in results})
            self.assertIn("guitar-absent-harmonic", {r.view for r in results})

    def test_a_failing_engine_is_recorded_as_fallback_not_raised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            classes = {"S00": "Guitar"}
            _write_track(root, classes)

            def broken(_features):
                raise RuntimeError("engine exploded")

            results = evaluate_track_views(
                root, _metadata(classes), [ChordRegion(0.0, 2.0, "C:maj")],
                {"broken": broken}, track_id="T", has_guitar=True)
            self.assertTrue(results)
            self.assertTrue(all(r.fell_back for r in results))


class AggregationTests(unittest.TestCase):
    def _rows(self):
        return [
            _result("rule-v3", "full-mix", "T1", True, detailed=0.50),
            _result("rule-v3", "full-mix", "T2", False, detailed=0.30),
            _result("rule-v3", "oracle-harmonic", "T1", True, detailed=0.70),
            _result("rule-v3", "oracle-harmonic", "T2", False, detailed=0.50),
            _result("rule-v3", "oracle-guitar", "T1", True, detailed=0.80),
            _result("rule-v3", "percussion-only", "T1", True, detailed=0.05),
        ]

    def test_aggregates_by_engine_and_view(self):
        out = aggregate(self._rows())
        self.assertIn("full-mix", out["byEngineView"]["rule-v3"])
        self.assertEqual(out["byEngineView"]["rule-v3"]["full-mix"]["tracks"], 2)
        self.assertAlmostEqual(
            out["byEngineView"]["rule-v3"]["full-mix"]["detailedAccuracy"], 0.40, places=4)

    def test_guitar_presence_is_reported_separately(self):
        out = aggregate(self._rows())
        presence = out["byGuitarPresence"]["rule-v3"]["full-mix"]
        self.assertAlmostEqual(presence["guitarPresent"]["detailedAccuracy"], 0.50, places=4)
        self.assertAlmostEqual(presence["guitarAbsent"]["detailedAccuracy"], 0.30, places=4)

    def test_no_harmony_view_is_judged_on_no_chord_only(self):
        out = aggregate(self._rows())
        control = out["noHarmonyControl"]["rule-v3"]["percussion-only"]
        self.assertIn("noChordF1", control)
        self.assertNotIn("detailedAccuracy", control)
        # It must not be folded into the guitar-presence accuracy tables.
        self.assertNotIn("percussion-only", out["byGuitarPresence"].get("rule-v3", {}))

    def test_fallbacks_are_counted_and_excluded_from_scores(self):
        rows = self._rows() + [_result("rule-v3", "full-mix", "T3", True, fell_back=True)]
        out = aggregate(rows)
        self.assertEqual(out["totalFallbacks"], 1)
        self.assertEqual(out["byEngineView"]["rule-v3"]["full-mix"]["tracks"], 2)

    def test_empty_input_does_not_divide_by_zero(self):
        out = aggregate([])
        self.assertEqual(out["scoredRows"], 0)
        self.assertEqual(out["byEngineView"], {})


class SeparationGapTests(unittest.TestCase):
    def test_gap_quantifies_what_better_separation_could_buy(self):
        out = aggregate([
            _result("rule-v3", "full-mix", "T1", True, detailed=0.40),
            _result("rule-v3", "oracle-harmonic", "T1", True, detailed=0.65),
            _result("rule-v3", "oracle-guitar", "T1", True, detailed=0.75),
        ])
        gaps = separation_gap(out)
        self.assertAlmostEqual(gaps["rule-v3"]["harmonicOracleGain"], 0.25, places=4)
        self.assertAlmostEqual(gaps["rule-v3"]["guitarOracleGain"], 0.35, places=4)

    def test_gap_is_omitted_when_a_reference_view_is_missing(self):
        out = aggregate([_result("rule-v3", "full-mix", "T1", True)])
        self.assertEqual(separation_gap(out), {})


class ReportTests(unittest.TestCase):
    def test_report_is_written_and_serialisable(self):
        import json

        with tempfile.TemporaryDirectory() as tmp:
            out = aggregate([
                _result("rule-v3", "full-mix", "T1", True),
                _result("rule-v3", "oracle-harmonic", "T1", True, detailed=0.7),
            ])
            path = write_report(out, Path(tmp) / "r.json", extra={"pilotSize": 1})
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pilotSize"], 1)
            self.assertIn("separationGap", payload)
            self.assertIn("full-mix", payload["views"])


if __name__ == "__main__":
    unittest.main()
