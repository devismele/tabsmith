"""Phase 2 over-segmentation diagnosis for full-v2.

Where do full-v2's excess chord regions come from? For the faithful full-v2
decoder (EMA smoothing + penalty-4 Viterbi) this measures, per capture:

* region-shape statistics (counts, short-region rates, A->B->A flicker)
* boundary-head vs decoded state-change precision/recall/F1 and their agreement
* an error taxonomy over predicted region changes (true vs false, by resulting
  region length and flicker)
* the evidence (top confidence, entropy, chord margin) immediately before FALSE
  transitions vs TRUE transitions

All from cached full-v2 output; no re-inference and no p00.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from ..decode import viterbi_decode
from ..metrics import _boundary_f1
from ..probability_smoothing import smooth_learned_probabilities
from .seg_metrics import _region_tuples

_TOL = 0.25


def _peak_boundaries(times, boundary, hop, threshold=0.5):
    """Predicted boundary times from the boundary head: local maxima above a
    threshold (simple peak pick, one boundary per contiguous run)."""
    out = []
    t = 1
    n = len(boundary)
    while t < n:
        if boundary[t] >= threshold and boundary[t] >= boundary[t - 1]:
            j = t
            while j + 1 < n and boundary[j + 1] >= threshold and boundary[j + 1] >= boundary[j]:
                j += 1
            out.append(float(times[j]))
            t = j + 1
        else:
            t += 1
    return out


def _nearest(value, candidates, tol):
    if not candidates:
        return None
    best = min(candidates, key=lambda c: abs(c - value))
    return best if abs(best - value) <= tol else None


def diagnose_capture(entries: list[Any], smoothing: dict) -> dict[str, Any]:
    """Aggregate diagnosis across all held-out captures for one capture view."""
    taxonomy = defaultdict(int)
    false_evidence = {"topConf": [], "entropy": [], "margin": [], "prevDurationS": [], "boundaryProb": []}
    true_evidence = {"topConf": [], "entropy": [], "margin": [], "boundaryProb": []}
    bh_hits = bh_pred = bh_ref = 0
    sc_hits = sc_pred = sc_ref = 0
    agree_changes = agree_boundary = 0
    total_extra = 0

    for e in entries:
        root, quality, nochord, boundary = smooth_learned_probabilities(
            e.root, e.quality, e.nochord, e.boundary, smoothing)
        predicted = viterbi_decode(e.times, root, quality, nochord, e.hop_seconds, 4.0)
        pred = _region_tuples(predicted)
        ref = _region_tuples(e.reference)
        ref_changes = [r[0] for r in ref[1:]]
        pred_changes = [p[0] for p in pred[1:]]
        total_extra += max(0, len(pred) - len(ref))

        # per-frame posterior for evidence at change frames
        chord = (root[:, :, None] * quality[:, None, :] * (1 - nochord[:, None, None])).reshape(len(root), -1)
        states = np.concatenate([chord, nochord[:, None]], axis=1)
        srt = np.sort(states, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        topconf = srt[:, -1]
        entropy = -np.sum(states * np.log(np.maximum(states, 1e-9)), axis=1) / np.log(37.0)

        # boundary-head precision/recall vs reference changes
        peaks = _peak_boundaries(e.times, boundary, e.hop_seconds)
        bh = _boundary_f1(ref_changes, peaks, _TOL)
        bh_ref += len(ref_changes)
        bh_pred += len(peaks)
        bh_hits += round(bh["recall"] * len(ref_changes)) if ref_changes else 0

        # state-change precision/recall vs reference changes
        sc = _boundary_f1(ref_changes, pred_changes, _TOL)
        sc_ref += len(ref_changes)
        sc_pred += len(pred_changes)
        sc_hits += round(sc["recall"] * len(ref_changes)) if ref_changes else 0

        # agreement: fraction of decoded changes with a boundary peak nearby
        for c in pred_changes:
            agree_changes += 1
            if _nearest(c, peaks, _TOL) is not None:
                agree_boundary += 1

        # taxonomy over predicted region changes
        for i in range(1, len(pred)):
            change_t = pred[i][0]
            fi = int(np.searchsorted(e.times, change_t))
            fi = min(max(fi, 0), len(e.times) - 1)
            ev = {
                "topConf": float(topconf[fi]), "entropy": float(entropy[fi]),
                "margin": float(margin[fi]), "boundaryProb": float(boundary[fi]),
            }
            true_change = _nearest(change_t, ref_changes, _TOL) is not None
            dur = pred[i][1] - pred[i][0]
            is_aba = i < len(pred) - 1 and pred[i - 1][2] == pred[i + 1][2] and pred[i][2] != pred[i - 1][2]
            if true_change:
                taxonomy["true_boundary"] += 1
                for k in true_evidence:
                    true_evidence[k].append(ev[k])
            else:
                for k in false_evidence:
                    if k == "prevDurationS":
                        false_evidence[k].append(pred[i - 1][1] - pred[i - 1][0])
                    else:
                        false_evidence[k].append(ev[k])
                if is_aba:
                    taxonomy["false_aba_flicker"] += 1
                elif dur <= e.hop_seconds * 1.5:
                    taxonomy["false_one_window"] += 1
                elif dur < 1.0:
                    taxonomy["false_short_multi_window"] += 1
                else:
                    taxonomy["false_long"] += 1

    def mean(x):
        return float(np.mean(x)) if x else 0.0

    bh_prec = bh_hits / bh_pred if bh_pred else 0.0
    bh_rec = bh_hits / bh_ref if bh_ref else 0.0
    sc_prec = sc_hits / sc_pred if sc_pred else 0.0
    sc_rec = sc_hits / sc_ref if sc_ref else 0.0
    return {
        "extraRegions": total_extra,
        "boundaryHead": {
            "precision": round(bh_prec, 4), "recall": round(bh_rec, 4),
            "f1": round(2 * bh_prec * bh_rec / (bh_prec + bh_rec), 4) if (bh_prec + bh_rec) else 0.0,
        },
        "stateChange": {
            "precision": round(sc_prec, 4), "recall": round(sc_rec, 4),
            "f1": round(2 * sc_prec * sc_rec / (sc_prec + sc_rec), 4) if (sc_prec + sc_rec) else 0.0,
        },
        "stateChangeBoundaryAgreement": round(agree_boundary / agree_changes, 4) if agree_changes else 0.0,
        "taxonomy": dict(taxonomy),
        "evidenceBeforeFalseTransition": {k: round(mean(v), 4) for k, v in false_evidence.items()},
        "evidenceBeforeTrueTransition": {k: round(mean(v), 4) for k, v in true_evidence.items()},
    }
