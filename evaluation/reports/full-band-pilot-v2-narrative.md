## Interpretation

**The preservation failure is solved, and rehearsal composition is what solved
it.** `rehearsal-heavy` does not merely stay inside the 2.0 pp root tolerance —
it *improves* GuitarSet root accuracy by **+6.04 pp** on microphone and
**+3.19 pp** on pickup, and detailed accuracy by **+11.59** and **+8.19 pp**,
while gaining **+12.17 pp** full-mix detailed accuracy and +0.114 no-chord F1 on
full-band audio. Against the pilot-v1 primary it converts a 4.8 pp root loss
into a 6.0 pp root gain for 2.66 pp of full-band detailed accuracy.

**The ablation rules out the cheap explanation.** `low-learning-rate` ran the
same 50/50 stream at 0.4x the step size and preserved *nothing*: root regressed
8.1 pp (microphone) and 10.75 pp (pickup), worse than the pilot-v1 primary, and
it also broke the detailed-accuracy gate. Drifting less is not what preserves
solo guitar; training on more of it is. Without this candidate,
`rehearsal-heavy`'s result could have been read as "any gentler fine-tune would
do", and that reading would have been wrong.

**The anchor did not bind.** `root-anchored-distillation` ran the identical
50/50 stream, schedule and seed as the pilot-v1 primary with a KL anchor to the
frozen v1 root head, and finished *worse* on the gate it targeted: 6.76 pp and
10.3 pp root regression against v1's 4.8 and 8.18. Its full-band detailed
accuracy also came in slightly below that primary (0.6367 against 0.6444). The
anchor term fell from 0.27 to 0.14 during training while the total loss sat
between 4 and 6, so at the frozen weight of 1.0 it was roughly two percent of
the objective — too weak to constrain anything. This bounds root distillation at
weight 1.0, exactly as the frozen configuration said it would; it does not
refute the mechanism, and the weight was deliberately not tuned because there
was no honest budget to tune it against.

**The binding constraint moved.** `rehearsal-heavy` fails on over-segmentation:
fragmentation rises +0.0568 (microphone) and +0.0209 (pickup) against a 0.015
allowance, and regions per minute rise +2.92 and +1.46 against 0.75. Every gate
it failed is a segmentation gate, and every accuracy gate it faced — on both
domains — it passed. That is the same wall the temporal-v2, segmental-v3 and
boundary-v3 studies each hit from a different direction: this model's residual
weakness is where it places boundaries, not what chords it names.

## What this changes

The full-band question and the fragmentation question are now separable in a way
they were not before pilot v2. A model exists that is better than v1 on
full-band audio *and* better than v1 on solo guitar in both accuracy metrics,
whose only remaining defect is that it changes chord too often. Prior studies
attacked fragmentation with decoders and boundary supervision and moved it by
±0.014; this candidate moved accuracy by 6 to 12 points and moved fragmentation
the wrong way by 0.057.

The next study is therefore over-segmentation control applied to a
rehearsal-heavy mixed-domain model — not another preservation mechanism, and not
another boundary-head intervention. `segmental-full` decoding was measured in
segmental-v3 to cut fragmentation 0.672 -> 0.649 on the full-v2 model without
costing accuracy; it has never been applied to a model in this family that
already passes the accuracy gates.

## Limits

- **`rehearsal-heavy`'s GuitarSet gain is measured on performers it rehearsed
  on.** p01-p05 are the performers v1 trained on and the ones this candidate
  trains on further, so +6.04 pp is a forgetting check that came out positive,
  **not** evidence of generalisation to unseen players. p00 remains sealed and is
  the only honest test of that claim, and it stays sealed because no candidate is
  eligible. This limit matters more now than in pilot v1, where the number moved
  the other way.
- Every candidate was still improving when the frozen 12-epoch budget ended, so
  all full-band figures are lower bounds rather than converged results.
- Reference labels are note-level (~0.6 s mean region), so full-band
  fragmentation and regions/minute are not comparable to GuitarSet and the
  full-band fragmentation gate is reported as not-evaluable.
- Slakh2100 is rendered from MIDI. Even a fully passing pilot would be
  synthetic-domain evidence and would not establish production readiness.
- The distillation result bounds one weight of one anchor, not the idea of
  anchoring.
