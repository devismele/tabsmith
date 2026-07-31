## Interpretation

**This is the first eligible candidate in the workstream.** `duration-viterbi`
applied to the frozen `rehearsal-heavy` weights passes all thirteen evaluable
gates, most by wide margins, against a gate set that has been reused unchanged
across three consecutive studies and that rejected every previous candidate.

**The control validates the harness.** `viterbi-penalty-4-control` reproduces
the frozen pilot-v2 `rehearsal-heavy` numbers exactly — fragmentation +0.0568 on
microphone, the same figure that rejected it — so the decoder axis is the only
thing that moved. Without that agreement none of the other rows would be
trustworthy.

**The win is genuinely Pareto, not a trade.** Against the control,
`duration-viterbi` gives up essentially nothing in accuracy (GuitarSet root
0.5802 -> 0.5784, detailed 0.4417 -> 0.4437; full-mix detailed 0.6178 -> 0.6174)
while removing the over-segmentation that rejected pilot v2: fragmentation
0.7439 -> 0.6873 on microphone and 0.7230 -> 0.6653 on pickup, regions per
minute 25.64 -> 22.17 and 24.64 -> 21.07.

**It is not degenerate.** A decoder can always pass a fragmentation gate by
emitting fewer, longer regions and destroying accuracy in the process. This one
lands at 22.17 regions/minute on microphone against v1's 22.72 and a reference
rate near 20, with detailed accuracy 11.79 pp *above* v1. It segments like v1
and labels far better.

**The simplest decoder won, which was not the expectation.** segmental-v3
identified `segmental-full` as its strongest over-segmentation control. On this
model it is the *worst* of the three: fragmentation 0.7514 on microphone, higher
than the control's 0.7439. `duration-boundary-viterbi` sits in between at
0.7197. Adding boundary gating and semi-Markov segmentation on top of the
duration constraint actively undid it here.

That is consistent with the boundary-v3 finding rather than in tension with it.
boundary-v3 concluded that this model family's boundary head, while individually
well calibrated (F1 0.817 raw), contributes an order of magnitude less to
decoded segmentation than the chord posterior does. The two candidates that read
the boundary head are exactly the two that failed; the one that ignores it and
constrains dwell time directly is the one that worked.

## What this changes

The workstream now has a configuration that beats the retained v1 on every
measured axis in both domains: full-band accuracy, solo-guitar accuracy, and
segmentation. Nothing has been promoted — this study selects a candidate, it
does not ship one.

The single measurement that would most change confidence is **p00**, the sealed
performer. Every preservation number here and in pilot v2 is measured on p01-p05,
which v1 trained on and which `rehearsal-heavy` rehearsed on further, so they are
forgetting checks rather than generalisation claims. Eligibility is precisely the
condition the seal was written to wait for, and unsealing is a separate frozen
decision rather than part of this study.

## Limits

- **The preservation result is still measured on rehearsed performers.** p01-p05
  are in the base model's training mixture. The +5.86 pp root gain is a
  forgetting check that came out strongly positive; it is not evidence of
  generalisation to an unseen player.
- **The decoder scalars were frozen against a different model.** They were
  transferred verbatim from segmental-v3, which tuned nothing on this model. That
  is what keeps the result honest, and it also means no candidate here is tuned
  for this model — a tuned decoder might do better, and could not be evaluated
  without fitting on the set the gates score.
- **Full-band evidence remains synthetic.** Slakh2100 is rendered from MIDI, its
  labels are note-level, and the full-band fragmentation gate remains
  not-evaluable for that reason.
- **The base model itself is unchanged and unpromoted.** This study varies only
  the decoder; the pilot-v2 conclusion to retain v1 as the shipped engine still
  stands until a promotion decision is taken on its own terms.
- Two gates remain not-evaluable for the reasons recorded in every pilot report,
  and are reported as such rather than counted as passes.
