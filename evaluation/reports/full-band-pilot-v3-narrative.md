## Interpretation

Every row above is the same frozen `duration-viterbi` decoder; only the weights
differ. Two configurations are eligible, and the one with the best accuracy of
any model measured in this workstream is not among them.

**More full-band data helps full-band accuracy.** `scale-600` trains on five
times as many compositions and improves full-mix detailed accuracy 0.6174 →
0.6255 while *reducing* fragmentation on both captures (0.6873 → 0.6723 mic,
0.6653 → 0.6341 pickup). It passes every gate.

**It also costs solo-guitar accuracy.** `scale-600` gives up GuitarSet root
0.5784 → 0.5581 and detailed 0.4437 → 0.3891 against `rehearsal-heavy-120`. The
domain ratio is identical, so this is what a fixed rehearsal share buys when the
other domain gets five times richer: per-epoch rehearsal is unchanged while the
full-band signal it competes with is far more varied. Both remain far above v1
(root 0.5198, detailed 0.3258), which is why both still clear the preservation
gates.

**The longer budget produces the best model on every accuracy axis, and it is
ineligible.** `scale-600-longer` is the strongest result this workstream has
measured — GuitarSet root 0.6120, GuitarSet detailed 0.4567, full-mix detailed
0.6399, all bests — and it fails the microphone fragmentation gate at 0.0155
against 0.015. **It misses by 0.0005.**

That gate is not being moved. It was frozen before any full-band result existed,
it has now bound four consecutive studies, and a 0.0005 miss is exactly the
situation where relaxing a bound would convert a real constraint into a
formality. The same gate was held against `precision-boundary-loss` in
boundary-v3 when it missed by 0.0008.

**The mechanism is worth naming.** Doubling the budget improved development loss
from 3.9879 to 3.8706 and improved every accuracy metric, while making the
decoded output less temporally stable. Better frame-level fit and worse
segmentation are not in tension here — a sharper posterior switches state more
readily, and the duration constraint that was sufficient at 12 epochs is not
sufficient at 24. This is the same axis segmental-v3 and boundary-v3 mapped from
other directions, reached this time by training longer rather than by decoding
differently.

**A noise floor, measured rather than assumed.** `scale-600` and
`scale-600-longer` share a seed, data and schedule for their first twelve
epochs, so those epochs should be identical. Their development loss at epoch 12
differs by 0.004 (3.9879 against 3.9918), which is CPU reduction-order
nondeterminism. Differences smaller than roughly 0.004 in development loss are
therefore not meaningful, and the 0.0005 fragmentation miss is well inside the
region where a rerun could plausibly land on either side of the line. That is an
argument for treating the candidate as unresolved, not for admitting it.

## The selection question this study deliberately does not answer

Two candidates are eligible and they are not ordered by the gates, which admit
rather than rank:

| | GuitarSet root (mic) | GuitarSet detailed (mic) | full-mix detailed | fragmentation (mic) |
|---|---|---|---|---|
| `rehearsal-heavy-120` | **0.5784** | **0.4437** | 0.6174 | 0.6873 |
| `scale-600` | 0.5581 | 0.3891 | **0.6255** | **0.6723** |

`rehearsal-heavy-120` is better on solo guitar; `scale-600` is better on
full-band audio and on segmentation. Choosing between them requires a criterion,
and no criterion for ranking eligible candidates was frozen in advance. Inventing
one now, with both results already visible, would be choosing the winner and then
writing the rule that selects it.

So this study reports two eligible configurations and stops there. A ranking rule
should be frozen before the comparison is made, and the obvious candidate for it
— relative weight on the guitar domain versus full-band audio — is a product
question about what Tabsmith is for, not a question the metrics can settle.

## Limits

- **p00 is spent**, so nothing here is a performer-level generalisation claim.
  The p00 confirmation showed GuitarSet gains on p01–p05 are substantially a
  rehearsal effect, and that applies to every GuitarSet number in this table.
- The full-band gain is the part that survived p00 scrutiny, and it is the axis
  where scaling helped most — but it is measured on MIDI-rendered audio and
  remains synthetic-domain evidence.
- The enlarged pool is a strict superset of the pilot's, so the comparison is
  nested rather than independent.
- `scale-600-longer` was still improving when its 24-epoch budget ended, so its
  accuracy figures are a lower bound and its fragmentation is likely still
  drifting in the wrong direction.
- Nothing is promoted. No production weights, application defaults or release
  gating changed.
