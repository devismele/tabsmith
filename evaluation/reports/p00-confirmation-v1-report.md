# p00 confirmation result

**Outcome: confirmed, not strongly confirmed.** All eight pre-declared checks
pass on the sealed performer. The preservation claim generalises; the *size* of
the GuitarSet accuracy gain does not.

Protocol `p00-confirmation-v1-20260731`, frozen and committed before p00 was
read. Subject: `rehearsal-heavy` weights decoded with `duration-viterbi`.
Baseline: v1 with its production-parity decoder. 120 p00 tracks (60 clips × 2
captures). Decoder priors derived from p01–p05 only.

## Measured on p00 (never trained on, never selected against)

| model | capture | root | detailed | fragmentation | regions/min |
|---|---|---|---|---|---|
| v1 | audio_mono-mic | 0.4947 | 0.3486 | 0.6931 | 22.68 |
| v1 | audio_mono-pickup_mix | 0.4954 | 0.3555 | 0.6895 | 23.07 |
| candidate | audio_mono-mic | 0.5002 | 0.3888 | 0.6894 | 23.07 |
| candidate | audio_mono-pickup_mix | 0.4892 | 0.3710 | 0.6650 | 21.60 |

| result | check | capture | measured | required |
|---|---|---|---|---|
| PASS | rootRegression | audio_mono-mic | −0.55 | <= 2.0 pp |
| PASS | detailedRegression | audio_mono-mic | −4.02 | <= 2.0 pp |
| PASS | fragmentationIncrease | audio_mono-mic | −0.0037 | <= 0.015 |
| PASS | regionsPerMinuteIncrease | audio_mono-mic | 0.394 | <= 0.75 |
| PASS | rootRegression | audio_mono-pickup_mix | 0.62 | <= 2.0 pp |
| PASS | detailedRegression | audio_mono-pickup_mix | −1.55 | <= 2.0 pp |
| PASS | fragmentationIncrease | audio_mono-pickup_mix | −0.0245 | <= 0.015 |
| PASS | regionsPerMinuteIncrease | audio_mono-pickup_mix | −1.477 | <= 0.75 |

## The finding: rehearsal explains most of the GuitarSet gain

This is the measurement the seal existed to produce, and it materially revises
how the pilot-v2 and decoder-study numbers should be read.

| metric | p01–p05 (rehearsed) | p00 (unseen) |
|---|---|---|
| root, microphone | **+6.04 pp** | **+0.55 pp** |
| root, pickup | **+3.19 pp** | **−0.62 pp** |
| detailed, microphone | **+11.79 pp** | **+4.02 pp** |
| detailed, pickup | **+8.19 pp** | **+1.55 pp** |

On an unseen player the root improvement essentially vanishes — flat on
microphone, marginally negative on pickup — and the detailed improvement shrinks
to roughly a third of what the rehearsed performers showed. The caveat recorded
in pilot v2 and repeated in the decoder study was the right one, and it was
larger than a footnote: **most of the headline GuitarSet gain was a rehearsal
effect on performers the model had already seen.**

## What genuinely survives

The result is not negative, and stating it as one would be equally wrong:

- **Preservation generalises.** Eight of eight checks pass on an unseen
  performer. The mixed-domain model does not forget solo guitar in a way that
  shows up on someone it never trained on; the worst movement anywhere is
  −0.62 pp of root accuracy, inside a 2.0 pp tolerance.
- **Segmentation genuinely improves on unseen data.** Fragmentation falls on
  both captures (−0.0037, −0.0245) and regions per minute falls on pickup
  (−1.48). This is the `duration-viterbi` effect, and unlike the accuracy gain it
  is not a rehearsal artifact.
- **Detailed accuracy still improves**, modestly but on both captures
  (+4.02, +1.55 pp).
- **The full-band capability is untouched by this finding.** The +12.13 pp
  full-mix detailed improvement is measured on Slakh compositions that are not in
  the training subset, and no part of it depends on GuitarSet rehearsal.

The honest summary: the configuration buys a large full-band improvement at
genuinely no cost on unseen solo guitar, plus a modest detailed-accuracy gain and
a real reduction in over-segmentation. It does **not** deliver the +6 pp root
improvement on guitar that p01–p05 suggested.

## Consequences

- p00 is now **spent**. It cannot serve as a virgin holdout again, and any later
  study that measures on it must say so rather than presenting it as held out.
- The eligibility decision stands as recorded — this measurement was
  confirmatory and could not, by protocol, change the selection.
- Nothing is promoted. No production weights, application defaults or release
  gating changed.
- Any future claim about solo-guitar accuracy improvement in this family needs a
  performer-level holdout that no longer exists in GuitarSet. That is a real cost
  of this measurement and was accepted deliberately when the trigger condition
  was met.
