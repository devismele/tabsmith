# Segmental-v3 over-segmentation diagnosis (full-v2)

Faithful full-v2 decoder (EMA smoothing + penalty-4 Viterbi) on p01-p05 held-out captures.
p00 sealed. Frozen full-v2 reference: mic rpm 22.21 / frag 0.6719; pickup rpm 22.42 / frag 0.6789;
v1-objective reference: mic rpm 19.857 / frag 0.6167; pickup rpm 19.910 / frag 0.6169.

## audio_mono-mic

- extra regions vs reference: 111
- boundary-head P/R/F1: 0.7436 / 0.3085 / 0.4361
- state-change P/R/F1: 0.688 / 0.6427 / 0.6646
- state-change/boundary agreement: 0.2692
- taxonomy: {"true_boundary": 2121, "false_long": 742, "false_aba_flicker": 116, "false_short_multi_window": 104}
- evidence before FALSE transition: {"topConf": 0.424, "entropy": 0.447, "margin": 0.1804, "prevDurationS": 2.6883, "boundaryProb": 0.2342}
- evidence before TRUE transition: {"topConf": 0.4771, "entropy": 0.3801, "margin": 0.2184, "boundaryProb": 0.4236}

## audio_mono-pickup_mix

- extra regions vs reference: 133
- boundary-head P/R/F1: 0.7413 / 0.3161 / 0.4432
- state-change P/R/F1: 0.6835 / 0.6452 / 0.6638
- state-change/boundary agreement: 0.2607
- taxonomy: {"true_boundary": 2129, "false_long": 760, "false_aba_flicker": 129, "false_short_multi_window": 97}
- evidence before FALSE transition: {"topConf": 0.4394, "entropy": 0.4309, "margin": 0.1932, "prevDurationS": 2.6857, "boundaryProb": 0.2298}
- evidence before TRUE transition: {"topConf": 0.4753, "entropy": 0.3768, "margin": 0.2128, "boundaryProb": 0.422}
