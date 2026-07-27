# Segmental-v3 candidate comparison (p01-p05 development)

root | detailed | frag | rpm | MAE ms | flicker | bF1@250 per capture.

## audio_mono-mic

| candidate | root | detailed | frag | rpm | MAE ms | flicker | bF1@250 |
|---|---|---|---|---|---|---|---|
| full-v2-existing | 0.7565 | 0.7081 | 0.6719 | 22.207 | 429.6 | 5 | 0.6572 |
| duration-viterbi | 0.7550 | 0.7071 | 0.6631 | 21.721 | 472.1 | 9 | 0.6750 |
| boundary-gated-viterbi | 0.7683 | 0.7185 | 0.6875 | 22.877 | 383.1 | 11 | 0.7421 |
| duration-boundary-viterbi | 0.7673 | 0.7182 | 0.6825 | 22.463 | 388.1 | 6 | 0.7444 |
| semi-markov-chord | 0.7491 | 0.7026 | 0.6417 | 20.638 | 519.3 | 5 | 0.6697 |
| semi-markov-chord-boundary | 0.7615 | 0.7146 | 0.6614 | 21.413 | 430.8 | 4 | 0.7417 |
| segmental-full | 0.7583 | 0.7133 | 0.6486 | 20.783 | 479.0 | 4 | 0.7320 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-existing: root [0.7285, 0.7844] rpm [21.444, 23.009] frag [0.6561, 0.6873]
- duration-viterbi: root [0.7272, 0.7828] rpm [20.962, 22.518] frag [0.6461, 0.6796]
- boundary-gated-viterbi: root [0.7409, 0.7956] rpm [22.044, 23.757] frag [0.6708, 0.7040]
- duration-boundary-viterbi: root [0.7397, 0.7949] rpm [21.654, 23.294] frag [0.6657, 0.6986]
- semi-markov-chord: root [0.7206, 0.7775] rpm [19.899, 21.406] frag [0.6247, 0.6600]
- semi-markov-chord-boundary: root [0.7339, 0.7896] rpm [20.645, 22.202] frag [0.6448, 0.6784]
- segmental-full: root [0.7300, 0.7863] rpm [20.044, 21.528] frag [0.6306, 0.6667]

## audio_mono-pickup_mix

| candidate | root | detailed | frag | rpm | MAE ms | flicker | bF1@250 |
|---|---|---|---|---|---|---|---|
| full-v2-existing | 0.7543 | 0.7051 | 0.6789 | 22.417 | 410.7 | 6 | 0.6564 |
| duration-viterbi | 0.7531 | 0.7039 | 0.6669 | 21.964 | 438.0 | 8 | 0.6769 |
| boundary-gated-viterbi | 0.7650 | 0.7152 | 0.6947 | 23.100 | 353.6 | 13 | 0.7460 |
| duration-boundary-viterbi | 0.7645 | 0.7145 | 0.6892 | 22.693 | 357.0 | 9 | 0.7468 |
| semi-markov-chord | 0.7455 | 0.6967 | 0.6417 | 20.756 | 524.9 | 7 | 0.6671 |
| semi-markov-chord-boundary | 0.7586 | 0.7095 | 0.6597 | 21.426 | 443.8 | 5 | 0.7355 |
| segmental-full | 0.7571 | 0.7089 | 0.6481 | 20.927 | 475.7 | 5 | 0.7280 |

95% performer-paired bootstrap CIs (root / rpm / frag):

- full-v2-existing: root [0.7270, 0.7801] rpm [21.572, 23.221] frag [0.6644, 0.6941]
- duration-viterbi: root [0.7253, 0.7785] rpm [21.086, 22.763] frag [0.6509, 0.6827]
- boundary-gated-viterbi: root [0.7366, 0.7900] rpm [22.214, 23.927] frag [0.6788, 0.7093]
- duration-boundary-viterbi: root [0.7352, 0.7899] rpm [21.823, 23.496] frag [0.6734, 0.7043]
- semi-markov-chord: root [0.7167, 0.7718] rpm [19.914, 21.484] frag [0.6239, 0.6587]
- semi-markov-chord-boundary: root [0.7303, 0.7842] rpm [20.578, 22.198] frag [0.6422, 0.6767]
- segmental-full: root [0.7282, 0.7829] rpm [20.088, 21.689] frag [0.6303, 0.6657]

## Pareto frontier

boundary-gated-viterbi, duration-boundary-viterbi, segmental-full, semi-markov-chord, semi-markov-chord-boundary

## Runtime

- full-v2-existing: 0.0017 s/audio-min (0.51s total)
- duration-viterbi: 0.0053 s/audio-min (1.60s total)
- boundary-gated-viterbi: 0.0044 s/audio-min (1.34s total)
- duration-boundary-viterbi: 0.0055 s/audio-min (1.68s total)
- semi-markov-chord: 0.0363 s/audio-min (11.03s total)
- semi-markov-chord-boundary: 0.0402 s/audio-min (12.19s total)
- segmental-full: 0.0415 s/audio-min (12.59s total)