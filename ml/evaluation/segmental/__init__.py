"""Segmental-harmony-v3 decoder study.

Evaluation-only, decoder-focused experiment layered on the frozen temporal-v2
full-v2 model. Nothing here changes the four-head model contract, production
weights, application defaults, release gating, or the existing hybrid.

The scientific question: preserve full-v2's chord-recognition and boundary
accuracy while reducing false or unsupported chord changes, reusing the existing
full-v2 probabilities and boundary outputs without retraining the network.
"""
