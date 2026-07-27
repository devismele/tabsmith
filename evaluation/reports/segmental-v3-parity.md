# Segmental-v3 TypeScript parity

No segmental decoder was selected for the application path: zero candidates passed the
frozen development gates, so v1 is retained (see `segmental-v3-selection.md`).

Per the protocol, an actual TypeScript decoder implementation and Python/TypeScript
parity fixtures are required only for a decoder *intended for the application path*.
Because nothing is promoted, no TypeScript decoder was added and no parity gate applies
in this cycle.

Portability note: the segmental decoders (`ml/evaluation/segmental/decoders.py`) are
implemented in pure numpy with no torch dependency and operate solely on the cached
four-head probability arrays, so they are portable to the TypeScript path if a future
frozen protocol selects one. The full-v2 model export path itself is unchanged; the
four-head model contract, production weights, application defaults, hybrid settings, and
release gating are untouched by this decoder-only study.
