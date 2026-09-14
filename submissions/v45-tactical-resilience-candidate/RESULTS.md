# V45 tactical-resilience candidate

V45 starts from V44's warm-up-safe V43 PVS engine. It targets the concrete
actual-game failure in `aichessathon-round-95-david-naylor.pgn`, without
changing the handcrafted evaluator or the perft-tested board substrate.

## Changes under test

- A one-ply, bounded extension for a quiet move that attacks an enemy queen or
  rook. Such a move cannot be seen by V43's capture-only quiescence search, yet
  it was the bridge `...Ne5!` in the lost game after `...Qe1+`.
- The same narrow class of move is exempt from late-move reduction and shallow
  quiet pruning at depth seven or below.
- Sparse castled pawn shelter facing an enemy queen and rook receives a capped
  1.5x soft-clock multiplier. This only changes time allocation, not scores.

The key regression is that after `36.g3? Rxf3 37.Rxf3 Qe1+ 38.Bf1`, the quiet
`...Ne5!` must keep a full-width ply when it is reached near the main-search
horizon; it cannot be resolved by V43's capture-only quiescence search alone.

## Checks completed

- `py_compile`, Ruff, and strict mypy pass for the two shipped files and the
  tactical regression.
- The regression confirms that `...Ne5` attacks the rook on f3 and that the
  exact sparse-shield position receives 1.51s -> 2.46s at a 27-second clock.
- A clean 27-second V45 replay of the actual position chose `36.Bf5`, not the
  losing `36.g3` (depth 10, 2,520,996 nodes). This is a tactical regression,
  not a strength claim.
- `core.py` is byte-identical to V44, so board representation, move generation,
  hashing, and SEE are unchanged.
- Two paired, varied-opening smoke games against the warm-up-fixed V43 were
  both clean threefold draws: `+0 =2 -0`, 50.0%. V45 initialised in 38.84s and
  42.95s (official allowance: 90s); there were no crashes, flags, or illegal
  moves. Records: `training/data/v45/v45_vs_v43_smoke_2/`.

This smoke result establishes reliability only. A larger paired V43 screen is
still required before calling V45 stronger.
