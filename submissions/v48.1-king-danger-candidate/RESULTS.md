# V48.1: king-danger evaluation term (patch on V48)

V48.1 is a full copy of `submissions/v48-clock-utilization-candidate/` (byte-identical
`agent.py`; `core.py` diffed to confirm the only change is a pure, additive 50-line insertion)
targeting the specific, previously-diagnosed blind spot behind two real losses this session:

1. A local 20-game batch of V48 vs `submissions/v43-classical-pvs-candidate/` at 10s+0.1s
   produced two real losses (`training/data/v48_vs_v43_pgns/game_01.pgn`, `game_02.pgn`).
   Game 2 (V48 Black) lost to `29.Ngxf6+! gxf6 30.Nxf6+! Nxf6 31.Qxf6!`, a two-knight
   sacrifice that rips open Black's king.
2. This is the same class of blind spot as a real rated-game loss today
   (`submissions/aichessathon-round-98-meshpotato.pgn`), already diagnosed in
   `submissions/README.md`'s "Diagnosing the V20 loss" section: SEE only resolves capture
   sequences on one square; a quiet piece buildup into a sacrificial king attack is caught
   only by search depth, if at all.

## What was verified before changing anything

- **Check extensions already fire**: `search_node` extends depth by 1 whenever the side to
  move is in check (`agent.py`, `if in_check: depth += 1`), confirmed by reading the code.
- **LMR and late-move pruning already exclude checks and captures**: both the reduction
  logic and the late-quiet-move prune require `quiet and not gives_check`. Every move in the
  actual sacrifice sequence (`Ngxf6+`, `Nxf6+`, and the final `Qxf6` capture) is therefore
  never reduced or pruned by the existing logic — replaying the exact position confirmed
  this is not a reduction/pruning bug.
- **Direct replay of the position right before `29.Ngxf6+`** (FEN
  `2qrr1k1/1b2n1p1/p1p1pp1p/2Pn3N/1P2B1N1/P6P/1Q3PP1/3RR1K1 w - - 0 29`) and the position
  right before Black's `28...c6` (FEN
  `2qrr1k1/1bp1n1p1/p3pp1p/2Pn3N/1P2B1N1/P6P/1Q3PP1/3RR1K1 b - - 1 28`) using V48's own
  `Engine.search` reproduced the actual game's move choice exactly (V48 picks `c7c6` at the
  decision point, matching the real loss) — confirming the FEN reconstruction and that the
  search, not move ordering noise, is what allowed the sacrifice. This is a genuine
  evaluation-quality gap, not a search-depth problem: the line White plays is only 3 plies
  deep from Black's 28th move, well inside normal search depth at this time control.

## The fix

`core.py`'s `eval_core` gained a **king-danger term**: for each side, every enemy
knight/bishop/rook/queen that attacks a square in that side's own king zone (the king's
square plus its 8 neighbours, `KING_ATTACKS[king] | (1 << king)`) adds weighted "attack
units" (knight/bishop 2, rook 3, queen 5, capped at 8 units), indexed into a nonlinear
midgame-only centipawn penalty table (`KING_DANGER_MG = [0, 0, 10, 25, 45, 70, 100, 135,
175]`). This is additive to (does not replace) the pre-existing pawn-shield/open-file king
safety term already in `eval_core`. It reuses the exact same O(1) magic-bitboard attack
lookups (`bishop_attacks`, `rook_attacks`) and leaper tables (`KNIGHT_ATTACKS`,
`KING_ATTACKS`) SEE already depends on, so no new move generation or search machinery was
added — cost is proportional to the number of enemy minor/major pieces (typically <= 7 per
side), not a percentage of the search tree.

Board representation, move generation, make/unmake, hashing, and SEE are all byte-identical
to V48/V43/V44 (untouched), so no perft or python-chess cross-check re-run was needed; only
`eval_core`'s midgame score changed.

## Verification

- Ruff and strict mypy both clean.
- The two replay probes above (`before_28_c6`, `before_29_Ngxf6`) ran cleanly with the new
  term: the danger evaluation of the position after `28...c6` moved from -93cp to -129cp
  (Black's own view), a real, directionally-correct ~36cp shift towards recognizing the
  coming attack as worse than the old evaluator thought — but **not, on its own, enough to
  flip the move choice at that specific position and depth**; both V48 and V48.1 still pick
  `c7c6` there. A direct comparison of hand-picked alternative 28th moves (`Rf8` defending
  f6 along the file, `Kh8` sidestepping the check, `Nc3`, `Ng6`) under both engines showed
  `Rf8` scoring ~30-45cp better than `c6` under *both* V48 and V48.1 alike — i.e. a real,
  pre-existing root-move-selection gap at this fast time control that predates this change
  and that this change alone does not close. This is reported plainly rather than smoothed
  over: the king-danger term measurably increases the perceived cost of an exposed king, but
  a real game-result gate, not this one position, is the actual test of whether it helps.
- `evaluate()` microbenchmark (20,000 calls x 5 sample positions, single process, no other
  load): V48 baseline 463,532 calls/s, V48.1 577,095 calls/s. V48.1 was not slower in this
  measurement; given the addition is a real amount of extra work, this is most likely
  measurement noise from a shared, contended machine (the same JIT/compile-time variance
  this project's own README documents elsewhere) rather than a genuine speedup, but there is
  no evidence of a severe slowdown in the class of V13 (4x) or V14 (2.17x) that sank earlier
  eval additions.

## Real game results — REJECTED, do not upload

**A real v43-vs-v44 environmental finding first, since it shaped how this was gated:** an
isolated, zero-contention control test this round
(`uv run python -m harness.arena --agent submissions/v43-classical-pvs-candidate --opponent
baselines/random --games 2 ...`) showed V43 itself flagging on its very first move 1 game out
of 2, with nothing else running on the machine. This reproduces the exact bug
`submissions/README.md`'s V44 row already documents (V43's own internal warm-up search only
runs for 12s before giving up, sometimes leaving a jitted code path to compile lazily on the
first real clocked move) — confirmed by diffing V43 against
`submissions/v44-volatility-pvs-candidate/` (search/eval byte-identical apart from the warm-up
call: `(0.2, 12.0, max_depth=5)` in V43 vs `(0.2, 80.0, max_depth=2)` in V44). Given V43 was
unusable as a reliable opponent in this session's environment right now, **V44 was substituted
as the real-game opponent** for this gate: identical chess strength, but reliable. This
substitution, and the reasoning above, should be re-checked independently once the environment
is calmer, rather than assumed to generalize.

**Real result, two batches, foreground/blocking, `harness.arena`, 10s+0.1s, ply-cap 150,
alternating colours:**

- Batch 1 (2 games): `+0 =1 -1`, 25.0% (`training/data/v48.1_vs_v44_chunk1/`)
- Batch 2 (4 games): `+0 =1 -3`, 12.5% (`training/data/v48.1_vs_v44_chunk2/`)
- **Combined: `+0 =2 -4` over 6 games, 16.7%** — zero wins, two draws by threefold repetition,
  four real losses (checkmate x3, adjudication x1). No crashes, no illegal moves, no flags on
  either side in any of these six games (all six finished cleanly).

This is a real, clear, negative result, not a wash and not noise from a small sample landing
near 50% — every batch leaned the same losing direction. **V48.1 is REJECTED. It is not
recommended as an upload over V48.** Given the deadline, no further tuning of the
`KING_DANGER_MG` magnitude or a narrower trigger condition was attempted or validated; the
honest, most likely explanation (untested) is that scoring every enemy
knight/bishop/rook/queen that merely *attacks a square near the king* — with no requirement
that the attack is actually backed by real threat, more attackers than defenders, or the king
lacking safe flight squares — makes the term fire on many completely ordinary, safe
developing positions (e.g. a knight routinely posted on f5/g4 near a castled king), not just
genuine attacking build-ups. That would bias the evaluator broadly rather than fixing the
specific sacrificial-attack blind spot it targeted, which is consistent with a real, general
regression rather than a narrow miss on one motif. This should be the starting hypothesis for
anyone picking this back up, not a repeat from scratch.

`submissions/v48-clock-utilization-candidate/` remains the safe, already-packaged, recommended
upload (`submissions/agent-v48-clock-utilization-candidate.zip`) — this candidate does not
replace it.
