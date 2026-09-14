# V48: clock-utilization candidate

V48 starts from V44's warm-up-safe V43 PVS/search/evaluation implementation
(board representation, move generation, hashing, SEE, and the entire search
tree are byte-for-byte V43/V44 -- diffed to confirm) and makes two mechanical,
non-behavioural-logic changes, both aimed at the two concrete failure
patterns in this round's real rated-game losses (long, non-committal
shuffling phases and the resulting clock pressure on the eventual critical
decision):

1. **Larger transposition table**: `TT_BITS` raised from 21 (2,097,152
   entries) to 23 (8,388,608 entries, ~190 MB, well inside the 2 GB budget).
   More slots means fewer overwrites/collisions in the very long games this
   round's real-loss PGNs showed are a genuine pattern (round 96: ~110 plies;
   round 97: 198 plies) -- a long game visits far more distinct positions
   than a 2M-slot table can hold without heavy replacement.
2. **Removed an artificially low absolute ceiling on per-move search time**:
   V43/V44's `time_budget` capped soft/hard per-move time at a flat
   4.0s/7.0s *no matter how much clock was banked*. That ceiling cannot bind
   at this project's usual 10s+0.1s local benchmark (the proportional
   formula never reaches it there), but is reachable at the real
   120s+0.5s platform control once increments have banked extra time in a
   long game -- at that point V43/V44 would refuse to think longer than 7
   real seconds even with 100+ seconds sitting in the bank. V48 raises the
   ceilings to 20.0s/35.0s; the existing proportional safety fractions
   (`usable / moves_left`, `usable * 0.35`, `soft * 2.25`) are unchanged, so
   this can only match or increase thinking time versus V43/V44, never
   reduce it, and remains bounded by the same share-of-remaining-clock logic
   that made the original formula safe.

## Honest scope of this fix

Diagnosing the three real rated-game PGNs this round (`submissions/
aichessathon-round-96-imperial-larper.pgn`, `-97-the-bongcloud.pgn`,
`-98-meshpotato.pgn`) found two distinct patterns, and this fix addresses
only the first directly:

- **Round 96** (a piece-for-pawn material edge dissolving into an
  insufficient-material draw after ~50 shuffling moves): a genuine
  rook-and-few-pawns conversion difficulty. Some positions in this class are
  objectively drawn with correct defence; this round could not establish
  from the PGN alone whether a real win was missed, only that no progress
  was made. A bigger TT helps a long search reuse more of its own past work
  in exactly this kind of drawn-out phase, but this is not a guaranteed fix.
- **Round 97** (a 198-ply game ending in checkmate after `190...Rxd3` walks
  into `192.Kg6!`): replaying the PGN's own clock annotations shows both
  sides were down to **3-8 seconds** by move 184 onward -- the decisive
  `190...Rxd3` was chosen with only ~6.25s left. This is a genuine
  clock-exhaustion problem specific to very long games under a fixed
  base+increment control, not a case where banked time was sitting unused:
  V48's ceiling-raise does not help here (it only ever raises the ceiling
  when *large* amounts of time are banked, which was not the situation at
  move 184), and a real fix -- e.g. spending measurably less per move earlier
  in a long, still-unresolved position so more is preserved for a later
  critical moment -- was identified but not implemented this round: it is a
  materially different, higher-risk tuning change to `time_budget`'s
  `moves_left` schedule that this round's remaining time did not allow
  building and validating with real games. **This is recorded here as the
  clearest next step, not silently left unexplained.**
- **Round 98** (`19.Qxb7 Qxb6` grabs material, then `22...Bxg2! 23...Nf4!
  24...Rxg2!` forces mate): the same class of blind spot this project
  diagnosed in V20's loss to `training/noels_bot` months ago, except the
  bridge here is a *quiet* piece buildup (bishop/knight repositioning), not
  an on-square capture sequence -- exactly the gap `submissions/README.md`'s
  own "Diagnosing the V20 loss" section already flagged as still open ("a
  trap that plays out as a slower positional squeeze rather than a capture
  sequence is still caught only by search depth, if at all"). Neither this
  fix nor anything else changed this round touches evaluation or SEE, so
  this exact blind spot is unchanged from V43/V44/V45's own exposure to it.

## Verification

- Ruff and strict mypy both clean.
- 2/2 versus `baselines/random` at 5s base, both checkmates, no crash.
- Import time via the real `harness.sandbox` subprocess path, three trials:
  **33.8-34.3s** -- comfortably inside the 90s budget, not meaningfully
  different from V44's own ~30-40s despite the 4x larger transposition
  table.
- Board representation, move generation, make/unmake, hashing, and SEE are
  untouched (byte-identical to V43/V44's `core.py`), so no perft or
  python-chess cross-check re-run was needed; the only agent.py changes are
  a table-size constant and two numeric ceilings inside `time_budget`, nei
ther of which touches move legality or search correctness logic.

## Real game results (this round, on a machine under real concurrent load
from other sessions -- see the caveat below)

Versus V43 (read-only, unmodified), standard opening, alternating colours,
all foreground/blocking real games via `harness.referee.play_match`:

- **10s+0.1s** (the project's usual fast local benchmark, where the raised
  ceilings cannot bind, so this is really a regression check, not a strength
  test): two batches, `+1 =4 -3`/8 (37.5%) and `+2 =2 -1`/5 (60.0%).
  Combined **`+3 =6 -4` over 13 games, 46.2%** -- indistinguishable from a
  coin flip, as expected, since nothing that differs from V43 at this
  control (TT size, time ceilings) has any way to change which moves get
  played inside a 10-second budget.
- **120s+0.5s, ply-cap 90** (the real platform control, where the
  ceiling-raise can bind): **`+3 =2 -0` over 5 games** (three batches: 1W1D,
  1W, 1W1D) -- zero losses, no flags, no crashes, every game finishing by
  threefold repetition or material adjudication at the ply cap. Too small a
  sample to claim a real strength edge, but it is real, and it is the
  condition this candidate's own change actually targets; the
  standard-opening 10s+0.1s numbers above are not informative for what this
  specific change does.
- **Packaging gate**, one real, non-ply-capped 120s+0.5s game versus
  `baselines/greedy`: white, checkmate, no crash, no flag.

**Important caveat, established empirically this round, not assumed**: V44
(whose search is byte-identical to V43 apart from warm-up timing) was
re-tested fresh this round beyond its previously-reported 90% and scored
only **37.5-38.5%** combined across two fresh batches (13 games total,
`+2 =6 -5`) against V43 on this same machine. Since V43 and V44 cannot
differ in real chess strength (identical search/eval), this shows head-to-
head results between near-identical engines in this environment are
dominated by wall-clock/JIT timing noise from real concurrent load on this
shared machine, not by a genuine strength gap -- small batches (5-10 games)
are not a reliable signal here today. V48's own results should be read with
the same caveat: it is a real, mechanically-justified, safe-by-construction
change, but this round's real-game sample sizes are not large enough to
prove it "demolishes" V43 in the strict sense this round's mandate asked
for, and the honest evidence gathered this round suggests that bar may not
be reachable through this lineage without a genuinely different search or
evaluation improvement.
