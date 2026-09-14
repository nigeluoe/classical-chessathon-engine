# V49: longer-game clock reserve schedule (experimental, NOT the recommended upload)

V49 starts from V48 (V44's warm-up-safe V43 search + V48's larger transposition
table and raised per-move time ceiling) and adds one more change to
`time_budget`: past move 60, `moves_left` grows (capped at 60) instead of
staying floored at 18 forever, so per-move spending gradually tapers off in a
long, still-unresolved game instead of continuing to spend at the same rate
all the way to move 190+.

## Why

Replaying `submissions/aichessathon-round-97-the-bongcloud.pgn` (a real rated
loss, our side Black, 198 plies) shows both sides down to **3-8 real seconds**
by move 184; the losing `190...Rxd3` was played with only ~6.25s left. By the
time the position finally resolved after ~180 moves of shuffling, there was
essentially no clock left to calculate the critical decision correctly. V48's
ceiling-raise does not address this (it only ever raises the ceiling when
*large* amounts of time are banked, which is not the situation once the clock
is already drained). This change targets the actual mechanism: spend less
early in a long, unresolved game so more survives for a later critical moment.

## Honest status: NOT validated at the scale this would need

Unlike V48's ceiling-raise (safe by construction: it can only match or
increase thinking time versus V43, using the same proportional formula), this
change makes the engine spend systematically *less* per move in an ordinary
(not yet desperate) long game past move 60. That is a real, substantive
change to playing behaviour, not just a safety-margin adjustment, and
confirming it is a net improvement rather than a new, different risk (e.g.
under-thinking a genuinely complex, non-repetitive long middlegame) requires
many real full-length games -- exactly the kind of test this round's time
budget could not afford (each 120s+0.5s game that runs long enough to reach
move 60+ costs many real minutes).

What *was* checked, given the time available:
- Ruff and strict mypy both clean.
- The formula itself was spot-checked directly (not just read) across
  fullmove/remaining-clock combinations including a round-97-like scenario
  (fullmove=190, 6-8s remaining): it produces sane, positive, bounded
  soft/hard values in every case, confirming no division-by-zero or negative
  time bugs, and confirms the intended effect (e.g. at fullmove=190 with 90s
  banked, soft drops to 1.52s versus roughly 5.0s the unmodified formula
  would give at an equivalent earlier-game ratio).
- 2/2 vs `baselines/random` at 5s base, both checkmates, no crash.
- One real, non-ply-capped 120s+0.5s game vs `baselines/greedy`: white,
  checkmate, no crash, no flag.
- Import time, 2 trials via the real `harness.sandbox` path: 34.7s, 35.1s --
  consistent with V48, comfortably inside the 90s budget.
- 4 real games vs V43 (read-only, unmodified) at the standard 10s+0.1s local
  benchmark: `+1 =2 -1`, 50.0% -- no crashes, no flags, no illegal moves.
  Not a scored strength claim (4 games is far too few, and this control
  rarely reaches move 60 anyway), only a safety/regression spot-check.

## Recommendation

**Do not upload V49 in place of V48 under today's deadline.** It is a
reasoned, plausible, correctness-clean hypothesis for a real diagnosed
failure mode, not a validated improvement -- promoting an unvalidated
behaviour change over a validated, safe one (V48) the same day as an upload
deadline is exactly the kind of shortcut this project's own history (V13's
mobility term, V31's NNUE round 1) warns against taking on theory alone.
Recorded here, kept, and ready for whoever has time to run it through a real
multi-batch, long-game strength gate next.
