# V52 activity and tactical-safety candidate

Status: **rejected after fresh-opening testing**. V52 scored **+1 =1 -6 (18.75%)**
over eight paired games against warm-up-fixed V43 at 10s + 0.1s. It won as Black
and drew as White in the Catalan, then lost both colours in the Scotch, Slav
and Ruy Lopez. All decisive games ended by checkmate, with no recorded flags,
illegal moves, initialization failures, crashes or voids. Initial imports were
56.65s for V52 and 44.20s for the opponent.

The targeted short-budget repairs did **not** generalize to playing strength.
The combined evaluator/extension changes are not promoted and no V52 archive
has been produced. V53 returns to V51's evaluator and search parameters to test
a separate, narrowly scoped transposition-depth correction.

## What the loss review established

V51's complete paired-opening match scored +4 =8 -6 against warm-up-fixed V43.
All six losses were reviewed with the local reference engine, initially at up to
depth 20 / 250ms per search. Early mistakes and starting assessments were then
rechecked at up to depth 24 / two seconds. Reference scores are estimates,
not proofs; attained depths and raw scores are retained.

- Italian: the bishop exchange `12.Bxd4` surrendered roughly two pawns of
  assessed value. Later `27.Rf3` missed the defensive `Bg8` continuation and
  changed an assessed drawable position to a clear loss.
- Queen's Gambit: `...Nxe3` overestimated a sacrifice; the reference preferred
  the forcing defensive `...Rd6`, attacking the queen.
- King's Indian: repeated passive choices missed bishop activity and defensive
  coordination. This remains a difficult area; V52 is not claimed to solve it.
- Caro-Kann ending: the passive `Rc3` was substantially worse than active `Rc7`.
- The final queen/pieces test start already left Black about six pawns worse
  by the reference after White's first move. It is unsuitable for a zero-loss
  target from approximately equal openings. The rook/minor start was also
  materially less balanced than the opening names implied.

Evidence: `training/data/v51/loss_audit/all_cases.json` and `confirmed.json`.
The reference executable is a development-only tool. No external engine,
search result table, or reference code is included in the submission.

## Changes

V52 retains V51's static cache and tactical move generator, and the unchanged
V44/V43 board substrate. The new evaluation scores pawn-safe mobility by piece
type, useful rook files, seventh-rank rook activity, and coordinated pressure
around the enemy king. King pressure requires multiple attacking pieces and
is reduced when the attacking side has no queen. Terms taper with material.

Search preserves quiet attacks on opposing queens and rooks near the horizon
and at the root. Such moves are exempt from late quiet pruning/reduction, and
can receive an extra ply near the horizon. A budget passed through recursion
limits these additions to **two per search path**. The detector is reused from
the team's V45; the explicit path budget avoids unbounded threat chasing.
No game-specific FENs or prescribed moves are embedded in the runtime.

## Revisions tested before the match

| Probe revision | Changes | 0.2s total reference-score change vs V51 | 1.0s change |
| --- | --- | ---: | ---: |
| Revision 0 | Mobility, rook files, king pressure | -233 cp | +2 cp |
| Revision 1, retained | Adds seventh rank and bounded threat search | +376 cp | -130 cp |
| Revision 2, rejected | Reduces extension budget from two to one | -305 cp | -118 cp |

Each column aggregates the same 13 replayed positions, with prior game positions
supplied to search. These are selected diagnostic loss positions, not an
unbiased strength benchmark. Identical choices share the same saved reference
assessment; changed choices receive equal one-second / depth-23 reference
budgets. Raw scores, mate fields and attained depths are saved.

Revision 1 finds `Bg8` and `Rc7` at 0.2s where V51 did not. It still regresses on
the one-second Queen's Gambit probe. Revision 2 lost the `Bg8` repair, but its
measured throughput was also much lower, so the trial does not isolate the
extension budget from host timing noise. Earlier source snapshots are retained
in `training/data/v52/revision0/` and `revision1/`.

Evidence: `training/data/v52/probe_*.json` and `graded_probes.json`. The retained
runtime hash matches `probe_v52_r1.json`.

## Verification and new-opening test

- 720 independently calculated python-chess activity/colour-mirror checks pass.
- A quiet rook move attacking a queen is explicitly recognized.
- Legal choices, timeout restoration, cache hits/collisions, promotion, en
  passant, mate and stalemate checks pass. Verification import: 65.15s.
- Ruff and strict mypy pass on both runtime files, using development-only Numba
  declarations for mypy.
- The new screen uses Catalan, Scotch, Slav and Ruy Lopez starts, each in both
  colours at 10s + 0.1s, against the same warm-up-fixed V43. These positions were
  not used to tune the loss repairs. Independent opening assessments were
  +39, +14, +28 and +42 cp for White, respectively, rather than the gross
  disadvantage in the previous queen/pieces fixture.

The screen uses isolated persistent workers and resets search state/history
between games through a development-only protocol command. It preserves the
existing referee and charges actual move wall time. Recorded match source
hashes, PGNs, timings and diagnostics are in `training/data/v52/holdout_screen/`.

No harness files, root agent or earlier archives were changed. No upload,
commit or push was performed. Only `agent.py` and `core.py` belong in a package;
all `dev/` files and reference engines are excluded.
