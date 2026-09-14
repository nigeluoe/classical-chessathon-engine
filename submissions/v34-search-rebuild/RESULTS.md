# External results: V34 optimization and rejected evaluation trial

Status: experimental, not proven stronger in matches and not a competition-winning claim.
Only Stockfish and Lc0 are used in this improvement loop. Older internal matches are not
acceptance evidence. All paths below are relative to the repository root.

## Baseline: real clocks, unrestricted external engines

The saved baseline is `training/data/v34/external-baseline-source.zip`.
Its runtime hashes are recorded in each result's `summary.json`.

| Opponent | Opening, both colours | Wins | Draws | Losses | Evidence |
| --- | --- | --- | --- | --- | --- |
| Stockfish 18 | Sicilian | 0 | 0 | 2 | `training/data/v34/stockfish_fullclock_before/` |
| Lc0 0.32.1, network 791556 | Sicilian | 0 | 0 | 2 | `training/data/v34/lc0_fullclock_before/` |

All four losses were checkmates, not flags, crashes, or illegal moves. Both players received
120s + 0.5s, one logical CPU, and no node/depth/move-time handicap. Stockfish used Skill Level
20 and UCI_LimitStrength false. Lc0 used its BLAS CPU backend and one search thread. UCI engines
received both actual clocks, increments, and game histories. Pondering was disabled. External
engines' lazy initialization, if any, was included in their first move; no Lc0 flag occurred.

The early positions showed overoptimistic scores for exposed kings and cramped pieces. For
example, against Stockfish as Black the baseline evaluated `17...Kh7` positively, while the
post-game reference judged the position clearly lost. The cheap post-game review has a depth
16 / 250ms cap; its scores are diagnostic estimates, not infallible labels.

## Sampled decision tests

Sampling uses every third candidate decision, up to 12 per game, rather than selecting moves
that a new feature happens to improve. Both candidates get a fresh TT/history and one second
of search per position. Stockfish reviews unrestricted and forced-choice searches at the same
root, depth 20 with a one-second cap each. Attained depths, negative raw score differences,
and mate distances are saved. Different attained depths introduce analysis noise.

| Build | Positions from | CP cases | Mean loss | Median nodes | Evidence |
| --- | --- | --- | --- | --- | --- |
| Baseline | Stockfish games | 20 (+2 mate) | 15.05 cp | 222,464.5 | `training/data/v34/reference_before/` |
| Mobility/king trial, before allocation optimization | Stockfish games | 20 (+2 mate) | 14.45 cp | 197,377.5 | `training/data/v34/reference_activity/` |
| Baseline | Lc0 games | 22 (+1 mate) | 30.82 cp | See JSON | `training/data/v34/reference_lc0_before/` |

The initial 0.60 cp difference is not convincing improvement, especially with about 11% fewer
nodes at equal time. That trial was not promoted. Its source is preserved in
`training/data/v34/activity-source-before-optimization.zip`.

The subsequent pawn-mask rewrite preserves that trial's evaluation exactly while removing
per-node allocations. Its combined reference test is `training/data/v34/reference_optimized/`.
Across the combined sample it scored 19.88 cp mean loss versus the baseline's 23.31 cp, but
this apparent gain did NOT survive the deeper review below.

## Rejected after deeper review

`training/data/v34/deeper_changed_moves/` contains a Stockfish MultiPV-2 search restricted to
the two candidate choices at each of the 15 changed positions, depth 22 with a four-second
cap. Unchanged choices need no pairwise comparison. Six changed choices improved, nine
worsened, with total score gain **-34 cp** across those 15 comparisons. Attained depths range
roughly 19-22 and sometimes differ by one within a pair, so this is still not an exact oracle.
It does not establish a statistically significant regression, but it fails to confirm a gain.

Consequently, mobility, king-pressure, and shelter changes were removed from the runtime
source. The rejected trial is preserved as
`training/data/v34/rejected-activity-optimized-source.zip`; its new-opening Stockfish match
logs are in `training/data/v34/stockfish_fullclock_activity/`. Do not upload that trial on the
strength of the optimistic shallow test. This is why deeper confirmation precedes promotion.

The retained working change is **allocation-free evaluation with exactly the original scores**.
Its results are separate: `training/data/v34/reference_speed_only/`. Known coarse king-safety
heuristics are preserved rather than claiming an unverified fix. The retained change is
tested for numeric parity against a slow, independent baseline evaluator.

## Correctness and environment

The optimization trial passed five exact perft stress positions, 1,494 randomized legal
move/make-unmake/hash checks, independent pawn-evaluation parity on those positions, mirrored
evaluation symmetry, mate/stalemate/TT/repetition/clock tests, Ruff, and strict mypy on both
runtime files with development-only Numba API typing. Direct verification import took 38.21s;
the first new Stockfish match initialized in 38.18s through the actual harness protocol.

The local virtual environment uses numpy 2.5.2, numba 0.67.0, chess 1.11.2, Python 3.13.5 on
Windows. The platform uses Python 3.12 on Linux and different hardware. The current public
docs specify 90s initialization; older repository notes saying 60s are historical. See
https://aichessathon.com/docs (checked 2026-09-09). The local harness files were not edited.

Match processes use core 0. The sampled-position tests use core 2 and may run concurrently;
these are local timing measurements, not isolated-hardware performance or Elo estimates.
Opening families are disjoint between baseline and new-opening tests, so their match results
are external stress tests, not a controlled causal estimate of the patch's playing strength.

The existing `submissions/agent-v34-search-rebuild.zip` predates these changes and is untouched.
No upload, commit, or push was performed. Development tools, reference data, engine binaries,
and the Lc0 network must never be added to a submission ZIP.

## Continuation: search efficiency and legal SEE

The continuation started from a source snapshot at
`training/data/v34/continuation-before-source/`. Three mechanical search changes are retained:

- constant-time de Bruijn bit scans replace per-square loops in `pop_lsb` and `king_square`;
- reusable non-PV TT bounds are consumed before move generation, with explicit guards for root,
  repetition, rule 50, mate-window collapse, and the ply cap;
- qsearch computes the checking-capture exception only when a losing SEE would otherwise prune.

The fixed-depth comparison preserves move, score, depth, and node count exactly on all 16 saved
Stockfish/Lc0 game positions. The first three-repeat run was visibly contaminated by local timing
noise and reported only 0.90x aggregate speed. A same-core seven-repeat rerun is therefore saved
separately and is the stronger measurement: `continuation_bench_rerun_before/` versus
`continuation_bench_rerun_after/` reports **1.68x aggregate** and **1.74x median** speed, with an
improvement on every position. The contradictory first run remains in the repository rather than
being hidden. The core-only scan cross-checks are in `core_scan_before.json` and
`core_scan_after.json`.

SEE also had a demonstrated correctness defect. In
`4k3/2p1r3/1B6/8/8/8/8/4R1K1 w - - 0 1`, `Bxc7` was scored -230 because the pinned rook on e7
was allowed to recapture; the legal result is +100 because `...Rxc7` exposes Black's king.
`least_valuable_legal_attacker` now rejects pinned recaptures and illegal king captures. The new
independent test actually plays capture sequences with python-chess and matched **11,376/11,376**
sampled captures, including 64 cases whose target had a pinned defender.

At fixed depth 7, legal SEE changed one unique choice across the 16-position set (the position is
duplicated between the two source matches); at the one-second budget that position still converged
to the saved build's `f1d3`, while the continued build reached depth 9 / 370,944 nodes rather than
the saved depth 8 / 266-272k nodes. The legal-SEE run retained effectively the same throughput as
the immediately preceding optimized run (about +1.3% aggregate nodes/second, within timing noise).

Final local gates passed: five perft positions, 1,494 randomized legal-set/make-unmake/hash/eval
checks, evaluation symmetry, mate/stalemate/futility/TT/repetition/clock regressions, the targeted
early-TT tests, Ruff, and strict mypy. A complete verification import took 40.67s; a separate
one-position process imported in 37.26s. Runtime SHA-256 values are
`agent.py=b6dfaa765f643d4abea1af6cb2113bcb96cabdc5ee26ad4cd61a3798220b5582` and
`core.py=ff4618dc2f1ad8e366d6790ba4463b9a65bc3e0d4712a4030fbeb41e770ad9b1`.

The new 45-position Stockfish review could not run in this session: sandboxed subprocess pipe
creation returned WinError 5, and the required elevated retry failed inside the approval service
before execution. No new Stockfish/Lc0 strength result exists for this source hash, so this
continuation is **not promoted or repackaged yet**. The two canonical rules pages also could not be
refetched for the same environment reason; the earlier 2026-09-09 rules note above remains the
latest successful local record, not a fresh verification from this continuation.
