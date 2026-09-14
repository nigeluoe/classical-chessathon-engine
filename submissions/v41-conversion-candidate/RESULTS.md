# V41 conversion candidate

Status: locally verified and stronger than v40 in the requested internal match; external-engine
validation is still pending, so this is not a competition-winning claim. All paths below are
relative to the repository root.

## Requested V41 self-play

V41 played the identical V41 source for 15 games at 10s + 0.1s. The candidate seat scored
**+6 =4 -5 (53.3%)**, close to the expected even result. Eleven games ended by checkmate, two by
threefold repetition, one by insufficient material, and one by the fifty-move rule. There were
no flags, crashes, illegal moves, or void games. The logs exposed asymmetric critical-position
time allocation and motivated the separate V42 candidate. Full evidence is in
`training/data/v41/v41_selfplay_15games/`.

## Changes and targeted replay

V41 retains V35's evaluation and search substrate. It adds referee-aligned claimable-threefold
detection and conversion-aware clock allocation when the side to move has a low-material
advantage. Replaying the exact histories and original clocks from the V35-v40 match changed the
first target from `32.Rd1` to `32.Qb5` at depth 11, and the second from `52...a4` to `52...Qxa2`
at depth 9. The fixed-depth depth-10 regression in the latter position remains `52...Kf7`.

## Requested 15-game rerun against v40

At the same 10s + 0.1s clocks, openings, colours, and 15-game ordering as the V35 match, V41
scored **+10 =5 -0 (83.3%)**, compared with V35's **+6 =8 -1 (66.7%)**. All ten wins were
checkmates. The five draws were four threefold repetitions and one fifty-move result. There were
no flags, crashes, illegal moves, or void games. Candidate initialization ranged from 28.89s to
64.27s during the batch.

The two originally targeted games both changed from draws to V41 checkmate wins. The remaining
draws are preserved as games 9, 10, 11, 12, and 15. Game 9 reached a difficult rook ending with
an advanced a-pawn but did not complete a plan before the fifty-move claim; it is a plausible
remaining conversion weakness. The other four are repetitions in positions not established as
wins. Without the blocked external Stockfish review, none is claimed as a provable missed win.

Evidence, PGNs, clocks, logs, source hashes, and machine metadata are in
`training/data/v41/v41_vs_v40_15games/summary.json`.

## Verification

Ruff and strict mypy pass. Five exact perft positions, 1,494 randomized legal/make-unmake/hash/
evaluation checks, TT guard tests, referee-repetition regressions, conversion clock bounds, and
11,376 independently played legal SEE captures all pass. Verification import took 38.33s. The
arena-tested runtime hashes are `agent.py=23d774f34ba8d69d9b3afbf6fb8237042ee8dacd04c4f7af93a9f4dc24a86588`
and `core.py=dfa351c528ad019211f65d1b211c3cd517b54eb5b9c1abfe2d22b720ea805910`.
The source-only archive `submissions/agent-v41-conversion.zip` contains exactly `agent.py` and
`core.py` at its root (77,300 bytes uncompressed, 22,105 bytes compressed). Archive SHA-256:
`7718e5f58c665d241b12aacee273379ddbb438cdbf1150711ec99a48fb5dc8ba`.

## Historical V35/V34 provenance and external evidence

Only Stockfish and Lc0 are used as external references in this improvement loop. The internal
match above is comparative evidence, not a replacement for that external gate.

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

## Requested internal arena: V35 versus V40

V35 played 15 games against `submissions/v40-search-plus-candidate` at 10s + 0.1s from the
arena's five opening families. Because 15 is odd, the first 14 games form seven reversed-colour
pairs and the final Sicilian game gives V35 one extra White: eight White games and seven Black.

Result: **V35 +6 =8 -1, 66.7%**. The opening breakdown was Italian +1 =3 -0,
Queen's Gambit +0 =3 -1, Sicilian +2 =1 -0, French +2 =0 -0, and English +1 =1 -0.
All seven decisive games ended by checkmate and all eight draws by threefold repetition. There
were no voids, crashes, illegal moves, or flags. V35 initialization ranged from 40.33s to 54.50s
(47.16s median); V40 ranged from 42.31s to 56.76s (50.59s median).

The complete PGNs, stderr logs, clocks, source hashes, and machine metadata are saved in
`training/data/v35/v35_vs_v40_15games/summary.json`. This requested internal match is useful
comparative evidence but does not replace the pending Stockfish/Lc0 external gate.

Packaged as `submissions/agent-v35-search-optimized.zip`, containing only `agent.py` and
`core.py` at the archive root (74,356 bytes uncompressed, 21,545 bytes compressed). Archive
SHA-256: `ccf01af426d5a9a5ec205ac7dbd920b6abbd61590c3632e63f78d548add4ff5b`.
