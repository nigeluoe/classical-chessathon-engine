# V50 history-guided search: not promoted

Six paired games against `training/data/v44/v43_warmfixed_opponent` at
10 seconds + 0.1 seconds finished **+1 =4 -1 (50.0%)**. This does not meet
the requested mostly-wins/zero-loss target or demonstrate a strength gain.

| Opening | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| Italian | 0 | 1 | 1 |
| Queen's Gambit | 0 | 2 | 0 |
| Sicilian | 1 | 1 | 0 |

Both decisive games ended by checkmate; all draws were threefold repetitions.
There were no initialization failures, flags, crashes, illegal moves or voids.
The opponent has V43's original search/evaluation with only import warm-up
repaired, so compilation flags are not counted as playing-strength wins.

Changes tested: defer clearly losing captures using legal SEE, penalize earlier
unsuccessful legal quiet moves after a quiet cutoff, and adapt late-move
reductions to the resulting signed history. Pruned and illegal moves are
excluded from the history penalties. The board substrate and evaluator are
unchanged from V44. Strict mypy passed on both runtime files.
After the match, two nested SEE conditions were combined into one equivalent
condition to satisfy Ruff; both runtime files now pass Ruff. Recorded match
hashes refer to the pre-formatting source, with identical search behaviour.

Raw PGNs, logs, per-move timings, initialization timings and tested source
hashes: `training/data/v50/history_screen/summary.json` and adjacent files.
The copied arena tool labels the candidate V43 in PGN headers; candidate
colour and source hashes in the JSON identify V50 unambiguously.

This was a local Windows shared-machine screen, not a platform performance
measurement. Initialization varied substantially under concurrent host load.
The final game overlapped an independent V51 verification process. No Elo
claim is made from these six games. No V50 upload package was produced.
