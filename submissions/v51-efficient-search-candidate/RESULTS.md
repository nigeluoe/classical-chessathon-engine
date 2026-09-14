# V51 efficient search candidate

The completed 18-game screen scored **+4 =8 -6 (44.4%)** against warm-up-fixed
V43. The requested mostly-wins/zero-loss target was **not achieved**. V51 has a
measured local speed improvement but no demonstrated playing-strength improvement.
It is not recommended as a replacement for V43 on these results.

## Completed match

| Opening | Wins | Draws | Losses |
| --- | ---: | ---: | ---: |
| Italian | 0 | 1 | 1 |
| Queen's Gambit | 0 | 1 | 1 |
| Sicilian | 1 | 1 | 0 |
| French | 0 | 2 | 0 |
| English | 1 | 1 | 0 |
| King's Indian | 0 | 1 | 1 |
| Caro-Kann | 1 | 0 | 1 |
| Rook/minor opening | 0 | 1 | 1 |
| Queen/pieces opening | 1 | 0 | 1 |

All ten decisive games ended by checkmate; all eight draws were repetitions.
No recorded game ended by flag, illegal move, initialization failure, crash, or
void. Two unexplained development-runner exits interrupted the batches between
recorded games; the remaining pairings were completed without dropping results.
The last two games used fresh processes and the original harness runner. V51
initialized in **60.95s and 61.68s** and completed both games cleanly (+1 -1).
Evidence: `training/data/v51/combined_screen.json`, with raw PGNs and logs in
`persistent_screen/`, `persistent_screen_part2/`, and `fresh_completion/`.

## Implementation

V51 starts from V44's warm-up-safe version of V43. It retains V43's evaluator,
move ordering, pruning parameters, time allocation, and transposition table.
The separate V50 history experiment was not incorporated: its six-game screen
finished +1 =4 -1.

- Non-check quiescence nodes generate captures and promotions directly, avoiding
  the discarded non-pawn quiet moves and castling checks. Checked positions still
  generate all evasions. The established pawn generator retains en passant and
  all four promotion choices.
- A 262,144-entry direct-mapped static-evaluation cache stores a full 64-bit
  position key and 32-bit score (3 MiB of array storage). Draw, repetition and
  halfmove-clock logic remain outside this cache. Entries with mismatched full
  keys cannot be reused. Engine reset invalidates all entries.
- The selected move supplies its own en-passant flag in quiescence pruning;
  V43 accidentally retained the last generated move's flag in that loop.

`core.py` is byte-identical to V44. Compared with the frozen V43 opponent, its
only difference is the module docstring; the remaining AST is identical.

## Correctness and speed

- The ordered tactical generator matches the original full generator's filtered
  moves on **2,648** seeded legal-play and special-move positions. The shipped
  function's AST exactly matches the independently tested draft.
- Seven search positions pass legal-move, board/metadata restoration, promotion,
  en-passant, mate, and stalemate checks. Cache hits and deliberately forced
  direct-map collisions match uncached evaluation. An interrupted search restores
  its input. Clock formulas are bounded at seven clock values.
- Verification import: **74.79 seconds** locally. Evidence:
  `training/data/v50/v51_search_checks.json`.
- Eight fixed positions, depth eight, three repeats each: all **24/24** searches
  match V43's move, score, completed depth **and node count**. Aggregate elapsed
  search time falls from **2.814s to 2.174s**: **1.294x throughput**, or **22.7% less
  time**. This is a local sequential timing sample on shared hardware, not an
  isolated-hardware or Elo estimate. Imports were 69.28s baseline and 67.31s V51.
  Evidence: `training/data/v51/bench_comparison.json` and the two adjacent raw
  benchmark JSON files.
- Ruff passes for the candidate directory; strict mypy passes on both runtime
  files and `dev/verify_search.py` using the development-only Numba declarations.

## Match methodology

`dev/screen.py` uses the unchanged `harness.referee.play_match` with two isolated
worker processes. Each module compiles once; every game resets search state
and clears both history globals. All move wall time is charged by the
referee. The extra reset command exists only in the development runner and is
never shipped. This measures playing behaviour without paying repeated JIT
startup per game; it does not replace fresh-process arena checks.

The earlier same-process two-module runner exited before producing any games.
It has no result and is not counted as an agent win, loss, or strength evidence.

The first persistent-worker batch completed eight games (+1 =5 -2) before the
parent runner exited without a ninth game or diagnostic. Those eight completed
records remain valid; no result is assigned to an unplayed game. A continuation
starts at the next paired opening. The reset command now calls `Engine.clear()`
instead of allocating replacement arrays, to avoid unnecessary memory churn;
the cause of the abrupt runner exit was not established. The continuation also
exited after eight completed games; the final pair used the original fresh-process
runner and finished normally. Consequently the reset change is not claimed as a
proven fix for the development-runner exits.

The opponent is `training/data/v44/v43_warmfixed_opponent`: V43 search/evaluation
with its import warm-up repaired. Historical compilation flags are not counted
as chess-strength wins. Games use paired colours, 10s + 0.1s, and the nine
development opening families. These are not the platform's undisclosed book.
Any 600-ply material adjudication from the local referee is recorded as a draw,
matching the current published documentation.

## Rules and packaging

The canonical markdown URLs were attempted but could not be fetched in this
session. The official rendered [documentation](https://aichessathon.com/docs)
and [rules](https://aichessathon.com/terms) were retrieved on 2026-09-11.
The documentation says 90s initialization, 120s + 0.5s, suspension during the
opponent's turn, and a draw at 600 plies. No harness files were edited here.

Only readable team-written `agent.py` and `core.py` belong in the archive.
No third-party engine, network, model weights, development runner, or match
data are shipped. The existing root agent and earlier submission archives
remain unchanged. No upload, commit or push has been performed.

`submissions/agent-v51-efficient-search-candidate.zip` contains exactly those
two files at the root: **23,367 bytes compressed, 86,238 bytes uncompressed**.
Archive integrity and exact equality to both source files were checked by the
builder. SHA-256:
`f336cdc57fdc8e443533e6678cb2de718bbf54c92317a5f1ac1fd028a25a67c5`.
This is an experimental archive, not a recommended replacement for V43.
