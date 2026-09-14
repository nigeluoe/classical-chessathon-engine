# V34: Classical Search Rebuild

V34 is a new search implementation on the team's tested V29 board substrate. The current
working source includes the score-preserving evaluation optimization described below; the older
`agent-v34-search-rebuild.zip` is a different source snapshot, not rebuilt automatically.
It is a candidate, not a claim of perfect play or a guaranteed competition winner.

## Files

- `agent.py`: public `get_move(fen, time_left_ms)` API, search, clocks, game history.
- `core.py`: bitboards, move generation, make/unmake, hashing, static evaluation, SEE.
- `dev/verify.py`: board and search regression checks; do not include in the upload.
- `dev/external_arena.py`: Stockfish/Lc0-only matches, real clocks, saved move review.
- `dev/reference_suite.py`: fixed-time decisions from sampled external games, rated by Stockfish.
- `dev/review_changes.py`: deeper, paired Stockfish review of every changed decision.
- `IMPROVEMENT_PLAN.md`: external testing, diagnosis, and acceptance process.
- `dev/arena.py`: historical internal comparator, not used for new strength decisions.
- `dev/build_zip.py`: builds and checks an archive containing only the two runtime modules.
- `dev/mypy.ini` and `dev/stubs/`: development-only typing for Numba's decorator/context API.

## How it chooses a move

The engine searches legal continuations with iterative deepening and principal-variation
alpha-beta search. Each completed depth supplies a legal fallback and the first move to try at
the next depth. Aspiration windows concentrate the next search around the previous score and
widen when that estimate is wrong. Root beta cutoffs stop work once a bound is proved.

Move generation uses 12 piece bitboards, precomputed knight/king attacks, and magic lookup
tables for sliding pieces. Hashes update incrementally when moves are made and unmade. The
same make/unmake functions handle castling, en passant, and all promotions. The hot search
loop runs in Numba; Python/chess objects are used at the API boundary.

Evaluation combines middlegame/endgame material and piece-square tables, game phase,
passed/isolated/doubled pawns, pawn shelter, king-file exposure, and bishop pairs. The values
and scoring semantics are unchanged from the external baseline. Precomputed pawn masks remove
the four temporary arrays and repeated file scans formerly allocated at every evaluation.
Independent square-based scoring tests verify the mask rewrite exactly.

A mobility/king-pressure/shelter trial was implemented, tested, and rejected because its
apparent shallow gain did not survive deeper Stockfish review. It is NOT in the current source.
The baseline's coarse king-shelter evaluation remains a limitation; the speed optimization
does not pretend to repair chess knowledge. There are no network weights or downloaded engines.

## Search details

- Per-ply move, score, and undo buffers are allocated once and reused throughout search.
- Move legality is checked when a move is searched. A separate early legal-move check protects
  mate/stalemate detection; quiescence does not validate every discarded quiet move.
- Quiescence searches captures, promotions, and every check evasion. SEE orders exchanges and
  skips losing nonchecking captures. Promotions and en passant bypass approximate SEE pruning.
- TT moves, captures, promotions, killers, and bounded history scores guide move ordering.
- The transposition table normalizes mate scores by ply and includes the halfmove clock in its
  key. Entries with repetition-sensitive history do not cut off directly.
- Narrow-window reverse futility is limited to depth three and excluded in check and mate
  windows. It is only applied after establishing that a legal move exists.
- Null-move pruning requires non-pawn material, a sufficient static score, a non-PV node,
  and no check. Consecutive null moves are blocked; artificial passes reset the repetition
  scan boundary.
- Late quiet moves may receive reduced-depth probes, with full-depth re-search on improvement.
  Check evasions, checking moves, and killers are protected from these reductions.
- Forward futility always searches one legal move and protects check evasions and PV nodes.
- Checkmate takes priority over the fifty-move rule. Stalemate and recognized insufficient
  material are scored as draws. Search has a maximum ply bound.

## Draws and time

The process remembers positions it has seen and the position after its own last move. It
matches the next FEN against legal opponent replies to reconstruct the observed game history.
Two earlier occurrences of a position are a draw; one earlier game occurrence alone is not.
A cycle inside a proposed search line is scored as a draw. Unobservable history before a
curated starting position cannot be recovered from a FEN. An unexpected new sequence clears
the tracked game history and TT depths.

Search has a soft target and a hard deadline calculated from the remaining clock. Stable best
moves can finish early. Deadlines leave overhead margin and do not have V33's abrupt reserve
switch. Forced single moves and tiny clocks return a legal fallback quickly. The last completed
iteration survives a timeout, and board state is restored while search unwinds.

Numba compilation is exercised at import, including iterative-deepening and UCI conversion
signatures. Warm-up table/history/killers are cleared before game one.

## Run the checks

From the repository root, with a Python containing numpy, numba, and python-chess:

```powershell
& .venv/Scripts/python.exe -u submissions/v34-search-rebuild/dev/verify.py
& .venv/Scripts/python.exe -u submissions/v34-search-rebuild/dev/external_arena.py `
  --opponent stockfish --games 2 --opening-offset 2 `
  --output training/data/v34/stockfish-fullclock-new
```

Use a new output directory for each run. Only Stockfish and Lc0 are external opponents; Noel's
bot and previous versions are not used for new strength decisions. Both sides get the actual
120s + 0.5s clocks, one CPU, and no strength/depth/node handicap. To test Lc0 on its CPU backend:

```powershell
& .venv/Scripts/python.exe -u submissions/v34-search-rebuild/dev/external_arena.py `
  --opponent lc0 --games 2 --opening-offset 2 `
  --output training/data/v34/lc0-fullclock-new
```

The runner uses the unchanged harness agent protocol, python-chess outcomes, a 90-second
initialization allowance, and a draw at 600 plies. These match the live docs checked on
2026-09-09: https://aichessathon.com/docs and https://aichessathon.com/terms. The requested
`.md` rules URLs were unavailable. CPU speed and opening selection remain local approximations;
the actual curated opening set is unpublished. The repository virtual environment has the
platform's NumPy/Numba/chess versions, but local Windows/Python 3.13 differs from the platform's
Linux/Python 3.12. Upload validation remains necessary.

The output folder contains `progress.json` (live phase, move, and clocks), `summary.json`
(completed games and settings), PGNs, stderr diagnostics, hashes, and `decisions.jsonl`.
Stockfish reviews decisions after the game, separately from either player's clock. The review
is depth/time limited; it is not infallible ground truth. Large errors are rechecked more deeply
before being used to justify a change. No external engine or review data goes into the ZIP.

## Verification so far

Five perft stress positions passed exactly, including castling/en-passant/promotion cases.
1,494 randomized positions matched python-chess legal moves and exact make/unmake/hash
restoration. Search regressions cover mate/stalemate, mate-distance TT storage, false-mate
futility, repetition/null boundaries, low-clock budgets, legal moves, and timeout unwinding.
The same 1,494 positions also verify pawn-mask evaluation against an independent, readable
square/file-loop implementation. Colour-mirrored total evaluations are checked.
External results and limitations are recorded in `RESULTS.md`.

## Submission contents

The upload needs exactly `agent.py` and `core.py` at the ZIP root. Development scripts,
logs, pycache files, and previous models are excluded. This build has no network, external
processes, native binaries, GPU dependency, or third-party engine implementation.
