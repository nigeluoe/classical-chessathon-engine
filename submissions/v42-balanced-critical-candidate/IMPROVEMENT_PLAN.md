# External-opponent improvement loop

Only Stockfish and Lc0 are opponents or reference engines for this work. Earlier-version
match scores are historical data; they are not the acceptance test for new improvements.
Noel's bot is excluded at the user's request. No process, network, weights, or move lookup
from either reference engine is included in a submission.

## 1. Establish honest external baselines

Run colour-paired games with both sides using the actual 120-second clock and 0.5-second
increment. The UCI opponent receives both real clocks and the full move history. Use one
thread and pin each playing process to one logical CPU. Disable strength reduction in
Stockfish (Skill Level 20, UCI_LimitStrength false). Use Lc0's CPU BLAS backend and record the
network fingerprint. There are no opponent node, depth, or fixed-per-move limits in matches.

Save source hashes, engine/network hashes, engine identities, opening moves, PGNs, every
move's clock use, and engine diagnostics. Resolve initialization/illegal-move/timeout faults
before interpreting chess strength. Games at the ply cap are drawn, not scored by material.

## 2. Find expensive mistakes

After a completed game, ask a separate Stockfish process for its best move and for the score
when restricted to our chosen move at the same root. Use the same depth/time limits in both
queries, clear its hash between queries, and record actual attained depths. Preserve negative
score deltas as analysis uncertainty. Store mate distances separately from centipawn loss.
Different moves alone are not errors: prioritize substantial score loss or missed/allowed mates.

Group the expensive positions by concrete cause: unsearched tactics, inaccurate exchanges,
king exposure, piece activity, pawn structure, or failure to convert a won ending. Preserve a
small regression set from those positions and use different openings for later validation.

## 3. Make measured changes

Repair a demonstrated correctness defect first. For evaluation weaknesses, add compact
bitboard features that can be computed without full legal move generation. For search
weaknesses, inspect the saved depth/node data and profile the hot paths. Compare a change
with its baseline against the external reference on the same diagnostic positions; do not
use candidate-versus-candidate games as proof of improvement.

Proposed features or search rules remain candidates until their runtime cost and move quality
are tested. A larger list of heuristics or a deeper nominal search is not itself a win.

## 4. Validate out of sample

Re-run correctness tests, strict typing with the Numba API stub, and lint. Test new openings
with reversed colours against both external opponents, using the same clocks, packages,
CPU settings, engine binaries, and Lc0 net. Record every loss and draw; do not cherry-pick.
When Stockfish wins all games, its move-quality analysis remains useful, but a zero match
score is still a zero match score. Small batches do not establish a reliable Elo estimate.

## 5. Package the tested candidate

Build only the runtime source modules into a ZIP, verify its contents, and document exactly
which results belong to its source hash. Keep any unresolved conversion or tactical weakness
visible. The current public contract is at https://aichessathon.com/docs; local hardware and
the unpublished tournament openings prevent an exact reproduction of the competition.

## Commands

```powershell
& .venv/Scripts/python.exe -u submissions/v34-search-rebuild/dev/external_arena.py `
  --opponent stockfish --games 2 --opening-offset 2 `
  --output training/data/v34/stockfish-fullclock-new

& .venv/Scripts/python.exe -u submissions/v34-search-rebuild/dev/external_arena.py `
  --opponent lc0 --games 2 --opening-offset 2 `
  --output training/data/v34/lc0-fullclock-new
```

Each run needs a fresh output directory. `summary.json`, `decisions.jsonl`, PGNs, and stderr
logs live in that directory. Post-game reference analysis defaults to depth 16 with a 250 ms
per-query cap; it is explicitly separate from the unhandicapped match time control.
