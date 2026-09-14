# V43 classical PVS candidate

V43 retains only the team's perft-tested bitboard substrate and replaces the move-selection
layer with a new PVS engine. It uses iterative deepening, aspiration windows, a 2M-entry
transposition table, quiescence search, null-move pruning, late-move reductions, shallow
futility pruning, killer/history ordering, and bounded clock management. It does not ponder;
the frozen qualifier rules suspend the process while the opponent thinks.

The first four-game build scored `+2 =2 -0` against V41. Auditing its two subsequent losses
found that the inherited evaluation treated a far-advanced pawn as full king shelter and
underpriced returning a minor to its starting square. Small symmetric development and
distance-weighted shelter terms changed the exact Sicilian loss from `...h4` to `...Nf5` at
the arena budget.

Post-change results against frozen V41 at 10s + 0.1s:

- Controlled Sicilian/French rerun: `+2 =2 -0`, 75.0%.
- Full five-opening paired run: `+4 =6 -0`, 70.0%.
- Four additional paired opening/FEN types: `+3 =4 -1`, 62.5%.
- Combined: `+9 =12 -1`, 68.2%; decisive games 9-1.

The only post-change loss began from the deliberately imbalanced queen-versus-pieces FEN,
which both engines evaluated at roughly +7 pawns for White; V43 won the color-reversed game.
There were no crashes, illegal moves, init failures, or flags. Evidence is under
`training/data/v43/`.
