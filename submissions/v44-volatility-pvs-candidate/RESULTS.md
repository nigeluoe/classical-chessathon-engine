# V44 reliability candidate

V44 is the frozen V43 PVS/search/evaluation implementation with one production
fix: its import warm-up permits the recursive search kernel to finish compiling
inside the published 90-second initialization allowance.  The old `12.0`-second
hard deadline could expire during lower-level compilation, leaving the recursive
kernel to compile on the first clocked move.

The submitted source uses a bounded depth-two warm-up with an 80-second hard
deadline, then clears warm-up state.  Search behaviour after initialization is
otherwise byte-for-byte V43, apart from the version diagnostic.

## Strength experiments rejected

All comparisons used `training/data/v44/v43_warmfixed_opponent`, a V43 copy
whose only change is the safe warm-up.  That prevents V43's historical
first-move compilation flag from being counted as chess strength.

- Pawn pressure / connected-passer terms: `+1 =2 -1`, 50.0% over four paired
  varied-opening games. Rejected.
- Quiet checking moves at principal-variation quiescence leaves: `+0 =3 -1`,
  37.5% over the same screen. Rejected.

Therefore V44 is a reliability repair, **not** a measured playing-strength
promotion over V43.  The strength target requires a new, independently
validated evaluator or search breakthrough; the rejected changes do not justify
an upload claim.

## Packaging and checks

`agent-v44-reliability-pvs.zip` contains only `agent.py` and `core.py`:
22,357 bytes compressed and 78,602 bytes uncompressed.  Ruff and strict mypy
both pass.  The functional `core.py` is unchanged from V43, retaining its
previous perft, legal-move, make/unmake, hash, and SEE gates.
