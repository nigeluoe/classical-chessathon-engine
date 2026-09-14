# V32 Large NNUE Candidate

V32 combines the V29 bitboard engine and search heuristics with a team-trained,
Stockfish-inspired NNUE. It is not packageable until a trained model has passed
the gates below and has been installed as `weights/nnue_v32_int.npz`.

## Preserved From V29

- Bitboard state, magic-bitboard slider attacks, legal move generation, and
  incremental Zobrist hashing.
- Iterative deepening alpha-beta/PVS, transposition table, quiescence, SEE,
  null-move pruning, late-move reductions, futility pruning, aspiration
  windows, and killer/history move ordering.
- V29's tapered handcrafted evaluator remains the safe fallback when weights
  are missing or fail shape validation.

## V32 Evaluator

- 32 mirrored king buckets x 10 relative non-king piece categories x 64
  oriented squares: 20,480 sparse feature rows.
- Two 512-value accumulators, updated incrementally after every move and
  restored on unmake. King moves refresh only the affected perspective.
- Int16 feature-transformer weights accumulated in int32. Dense int8 weights
  are dequantised once at import; the `1024 -> 32 -> 32 -> 1` head runs in
  Numba at search leaves.
- If `weights/nnue_v32_int.npz` is absent, NNUE is disabled rather than using
  random weights. This keeps the scaffold behaviour equivalent to V29.

## Model Handoff

```powershell
& 'C:\Joe\Anaconda\python.exe' training/install_v32_weights.py `
  --model training/models/v32_nnue_int.npz
```

Then run, in order:

```powershell
uv run python training/test_v32_nnue_incremental.py
uv run python training/test_v32_nnue_weights_parity.py
uv run python -m harness.arena --agent submissions/v32-large-nnue-candidate `
  --opponent submissions/v29-magic-bitboards-candidate --games 40
```

Only package V32 if the accumulator and forward-parity gates pass, import time
has margin below the platform limit, and the 40-game V32-versus-V29 result is
a clear win rather than a wash.
