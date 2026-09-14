# V46 NNUE-blend candidate

V46 is deliberately a measurement candidate, not a claimed replacement for
V43/V45. It keeps V45's PVS, tactical-resilience changes, and handcrafted
evaluation, then adds 12.5% of a compact team-trained NNUE score at every
static-evaluation point.

## Model

- `v45_nnue_256_finetune_int.npz` is our compact two-accumulator model trained
  on the completed stratified Lichess corpus.
- Its best held-out result was 138.1cp MAE (980,000 positions), at fine-tune
  epoch four.
- The submission ships the 9.8MB quantised model plus readable source; its
  uncompressed size is comfortably below the platform limit.

## Runtime design

The search does **not** encode the board or sum 30 features at every leaf.
It owns two `int32[256]` accumulators. Before each make, their old values are
saved in a per-ply stack; only moved, captured, promoted, or castling-rook
features are adjusted. A king move rebuilds only that king's perspective,
because it changes its king-relative feature IDs. Unmake restores the saved
accumulators exactly.

## Required gates

1. `dev/test_nnue_incremental.py` must pass, checking every legal move from
   normal, castling, promotion, and en-passant positions against a clean
   rebuild.
2. V46 must initialise within the platform budget and retain V45's completed
   search depth/nodes-per-second on the fixed position suite.
3. It must retain the V45 sparse-king tactical regression.
4. Only then run a 20-game paired varied-opening match against warm-fixed V43.

Failure of any speed or match gate rejects V46. The NNUE is an experiment to
test whether its positional signal survives real alpha-beta search; validation
MAE alone is not evidence of chess strength.
