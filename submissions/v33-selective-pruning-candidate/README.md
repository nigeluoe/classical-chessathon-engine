# V33: Selective Pruning Candidate

V33 starts from V29, the strongest validated classical candidate. It intentionally does not
use NNUE or any external engine at runtime.

## What changed

- **More useful clock use:** V29 never searches longer than three seconds. V33 uses the actual
  remaining competition clock, keeps a ten-second reserve, and caps any move at 4.25 seconds.
  At a short local test clock it intentionally falls back to V29's tested allocation instead of
  treating the whole clock as reserve.
- **Reverse futility pruning:** at narrow non-principal-variation nodes, an evaluation already
  safely above beta can cut off. This releases nodes for the tactical line the engine believes
  is best.
- **Late-move pruning:** after several quiet alternatives fail at a narrow node, later quiet
  alternatives are skipped. Captures, promotions, checks, check evasions, and principal
  variation nodes are never skipped by this rule.

The changes are deliberately bounded. A strong result against V29 must be demonstrated through
colour-balanced games before this candidate is promoted or submitted; no chess engine can
honestly guarantee a win in every game against another strong deterministic engine.

## Required validation

Run legal-move/perft tests first, then use the local multi-opening arena and alternate colours.
The real platform's curated positions are unpublished, so local results are evidence rather
than a reproduction of rated play.
