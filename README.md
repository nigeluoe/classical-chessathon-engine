# Classical Chess Engine for AI Chessathon

A high-performance chess engine built for **AI Chessathon**, hosted by **Optiver**. Our team finished in the **top 100 of approximately 450 teams**.

This repository documents a deliberately classical-engine approach. Modern top engines commonly rely on efficiently updatable neural-network evaluation (NNUE) or other neural methods. We set out to test a different proposition: how competitive can a carefully engineered, fully interpretable alpha-beta engine be when it has no neural evaluator at runtime?

The result is a compact Python engine that combines bitboards, magic-bitboard attack lookups, a handcrafted tapered evaluator, and an aggressively optimised principal-variation search. This repository preserves the complete source lineage, from the original alpha-beta baseline through 53 numbered experiments. V44 is the primary runnable classical reference because its release record is complete and its warm-up path was hardened for the competition environment.

## Project outcome

- **Competition:** AI Chessathon, hosted by Optiver
- **Result:** top 100 among roughly 450 teams
- **Engine approach:** classical search and handcrafted evaluation; no NNUE or neural-network inference at runtime
- **Primary question:** whether disciplined systems engineering can make a non-neural engine competitive under a constrained, single-core environment

The project was as much an engineering exercise as a chess one. Strength improvements were only retained when they were accompanied by legality checks, timed execution, and head-to-head testing. Promising changes that did not survive those gates were recorded and rejected rather than folded into the final candidate.

## What is included

`submissions/` contains the complete, chronological experiment record. Each directory preserves the source available for that iteration; models, tablebases, generated data and upload archives are deliberately excluded. See [the evolution record](docs/EVOLUTION.md) before comparing versions: some experiments are runnable releases, while NNUE and tablebase experiments are source-preserved research snapshots whose local assets are intentionally absent.

`submissions/v44-volatility-pvs-candidate/` is the primary runnable classical release:

```text
submissions/v44-volatility-pvs-candidate/
├── agent.py     # Search, evaluation adjustments, clock management, entry point
├── core.py      # Bitboards, legal move generation, magic attacks, hashing and SEE
└── RESULTS.md   # Release rationale, checks and experiment record
```

The engine exposes the competition interface:

```python
def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation."""
```

`harness/` contains the local game runner and referee used to exercise agents before release. `baselines/` contains lightweight comparison opponents. Large training corpora, caches, model artefacts and packaged uploads are intentionally excluded from the public release; every iteration's source is retained.

## Engine design

### Board representation and legality

The hot path uses twelve piece bitboards rather than Python chess objects. It provides:

- precomputed pawn, knight and king attack tables;
- magic-bitboard lookup tables for bishop, rook and queen attacks;
- pseudo-legal generation followed by king-safety validation;
- make/unmake with en-passant, castling and promotion handling; and
- an incremental Zobrist position hash.

The representation was verified separately from search. Perft positions, move-for-move cross-checks against `python-chess`, make/unmake round trips and incremental-hash checks were used to catch correctness regressions before any strength testing.

### Search

Move selection is an iterative-deepening principal-variation search (PVS) with alpha-beta pruning. The main ingredients are:

- aspiration windows around the previous iteration's score;
- a transposition table with depth- and bound-aware entries;
- capture-oriented quiescence search;
- null-move pruning, late-move reductions and guarded shallow futility pruning;
- transposition-table, killer-move and history-heuristic move ordering; and
- explicit soft and hard clock budgets with a legal-move fallback.

Selective search is useful only if it remains safe under time pressure. The implementation therefore checks a deadline from inside compiled search code and always retains a previously completed legal root move.

### Evaluation

The evaluator is deliberately transparent. It blends middle-game and end-game scores according to remaining material, with terms for:

- material and piece-square tables;
- doubled, isolated and passed pawns;
- pawn shelter and king-adjacent open files;
- bishop-pair value;
- development-sensitive adjustments in queenful positions; and
- a modest side-to-move initiative.

This is not an attempt to claim that a handcrafted evaluation supersedes NNUE. Rather, it makes every score component inspectable, fast enough to search deeply, and suitable for analysing why a particular change helped or failed.

## Reliability work in V44

V44 retains V43's search, evaluator and move-generation substrate. Its release change was operational: the compilation warm-up was extended so the recursive search kernel reliably finished compiling before the first timed move. This avoided a failure mode where compilation could consume a player's clock on the opening move.

The release was deliberately conservative. Two potential strength changes were screened and declined:

| Experiment | Paired-game result | Decision |
| --- | ---: | --- |
| Pawn-pressure and connected-passer terms | `+1 =2 -1` (50.0%) | Not promoted |
| Quiet checks at PVS quiescence leaves | `+0 =3 -1` (37.5%) | Not promoted |

The V44 archive contains only `agent.py` and `core.py` (78.6 KB uncompressed). Ruff and strict mypy checks passed, and the underlying board substrate had already passed perft, legal-move, make/unmake, hashing and static-exchange-evaluation gates. See [the release record](submissions/v44-volatility-pvs-candidate/RESULTS.md) for the precise scope and results.

## Lessons from the experiment

The most useful conclusion was not that handcrafted evaluation "beats" neural evaluation. NNUE remains a dominant practical approach in modern chess-engine evaluation because it delivers a strong positional signal at low incremental cost. Our result was narrower and more useful: under a fixed compute budget, a classical engine can still become meaningfully competitive when representation, move generation, search, validation and time management are treated as one system.

Several recurring lessons shaped the project:

1. **Search depth is a feature.** An evaluation term that appears sensible can be harmful if it slows the hottest path enough to lose a ply of search.
2. **Correctness precedes Elo.** Fast sliding attacks, incremental state and pruning all need independent checks; a rare legal-move bug invalidates every strength claim.
3. **Ablation beats intuition.** Changes were measured against a warm-up-safe baseline, so a compilation or harness effect was not mistaken for chess strength.
4. **Operational details matter.** Compilation, memory allocation and clock reserve policies can decide a game as surely as a positional term.

## Running a local game

The project uses Python 3.12 and `uv` for local development. The original competition environment provided compatible versions of `python-chess`, NumPy and Numba.

```bash
uv sync
uv run python -m harness.play \
  --white submissions/v44-volatility-pvs-candidate \
  --black baselines/greedy
```

To run a small alternating-colour screen:

```bash
uv run python -m harness.arena \
  --agent submissions/v44-volatility-pvs-candidate \
  --opponent baselines/greedy \
  --games 20
```

The first import intentionally spends time compiling Numba kernels. That work is part of the deployment design, not a benchmark anomaly: it keeps compilation out of the move clock.

## Scope and attribution

This is a competition/research artefact, not a general-purpose chess engine or a claim of parity with Stockfish, Leela Chess Zero, or NNUE-based engines. No external engine binary, external search result or neural model is required at runtime by the published V44 candidate.

The project began from the [AI Chessathon starter](https://github.com/advitrocks9/aichessathon-starter); its original MIT licence is retained. The engine work and research record in this repository were developed as part of the AI Chessathon effort.
