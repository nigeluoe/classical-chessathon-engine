# Engine evolution and experiment record

This repository retains the source of the full AI Chessathon development lineage, not merely the final classical release. The version history is intentionally candid: a candidate was promoted only when the available validation supported it; a useful idea that failed its tests remains visible as a rejected experiment.

The competition project finished in the top 100 of approximately 450 teams. It was created for AI Chessathon, hosted by Optiver.

## How to read this record

- **Promoted** means the change cleared the team's available correctness and head-to-head gates at the time. It does not claim a universal Elo gain.
- **Wash** means the result was too close to distinguish from noise or did not meet the promotion threshold.
- **Rejected** means a measured regression, reliability issue, or unqualified implementation stopped the branch.
- Version numbers describe the real development chronology, so some numbers are absent and two V42 directories remain: they record a naming collision between independent experiments rather than being silently rewritten.
- All results are local experimental evidence. Different versions were tested with different opponents, openings, clock budgets and hardware load; scores should not be compared across rows as a single rating list.

All iteration source is under [`submissions/`](../submissions/). Generated corpora, trained weights, tablebases, external engines, PGNs and upload archives are excluded. This keeps the public repository reviewable and avoids presenting a model-dependent snapshot as runnable when its experimental asset is absent.

## Chronological summary

| Version | Hypothesis and engineering change | Recorded outcome | Status |
| --- | --- | --- | --- |
| [V1](../submissions/v1-original) | Readable alpha-beta baseline with material and piece-square evaluation. | Original submitted baseline. | Historical baseline |
| [V2](../submissions/v2-pvs-baseline) | Added PVS, transposition table, killer/history ordering, quiescence and safer time allocation. | Beat V1 in the early local comparison. | Foundation |
| [V3](../submissions/v3-candidate) | First king-relative residual NNUE, rebuilding NumPy features at each leaf. | Lost the first clean V3-vs-V2 pair, `0–2`, after an inference repair. | Rejected: leaf cost dominated |
| [V4](../submissions/v4-incremental-candidate) | Incremental NNUE accumulators for normal moves and special moves. | 7,975 accumulator checks passed, but depth fell from 3 to 2; 25% in the fast V4-vs-V2 screen with two flags. | Rejected: Python state copying cost too much |
| [V5 search](../submissions/v5-search-candidate) | Tested null-move pruning, LMR, aspiration and futility together with pawn structure. | Did not convincingly beat V2. | Rejected combined branch |
| [V5 pruning](../submissions/v5-pruning-candidate) | Isolated the search-pruning changes from pawn evaluation. | `+9 =6 -5`, 60.0% vs V2 over 20 games. | Promising, superseded |
| [V6](../submissions/v6-pruning-candidate) | Used a TT entry before null-move work and delayed futility evaluation. | `+6 =13 -1`, 62.5% vs V2; lint, type and smoke checks passed. | Promoted |
| [V7](../submissions/v7-draw-aware-candidate) | Root-level threefold handling plus a safe quiescence timeout unwind. | Aggressive draw-decline policy: 42.5% vs V6. Conservative follow-up: 55.0%, still a wash. | Rejected / documented policy correction |
| [V8](../submissions/v8-ponder-candidate) | Background pondering. | 47.5% vs V6. Later harness review showed the local runner could not model opponent-clock pondering fully. | Inconclusive |
| [V9](../submissions/v9-tablebase-candidate) | Small Syzygy endgame-tablebase experiment. | 47.5% vs V6. | Wash |
| V10 | Offline NNUE/training exploration; no standalone submission directory was retained. | Informed later network experiments. | Research-only |
| [V11](../submissions/v11-tapered-eval-candidate) | Tapered PeSTO-style midgame/endgame material and piece-square evaluation. | `+12 =7 -1`, 77.5% vs V6; evaluation was also faster in the measured microbenchmark. | Promoted |
| [V12](../submissions/v12-tapered-draw-aware-candidate) | Re-applied the conservative V7 repetition policy to V11. | `+7 =6 -7`, 50.0% vs V11. | Wash |
| [V13](../submissions/v13-mobility-candidate) | Added mobility by generating legal moves in the evaluator. | Evaluation became about 4x slower; `+1 =2 -17`, 10.0% vs V11. | Rejected |
| [V14](../submissions/v14-pawn-structure-candidate) | Added doubled, isolated and passed-pawn terms. | Evaluation about 2.17x slower; `+0 =9 -11`, 22.5% vs V11. | Rejected |
| [V15](../submissions/v15-king-safety-candidate) | Pawn shelter and king-adjacent open-file terms. | `+8 =2 -10`, 45.0% vs V11. | Not promoted |
| [V16](../submissions/v16-rook-open-file-candidate) | Added rook open/semi-open file bonuses. | `+6 =3 -11`, 37.5% vs V11. | Rejected |
| [V17](../submissions/v17-numba-eval-candidate) | Jitted the existing tapered evaluator without adding chess terms. | `+11 =6 -3`, 70.0% vs V11. | Promoted |
| [V18](../submissions/v18-numba-king-safety-candidate) | Retried king safety on the faster V17 substrate. | `+9 =4 -7`, 55.0% vs V17. | Wash |
| [V19](../submissions/v19-numba-king-rook-candidate) | Combined king safety and rook-file activity. | `+0 =5 -15`, 12.5% vs V17. | Rejected |
| [V20](../submissions/v20-numba-rich-eval-candidate) | Moved pawn structure and king safety into the same jitted evaluator pass. | Became the operational classical reference for the next rebuild. | Promoted foundation |
| [V21](../submissions/v21-check-extension-candidate) | First bounded check-extension exploration on the V17 branch. | Superseded before an independent strength result was recorded. | Research snapshot |
| [V22](../submissions/v22-check-extension-candidate) | Rebased the bounded check extension on V20. | Preserved for comparison; not promoted independently. | Research snapshot |
| [V23](../submissions/v23-jitted-mobility-candidate) | Retried mobility inside compiled code. | A mixed result exposed an emergency-fallback issue rather than a clear mobility gain. | Diagnostic branch |
| [V24](../submissions/v24-safe-fallback-candidate) | Repaired the depth-zero emergency move fallback. | Removed a reproducible arbitrary-legal-move failure mode. | Reliability fix |
| [V25](../submissions/v25-jitted-mobility-v24-candidate) | Retested compiled mobility after the fallback repair. | Did not provide a clear promotion result. | Not promoted |
| [V26](../submissions/v26-bitboard-core-candidate) | Rebuilt the hot path around 12 bitboards and Numba move generation, make/unmake and search. | Perft/cross-check gates passed; roughly 38–45x search-node-rate improvement; `+16 =4 -0`, 90.0% vs V20. | Promoted |
| [V27](../submissions/v27-bitboard-pruning-candidate) | Ported full pruning and move ordering onto V26. | `+10 =10 -1`, 71.4% vs V26. | Promoted |
| [V28](../submissions/v28-see-candidate) | Added Static Exchange Evaluation to quiescence and capture ordering. | 148,086 sampled move checks with zero mismatches; `+10 =7 -4`, 64.3% vs V27. | Promoted |
| [V29](../submissions/v29-magic-bitboards-candidate) | Replaced sliding-ray scans with validated magic-bitboard lookups. | `+13 =13 -7`, 59.1% vs V28 over 33 games. | Promoted |
| [V30](../submissions/v30-rfp-lmp-candidate) | Explored reverse futility and late-move pruning on the bitboard line. | Valuable design continuation, later revisited in V33/V40/V42. | Research snapshot |
| [V31](../submissions/v31-nnue-candidate) | Revisited incremental NNUE with a smaller feature table and corrected update path. | Accumulator correctness and speed blockers improved, but the trained network lost decisively to the classical V29 baseline. | Rejected |
| [V32](../submissions/v32-large-nnue-candidate) | Larger NNUE design with incremental accumulators and safe classical fallback. | Source retained; a qualifying model was not packaged. | Source-only research snapshot |
| [V33](../submissions/v33-selective-pruning-candidate) | Conservative selective pruning and revised clock use on V29. | Required a clear colour-balanced win before promotion; not established. | Experimental |
| [V34](../submissions/v34-search-rebuild) | Independently rewrote the V29 search to reduce allocation and bookkeeping cost. | Strong on a varied-opening screen but mixed overall; external-engine tests remained exploratory. | Experimental benchmark |
| [V35](../submissions/v35-search-optimized-candidate) | Optimised V34 search and added legal-recapture SEE. | Continued the allocation-light search line. | Experimental continuation |
| V36–V39 | No persisted submission directories. | Labels were reserved during parallel exploration. | No public source snapshot |
| [V40](../submissions/v40-search-plus-candidate) | First RFP/search-plus branch. | Later collided in name with a parallel branch; retained unchanged as a historical snapshot. | Historical snapshot |
| [V41](../submissions/v41-conversion-candidate) | Referee-aligned threefold handling and extra time for low-material conversion. | `+10 =5 -0`, 83.3% vs V40 in the recorded internal screen. | Locally positive |
| [V42 balanced](../submissions/v42-balanced-critical-candidate) | Symmetric critical-position time allocation. | `+7 =3 -5`, 56.7% vs V41; positive but not conclusive. | Experimental |
| [V42 search-plus](../submissions/v42-search-plus-candidate) | Separate V29-based RFP/futility/allocation branch using the duplicated version label. | Retained exactly to make the naming collision auditable. | Historical snapshot |
| [V43](../submissions/v43-classical-pvs-candidate) | Independently rebuilt classical PVS on the verified bitboard substrate. | `+9 =12 -1`, 68.2% vs V41 across paired opening types. | Promoted |
| [V44](../submissions/v44-volatility-pvs-candidate) | Hardened the Numba warm-up so recursive search compiled before timed play. | No chess-strength claim: two added-evaluation experiments were rejected. | Primary reproducible release |
| [V45](../submissions/v45-tactical-resilience-candidate) | Protected quiet queen/rook attacks from shallow pruning and added bounded tactical extensions. | Solved a targeted replay; smoke result was not enough for a strength claim. | Experimental |
| [V46](../submissions/v46-nnue-blend-candidate) | Blended a small, team-trained NNUE signal into the classical score. | Model asset is deliberately excluded; this is source-only research. | Measurement candidate |
| [V47](../submissions/v47-ponder-conversion-candidate) | Intended ponder/conversion continuation. | Source is byte-identical to V45 and the branch was not completed. | Abandoned snapshot |
| [V48](../submissions/v48-clock-utilization-candidate) | Enlarged the TT and removed an overly low per-move ceiling for long games. | Safe mechanical change; limited real-clock screen was positive but too small for a strength claim. | Recommended competition-era release |
| [V48.1](../submissions/v48.1-king-danger-candidate) | Added a king-danger term for sacrificial-attack blind spots. | Targeted improvement did not generalise; rejected. | Rejected |
| [V49](../submissions/v49-reserve-schedule-candidate) | Reserved more clock for very long unresolved games. | Plausible but not sufficiently validated before the deadline. | Experimental |
| [V50](../submissions/v50-history-search-candidate) | Optimised tactical generation, static caching and en-passant handling. | `+1 =4 -1`, 50.0% vs warm-up-fixed V43. | Not promoted |
| [V51](../submissions/v51-efficient-search-candidate) | Refined the V50 efficient search path. | Faster locally, but `+4 =8 -6`, 44.4% vs V43. | Not promoted |
| [V52](../submissions/v52-activity-safety-candidate) | Added activity, king-pressure and bounded threat-search ideas. | `+1 =1 -6`, 18.75% on fresh-opening testing. | Rejected |
| [V53](../submissions/v53-search-integrity-candidate) | Corrected transposition-depth semantics while returning to a stable classical evaluator. | Source preserved; no completed strength screen was recorded. | Final research snapshot |

## Reusable technical findings

The lineage produced a number of practical findings that matter beyond a single engine build:

1. **Speed and evaluation quality cannot be separated.** The V13–V16 experiments showed that an evaluator with more chess knowledge may weaken the engine when its cost removes search depth. V17 and V20 demonstrated that moving work into the compiled path can change that trade-off.
2. **Representation was the largest leverage point.** Replacing object-heavy board operations with a Numba bitboard core in V26 made a greater measured difference than a series of small evaluation additions.
3. **Correctness gates enable aggressive optimisation.** Perft, full state restoration, incremental hash checks, legal-move cross-checks and a reference SEE test made it possible to optimise the hot path without trusting silent assumptions.
4. **Small match samples are not a rating system.** Repeated matches exposed timing noise, opening sensitivity and the weakness of self-lineage-only benchmarks. The result record therefore uses promotion language cautiously.
5. **NNUE was investigated rather than ignored.** Multiple NNUE designs were built and validated for incremental-update correctness. They were not adopted for the released classical line because the available runtime/training trade-off did not clear the engine's empirical gates.

## Source and reproducibility notes

Most classical snapshots can be run through the local harness by selecting their directory as the agent. Some historical directories require an intentionally excluded asset:

- V9 references local Syzygy tablebases.
- V3, V4, V31, V32 and V46 contain NNUE research code; their local model weights are not published here.
- Earlier Python-chess versions and later bitboard candidates have different dependency and compilation characteristics; use the version's own source notes before treating one benchmark command as universal.

For a small, reproducible classical starting point, use V44. For the most complete source history, browse the individual version directories alongside this document.
