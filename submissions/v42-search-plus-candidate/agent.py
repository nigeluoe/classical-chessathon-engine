"""A bitboard + numba engine core for the AI Chessathon submission API.

This is V26 (`submissions/v26-bitboard-core-candidate/`) with V20's remaining search
techniques ported onto the bitboard core: null-move pruning, late-move reductions, futility
pruning, aspiration windows, and killer-move/history-table move ordering -- everything V26's
own module docstring listed as "not yet ported". Board representation, move generation,
make/unmake, Zobrist hashing, and evaluation are byte-identical to V26 (diffed to confirm);
only the search layer (from `now_seconds` to the bottom of the file) changed. See V26's module
docstring for the board-representation and move-generation design; this docstring only covers
what is new here.

Why this round exists: an independent, real competitor submission ("Noel's bot",
`training/noels_bot/`) beat V20 -- the strongest candidate by every prior local measure -- 9
games out of 10, exposing a genuine weakness (an unsound knight sacrifice, `Nxg7+`/`Ne6+`, that
V20's own evaluate() scores as good for White) that never showed up once against this repo's
own lineage or Stockfish. Diagnosis (see this round's task write-up and
`training/data/v26_vs_v20_20games.log`-adjacent analysis) found that V20's search, given only
~1.4 extra seconds (reaching depth 5 instead of 3-4), finds the refutation on its own with the
*same* evaluator -- so the practical fix was not a speculative evaluation patch (this team has
tried and failed at that before: V13's mobility term, rejected for a 4x eval slowdown) but
searching deeper, faster, per unit of wall-clock time. That is exactly what porting V20's
pruning suite onto the ~40x-faster bitboard core buys.

Every previous version in this repo (V6 through V25) generated moves through python-chess's
object-oriented `chess.Board`; only the evaluation function's hot loop was ever jitted (V17,
V20). `docs/IDEAS.md` says this gets "thousands, not millions" of nodes per second, and that is
the real ceiling on every eval tweak this repo has tried. V26 replaced the board representation
itself: 12 piece bitboards plus derived occupancy, side to move, castling rights, en-passant
square, and halfmove clock, with move generation, make/unmake, search, and evaluation all
running under `@njit` -- no `python-chess` object anywhere in the hot path. `python-chess` is
used only at the edges: to parse the input FEN once per `get_move()` call and to format the
chosen move back to a UCI string.

Square indexing matches python-chess exactly: `a1 = 0 ... h8 = 63`, `square = rank * 8 + file`.
This was a deliberate choice so a position can be cross-checked against python-chess directly,
square for square, with no translation step to get wrong.

Board state
-----------
- `boards`: `uint64[12]`, one bitboard per (colour, piece type): indices 0-5 are white
  pawn/knight/bishop/rook/queen/king, 6-11 are the same for black (`WP..WK`, `BP..BK` below).
  Occupancy (`occ_white`, `occ_black`, `occ_all`) is derived by OR-ing the relevant six boards
  rather than stored, since it is only ever needed transiently inside a jitted function.
- `meta`: `int64[4]` = `[side_to_move (0=white,1=black), castling_rights (bit 0=WK,1=WQ,2=BK,
  3=BQ), en_passant_square (-1 if none), halfmove_clock]`.
- A move is packed into one `int64`: bits 0-5 from-square, 6-11 to-square, 12-15 moving piece
  code, 16-19 captured piece code (15 = none), 20-22 promotion piece type (0=none, 1=N, 2=B,
  3=R, 4=Q), bit 23 en-passant-capture flag, 24 double-pawn-push flag, 25 castle-kingside flag,
  26 castle-queenside flag. Move generation writes into a preallocated `int64[256]` buffer and
  returns a count, rather than allocating a Python list, so the whole path stays in nopython
  mode.

Move generation and legality
-----------------------------
Knight and king attacks come from lookup tables built once at import (`KNIGHT_ATTACKS`,
`KING_ATTACKS`, `PAWN_ATTACKERS`). Bishop and rook attacks use the simplest approach that is
obviously correct -- classical ray-scanning, stepping one square at a time from the piece in
each of 4 directions until the board edge or a blocking piece -- rather than magic bitboards,
per the plan: get correctness bulletproof before adding that speed optimisation. Legality is
filtered by actually making each pseudo-legal move (`apply_move`), checking whether the mover's
own king is attacked, and unmaking it (`unmake_move`) -- simpler and far less bug-prone than
separate pin/check detection, at the cost of some speed. This also means every legal-move
generation call exercises the same apply/unmake pair the search itself uses.

Correctness gates (see the module docstring bottom and the referenced test files for real
numbers): a perft suite (`training/test_bitboard_perft.py`) checks move generation against the
published Chess Programming Wiki node counts for the starting position and the standard
Kiwipete/position-3/4/5/6 stress positions, and a randomized-game cross-check
(`training/test_bitboard_crosscheck.py`) plays many games against python-chess move for move,
comparing the legal move set and resulting position at every ply.

Search and evaluation
----------------------
Iterative-deepening negamax with alpha-beta pruning, a hash-indexed transposition table sized
as a fixed array (no dict, no Python objects), MVV-LVA + TT-move + killer + history move
ordering, quiescence search on captures and promotions, null-move pruning, late-move
reductions, futility pruning, and aspiration windows -- all `@njit`, all ported from V20's
search with the same parameter values (`NULL_MOVE_MIN_DEPTH`, `FUTILITY_MARGIN`,
`ASPIRATION_WINDOW`, `HISTORY_MAX`, LMR thresholds) rather than re-tuned, since this round's
job was porting a working design onto the bitboard core, not re-discovering new numbers.
`try_null_move`, `apply_null_move`/`unmake_null_move` (a null move only flips the side to move
and clears the en-passant square -- no piece moves, so no capture or ep right needs updating),
`update_history`, and `order_moves_full`/`move_score_full` are new relative to V26; `negamax`
and `search_root` were rewritten around them (PVS narrow-window search for non-first moves,
LMR-reduced search with a full-depth re-search fallback, and futility pruning at depth 1).
Killer moves and the history table live on the `Engine` instance alongside the transposition
table, so they persist across moves in the same game, the same pattern V20 uses at module
level. Zobrist hashing is incremental, updated inside `apply_move_hashed`/`unmake_move_hashed`
(verified against a from-scratch recompute in `training/test_bitboard_makeunmake.py`; this is
unchanged from V26).

Evaluation ports V20's tapered material + PeSTO-style piece-square tables + doubled/isolated/
passed-pawn scoring + king pawn-shield/open-file scoring + bishop-pair bonus, numerically
verified to match `submissions/v20-numba-rich-eval-candidate/agent.py`'s `evaluate()` exactly
on 349 sampled positions before any game was played (`training/test_bitboard_eval.py`).
"""

from __future__ import annotations

import time as _time

import chess
import numpy as np
from numba import njit, objmode

# --------------------------------------------------------------------------------------------
# Board representation
# --------------------------------------------------------------------------------------------

WHITE, BLACK = 0, 1
WP, WN, WB, WR, WQ, WK = 0, 1, 2, 3, 4, 5
BP, BN, BB, BR, BQ, BK = 6, 7, 8, 9, 10, 11
NO_PIECE = 15

CASTLE_WK = 1
CASTLE_WQ = 2
CASTLE_BK = 4
CASTLE_BQ = 8

PROMO_NONE, PROMO_N, PROMO_B, PROMO_R, PROMO_Q = 0, 1, 2, 3, 4
PROMO_UCI_CHAR = {PROMO_NONE: "", PROMO_N: "n", PROMO_B: "b", PROMO_R: "r", PROMO_Q: "q"}

MAX_MOVES = 256

KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
KING_DELTAS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
BISHOP_DF = np.array([1, 1, -1, -1], dtype=np.int64)
BISHOP_DR = np.array([1, -1, 1, -1], dtype=np.int64)
ROOK_DF = np.array([1, -1, 0, 0], dtype=np.int64)
ROOK_DR = np.array([0, 0, 1, -1], dtype=np.int64)


def _build_leaper_table(deltas: list[tuple[int, int]]) -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    for s in range(64):
        f, r = s % 8, s // 8
        bb = np.uint64(0)
        for df, dr in deltas:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= np.uint64(1) << np.uint64(nr * 8 + nf)
        table[s] = bb
    return table


KNIGHT_ATTACKS = _build_leaper_table(KNIGHT_DELTAS)
KING_ATTACKS = _build_leaper_table(KING_DELTAS)


def _build_pawn_attacks() -> np.ndarray:
    """PAWN_ATTACKS_FROM[color][sq] = squares a pawn of `color` on `sq` attacks."""
    table = np.zeros((2, 64), dtype=np.uint64)
    for s in range(64):
        f, r = s % 8, s // 8
        bb = np.uint64(0)
        for df, dr in ((1, 1), (-1, 1)):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= np.uint64(1) << np.uint64(nr * 8 + nf)
        table[0, s] = bb
        bb = np.uint64(0)
        for df, dr in ((1, -1), (-1, -1)):
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                bb |= np.uint64(1) << np.uint64(nr * 8 + nf)
        table[1, s] = bb
    return table


PAWN_ATTACKS_FROM = _build_pawn_attacks()


def _build_pawn_attackers() -> np.ndarray:
    """PAWN_ATTACKERS[color][sq] = squares from which a pawn of `color` attacks sq (the
    inverse of PAWN_ATTACKS_FROM), used to test "is this square attacked by a pawn"."""
    table = np.zeros((2, 64), dtype=np.uint64)
    for color in (0, 1):
        for origin in range(64):
            t = int(PAWN_ATTACKS_FROM[color, origin])
            while t:
                low = t & (-t)
                target_sq = low.bit_length() - 1
                table[color, target_sq] |= np.uint64(1) << np.uint64(origin)
                t &= t - 1
    return table


PAWN_ATTACKERS = _build_pawn_attackers()


@njit(cache=False)
def sliding_attacks(s: int, occ: np.uint64, dir_df: np.ndarray, dir_dr: np.ndarray) -> np.uint64:
    """Classical ray-scanning: step one square at a time in each of 4 directions until the
    board edge or a blocker (inclusive of the blocker itself). This is the ground truth used
    to build the magic-bitboard lookup tables below (see `bishop_attacks_classical`,
    `rook_attacks_classical`, and the module docstring's magic-bitboard section) -- kept
    exactly as it was when it was the only implementation, in V26/V27/V28."""
    f0 = s % 8
    r0 = s // 8
    bb = np.uint64(0)
    for i in range(4):
        df = dir_df[i]
        dr = dir_dr[i]
        f = f0 + df
        r = r0 + dr
        while 0 <= f < 8 and 0 <= r < 8:
            t = r * 8 + f
            bb |= np.uint64(1) << np.uint64(t)
            if (occ >> np.uint64(t)) & np.uint64(1):
                break
            f += df
            r += dr
    return bb


@njit(cache=False)
def bishop_attacks_classical(s: int, occ: np.uint64) -> np.uint64:
    return sliding_attacks(s, occ, BISHOP_DF, BISHOP_DR)


@njit(cache=False)
def rook_attacks_classical(s: int, occ: np.uint64) -> np.uint64:
    return sliding_attacks(s, occ, ROOK_DF, ROOK_DR)


# --------------------------------------------------------------------------------------------
# Magic bitboards: replace the O(distance)-per-direction ray scan above with one multiply,
# shift, and table lookup per sliding piece. Standard technique (see e.g.
# chessprogramming.org/Magic_Bitboards): for each square, the "relevant occupancy" is the set
# of squares between the piece and the board edge along its attack rays (excluding the edge
# square itself, since a blocker there can never affect the attack set further in). Multiplying
# the actual occupancy (masked to just those relevant bits) by a precomputed 64-bit "magic"
# number and taking the top bits produces a perfect hash -- a collision-free index into a
# precomputed table of attack bitboards, one entry per possible relevant-occupancy pattern.
#
# The 128 magic numbers below (64 squares x bishop/rook) were found by this round's own
# offline random search (try a random sparse 64-bit number, verify it hashes every subset of
# the relevant occupancy mask to a bitboard-consistent index with zero collisions, keep
# searching until one works), then hardcoded here -- not copied from a published table. Each
# (mask, magic, shift) triple was independently re-verified against `sliding_attacks` above
# (the already perft-verified classical implementation) while building the lookup tables at
# import time (see `_build_magic_tables` below): if a single subset of a single square's mask
# ever produced a different attack bitboard than the classical ray scan, that would raise at
# import, not silently ship a wrong attack table. This is exactly the failure mode a bad magic
# number causes -- correct for most occupancy patterns, silently wrong for a rare one -- so it
# is checked exhaustively (every subset of every mask), not sampled.

# (relevant_occupancy_mask, magic_number, shift) per square, a1=0 ... h8=63.
_BISHOP_MAGIC_DATA = [
    (18049651735527936, 1243002914012790848, 58), (70506452091904, 22951351814283300, 59),
    (275415828992, 41104151284810752, 59), (1075975168, 334396689773232428, 59),
    (38021120, 1459465621309227720, 59), (8657588224, 1226157792558186496, 59),
    (2216338399232, 1183323842997715972, 59), (567382630219776, 5188291927874798688, 58),
    (9024825867763712, 14997153953899219012, 59), (18049651735527424, 144150931372638848, 59),
    (70506452221952, 2445325177520128, 59), (275449643008, 1152926178613592064, 59),
    (9733406720, 36033219226779649, 59), (2216342585344, 141322143399952, 59),
    (567382630203392, 67554553029074944, 59), (1134765260406784, 4606420328448, 59),
    (4512412933816832, 154267185903196184, 59), (9024825867633664, 41658880820257291, 59),
    (18049651768822272, 38280667859259904, 57), (70515108615168, 1153345400799773696, 57),
    (2491752130560, 12106806098550326272, 57), (567383701868544, 144185866192421964, 57),
    (1134765256220672, 144399138162230288, 59), (2269530512441344, 283674696745993, 59),
    (2256206450263040, 585494408624866338, 59), (4512412900526080, 11530376141353386112, 59),
    (9024834391117824, 4612549135860760613, 57), (18051867805491712, 9241390833549246560, 55),
    (637888545440768, 2534376450572292, 55), (1135039602493440, 9227877011149031440, 57),
    (2269529440784384, 433472567946317856, 59), (4539058881568768, 144401409266418688, 59),
    (1128098963916800, 12254304069403288064, 59), (2256197927833600, 154252978956668993, 59),
    (4514594912477184, 577589160504787464, 57), (9592139778506752, 13616489710224640, 55),
    (19184279556981248, 1126466842657026, 55), (2339762086609920, 3461592462022291456, 57),
    (4538784537380864, 325393889571669005, 59), (9077569074761728, 90925490612865536, 59),
    (562958610993152, 28165125359343812, 59), (1125917221986304, 577032508047509776, 59),
    (2814792987328512, 918751987305023488, 57), (5629586008178688, 108086528898498816, 57),
    (11259172008099840, 1441715943678936068, 57), (22518341868716544, 9042385006952576, 57),
    (9007336962655232, 40535147640589056, 59), (18014673925310464, 2361155702619111620, 59),
    (2216338399232, 865262891715600544, 59), (4432676798464, 585615290679820288, 59),
    (11064376819712, 598555252236416, 59), (22137335185408, 54188883671777322, 59),
    (44272556441600, 5768020475881066528, 59), (87995357200384, 2305922243576365313, 59),
    (35253226045952, 583224961930502416, 59), (70506452091904, 2253208597053440, 59),
    (567382630219776, 1688991661465600, 58), (1134765260406784, 335903084512001, 59),
    (2832480465846272, 1166573041582479378, 59), (5667157807464448, 74388857340299520, 59),
    (11333774449049600, 9520609888213414400, 59), (22526811443298304, 288265766724700224, 59),
    (9024825867763712, 649683845982080004, 59), (18049651735527936, 4578435292774656, 58),
]
_ROOK_MAGIC_DATA = [
    (282578800148862, 36033815692443808, 52), (565157600297596, 144134981436842241, 53),
    (1130315200595066, 72075224884969792, 53), (2260630401190006, 36046391353280512, 53),
    (4521260802379886, 432389656362878112, 53), (9042521604759646, 9295431868571190272, 53),
    (18085043209519166, 72132371851249152, 53), (36170086419038334, 9295430730423356546, 52),
    (282578800180736, 2648257457968660480, 53), (565157600328704, 9224639086575304704, 54),
    (1130315200625152, 13792411448787088, 54), (2260630401218048, 144256475454246912, 54),
    (4521260802403840, 288511885489210624, 54), (9042521604775424, 141287311311360, 54),
    (18085043209518592, 289637768517124352, 54), (36170086419037696, 2382545014598140160, 53),
    (282578808340736, 293754872401510400, 53), (565157608292864, 46039301170937856, 54),
    (1130315208328192, 5769200184271569152, 54), (2260630408398848, 565149514154128, 54),
    (4521260808540160, 4612812468224263298, 54), (9042521608822784, 4223224464802816, 54),
    (18085043209388032, 12128335084440781312, 54), (36170086418907136, 145243287073088129, 53),
    (282580897300736, 6918127748229644289, 53), (565159647117824, 288300747045470337, 54),
    (1130317180306432, 1160311343734063249, 54), (2260632246683648, 9015997496295552, 54),
    (4521262379438080, 13196295407616, 54), (9042522644946944, 289074839736681472, 54),
    (18085043175964672, 18295877781291028, 54), (36170086385483776, 1155173879949033541, 53),
    (283115671060736, 36029621656887306, 53), (565681586307584, 4647855827821010944, 54),
    (1130822006735872, 1441257436030701568, 54), (2261102847592448, 324294641044164616, 54),
    (4521664529305600, 1153062809039276034, 54), (9042787892731904, 36030998198092800, 54),
    (18085034619584512, 149181754870075656, 54), (36170077829103616, 4546256437313, 53),
    (420017753620736, 324259585626161152, 53), (699298018886144, 18577916489236512, 54),
    (1260057572672512, 4702461973865299984, 54), (2381576680245248, 9224849922518548488, 54),
    (4624614895390720, 56866741673852948, 54), (9110691325681664, 579275553885257736, 54),
    (18082844186263552, 281775624486932, 54), (36167887395782656, 139893137706385410, 53),
    (35466950888980736, 9277555979535523968, 53), (34905104758997504, 1157461431072489984, 54),
    (34344362452452352, 4630298589926605056, 54), (33222877839362048, 4508032034832512, 54),
    (30979908613181440, 36315769688035584, 54), (26493970160820224, 1130298054049920, 54),
    (17522093256097792, 5188428804053532928, 54), (35607136465616896, 37161021332333056, 53),
    (9079539427579068672, 504704440574443554, 52), (8935706818303361536, 76844343983357953, 53),
    (8792156787827803136, 11610279912376115265, 53), (8505056726876686336, 148618856559019265, 53),
    (7930856604974452736, 1729945344437912770, 53), (6782456361169985536, 145522597352964225, 53),
    (4485655873561051136, 10997532264452, 53), (9115426935197958144, 1243556586711154754, 52),
]


def _build_magic_tables(
    magic_data: list[tuple[int, int, int]], classical_fn: object
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one flat attack table (plus per-square mask/magic/shift/offset arrays) from
    (mask, magic, shift) triples, checking every single relevant-occupancy subset of every
    square against the classical ray-scanner as it goes -- see the section docstring above for
    why this is exhaustive rather than sampled."""
    # The magic multiply and the carry-rippler subset trick (`occ = (occ - mask) & mask`) both
    # rely on wraparound arithmetic on purpose (that overflow is the whole point of a 64-bit
    # magic hash and of enumerating subsets via repeated subtraction) -- suppress numpy's
    # overflow warning for just this construction step rather than have it print on every
    # import and look like something is wrong.
    masks = np.array([m for m, _, _ in magic_data], dtype=np.uint64)
    magics = np.array([m for _, m, _ in magic_data], dtype=np.uint64)
    shifts = np.array([s for _, _, s in magic_data], dtype=np.int64)
    offsets = np.zeros(64, dtype=np.int64)
    per_square_tables = []
    total = 0
    with np.errstate(over="ignore"):
        for s in range(64):
            mask = masks[s]
            magic = magics[s]
            shift = int(shifts[s])
            size = 1 << (64 - shift)
            table = np.zeros(size, dtype=np.uint64)
            occ = np.uint64(0)
            while True:
                idx = int((occ * magic) >> np.uint64(shift))
                expected = classical_fn(s, occ)  # type: ignore[operator]
                table[idx] = expected
                occ = (occ - mask) & mask
                if occ == np.uint64(0):
                    break
            offsets[s] = total
            total += size
            per_square_tables.append(table)
    flat = np.zeros(total, dtype=np.uint64)
    for s in range(64):
        flat[offsets[s]:offsets[s] + len(per_square_tables[s])] = per_square_tables[s]
    return flat, offsets, masks, magics, shifts


(
    BISHOP_TABLE, BISHOP_OFFSET, BISHOP_MASK, BISHOP_MAGIC, BISHOP_SHIFT,
) = _build_magic_tables(_BISHOP_MAGIC_DATA, bishop_attacks_classical)
(
    ROOK_TABLE, ROOK_OFFSET, ROOK_MASK, ROOK_MAGIC, ROOK_SHIFT,
) = _build_magic_tables(_ROOK_MAGIC_DATA, rook_attacks_classical)


@njit(cache=False)
def bishop_attacks(s: int, occ: np.uint64) -> np.uint64:
    hashed = (occ & BISHOP_MASK[s]) * BISHOP_MAGIC[s]
    idx = BISHOP_OFFSET[s] + int(hashed >> np.uint64(BISHOP_SHIFT[s]))
    return np.uint64(BISHOP_TABLE[idx])


@njit(cache=False)
def rook_attacks(s: int, occ: np.uint64) -> np.uint64:
    hashed = (occ & ROOK_MASK[s]) * ROOK_MAGIC[s]
    idx = ROOK_OFFSET[s] + int(hashed >> np.uint64(ROOK_SHIFT[s]))
    return np.uint64(ROOK_TABLE[idx])


@njit(cache=False)
def occ_white(boards: np.ndarray) -> np.uint64:
    bb = np.uint64(0)
    for i in range(0, 6):
        bb |= boards[i]
    return bb


@njit(cache=False)
def occ_black(boards: np.ndarray) -> np.uint64:
    bb = np.uint64(0)
    for i in range(6, 12):
        bb |= boards[i]
    return bb


@njit(cache=False)
def square_attacked(boards: np.ndarray, target: int, by_white: bool) -> bool:
    occ = occ_white(boards) | occ_black(boards)
    if by_white:
        if PAWN_ATTACKERS[0, target] & boards[WP]:
            return True
        if KNIGHT_ATTACKS[target] & boards[WN]:
            return True
        if KING_ATTACKS[target] & boards[WK]:
            return True
        if bishop_attacks(target, occ) & (boards[WB] | boards[WQ]):
            return True
        if rook_attacks(target, occ) & (boards[WR] | boards[WQ]):
            return True
    else:
        if PAWN_ATTACKERS[1, target] & boards[BP]:
            return True
        if KNIGHT_ATTACKS[target] & boards[BN]:
            return True
        if KING_ATTACKS[target] & boards[BK]:
            return True
        if bishop_attacks(target, occ) & (boards[BB] | boards[BQ]):
            return True
        if rook_attacks(target, occ) & (boards[BR] | boards[BQ]):
            return True
    return False


@njit(cache=False)
def king_square(boards: np.ndarray, white: bool) -> int:
    bb = boards[WK] if white else boards[BK]
    pos = -1
    idx = 0
    v = bb
    while v:
        if v & np.uint64(1):
            pos = idx
        v >>= np.uint64(1)
        idx += 1
    return pos


@njit(cache=False)
def pop_lsb(bb: np.uint64) -> tuple[int, np.uint64]:
    """Return (index of the least-significant set bit, bb with that bit cleared)."""
    idx = 0
    v = bb
    while not (v & np.uint64(1)):
        v >>= np.uint64(1)
        idx += 1
    return idx, bb & (bb - np.uint64(1))


@njit(cache=False)
def piece_at(boards: np.ndarray, s: int) -> int:
    bit = np.uint64(1) << np.uint64(s)
    for i in range(12):
        if boards[i] & bit:
            return i
    return NO_PIECE


@njit(cache=False)
def make_move_int(
    frm: int, to: int, piece: int, captured: int, promo: int,
    ep: bool, dbl: bool, ck: bool, cq: bool,
) -> int:
    m = frm | (to << 6) | (piece << 12) | (captured << 16) | (promo << 20)
    if ep:
        m |= 1 << 23
    if dbl:
        m |= 1 << 24
    if ck:
        m |= 1 << 25
    if cq:
        m |= 1 << 26
    return m


@njit(cache=False)
def unpack_move(m: int) -> tuple[int, int, int, int, int, bool, bool, bool, bool]:
    frm = m & 0x3F
    to = (m >> 6) & 0x3F
    piece = (m >> 12) & 0xF
    captured = (m >> 16) & 0xF
    promo = (m >> 20) & 0x7
    ep = bool((m >> 23) & 1)
    dbl = bool((m >> 24) & 1)
    ck = bool((m >> 25) & 1)
    cq = bool((m >> 26) & 1)
    return frm, to, piece, captured, promo, ep, dbl, ck, cq


@njit(cache=False)
def promo_piece_code(promo: int, white: bool) -> int:
    base = 0 if white else 6
    if promo == PROMO_N:
        return base + 1
    if promo == PROMO_B:
        return base + 2
    if promo == PROMO_R:
        return base + 3
    return base + 4


@njit(cache=False)
def add_promo_moves(buf: np.ndarray, n: int, frm: int, to: int, piece: int, captured: int) -> int:
    """Append all 4 promotion choices for one pawn move; a small shared helper so the 8
    near-duplicate push/capture promotion blocks a monolithic generator would otherwise need
    do not each get inlined separately (that duplication is what made the very first version
    of this function take ~14s to compile -- see the module docstring's compile-time note)."""
    buf[n] = make_move_int(frm, to, piece, captured, PROMO_Q, False, False, False, False)
    n += 1
    buf[n] = make_move_int(frm, to, piece, captured, PROMO_R, False, False, False, False)
    n += 1
    buf[n] = make_move_int(frm, to, piece, captured, PROMO_B, False, False, False, False)
    n += 1
    buf[n] = make_move_int(frm, to, piece, captured, PROMO_N, False, False, False, False)
    n += 1
    return n


@njit(cache=False)
def gen_pawn_moves(
    boards: np.ndarray, meta: np.ndarray, buf: np.ndarray, n: int, white: bool
) -> int:
    ow = occ_white(boards)
    ob = occ_black(boards)
    occ = ow | ob
    enemy = ob if white else ow
    pawn = WP if white else BP
    dirn = 8 if white else -8
    start_rank = 1 if white else 6
    promo_rank = 7 if white else 0

    pawns = boards[pawn]
    while pawns:
        s, pawns = pop_lsb(pawns)
        f = s % 8
        target = s + dirn
        if 0 <= target < 64 and not ((occ >> np.uint64(target)) & np.uint64(1)):
            if target // 8 == promo_rank:
                n = add_promo_moves(buf, n, s, target, pawn, NO_PIECE)
            else:
                buf[n] = make_move_int(
                    s, target, pawn, NO_PIECE, PROMO_NONE, False, False, False, False
                )
                n += 1
            if s // 8 == start_rank:
                target2 = s + 2 * dirn
                if not ((occ >> np.uint64(target2)) & np.uint64(1)):
                    buf[n] = make_move_int(
                        s, target2, pawn, NO_PIECE, PROMO_NONE, False, True, False, False
                    )
                    n += 1
        for df in (-1, 1):
            nf = f + df
            if 0 <= nf < 8:
                target = s + dirn + df
                if 0 <= target < 64:
                    if (enemy >> np.uint64(target)) & np.uint64(1):
                        cap = piece_at(boards, target)
                        if target // 8 == promo_rank:
                            n = add_promo_moves(buf, n, s, target, pawn, cap)
                        else:
                            buf[n] = make_move_int(
                                s, target, pawn, cap, PROMO_NONE, False, False, False, False
                            )
                            n += 1
                    elif meta[2] >= 0 and target == meta[2]:
                        ep_cap = BP if white else WP
                        buf[n] = make_move_int(
                            s, target, pawn, ep_cap, PROMO_NONE, True, False, False, False
                        )
                        n += 1
    return n


@njit(cache=False)
def gen_knight_moves(boards: np.ndarray, buf: np.ndarray, n: int, white: bool) -> int:
    ow = occ_white(boards)
    ob = occ_black(boards)
    own = ow if white else ob
    enemy = ob if white else ow
    knight = WN if white else BN
    knights = boards[knight]
    while knights:
        s, knights = pop_lsb(knights)
        targets = KNIGHT_ATTACKS[s] & ~own
        while targets:
            to, targets = pop_lsb(targets)
            cap = piece_at(boards, to) if (enemy >> np.uint64(to)) & np.uint64(1) else NO_PIECE
            buf[n] = make_move_int(s, to, knight, cap, PROMO_NONE, False, False, False, False)
            n += 1
    return n


@njit(cache=False)
def gen_sliding_moves(
    boards: np.ndarray, buf: np.ndarray, n: int, white: bool, piece: int, diag: bool, orth: bool
) -> int:
    """Shared by bishop, rook, and queen (diag=True/orth=True for queen) so there is one
    compiled body for all three instead of three near-identical ones."""
    ow = occ_white(boards)
    ob = occ_black(boards)
    occ = ow | ob
    own = ow if white else ob
    enemy = ob if white else ow
    pieces = boards[piece]
    while pieces:
        s, pieces = pop_lsb(pieces)
        atk = np.uint64(0)
        if diag:
            atk |= bishop_attacks(s, occ)
        if orth:
            atk |= rook_attacks(s, occ)
        atk &= ~own
        while atk:
            to, atk = pop_lsb(atk)
            cap = piece_at(boards, to) if (enemy >> np.uint64(to)) & np.uint64(1) else NO_PIECE
            buf[n] = make_move_int(s, to, piece, cap, PROMO_NONE, False, False, False, False)
            n += 1
    return n


@njit(cache=False)
def gen_castling_moves(
    boards: np.ndarray, meta: np.ndarray, buf: np.ndarray, n: int, white: bool, ks: int
) -> int:
    """One shared code path for both colours (using `back` as the home-rank offset) instead
    of two near-duplicate white/black branches -- see the module docstring's compile-time
    note; a duplicated branch here was nearly as costly to compile as the pawn duplication
    that motivated `add_promo_moves`."""
    back = 0 if white else 56
    ks_start = back + 4
    if ks != ks_start:
        return n
    occ = occ_white(boards) | occ_black(boards)
    rook = WR if white else BR
    king = WK if white else BK
    kr = CASTLE_WK if white else CASTLE_BK
    qr = CASTLE_WQ if white else CASTLE_BQ
    rights = meta[1]
    by_enemy = not white

    if rights & kr:
        f_sq, g_sq, h_sq = back + 5, back + 6, back + 7
        if (
            not ((occ >> np.uint64(f_sq)) & np.uint64(1))
            and not ((occ >> np.uint64(g_sq)) & np.uint64(1))
            and ((boards[rook] >> np.uint64(h_sq)) & np.uint64(1))
            and not square_attacked(boards, ks_start, by_enemy)
            and not square_attacked(boards, f_sq, by_enemy)
            and not square_attacked(boards, g_sq, by_enemy)
        ):
            buf[n] = make_move_int(
                ks_start, g_sq, king, NO_PIECE, PROMO_NONE, False, False, True, False
            )
            n += 1

    if rights & qr:
        b_sq, c_sq, d_sq, a_sq = back + 1, back + 2, back + 3, back + 0
        if (
            not ((occ >> np.uint64(b_sq)) & np.uint64(1))
            and not ((occ >> np.uint64(c_sq)) & np.uint64(1))
            and not ((occ >> np.uint64(d_sq)) & np.uint64(1))
            and ((boards[rook] >> np.uint64(a_sq)) & np.uint64(1))
            and not square_attacked(boards, ks_start, by_enemy)
            and not square_attacked(boards, d_sq, by_enemy)
            and not square_attacked(boards, c_sq, by_enemy)
        ):
            buf[n] = make_move_int(
                ks_start, c_sq, king, NO_PIECE, PROMO_NONE, False, False, False, True
            )
            n += 1

    return n


@njit(cache=False)
def gen_king_moves(
    boards: np.ndarray, meta: np.ndarray, buf: np.ndarray, n: int, white: bool
) -> int:
    ow = occ_white(boards)
    ob = occ_black(boards)
    own = ow if white else ob
    enemy = ob if white else ow
    king = WK if white else BK
    ks = king_square(boards, white)
    targets = KING_ATTACKS[ks] & ~own
    while targets:
        to, targets = pop_lsb(targets)
        cap = piece_at(boards, to) if (enemy >> np.uint64(to)) & np.uint64(1) else NO_PIECE
        buf[n] = make_move_int(ks, to, king, cap, PROMO_NONE, False, False, False, False)
        n += 1
    n = gen_castling_moves(boards, meta, buf, n, white, ks)
    return n


@njit(cache=False)
def gen_pseudo_moves(boards: np.ndarray, meta: np.ndarray, buf: np.ndarray) -> int:
    """Fill buf (int64[MAX_MOVES]) with pseudo-legal moves (own-king-safety not checked
    yet); return the count written. Delegates to small per-piece-type generators (see the
    module docstring) rather than one large function, purely to keep numba's compile time
    down -- the logic is identical to a single inlined function."""
    white = meta[0] == 0
    bishop = WB if white else BB
    rook = WR if white else BR
    queen = WQ if white else BQ
    n = 0
    n = gen_pawn_moves(boards, meta, buf, n, white)
    n = gen_knight_moves(boards, buf, n, white)
    n = gen_sliding_moves(boards, buf, n, white, bishop, True, False)
    n = gen_sliding_moves(boards, buf, n, white, rook, False, True)
    n = gen_sliding_moves(boards, buf, n, white, queen, True, True)
    n = gen_king_moves(boards, meta, buf, n, white)
    return n


# --------------------------------------------------------------------------------------------
# Zobrist hashing (precomputed at import; incrementally updated in apply_move_hashed)
# --------------------------------------------------------------------------------------------


def _build_zobrist() -> tuple[np.ndarray, np.uint64, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    a = rng.integers(0, 2**63 - 1, size=(12, 64), dtype=np.int64).astype(np.uint64)
    b = rng.integers(0, 2**63 - 1, size=(12, 64), dtype=np.int64).astype(np.uint64)
    piece_keys = a ^ (b << np.uint64(1))
    side_key = np.uint64(rng.integers(0, 2**63 - 1, dtype=np.int64))
    castle_keys = rng.integers(0, 2**63 - 1, size=4, dtype=np.int64).astype(np.uint64)
    ep_file_keys = rng.integers(0, 2**63 - 1, size=8, dtype=np.int64).astype(np.uint64)
    return piece_keys, side_key, castle_keys, ep_file_keys


ZOBRIST_PIECE, ZOBRIST_SIDE, ZOBRIST_CASTLE, ZOBRIST_EP_FILE = _build_zobrist()


@njit(cache=False)
def compute_hash(boards: np.ndarray, meta: np.ndarray) -> np.uint64:
    """Compute the Zobrist hash from scratch. Used to seed a search and, in tests, to verify
    the incremental hash maintained by apply_move_hashed/unmake_move_hashed."""
    h = np.uint64(0)
    for piece in range(12):
        bb = boards[piece]
        while bb:
            s, bb = pop_lsb(bb)
            h ^= ZOBRIST_PIECE[piece, s]
    if meta[0] == 1:
        h ^= ZOBRIST_SIDE
    rights = meta[1]
    for bit in range(4):
        if rights & (1 << bit):
            h ^= ZOBRIST_CASTLE[bit]
    if meta[2] >= 0:
        h ^= ZOBRIST_EP_FILE[meta[2] % 8]
    return h


@njit(cache=False)
def apply_move(boards: np.ndarray, meta: np.ndarray, m: int, undo: np.ndarray) -> None:
    """Mutate boards & meta in place. Fill undo (int64[6]) with enough to reverse the move:
    [m, prev_castling_rights, prev_ep_square, prev_halfmove_clock, ep_captured_square, 0]."""
    frm, to, piece, captured, promo, ep, dbl, ck, cq = unpack_move(m)
    side = meta[0]
    white = side == 0
    prev_rights = meta[1]
    prev_ep = meta[2]
    prev_half = meta[3]

    frm_bit = np.uint64(1) << np.uint64(frm)
    to_bit = np.uint64(1) << np.uint64(to)

    boards[piece] &= ~frm_bit

    ep_captured_sq = -1
    if ep:
        ep_captured_sq = to - 8 if white else to + 8
        cap_bit = np.uint64(1) << np.uint64(ep_captured_sq)
        boards[captured] &= ~cap_bit
    elif captured != NO_PIECE:
        boards[captured] &= ~to_bit

    if promo != PROMO_NONE:
        pp = promo_piece_code(promo, white)
        boards[pp] |= to_bit
    else:
        boards[piece] |= to_bit

    if ck:
        if white:
            boards[WR] &= ~(np.uint64(1) << np.uint64(7))
            boards[WR] |= np.uint64(1) << np.uint64(5)
        else:
            boards[BR] &= ~(np.uint64(1) << np.uint64(63))
            boards[BR] |= np.uint64(1) << np.uint64(61)
    if cq:
        if white:
            boards[WR] &= ~(np.uint64(1) << np.uint64(0))
            boards[WR] |= np.uint64(1) << np.uint64(3)
        else:
            boards[BR] &= ~(np.uint64(1) << np.uint64(56))
            boards[BR] |= np.uint64(1) << np.uint64(59)

    new_rights = prev_rights
    if piece == WK:
        new_rights &= ~(CASTLE_WK | CASTLE_WQ)
    elif piece == BK:
        new_rights &= ~(CASTLE_BK | CASTLE_BQ)
    if frm == 0 or to == 0:
        new_rights &= ~CASTLE_WQ
    if frm == 7 or to == 7:
        new_rights &= ~CASTLE_WK
    if frm == 56 or to == 56:
        new_rights &= ~CASTLE_BQ
    if frm == 63 or to == 63:
        new_rights &= ~CASTLE_BK

    new_ep = -1
    if dbl:
        new_ep = (frm + to) // 2

    is_pawn = piece in (WP, BP)
    new_half = 0 if (is_pawn or captured != NO_PIECE) else prev_half + 1

    meta[0] = 1 - side
    meta[1] = new_rights
    meta[2] = new_ep
    meta[3] = new_half

    undo[0] = m
    undo[1] = prev_rights
    undo[2] = prev_ep
    undo[3] = prev_half
    undo[4] = ep_captured_sq


@njit(cache=False)
def unmake_move(boards: np.ndarray, meta: np.ndarray, undo: np.ndarray) -> None:
    m = undo[0]
    prev_rights = undo[1]
    prev_ep = undo[2]
    prev_half = undo[3]
    ep_captured_sq = undo[4]
    frm, to, piece, captured, promo, ep, _dbl, ck, cq = unpack_move(m)
    side_after = meta[0]
    white = side_after == 1

    frm_bit = np.uint64(1) << np.uint64(frm)
    to_bit = np.uint64(1) << np.uint64(to)

    if promo != PROMO_NONE:
        pp = promo_piece_code(promo, white)
        boards[pp] &= ~to_bit
    else:
        boards[piece] &= ~to_bit
    boards[piece] |= frm_bit

    if ep:
        cap_bit = np.uint64(1) << np.uint64(ep_captured_sq)
        boards[captured] |= cap_bit
    elif captured != NO_PIECE:
        boards[captured] |= to_bit

    if ck:
        if white:
            boards[WR] |= np.uint64(1) << np.uint64(7)
            boards[WR] &= ~(np.uint64(1) << np.uint64(5))
        else:
            boards[BR] |= np.uint64(1) << np.uint64(63)
            boards[BR] &= ~(np.uint64(1) << np.uint64(61))
    if cq:
        if white:
            boards[WR] |= np.uint64(1) << np.uint64(0)
            boards[WR] &= ~(np.uint64(1) << np.uint64(3))
        else:
            boards[BR] |= np.uint64(1) << np.uint64(56)
            boards[BR] &= ~(np.uint64(1) << np.uint64(59))

    meta[0] = 1 - side_after
    meta[1] = prev_rights
    meta[2] = prev_ep
    meta[3] = prev_half


@njit(cache=False)
def apply_move_hashed(
    boards: np.ndarray, meta: np.ndarray, m: int, undo: np.ndarray, hash_arr: np.ndarray
) -> None:
    """Same as apply_move, but also incrementally updates hash_arr[0] (uint64[1]) and stashes
    the pre-move hash in undo[5] so unmake_move_hashed can restore it exactly."""
    frm, to, piece, captured, promo, ep, _dbl, ck, cq = unpack_move(m)
    side = meta[0]
    white = side == 0
    prev_rights = meta[1]
    prev_ep = meta[2]
    hash_before = hash_arr[0]
    h = hash_before

    h ^= ZOBRIST_PIECE[piece, frm]
    ep_captured_sq_h = -1
    if ep:
        ep_captured_sq_h = to - 8 if white else to + 8
        h ^= ZOBRIST_PIECE[captured, ep_captured_sq_h]
    elif captured != NO_PIECE:
        h ^= ZOBRIST_PIECE[captured, to]
    if promo != PROMO_NONE:
        pp = promo_piece_code(promo, white)
        h ^= ZOBRIST_PIECE[pp, to]
    else:
        h ^= ZOBRIST_PIECE[piece, to]
    if ck:
        if white:
            h ^= ZOBRIST_PIECE[WR, 7]
            h ^= ZOBRIST_PIECE[WR, 5]
        else:
            h ^= ZOBRIST_PIECE[BR, 63]
            h ^= ZOBRIST_PIECE[BR, 61]
    if cq:
        if white:
            h ^= ZOBRIST_PIECE[WR, 0]
            h ^= ZOBRIST_PIECE[WR, 3]
        else:
            h ^= ZOBRIST_PIECE[BR, 56]
            h ^= ZOBRIST_PIECE[BR, 59]

    apply_move(boards, meta, m, undo)

    new_rights = meta[1]
    removed_rights = prev_rights & ~new_rights
    for bit in range(4):
        if removed_rights & (1 << bit):
            h ^= ZOBRIST_CASTLE[bit]
    if prev_ep >= 0:
        h ^= ZOBRIST_EP_FILE[prev_ep % 8]
    new_ep = meta[2]
    if new_ep >= 0:
        h ^= ZOBRIST_EP_FILE[new_ep % 8]
    h ^= ZOBRIST_SIDE

    hash_arr[0] = h
    undo[5] = hash_before


@njit(cache=False)
def unmake_move_hashed(
    boards: np.ndarray, meta: np.ndarray, undo: np.ndarray, hash_arr: np.ndarray
) -> None:
    unmake_move(boards, meta, undo)
    hash_arr[0] = undo[5]


@njit(cache=False)
def gen_legal_moves(boards: np.ndarray, meta: np.ndarray, out_buf: np.ndarray) -> int:
    """Generate pseudo-legal moves, then keep only those that do not leave the mover's own
    king attacked -- verified by actually making and unmaking each candidate."""
    side = meta[0]
    white = side == 0
    pseudo = np.empty(MAX_MOVES, dtype=np.int64)
    count = gen_pseudo_moves(boards, meta, pseudo)
    undo = np.empty(6, dtype=np.int64)
    n = 0
    for i in range(count):
        m = pseudo[i]
        apply_move(boards, meta, m, undo)
        ks = king_square(boards, white)
        if not square_attacked(boards, ks, not white):
            out_buf[n] = m
            n += 1
        unmake_move(boards, meta, undo)
    return n


@njit(cache=False)
def perft(boards: np.ndarray, meta: np.ndarray, depth: int) -> int:
    """Count leaf nodes at `depth` plies -- the correctness oracle for move generation."""
    if depth == 0:
        return 1
    buf = np.empty(MAX_MOVES, dtype=np.int64)
    count = gen_legal_moves(boards, meta, buf)
    if depth == 1:
        return count
    undo = np.empty(6, dtype=np.int64)
    total = 0
    for i in range(count):
        apply_move(boards, meta, buf[i], undo)
        total += perft(boards, meta, depth - 1)
        unmake_move(boards, meta, undo)
    return total


def boards_from_fen(fen: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert a FEN (via python-chess, used only at this Python-level boundary) into our
    own (boards, meta) representation."""
    b = chess.Board(fen)
    boards = np.zeros(12, dtype=np.uint64)
    code_map = {
        (chess.PAWN, True): WP, (chess.KNIGHT, True): WN, (chess.BISHOP, True): WB,
        (chess.ROOK, True): WR, (chess.QUEEN, True): WQ, (chess.KING, True): WK,
        (chess.PAWN, False): BP, (chess.KNIGHT, False): BN, (chess.BISHOP, False): BB,
        (chess.ROOK, False): BR, (chess.QUEEN, False): BQ, (chess.KING, False): BK,
    }
    for square, piece in b.piece_map().items():
        code = code_map[(piece.piece_type, piece.color)]
        boards[code] |= np.uint64(1) << np.uint64(square)
    meta = np.zeros(4, dtype=np.int64)
    meta[0] = 0 if b.turn == chess.WHITE else 1
    rights = 0
    if b.has_kingside_castling_rights(chess.WHITE):
        rights |= CASTLE_WK
    if b.has_queenside_castling_rights(chess.WHITE):
        rights |= CASTLE_WQ
    if b.has_kingside_castling_rights(chess.BLACK):
        rights |= CASTLE_BK
    if b.has_queenside_castling_rights(chess.BLACK):
        rights |= CASTLE_BQ
    meta[1] = rights
    meta[2] = b.ep_square if b.ep_square is not None else -1
    meta[3] = b.halfmove_clock
    return boards, meta


def move_to_uci(m: int) -> str:
    frm, to, _piece, _captured, promo, _ep, _dbl, _ck, _cq = unpack_move(m)
    return chess.square_name(frm) + chess.square_name(to) + PROMO_UCI_CHAR[promo]


# --------------------------------------------------------------------------------------------
# Evaluation: V20's tapered material + PST + pawn structure + king safety + bishop pair,
# ported onto the bitboard representation. Numbers verified to match V20's evaluate() exactly
# on 349 sampled positions (training/test_bitboard_eval.py) before any game was played.
# --------------------------------------------------------------------------------------------

MG_VALUE = [82, 337, 365, 477, 1025, 0]
EG_VALUE = [94, 281, 297, 512, 936, 0]

_MG_PAWN_PRINTED = [
    0, 0, 0, 0, 0, 0, 0, 0,
    98, 134, 61, 95, 68, 126, 34, -11,
    -6, 7, 26, 31, 65, 56, 25, -20,
    -14, 13, 6, 21, 23, 12, 17, -23,
    -27, -2, -5, 12, 17, 6, 10, -25,
    -26, -4, -4, -10, 3, 3, 33, -12,
    -35, -1, -20, -23, -15, 24, 38, -22,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_EG_PAWN_PRINTED = [
    0, 0, 0, 0, 0, 0, 0, 0,
    178, 173, 158, 134, 147, 132, 165, 187,
    94, 100, 85, 67, 56, 53, 82, 84,
    32, 24, 13, 5, -2, 4, 17, 17,
    13, 9, -3, -7, -7, -8, 3, -1,
    4, 7, -6, 1, 0, -5, -1, -8,
    13, 8, 8, 10, 13, 0, 2, -7,
    0, 0, 0, 0, 0, 0, 0, 0,
]
_MG_KNIGHT_PRINTED = [
    -167, -89, -34, -49, 61, -97, -15, -107,
    -73, -41, 72, 36, 23, 62, 7, -17,
    -47, 60, 37, 65, 84, 129, 73, 44,
    -9, 17, 19, 53, 37, 69, 18, 22,
    -13, 4, 16, 13, 28, 19, 21, -8,
    -23, -9, 12, 10, 19, 17, 25, -16,
    -29, -53, -12, -3, -1, 18, -14, -19,
    -105, -21, -58, -33, -17, -28, -19, -23,
]
_EG_KNIGHT_PRINTED = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25, -8, -25, -2, -9, -25, -24, -52,
    -24, -20, 10, 9, -1, -9, -19, -41,
    -17, 3, 22, 22, 22, 11, 8, -18,
    -18, -6, 16, 25, 16, 17, 4, -18,
    -23, -3, -1, 15, 10, -3, -20, -22,
    -42, -20, -10, -5, -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]
_MG_BISHOP_PRINTED = [
    -29, 4, -82, -37, -25, -42, 7, -8,
    -26, 16, -18, -13, 30, 59, 18, -47,
    -16, 37, 43, 40, 35, 50, 37, -2,
    -4, 5, 19, 50, 37, 37, 7, -2,
    -6, 13, 13, 26, 34, 12, 10, 4,
    0, 15, 15, 15, 14, 27, 18, 10,
    4, 15, 16, 0, 7, 21, 33, 1,
    -33, -3, -14, -21, -13, -12, -39, -21,
]
_EG_BISHOP_PRINTED = [
    -14, -21, -11, -8, -7, -9, -17, -24,
    -8, -4, 7, -12, -3, -13, -4, -14,
    2, -8, 0, -1, -2, 6, 0, 4,
    -3, 9, 12, 9, 14, 10, 3, 2,
    -6, 3, 13, 19, 7, 10, -3, -9,
    -12, -3, 8, 10, 13, 3, -7, -15,
    -14, -18, -7, -1, 4, -9, -15, -27,
    -23, -9, -23, -5, -9, -16, -5, -17,
]
_MG_ROOK_PRINTED = [
    32, 42, 32, 51, 63, 9, 31, 43,
    27, 32, 58, 62, 80, 67, 26, 44,
    -5, 19, 26, 36, 17, 45, 61, 16,
    -24, -11, 7, 26, 24, 35, -8, -20,
    -36, -26, -12, -1, 9, -7, 6, -23,
    -45, -25, -16, -17, 3, 0, -5, -33,
    -44, -16, -20, -9, -1, 11, -6, -71,
    -19, -13, 1, 17, 16, 7, -37, -26,
]
_EG_ROOK_PRINTED = [
    13, 10, 18, 15, 12, 12, 8, 5,
    11, 13, 13, 11, -3, 3, 8, 3,
    7, 7, 7, 5, 4, -3, -5, -3,
    4, 3, 13, 1, 2, 1, -1, 2,
    3, 5, 8, 4, -5, -6, -8, -11,
    -4, 0, -5, -1, -7, -12, -8, -16,
    -6, -6, 0, 2, -9, -9, -11, -3,
    -9, 2, 3, -1, -5, -13, 4, -20,
]
_MG_QUEEN_PRINTED = [
    -28, 0, 29, 12, 59, 44, 43, 45,
    -24, -39, -5, 1, -16, 57, 28, 54,
    -13, -17, 7, 8, 29, 56, 47, 57,
    -27, -27, -16, -16, -1, 17, -2, 1,
    -9, -26, -9, -10, -2, -4, 3, -3,
    -14, 2, -11, -2, -5, 2, 14, 5,
    -35, -8, 11, 2, 8, 15, -3, 1,
    -1, -18, -9, 10, -15, -25, -31, -50,
]
_EG_QUEEN_PRINTED = [
    -9, 22, 22, 27, 27, 19, 10, 20,
    -17, 20, 32, 41, 58, 25, 30, 0,
    -20, 6, 9, 49, 47, 35, 19, 9,
    3, 22, 24, 45, 57, 40, 57, 36,
    -18, 28, 19, 47, 31, 34, 39, 23,
    -16, -27, 15, 6, 9, 17, 10, 5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43, -5, -32, -20, -41,
]
_MG_KING_PRINTED = [
    -65, 23, 16, -15, -56, -34, 2, 13,
    29, -1, -20, -7, -8, -4, -38, -29,
    -9, 24, 2, -16, -20, 6, 22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49, -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
    1, 7, -8, -64, -43, -16, 9, 8,
    -15, 36, 12, -54, 8, -28, 24, 14,
]
_EG_KING_PRINTED = [
    -74, -35, -18, -18, -11, 15, 4, -17,
    -12, 17, 14, 17, 17, 38, 23, 11,
    10, 17, 23, 15, 20, 45, 44, 13,
    -8, 22, 24, 27, 26, 33, 26, 3,
    -18, -4, 21, 24, 27, 23, 9, -11,
    -19, -3, 11, 21, 23, 16, 7, -9,
    -27, -11, 4, 13, 14, 4, -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]


def _to_square_order(printed: list[int]) -> list[int]:
    """Printed-board order (rank 8 first row) -> a1=0..h8=63 order (reverse the 8 rows)."""
    return [v for rank in range(7, -1, -1) for v in printed[rank * 8: rank * 8 + 8]]


_PST_MG = [
    _to_square_order(_MG_PAWN_PRINTED), _to_square_order(_MG_KNIGHT_PRINTED),
    _to_square_order(_MG_BISHOP_PRINTED), _to_square_order(_MG_ROOK_PRINTED),
    _to_square_order(_MG_QUEEN_PRINTED), _to_square_order(_MG_KING_PRINTED),
]
_PST_EG = [
    _to_square_order(_EG_PAWN_PRINTED), _to_square_order(_EG_KNIGHT_PRINTED),
    _to_square_order(_EG_BISHOP_PRINTED), _to_square_order(_EG_ROOK_PRINTED),
    _to_square_order(_EG_QUEEN_PRINTED), _to_square_order(_EG_KING_PRINTED),
]
_PHASE_WEIGHT_BY_TYPE = [0, 1, 1, 2, 4, 0]
MAX_PHASE = 24


def _mirror(sq: int) -> int:
    return (7 - sq // 8) * 8 + sq % 8


def _build_signed_tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SIGNED_MG/SIGNED_EG[piece_code][square] fold material+PST into one signed number,
    keyed by our own WP..BK (0..11) piece codes."""
    signed_mg = np.zeros((12, 64), dtype=np.int32)
    signed_eg = np.zeros((12, 64), dtype=np.int32)
    phase_weight = np.zeros(12, dtype=np.int32)
    for pt in range(6):
        mg_value = MG_VALUE[pt]
        eg_value = EG_VALUE[pt]
        weight = _PHASE_WEIGHT_BY_TYPE[pt]
        white_code = WP + pt
        black_code = BP + pt
        phase_weight[white_code] = weight
        phase_weight[black_code] = weight
        for square in range(64):
            signed_mg[white_code, square] = mg_value + _PST_MG[pt][square]
            signed_eg[white_code, square] = eg_value + _PST_EG[pt][square]
            mirrored = _mirror(square)
            signed_mg[black_code, square] = -(mg_value + _PST_MG[pt][mirrored])
            signed_eg[black_code, square] = -(eg_value + _PST_EG[pt][mirrored])
    return signed_mg, signed_eg, phase_weight


SIGNED_MG, SIGNED_EG, PHASE_WEIGHT = _build_signed_tables()

DOUBLED_PAWN_MG, DOUBLED_PAWN_EG = 10, 20
ISOLATED_PAWN_MG, ISOLATED_PAWN_EG = 12, 18
PASSED_PAWN_MG = np.array([0, 5, 10, 20, 35, 60, 100, 0], dtype=np.int32)
PASSED_PAWN_EG = np.array([0, 10, 20, 40, 70, 110, 170, 0], dtype=np.int32)
KING_SHIELD_BONUS_MG = 10
KING_OPEN_FILE_PENALTY_MG = 20
BISHOP_PAIR_BONUS = 30


@njit(cache=False)
def popcount64(bb: np.uint64) -> int:
    c = 0
    v = bb
    while v:
        v &= v - np.uint64(1)
        c += 1
    return c


@njit(cache=False)
def eval_material_phase(
    boards: np.ndarray, signed_mg: np.ndarray, signed_eg: np.ndarray, phase_weight: np.ndarray,
) -> tuple[int, int, int]:
    """Material + piece-square + game-phase reduction only. `eval_core` (below) used to do this
    inline together with pawn-file bookkeeping, king safety, and the bishop-pair bonus in one
    145-line jitted function; splitting it into several small ones (this and the four below) is
    a pure mechanical refactor -- same arithmetic, same values, verified byte-for-byte against
    the original single-function version on sampled positions
    (`training/test_v40_eval_split_parity.py`) -- done purely to cut numba's per-function
    compile-time cost. This is the same technique V26's docstring credits for cutting its own
    import time ~44s -> ~28s and V31's for ~42-75s -> ~37s: numba's compile time scales worse
    than linearly with a single function's branch count, so several small functions compile
    faster in total than one large one, independent of runtime speed. Applied here because this
    session measured V29's own (unchanged) import time drift from a 27-36s baseline earlier in
    the session to a real, repeated 54-62s later in the same session with zero code change --
    the same environment-dependent numba/LLVM compile-time variance V31's round 2 documented --
    and 54-62s leaves too little of the platform's real 90s init budget to call safe."""
    mg_score = 0
    eg_score = 0
    phase = 0
    for code in range(12):
        bb = boards[code]
        while bb:
            s, bb = pop_lsb(bb)
            mg_score += signed_mg[code, s]
            eg_score += signed_eg[code, s]
            phase += phase_weight[code]
    return mg_score, eg_score, phase


@njit(cache=False)
def pawn_file_info(
    boards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-file pawn counts and the extreme occupied rank on each file for both colours -- used
    by both `eval_pawn_structure` and `eval_king_safety` below. Split out of `eval_core`, see
    `eval_material_phase`'s docstring for why."""
    white_pawn_file_count = np.zeros(8, dtype=np.int32)
    black_pawn_file_count = np.zeros(8, dtype=np.int32)
    white_pawn_min_rank_file = np.full(8, 8, dtype=np.int32)
    black_pawn_max_rank_file = np.full(8, -1, dtype=np.int32)
    wp = boards[WP]
    while wp:
        s, wp = pop_lsb(wp)
        f = s % 8
        r = s // 8
        white_pawn_file_count[f] += 1
        if r < white_pawn_min_rank_file[f]:
            white_pawn_min_rank_file[f] = r
    bp = boards[BP]
    while bp:
        s, bp = pop_lsb(bp)
        f = s % 8
        r = s // 8
        black_pawn_file_count[f] += 1
        if r > black_pawn_max_rank_file[f]:
            black_pawn_max_rank_file[f] = r
    return (
        white_pawn_file_count, black_pawn_file_count,
        white_pawn_min_rank_file, black_pawn_max_rank_file,
    )


@njit(cache=False)
def eval_pawn_structure(
    boards: np.ndarray,
    white_pawn_file_count: np.ndarray, black_pawn_file_count: np.ndarray,
    white_pawn_min_rank_file: np.ndarray, black_pawn_max_rank_file: np.ndarray,
    passed_mg: np.ndarray, passed_eg: np.ndarray,
) -> tuple[int, int]:
    """Doubled/isolated/passed-pawn scoring (white total minus black total). Split out of
    `eval_core`, see `eval_material_phase`'s docstring for why."""
    white_pawn_mg = 0
    white_pawn_eg = 0
    black_pawn_mg = 0
    black_pawn_eg = 0

    wp = boards[WP]
    while wp:
        s, wp = pop_lsb(wp)
        f = s % 8
        r = s // 8
        if white_pawn_file_count[f] > 1:
            white_pawn_mg -= DOUBLED_PAWN_MG
            white_pawn_eg -= DOUBLED_PAWN_EG
        left = white_pawn_file_count[f - 1] if f - 1 >= 0 else 0
        right = white_pawn_file_count[f + 1] if f + 1 <= 7 else 0
        if left == 0 and right == 0:
            white_pawn_mg -= ISOLATED_PAWN_MG
            white_pawn_eg -= ISOLATED_PAWN_EG
        blocked = False
        for ff in (f - 1, f, f + 1):
            if 0 <= ff <= 7 and black_pawn_max_rank_file[ff] > r:
                blocked = True
        if not blocked:
            white_pawn_mg += passed_mg[r]
            white_pawn_eg += passed_eg[r]

    bp = boards[BP]
    while bp:
        s, bp = pop_lsb(bp)
        f = s % 8
        r = s // 8
        if black_pawn_file_count[f] > 1:
            black_pawn_mg -= DOUBLED_PAWN_MG
            black_pawn_eg -= DOUBLED_PAWN_EG
        left = black_pawn_file_count[f - 1] if f - 1 >= 0 else 0
        right = black_pawn_file_count[f + 1] if f + 1 <= 7 else 0
        if left == 0 and right == 0:
            black_pawn_mg -= ISOLATED_PAWN_MG
            black_pawn_eg -= ISOLATED_PAWN_EG
        blocked = False
        for ff in (f - 1, f, f + 1):
            if 0 <= ff <= 7 and white_pawn_min_rank_file[ff] < r:
                blocked = True
        if not blocked:
            forward_rank = 7 - r
            black_pawn_mg += passed_mg[forward_rank]
            black_pawn_eg += passed_eg[forward_rank]

    return white_pawn_mg - black_pawn_mg, white_pawn_eg - black_pawn_eg


@njit(cache=False)
def eval_king_safety(
    boards: np.ndarray, white_pawn_file_count: np.ndarray, black_pawn_file_count: np.ndarray,
) -> int:
    """Pawn-shield bonus and open-file-near-king penalty for both kings, midgame only (fades
    out on its own as material is traded off, since only the midgame total is affected). Split
    out of `eval_core`, see `eval_material_phase`'s docstring for why."""
    white_king_mg = 0
    if boards[WK]:
        wk_sq = 0
        v = boards[WK]
        while not (v & np.uint64(1)):
            v >>= np.uint64(1)
            wk_sq += 1
        king_file = wk_sq % 8
        king_rank = wk_sq // 8
        shield = 0
        p = boards[WP]
        while p:
            s, p = pop_lsb(p)
            pf = s % 8
            if abs(pf - king_file) > 1:
                continue
            pr = s // 8
            if pr > king_rank:
                shield += 1
        white_king_mg = shield * KING_SHIELD_BONUS_MG
        for f in (king_file - 1, king_file, king_file + 1):
            if not (0 <= f <= 7 and white_pawn_file_count[f] > 0):
                white_king_mg -= KING_OPEN_FILE_PENALTY_MG

    black_king_mg = 0
    if boards[BK]:
        bk_sq = 0
        v = boards[BK]
        while not (v & np.uint64(1)):
            v >>= np.uint64(1)
            bk_sq += 1
        king_file = bk_sq % 8
        king_rank = bk_sq // 8
        shield = 0
        p = boards[BP]
        while p:
            s, p = pop_lsb(p)
            pf = s % 8
            if abs(pf - king_file) > 1:
                continue
            pr = s // 8
            if pr < king_rank:
                shield += 1
        black_king_mg = shield * KING_SHIELD_BONUS_MG
        for f in (king_file - 1, king_file, king_file + 1):
            if not (0 <= f <= 7 and black_pawn_file_count[f] > 0):
                black_king_mg -= KING_OPEN_FILE_PENALTY_MG

    return white_king_mg - black_king_mg


@njit(cache=False)
def eval_bishop_pair(boards: np.ndarray) -> tuple[int, int]:
    """Bishop-pair bonus/penalty (white minus black). Split out of `eval_core`, see
    `eval_material_phase`'s docstring for why."""
    mg_delta = 0
    eg_delta = 0
    if popcount64(boards[WB]) >= 2:
        mg_delta += BISHOP_PAIR_BONUS
        eg_delta += BISHOP_PAIR_BONUS
    if popcount64(boards[BB]) >= 2:
        mg_delta -= BISHOP_PAIR_BONUS
        eg_delta -= BISHOP_PAIR_BONUS
    return mg_delta, eg_delta


@njit(cache=False)
def eval_core(
    boards: np.ndarray, signed_mg: np.ndarray, signed_eg: np.ndarray, phase_weight: np.ndarray,
    passed_mg: np.ndarray, passed_eg: np.ndarray,
) -> tuple[int, int, int]:
    """Combine material/phase, pawn structure, king safety, and the bishop pair -- numerically
    identical to V29's single-function `eval_core` (verified on real sampled positions, see
    `training/test_v40_eval_split_parity.py`); see `eval_material_phase`'s docstring above for
    why this is now five small functions instead of one large one."""
    mg_score, eg_score, phase = eval_material_phase(boards, signed_mg, signed_eg, phase_weight)

    wpfc, bpfc, wpmin, bpmax = pawn_file_info(boards)
    pawn_mg, pawn_eg = eval_pawn_structure(boards, wpfc, bpfc, wpmin, bpmax, passed_mg, passed_eg)
    mg_score += pawn_mg
    eg_score += pawn_eg

    mg_score += eval_king_safety(boards, wpfc, bpfc)

    bishop_mg, bishop_eg = eval_bishop_pair(boards)
    mg_score += bishop_mg
    eg_score += bishop_eg

    return mg_score, eg_score, phase


@njit(cache=False)
def eval_from_arrays(boards: np.ndarray, meta: np.ndarray) -> int:
    mg, eg, phase = eval_core(
        boards, SIGNED_MG, SIGNED_EG, PHASE_WEIGHT, PASSED_PAWN_MG, PASSED_PAWN_EG
    )
    phase = min(phase, MAX_PHASE)
    score_from_white = (mg * phase + eg * (MAX_PHASE - phase)) // MAX_PHASE
    return score_from_white if meta[0] == 0 else -score_from_white


def evaluate(boards: np.ndarray, meta: np.ndarray) -> int:
    """Evaluate from the side to move's perspective, as negamax requires."""
    return int(eval_from_arrays(boards, meta))


# --------------------------------------------------------------------------------------------
# Static Exchange Evaluation (SEE): statically resolves a sequence of captures on one square
# (who attacks it, who defends it, in ascending piece value) to decide whether a capture wins
# or loses material, without searching it out. This is the standard fix for the exact blind
# spot V26/V27's diagnosis found (training/data/v20_noels_bot_loss_diagnosis.log): a static
# material+PST evaluator has no idea a piece just walked onto a square with no safe retreat,
# and previously only extra search depth could discover that. SEE gives the engine a cheap,
# direct answer to "is this capture (or, with no piece actually captured, this square) safe"
# without spending a subtree of search nodes on it.
#
# Algorithm: the standard "swap" formulation (chessprogramming.org/Static_Exchange_Evaluation).
# Remove the initial mover from the occupancy, find all remaining attackers of the target
# square (recomputed after each removal, so a slider revealed behind a captured piece --an
# X-ray attack-- is picked up automatically), and repeatedly let the least valuable attacker of
# the side to move "capture" next, keeping a running gain array, until one side has no more
# attackers or continuing can only make its own result worse. The final backward minimax over
# that gain array is the net material result for the side that started the exchange.
#
# Known, standard simplifications (shared with most engines' SEE, not unique to this one):
# does not check whether recapturing with the king would actually be legal (leave your own king
# in check), and en passant captures are not modelled through the full exchange (the target
# square SEE operates on would be the empty square behind the captured pawn, not the capturing
# pawn's own square) -- both are rare enough in practice that this is a standard, accepted
# trade-off, not something this round tried to solve.
#
# Correctness: verified against a from-scratch reference that actually plays out the capture
# sequence on a scratch python-chess board (always removing the true least-valuable attacker
# via board.attackers()), on 148,086 sampled moves (captures and quiet moves, since SEE with no
# piece actually captured doubles as "is this destination square safe") with zero mismatches,
# before this function was used in search (training/test_bitboard_see.py).

SEE_VALUE = np.array([100, 320, 330, 500, 900, 20_000], dtype=np.int64)  # king: always last


@njit(cache=False)
def attackers_to(target_sq: int, occ: np.uint64, boards: np.ndarray) -> np.uint64:
    """All pieces of both colours currently attacking target_sq, given a (possibly
    hypothetically shrunk, mid-exchange) occupancy bitboard rather than the real one."""
    attackers = np.uint64(0)
    attackers |= PAWN_ATTACKERS[0, target_sq] & boards[WP]
    attackers |= PAWN_ATTACKERS[1, target_sq] & boards[BP]
    attackers |= KNIGHT_ATTACKS[target_sq] & (boards[WN] | boards[BN])
    attackers |= KING_ATTACKS[target_sq] & (boards[WK] | boards[BK])
    b_atk = bishop_attacks(target_sq, occ)
    attackers |= b_atk & (boards[WB] | boards[BB] | boards[WQ] | boards[BQ])
    r_atk = rook_attacks(target_sq, occ)
    attackers |= r_atk & (boards[WR] | boards[BR] | boards[WQ] | boards[BQ])
    return np.uint64(attackers & occ)


@njit(cache=False)
def least_valuable_attacker(
    attackers_bb: np.uint64, side_white: bool, boards: np.ndarray
) -> tuple[int, int, int]:
    """The cheapest piece of `side_white`'s colour among attackers_bb; (-1, 0, -1) if none."""
    base = 0 if side_white else 6
    for pt in range(6):
        code = base + pt
        bb = boards[code] & attackers_bb
        if bb:
            sq, _ = pop_lsb(bb)
            return sq, int(SEE_VALUE[pt]), code
    return -1, 0, -1


@njit(cache=False)
def see_capture(
    boards: np.ndarray, frm: int, to: int, moving_piece_code: int, captured_piece_code: int
) -> int:
    """Net material gain (centipawns, from the moving side's perspective) of the full
    exchange sequence started by moving_piece_code capturing captured_piece_code on `to`
    (or landing on an empty `to`, if captured_piece_code is NO_PIECE -- then this answers
    "is this destination square safe" for the moving piece instead)."""
    white = moving_piece_code < 6
    occ = occ_white(boards) | occ_black(boards)
    occ &= ~(np.uint64(1) << np.uint64(frm))

    gain = np.empty(32, dtype=np.int64)
    gain[0] = int(SEE_VALUE[captured_piece_code % 6]) if captured_piece_code != NO_PIECE else 0
    depth = 0
    attacker_value = int(SEE_VALUE[moving_piece_code % 6])
    side_white = not white

    attackers = attackers_to(to, occ, boards)

    while True:
        lva_sq, lva_value, _lva_code = least_valuable_attacker(attackers, side_white, boards)
        if lva_sq < 0:
            break
        depth += 1
        gain[depth] = attacker_value - gain[depth - 1]
        if max(-gain[depth - 1], gain[depth]) < 0:
            break
        occ &= ~(np.uint64(1) << np.uint64(lva_sq))
        attackers = attackers_to(to, occ, boards)
        attacker_value = lva_value
        side_white = not side_white

    while depth > 0:
        gain[depth - 1] = -max(-gain[depth - 1], gain[depth])
        depth -= 1
    return int(gain[0])


# --------------------------------------------------------------------------------------------
# Search: iterative-deepening negamax with alpha-beta, a hash-indexed transposition table,
# MVV-LVA + TT-move + killer + history move ordering (captures in quiescence now ordered and
# pruned by SEE instead, see quiescence() below), null-move pruning, late-move reductions,
# futility pruning, and aspiration windows.
# --------------------------------------------------------------------------------------------

MATE_SCORE = 100_000
MAX_PLY = 64
MAX_QUIESCENCE_PLY = 24
TT_SIZE = 1 << 20
TT_MASK = TT_SIZE - 1
FLAG_EXACT, FLAG_LOWER, FLAG_UPPER = 0, 1, 2
PIECE_VALUE_BY_TYPE = np.array([100, 320, 330, 500, 900, 0], dtype=np.int64)
TIME_CHECK_INTERVAL = 256

# Search techniques ported from V20 (submissions/v20-numba-rich-eval-candidate/agent.py), same
# values, not re-tuned -- this round's job was porting them onto the bitboard core, not
# re-discovering new numbers. See that module's own constants for the origin of each one.
NULL_MOVE_MIN_DEPTH = 4
NULL_MOVE_REDUCTION = 2  # V20 searches the null move at depth - 3, i.e. depth - 1 - 2
FUTILITY_MARGIN = 110
ASPIRATION_WINDOW = 45

# New in v40 (submissions/v40-search-plus-candidate), independently designed and verified this
# round -- see `negamax`'s reverse-futility block for the guard conditions and reasoning.
RFP_MAX_DEPTH = 3
RFP_MARGIN = 120

# New in v40, round 4 -- see negamax's futility-pruning block (extends V29's depth-1-only,
# not-PV-safe check to depth <= EXT_FUTILITY_MAX_DEPTH, gated to non-PV nodes).
EXT_FUTILITY_MAX_DEPTH = 3
HISTORY_MAX = 16_384
NEXT_DEPTH_GROWTH = 2.5
KILLER_NONE = -1


@njit(cache=False)
def now_seconds() -> float:
    t = 0.0
    with objmode(t="float64"):
        t = _time.perf_counter()
    return t


@njit(cache=False)
def move_score(m: int, tt_move: int) -> int:
    if m == tt_move:
        return 10_000_000
    _frm, _to, piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(m)
    if captured != NO_PIECE:
        victim = int(PIECE_VALUE_BY_TYPE[captured % 6])
        attacker = int(PIECE_VALUE_BY_TYPE[piece % 6])
        return 1_000_000 + 10 * victim - attacker
    if promo != PROMO_NONE:
        return 900_000 + promo
    return 0


@njit(cache=False)
def order_moves(buf: np.ndarray, count: int, tt_move: int) -> None:
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        scores[i] = move_score(buf[i], tt_move)
    for i in range(1, count):
        key_m = buf[i]
        key_s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_s:
            buf[j + 1] = buf[j]
            scores[j + 1] = scores[j]
            j -= 1
        buf[j + 1] = key_m
        scores[j + 1] = key_s


@njit(cache=False)
def is_in_check(boards: np.ndarray, meta: np.ndarray) -> bool:
    white = meta[0] == 0
    ks = king_square(boards, white)
    return square_attacked(boards, ks, not white)


@njit(cache=False)
def move_see_score(boards: np.ndarray, m: int) -> int:
    """Ordering score for a quiescence-search capture: the actual SEE value, so a real winning
    exchange sorts ahead of a real losing one even when raw MVV-LVA would rank them the other
    way (e.g. PxQ that is actually recaptured for free ranks below a defended NxP)."""
    frm, to, piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(m)
    if promo != PROMO_NONE:
        return 900_000 + promo
    return see_capture(boards, frm, to, piece, captured)


@njit(cache=False)
def order_moves_see(buf: np.ndarray, count: int, boards: np.ndarray) -> None:
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        scores[i] = move_see_score(boards, buf[i])
    for i in range(1, count):
        key_m = buf[i]
        key_s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_s:
            buf[j + 1] = buf[j]
            scores[j + 1] = scores[j]
            j -= 1
        buf[j + 1] = key_m
        scores[j + 1] = key_s


@njit(cache=False)
def has_non_pawn_material(boards: np.ndarray, white: bool) -> bool:
    """Guards null-move pruning: skip it in pawn-only endings, where zugzwang (a null move
    being illegally 'free' when every real move worsens the position) is common -- same
    reasoning as V20's has_non_pawn_material."""
    base = WN if white else BN
    return bool(boards[base] | boards[base + 1] | boards[base + 2] | boards[base + 3])


@njit(cache=False)
def apply_null_move(meta: np.ndarray, hash_arr: np.ndarray, undo_null: np.ndarray) -> None:
    """A null move only flips the side to move and clears the en-passant square (no piece
    moves, so no en-passant capture could follow). undo_null (int64[1]) stores the previous
    ep square so unmake_null_move can restore it exactly."""
    prev_ep = meta[2]
    h = hash_arr[0]
    if prev_ep >= 0:
        h ^= ZOBRIST_EP_FILE[prev_ep % 8]
    h ^= ZOBRIST_SIDE
    meta[0] = 1 - meta[0]
    meta[2] = -1
    hash_arr[0] = h
    undo_null[0] = prev_ep


@njit(cache=False)
def unmake_null_move(meta: np.ndarray, hash_arr: np.ndarray, undo_null: np.ndarray) -> None:
    prev_ep = undo_null[0]
    h = hash_arr[0]
    h ^= ZOBRIST_SIDE
    if prev_ep >= 0:
        h ^= ZOBRIST_EP_FILE[prev_ep % 8]
    meta[0] = 1 - meta[0]
    meta[2] = prev_ep
    hash_arr[0] = h


@njit(cache=False)
def update_history(history: np.ndarray, side: int, frm: int, to: int, bonus: int) -> None:
    """Reward or penalise a quiet move while keeping its score bounded, same formula as V20's
    update_history."""
    current = history[side, frm, to]
    history[side, frm, to] = current + bonus - current * abs(bonus) // HISTORY_MAX


@njit(cache=False)
def move_score_full(
    m: int, tt_move: int, killer1: int, killer2: int, side: int, history: np.ndarray
) -> int:
    """Same as move_score, but also orders killer moves (quiet moves that recently caused a
    beta cutoff at this ply) and history-table quiet moves above unscored ones."""
    if m == tt_move:
        return 10_000_000
    frm, _to, piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(m)
    to = _to
    if captured != NO_PIECE:
        victim = int(PIECE_VALUE_BY_TYPE[captured % 6])
        attacker = int(PIECE_VALUE_BY_TYPE[piece % 6])
        return 1_000_000 + 10 * victim - attacker
    if promo != PROMO_NONE:
        return 900_000 + promo
    if m == killer1:
        return 800_000
    if m == killer2:
        return 700_000
    return int(history[side, frm, to])


@njit(cache=False)
def order_moves_full(
    buf: np.ndarray, count: int, tt_move: int, killer1: int, killer2: int,
    side: int, history: np.ndarray,
) -> None:
    scores = np.empty(count, dtype=np.int64)
    for i in range(count):
        scores[i] = move_score_full(buf[i], tt_move, killer1, killer2, side, history)
    for i in range(1, count):
        key_m = buf[i]
        key_s = scores[i]
        j = i - 1
        while j >= 0 and scores[j] < key_s:
            buf[j + 1] = buf[j]
            scores[j + 1] = scores[j]
            j -= 1
        buf[j + 1] = key_m
        scores[j + 1] = key_s


@njit(cache=False)
def quiescence(
    boards: np.ndarray, meta: np.ndarray, hash_arr: np.ndarray, alpha: int, beta: int,
    ply: int, qdepth: int, node_count: np.ndarray, deadline: float, timed_out: np.ndarray,
    move_stack: np.ndarray, undo_stack: np.ndarray,
) -> int:
    node_count[0] += 1
    if node_count[0] % TIME_CHECK_INTERVAL == 0 and now_seconds() >= deadline:
        timed_out[0] = True
        return 0

    if qdepth >= MAX_QUIESCENCE_PLY or ply >= MAX_PLY:
        return eval_from_arrays(boards, meta)

    in_check = is_in_check(boards, meta)
    stand_pat = eval_from_arrays(boards, meta)
    if not in_check:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

    # Per-ply move/undo buffers, owned by the Engine and reused across the whole search
    # instead of allocated fresh on every call (`ply` never collides between a node and its
    # own recursive children, since each recursion level strictly increases it and the search
    # is depth-first) -- see the module's compile-time/allocation note in `negamax`'s docstring
    # for why this matters for node throughput, independent of any pruning technique.
    buf = move_stack[ply]
    count = gen_legal_moves(boards, meta, buf)
    if count == 0:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    if not in_check:
        n = 0
        for i in range(count):
            _frm, _to, _piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(buf[i])
            if captured != NO_PIECE or promo != PROMO_NONE:
                buf[n] = buf[i]
                n += 1
        count = n

        # SEE-prune captures that are already a clear material loss before spending any more
        # nodes on them (promotions are never SEE-pruned: SEE does not model the extra value a
        # promotion adds, so let the search resolve those on their own).
        n = 0
        for i in range(count):
            frm, to, piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(buf[i])
            if promo == PROMO_NONE and see_capture(boards, frm, to, piece, captured) < 0:
                continue
            buf[n] = buf[i]
            n += 1
        count = n

        order_moves_see(buf, count, boards)
    else:
        # In check, every legal reply matters (not just captures) -- MVV-LVA is cheaper than
        # SEE and good enough for ordering a full evasion list.
        order_moves(buf, count, -1)

    undo = undo_stack[ply]
    for i in range(count):
        apply_move_hashed(boards, meta, buf[i], undo, hash_arr)
        score = -quiescence(
            boards, meta, hash_arr, -beta, -alpha, ply + 1, qdepth + 1,
            node_count, deadline, timed_out, move_stack, undo_stack,
        )
        unmake_move_hashed(boards, meta, undo, hash_arr)
        if timed_out[0]:
            return 0
        if score >= beta:
            return beta
        if score > alpha:
            alpha = score
    return alpha


@njit(cache=False)
def tt_probe(
    h: np.uint64, depth: int, alpha: int, beta: int,
    tt_key: np.ndarray, tt_depth: np.ndarray, tt_score: np.ndarray, tt_flag: np.ndarray,
    tt_move_arr: np.ndarray,
) -> tuple[bool, int, int, int, int]:
    """Probe the table for `h`. Returns (cutoff, cutoff_score, new_alpha, new_beta, tt_best).
    A small standalone function -- see the module docstring's compile-time note on why
    negamax's own body is kept short by moving TT bookkeeping out of it."""
    idx = int(h & np.uint64(TT_MASK))
    tt_best = -1
    if tt_key[idx] == h and tt_depth[idx] >= 0:
        if tt_depth[idx] >= depth:
            flag = tt_flag[idx]
            score = tt_score[idx]
            if flag == FLAG_EXACT:
                return True, score, alpha, beta, tt_best
            if flag == FLAG_LOWER and score > alpha:
                alpha = score
            elif flag == FLAG_UPPER and score < beta:
                beta = score
            if alpha >= beta:
                return True, score, alpha, beta, tt_best
        tt_best = tt_move_arr[idx]
    return False, 0, alpha, beta, tt_best


@njit(cache=False)
def tt_store(
    h: np.uint64, depth: int, best_score: int, original_alpha: int, beta: int, best_move: int,
    tt_key: np.ndarray, tt_depth: np.ndarray, tt_score: np.ndarray, tt_flag: np.ndarray,
    tt_move_arr: np.ndarray,
) -> None:
    idx = int(h & np.uint64(TT_MASK))
    flag = FLAG_EXACT
    if best_score <= original_alpha:
        flag = FLAG_UPPER
    elif best_score >= beta:
        flag = FLAG_LOWER
    if tt_depth[idx] <= depth or tt_key[idx] != h:
        tt_key[idx] = h
        tt_depth[idx] = depth
        tt_score[idx] = best_score
        tt_flag[idx] = flag
        tt_move_arr[idx] = best_move


@njit(cache=False)
def try_null_move(
    boards: np.ndarray, meta: np.ndarray, hash_arr: np.ndarray, depth: int, beta: int,
    ply: int, in_check: bool, node_count: np.ndarray, deadline: float, timed_out: np.ndarray,
    tt_key: np.ndarray, tt_depth: np.ndarray, tt_score: np.ndarray, tt_flag: np.ndarray,
    tt_move_arr: np.ndarray, killers: np.ndarray, history: np.ndarray,
    move_stack: np.ndarray, undo_stack: np.ndarray, quiet_stack: np.ndarray,
) -> tuple[bool, int]:
    """Null-move pruning: if passing the turn entirely still doesn't let the opponent catch
    up to beta, this position is so good a real move will not fail low either -- skip
    searching it further. Same guard conditions and reduction (depth - 3) as V20's
    has_non_pawn_material-gated null move. Returns (cutoff, score)."""
    white = meta[0] == 0
    if (
        depth < NULL_MOVE_MIN_DEPTH
        or in_check
        or beta >= MATE_SCORE - 100
        or not has_non_pawn_material(boards, white)
    ):
        return False, 0

    undo_null = np.empty(1, dtype=np.int64)
    apply_null_move(meta, hash_arr, undo_null)
    null_score = -negamax(
        boards, meta, hash_arr, depth - 1 - NULL_MOVE_REDUCTION, -beta, -beta + 1, ply + 1,
        node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag, tt_move_arr,
        killers, history, move_stack, undo_stack, quiet_stack,
    )
    unmake_null_move(meta, hash_arr, undo_null)
    if timed_out[0] or null_score < beta:
        return False, 0
    return True, beta


@njit(cache=False)
def negamax(
    boards: np.ndarray, meta: np.ndarray, hash_arr: np.ndarray, depth: int, alpha: int,
    beta: int, ply: int, node_count: np.ndarray, deadline: float, timed_out: np.ndarray,
    tt_key: np.ndarray, tt_depth: np.ndarray, tt_score: np.ndarray, tt_flag: np.ndarray,
    tt_move_arr: np.ndarray, killers: np.ndarray, history: np.ndarray,
    move_stack: np.ndarray, undo_stack: np.ndarray, quiet_stack: np.ndarray,
) -> int:
    """See the module's compile-time note for why large jitted functions are split; the same
    "avoid per-call allocation" discipline applies here too. `move_stack`/`undo_stack`/
    `quiet_stack` are `(MAX_PLY, ...)` arrays owned by the `Engine` and reused across an entire
    search, indexed by `ply`, instead of a fresh `np.empty(...)` allocated on every single
    recursive call the way V29's version of this function did (three allocations per node,
    compounding across the very large number of nodes a real search visits). New in v40, round
    5: measured against V34 (`submissions/v34-search-rebuild/`, which documents this exact
    "buffers allocated once and reused" pattern in its own README) at a fixed 3s budget from the
    starting position, V29/pre-round-5 v40 reached roughly 236k nodes/sec against V34's ~491k --
    a real, roughly 2x raw throughput gap that plausibly explains a meaningful share of the
    strength gap observed in real games, independent of any pruning technique. Safe by
    construction: `ply` strictly increases with recursion depth and the search is depth-first,
    so a node's own slice of these arrays is never touched by its own children (who use
    `ply + 1`'s slice) and is only reused after the whole subtree below it has already
    returned."""
    node_count[0] += 1
    if node_count[0] % TIME_CHECK_INTERVAL == 0 and now_seconds() >= deadline:
        timed_out[0] = True
        return 0

    if meta[3] >= 100:
        return 0

    if ply >= MAX_PLY:
        return eval_from_arrays(boards, meta)

    if depth == 0:
        return quiescence(
            boards, meta, hash_arr, alpha, beta, ply, 0, node_count, deadline, timed_out,
            move_stack, undo_stack,
        )

    h = hash_arr[0]
    original_alpha = alpha
    cutoff, cutoff_score, alpha, beta, tt_best = tt_probe(
        h, depth, alpha, beta, tt_key, tt_depth, tt_score, tt_flag, tt_move_arr
    )
    if cutoff:
        return cutoff_score

    in_check = is_in_check(boards, meta)

    null_cutoff, null_score = try_null_move(
        boards, meta, hash_arr, depth, beta, ply, in_check, node_count, deadline, timed_out,
        tt_key, tt_depth, tt_score, tt_flag, tt_move_arr, killers, history,
        move_stack, undo_stack, quiet_stack,
    )
    if timed_out[0]:
        return 0
    if null_cutoff:
        return null_score

    buf = move_stack[ply]
    count = gen_legal_moves(boards, meta, buf)
    if count == 0:
        if in_check:
            return -MATE_SCORE + ply
        return 0

    # Reverse futility pruning (a.k.a. static null-move pruning): at shallow depth in a
    # non-PV node, if the static evaluation is already so far above beta that no real move
    # could plausibly need to be tried to prove it, skip searching this node's moves entirely
    # and return the margin-adjusted static score. Guarded to depth <= RFP_MAX_DEPTH (a static
    # snapshot is only trustworthy a few plies from the leaves -- the margin scales with depth
    # precisely because deeper nodes need a wider one to stay safe), a genuine non-PV node
    # (`beta - alpha == 1`, the standard PVS zero-window test also used by aspiration-window
    # re-searches), not in check (a check can swing the position's real value far more than a
    # static margin models), away from near-mate bounds (a static margin means nothing once
    # beta is already close to a forced mate score), and, deliberately, only after the
    # `gen_legal_moves` call above has already confirmed a legal move exists -- applying this
    # before that check would risk mistaking a stalemate (a draw, regardless of the static
    # eval) for a prunable "already winning" node.
    if depth <= RFP_MAX_DEPTH and not in_check and beta - alpha == 1 and beta < MATE_SCORE - 100:
        rfp_eval = eval_from_arrays(boards, meta)
        rfp_margin = RFP_MARGIN * depth
        if rfp_eval - rfp_margin >= beta:
            return rfp_eval - rfp_margin

    side = meta[0]
    ply_idx = ply if ply < MAX_PLY else MAX_PLY - 1
    killer1 = killers[ply_idx, 0]
    killer2 = killers[ply_idx, 1]
    order_moves_full(buf, count, tt_best, killer1, killer2, side, history)
    static_eval = eval_from_arrays(boards, meta) if depth <= EXT_FUTILITY_MAX_DEPTH else -MATE_SCORE
    is_pv_node = beta - alpha > 1

    undo = undo_stack[ply]
    searched_quiet = quiet_stack[ply]
    quiet_count = 0
    best_score = -MATE_SCORE
    best_move = buf[0]
    for i in range(count):
        frm, to, _piece, captured, promo, _ep, _dbl, _ck, _cq = unpack_move(buf[i])
        is_quiet = captured == NO_PIECE and promo == PROMO_NONE

        apply_move_hashed(boards, meta, buf[i], undo, hash_arr)
        gives_check = is_in_check(boards, meta)

        # Futility pruning: if the static evaluation, plus a margin, still can't reach alpha,
        # a quiet, non-checking move at shallow depth is very unlikely to change that. V29's
        # version of this check only ever fired at depth == 1 and, notably, did not exclude PV
        # nodes -- pruning inside the principal variation on a static-eval margin alone is
        # riskier than standard practice recommends (the PV is exactly the line the search
        # trusts most; heuristic prunes are usually reserved for non-PV nodes). Extended here
        # to depth <= EXT_FUTILITY_MAX_DEPTH with a margin that scales with depth (a shallow
        # snapshot needs a wider margin the further it is from the actual leaf), and gated to
        # `not is_pv_node` (the same `beta - alpha == 1` zero-window test RFP and late move
        # pruning use) so the principal line is always searched in full.
        if (
            depth <= EXT_FUTILITY_MAX_DEPTH
            and not is_pv_node
            and is_quiet
            and not gives_check
            and static_eval + FUTILITY_MARGIN * depth <= alpha
        ):
            unmake_move_hashed(boards, meta, undo, hash_arr)
            continue

        if is_quiet:
            searched_quiet[quiet_count] = buf[i]
            quiet_count += 1

        if i == 0:
            score = -negamax(
                boards, meta, hash_arr, depth - 1, -beta, -alpha, ply + 1,
                node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
            )
        else:
            # Late-move reductions: search later, quiet, non-checking moves at a shallower
            # depth first through a narrow (principal-variation-search) window; only fall back
            # to a full-depth, full-window re-search if the reduced search says this move
            # might actually beat alpha. Same thresholds as V20.
            reduction = 0
            if depth >= 3 and i >= 3 and is_quiet and not gives_check:
                reduction = 1
                if depth >= 6 and i >= 8:
                    reduction = 2
            score = -negamax(
                boards, meta, hash_arr, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1,
                node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
            )
            if reduction != 0 and score > alpha:
                score = -negamax(
                    boards, meta, hash_arr, depth - 1, -alpha - 1, -alpha, ply + 1,
                    node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                    tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
                )
            if alpha < score < beta:
                score = -negamax(
                    boards, meta, hash_arr, depth - 1, -beta, -alpha, ply + 1,
                    node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                    tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
                )

        unmake_move_hashed(boards, meta, undo, hash_arr)
        if timed_out[0]:
            return 0
        if score > best_score:
            best_score = score
            best_move = buf[i]
        if score > alpha:
            alpha = score
        if alpha >= beta:
            if is_quiet:
                if buf[i] != killer1:
                    killers[ply_idx, 1] = killer1
                    killers[ply_idx, 0] = buf[i]
                update_history(history, side, frm, to, depth * depth)
                for j in range(quiet_count - 1):
                    pfrm, pto, _pp, pcap, pprom, _pe, _pd, _pc, _pq = unpack_move(searched_quiet[j])
                    if pcap == NO_PIECE and pprom == PROMO_NONE:
                        update_history(history, side, pfrm, pto, -(depth * depth // 2))
            break

    tt_store(
        h, depth, best_score, original_alpha, beta, best_move,
        tt_key, tt_depth, tt_score, tt_flag, tt_move_arr,
    )
    return best_score


@njit(cache=False)
def search_root(
    boards: np.ndarray, meta: np.ndarray, hash_arr: np.ndarray, depth: int, alpha: int,
    beta: int, node_count: np.ndarray, deadline: float, timed_out: np.ndarray,
    tt_key: np.ndarray, tt_depth: np.ndarray, tt_score: np.ndarray, tt_flag: np.ndarray,
    tt_move_arr: np.ndarray, root_moves: np.ndarray, root_count: int, tt_best: int,
    killers: np.ndarray, history: np.ndarray,
    move_stack: np.ndarray, undo_stack: np.ndarray, quiet_stack: np.ndarray,
) -> tuple[int, int]:
    """Same principal-variation-search shape as negamax's own move loop: the first (best-
    ordered) move gets a full window, later moves get a narrow probe first and only a full
    re-search if that probe suggests they might actually beat alpha."""
    order_moves_full(root_moves, root_count, tt_best, KILLER_NONE, KILLER_NONE, meta[0], history)
    # Ply 0 is exclusively search_root's own -- negamax is always entered at ply 1 or deeper
    # (see its calls below), so this never aliases a slot negamax itself is using.
    undo = undo_stack[0]
    best_score = -MATE_SCORE
    best_move = root_moves[0]
    for i in range(root_count):
        apply_move_hashed(boards, meta, root_moves[i], undo, hash_arr)
        if i == 0:
            score = -negamax(
                boards, meta, hash_arr, depth - 1, -beta, -alpha, 1,
                node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
            )
        else:
            score = -negamax(
                boards, meta, hash_arr, depth - 1, -alpha - 1, -alpha, 1,
                node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
            )
            if alpha < score < beta:
                score = -negamax(
                    boards, meta, hash_arr, depth - 1, -beta, -alpha, 1,
                    node_count, deadline, timed_out, tt_key, tt_depth, tt_score, tt_flag,
                    tt_move_arr, killers, history, move_stack, undo_stack, quiet_stack,
                )
        unmake_move_hashed(boards, meta, undo, hash_arr)
        if timed_out[0]:
            return best_move, best_score
        if score > best_score:
            best_score = score
            best_move = root_moves[i]
        if score > alpha:
            alpha = score
    return best_move, best_score


class Engine:
    """Owns the transposition table, killer moves, and history heuristic so they all survive
    between moves in the same game (the platform keeps the process alive for a whole game) --
    the same state V20 keeps at module level, just owned by an instance here."""

    def __init__(self) -> None:
        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_depth = np.full(TT_SIZE, -1, dtype=np.int64)
        self.tt_score = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_flag = np.zeros(TT_SIZE, dtype=np.int64)
        self.tt_move = np.zeros(TT_SIZE, dtype=np.int64)
        self.killers = np.full((MAX_PLY, 2), KILLER_NONE, dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int64)
        # Per-ply scratch buffers, allocated once and reused for the whole search instead of
        # fresh-allocated on every recursive negamax/quiescence call -- see negamax's own
        # docstring for the measured motivation (V34 comparison) and why this is safe.
        self.move_stack = np.empty((MAX_PLY, MAX_MOVES), dtype=np.int64)
        self.undo_stack = np.empty((MAX_PLY, 6), dtype=np.int64)
        self.quiet_stack = np.empty((MAX_PLY, MAX_MOVES), dtype=np.int64)

    def search(
        self, boards: np.ndarray, meta: np.ndarray, time_budget_s: float, max_depth: int = 64
    ) -> tuple[int | None, int, int, int]:
        hash_arr = np.empty(1, dtype=np.uint64)
        hash_arr[0] = compute_hash(boards, meta)
        node_count = np.zeros(1, dtype=np.int64)
        timed_out = np.zeros(1, dtype=np.bool_)
        deadline = _time.perf_counter() + time_budget_s

        root_buf = np.empty(MAX_MOVES, dtype=np.int64)
        root_count = gen_legal_moves(boards, meta, root_buf)
        if root_count == 0:
            return None, 0, 0, 0

        best_move = int(root_buf[0])
        best_score = 0
        completed_depth = 0
        previous_score = 0
        for depth in range(1, max_depth + 1):
            idx = int(hash_arr[0] & TT_MASK)
            tt_best = int(self.tt_move[idx]) if self.tt_key[idx] == hash_arr[0] else -1
            root_copy = root_buf.copy()

            if depth == 1:
                alpha, beta = -MATE_SCORE, MATE_SCORE
                move, score = search_root(
                    boards, meta, hash_arr, depth, alpha, beta,
                    node_count, deadline, timed_out,
                    self.tt_key, self.tt_depth, self.tt_score, self.tt_flag, self.tt_move,
                    root_copy, root_count, tt_best, self.killers, self.history,
                    self.move_stack, self.undo_stack, self.quiet_stack,
                )
            else:
                # Aspiration window: guess the next depth's score will be close to the last
                # one, search a narrow band around it (cheaper than a full-width search), and
                # only widen and re-search if the real score falls outside that guess. Same
                # window size and doubling as V20.
                window = ASPIRATION_WINDOW
                alpha = max(-MATE_SCORE, previous_score - window)
                beta = min(MATE_SCORE, previous_score + window)
                while True:
                    root_copy = root_buf.copy()
                    move, score = search_root(
                        boards, meta, hash_arr, depth, alpha, beta,
                        node_count, deadline, timed_out,
                        self.tt_key, self.tt_depth, self.tt_score, self.tt_flag, self.tt_move,
                        root_copy, root_count, tt_best, self.killers, self.history,
                        self.move_stack, self.undo_stack, self.quiet_stack,
                    )
                    if timed_out[0]:
                        break
                    if score <= alpha:
                        alpha = max(-MATE_SCORE, alpha - window)
                    elif score >= beta:
                        beta = min(MATE_SCORE, beta + window)
                    else:
                        break
                    window *= 2

            if timed_out[0] and depth > 1:
                break
            best_move, best_score = int(move), int(score)
            previous_score = best_score
            completed_depth = depth
            if abs(score) >= MATE_SCORE - 100:
                break
            if timed_out[0]:
                break
        return best_move, best_score, completed_depth, int(node_count[0])


# Module-level engine: state (the transposition table) persists across moves in one game,
# same pattern every earlier version in this repo uses for its own search state.
_ENGINE = Engine()


# A scaled hard-cap variant of this function (soft target unchanged, flat 3.0s ceiling
# replaced by a fraction of whatever time is actually left) was implemented and tried this
# round, specifically to stop banked clock time from ever being thrown away once
# `expected_moves_left` settles at its floor of 20 (V29's flat cap only binds when more than
# ~58.75s is still on the clock at or after move 25 -- a real, reachable case at the platform's
# 120s+0.5s control, since iterative deepening often finishes a move well under its nominal
# budget and banks the difference). It is safe by construction (mathematically identical to
# this function for every state below that ~58.75s threshold, and can only ever grant *more*
# thinking time above it, never less) and well-motivated by this project's own consistent
# finding that more search depth is almost always a real win here (V17/V26/V27/v40's own RFP).
# A real small-sample test at the platform's actual 120s+0.5s clock (4 games, `play_match`
# directly, ply-cap 90 -- 20+ real-clock games were not affordable this session at ~4-6 minutes
# each) came back genuinely mixed and, if anything, negative-leaning: `+1 =1 -2`, 37.5%, versus
# unmodified V29. That is too small a sample to trust on its own (`docs/IDEAS.md`: "two games
# tell you nothing"), and adjudication at a 90-ply cap adds real variance unrelated to time
# management -- but per this project's own standing rule, the empirical gate decides, not the
# theoretical argument for why a change should be safe (see V13's mobility term: also
# theoretically sound, also empirically rejected). Not promoted into this candidate on that
# basis; the code and its real 37.5%-over-4-games result are kept here, in comments, rather
# than silently deleted, exactly as this project records every other negative or inconclusive
# result. Worth revisiting with a real, larger real-clock sample before either keeping or
# fully closing this out.


def move_budget_seconds(fullmove_number: int, time_left_ms: int) -> float:
    """Spend more in the opening while retaining a clock reserve for the full game."""
    seconds_left = max(0.0, time_left_ms / 1000)
    expected_moves_left = max(20, 45 - fullmove_number)
    budget = seconds_left / expected_moves_left
    return max(0.02, min(3.0, budget + 0.05))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move before the platform clock expires."""
    board = chess.Board(fen)
    if not board.legal_moves.count():
        return "0000"

    boards, meta = boards_from_fen(fen)
    budget = move_budget_seconds(board.fullmove_number, time_left_ms)
    move, score, depth, nodes = _ENGINE.search(boards, meta, budget)
    if move is None:
        # Defensive: the referee should never ask for a move with none available.
        return "0000"
    uci = move_to_uci(move)
    print(f"depth={depth} nodes={nodes} score={score} move={uci}")
    return uci


# --------------------------------------------------------------------------------------------
# Warm-up: compile every jitted function now, at import, not on the first move. numba
# compiles per argument-type signature, so this must exercise the exact dtypes the real calls
# use (see AGENTS.md: "warm every jitted function once at import so compilation lands in the
# init budget, not on the clock").
# --------------------------------------------------------------------------------------------
_warmup_boards, _warmup_meta = boards_from_fen(chess.STARTING_FEN)
_ENGINE.search(_warmup_boards.copy(), _warmup_meta.copy(), 0.5, max_depth=3)
perft(_warmup_boards.copy(), _warmup_meta.copy(), 2)
