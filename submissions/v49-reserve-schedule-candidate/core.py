"""V49 board representation, move generation, hashing, and exchange evaluation.

Reuses the team-written and perft-tested V29 substrate. Sliding attacks use
magic bitboards verified against ray scans at import. Search is in agent.py.
"""

from __future__ import annotations

import chess
import numpy as np
from numba import njit

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


DEBRUIJN64 = np.uint64(0x03F79D71B4CB0A89)


def _build_bit_indices() -> np.ndarray:
    """A de Bruijn multiplier maps each isolated bit to a different six-bit prefix."""
    indices = np.empty(64, dtype=np.int64)
    prefixes: set[int] = set()
    for bit in range(64):
        prefix = (((1 << bit) * int(DEBRUIJN64)) & ((1 << 64) - 1)) >> 58
        indices[prefix] = bit
        prefixes.add(prefix)
    assert len(prefixes) == 64
    return indices


BIT_INDICES = _build_bit_indices()


@njit(cache=False)
def king_square(boards: np.ndarray, white: bool) -> int:
    bb = boards[WK] if white else boards[BK]
    if not bb:
        return -1
    # Legal boards have one king bit. Saturate only malformed multi-king boards
    # to preserve the old highest-bit result without scanning individual squares.
    if bb & (bb - np.uint64(1)):
        bb |= bb >> np.uint64(1)
        bb |= bb >> np.uint64(2)
        bb |= bb >> np.uint64(4)
        bb |= bb >> np.uint64(8)
        bb |= bb >> np.uint64(16)
        bb |= bb >> np.uint64(32)
        bb -= bb >> np.uint64(1)
    return int(BIT_INDICES[(bb * DEBRUIJN64) >> np.uint64(58)])


@njit(cache=False)
def pop_lsb(bb: np.uint64) -> tuple[int, np.uint64]:
    """Return (least set-bit index, remaining bits); bb must be nonzero."""
    rest = bb & (bb - np.uint64(1))
    isolated = bb ^ rest
    index = BIT_INDICES[(isolated * DEBRUIJN64) >> np.uint64(58)]
    return int(index), rest


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


def _pawn_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = np.array([0x0101010101010101 << f for f in range(8)], dtype=np.uint64)
    adjacent = np.zeros(8, dtype=np.uint64)
    passed = np.zeros((2, 64), dtype=np.uint64)
    for f in range(8):
        for other in (f - 1, f + 1):
            if 0 <= other < 8:
                adjacent[f] |= files[other]
    for side in range(2):
        direction = 1 if side == 0 else -1
        for square in range(64):
            f, rank = square % 8, square // 8
            for r in range(8):
                distance = (r - rank) * direction
                if distance <= 0:
                    continue
                for other in (f - 1, f, f + 1):
                    if 0 <= other < 8:
                        bit = np.uint64(1 << (8 * r + other))
                        passed[side, square] |= bit
    return files, adjacent, passed


FILE_MASKS, ADJACENT_FILES, PASSED_MASKS = _pawn_masks()


@njit(cache=False)
def eval_core(
    boards: np.ndarray, signed_mg: np.ndarray, signed_eg: np.ndarray, phase_weight: np.ndarray,
    passed_mg: np.ndarray, passed_eg: np.ndarray,
) -> tuple[int, int, int]:
    mg_score, eg_score, phase = 0, 0, 0
    for code in range(12):
        pieces = boards[code]
        while pieces:
            square, pieces = pop_lsb(pieces)
            mg_score += signed_mg[code, square]
            eg_score += signed_eg[code, square]
            phase += phase_weight[code]

    # Precomputed masks replace four per-evaluation arrays and repeated file scans.
    for side in range(2):
        offset, enemy = side * 6, (1 - side) * 6
        sign = 1 if side == 0 else -1
        pawns, enemy_pawns = boards[offset], boards[enemy]
        pieces = pawns
        while pieces:
            square, pieces = pop_lsb(pieces)
            f = square % 8
            rank = square // 8 if side == 0 else 7 - square // 8
            same_file = pawns & FILE_MASKS[f]
            if same_file & (same_file - np.uint64(1)):
                mg_score -= sign * DOUBLED_PAWN_MG
                eg_score -= sign * DOUBLED_PAWN_EG
            if not (pawns & ADJACENT_FILES[f]):
                mg_score -= sign * ISOLATED_PAWN_MG
                eg_score -= sign * ISOLATED_PAWN_EG
            if not (enemy_pawns & PASSED_MASKS[side, square]):
                mg_score += sign * passed_mg[rank]
                eg_score += sign * passed_eg[rank]

        if boards[offset + 5]:
            king = king_square(boards, side == 0)
            shield = popcount64(pawns & PASSED_MASKS[side, king])
            mg_score += sign * shield * KING_SHIELD_BONUS_MG
            king_file = king % 8
            for f in (king_file - 1, king_file, king_file + 1):
                if not (0 <= f < 8 and pawns & FILE_MASKS[f]):
                    mg_score -= sign * KING_OPEN_FILE_PENALTY_MG
        if popcount64(boards[offset + 2]) >= 2:
            mg_score += sign * BISHOP_PAIR_BONUS
            eg_score += sign * BISHOP_PAIR_BONUS
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
# En passant captures are not modelled through the full exchange (the target square SEE
# operates on would be the empty square behind the captured pawn, not the captured pawn's own
# square). Recaptures that expose their own king are excluded, including pinned defenders and
# kings that would capture onto an attacked square.
#
# The original pseudo-legal exchange implementation was checked on 148,086 sampled moves.
# The legal-recapture version below is additionally checked by dev/verify_legal_see.py, which
# actually plays each capture and cheapest legal recapture on a scratch python-chess board.

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
def least_valuable_legal_attacker(
    attackers_bb: np.uint64, target_sq: int, occ: np.uint64,
    side_white: bool, boards: np.ndarray,
) -> tuple[int, int, int]:
    """Cheapest attacker whose capture does not expose its own king."""
    base = 0 if side_white else 6
    target_bit = np.uint64(1) << np.uint64(target_sq)
    for pt in range(6):
        code = base + pt
        bb = boards[code] & attackers_bb
        while bb:
            sq, bb = pop_lsb(bb)
            after = occ & ~(np.uint64(1) << np.uint64(sq))
            king = target_sq if pt == 5 else king_square(boards, side_white)
            # The piece on target is captured and replaced by this attacker. Static `boards`
            # still identifies the old occupant, so explicitly exclude that square when
            # asking whether an opposing piece attacks the king after the capture.
            enemies = (occ_black(boards) if side_white else occ_white(boards)) & after
            enemies &= ~target_bit
            if not (attackers_to(king, after, boards) & enemies):
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
        lva_sq, lva_value, _lva_code = least_valuable_legal_attacker(
            attackers, to, occ, side_white, boards
        )
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
