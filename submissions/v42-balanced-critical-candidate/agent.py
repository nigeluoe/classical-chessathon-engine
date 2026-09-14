"""V42: V41 conversion search with symmetric critical-position time.

The board/evaluation substrate is the team's V29 code in core.py. This search
is our implementation, not a port of a third-party engine. No model is needed.
"""

from __future__ import annotations

import time

import chess
import core as c
import numpy as np
from numba import njit, objmode

MATE = 100_000
MATE_BOUND = MATE - 128
MAX_PLY = 96
TT_SIZE = 1 << 20
TT_MASK = np.uint64(TT_SIZE - 1)
EXACT, LOWER, UPPER = 0, 1, 2
VALUES = np.array([100, 320, 330, 500, 900, 20000], dtype=np.int64)
HISTORY_MAX = 16384
CLOCK_SALT = np.uint64(0x9E3779B97F4A7C15)
CONVERSION_SCORE = 120
CONVERSION_PIECES = 16
DEFENSE_ALERT_SCORE = 100


@njit(cache=False)
def now() -> float:
    stamp = 0.0
    with objmode(stamp="float64"):
        stamp = time.perf_counter()
    return stamp


@njit(cache=False)
def check_time(nodes: np.ndarray, aborted: np.ndarray, deadline: float) -> bool:
    nodes[0] += 1
    if nodes[0] % 256 == 0 and now() >= deadline:
        aborted[0] = True
    return bool(aborted[0])


@njit(cache=False)
def mate_to_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score + ply
    if score <= -MATE_BOUND:
        return score - ply
    return score


@njit(cache=False)
def mate_from_table(score: int, ply: int) -> int:
    if score >= MATE_BOUND:
        return score - ply
    if score <= -MATE_BOUND:
        return score + ply
    return score


@njit(cache=False)
def insufficient(boards: np.ndarray) -> bool:
    if boards[c.WP] | boards[c.BP] | boards[c.WR] | boards[c.BR] | boards[c.WQ] | boards[c.BQ]:
        return False
    knights = boards[c.WN] | boards[c.BN]
    bishops = boards[c.WB] | boards[c.BB]
    if c.popcount64(knights | bishops) <= 1:
        return True
    if knights:
        return False
    dark = np.uint64(0xAA55AA55AA55AA55)
    return not (bishops & dark) or not (bishops & ~dark)


@njit(cache=False)
def repetition_count(
    path: np.ndarray, sp: int, reversible: int, barrier: int, root_sp: int,
) -> int:
    """Two past occurrences draw; a cycle entirely inside search is also scored as draw.

    A null move advances the barrier, so an artificial pass cannot fabricate repetition.
    One historical occurrence alone is not an automatic draw.
    """
    matches = 0
    for index in range(sp - 2, max(barrier, sp - reversible) - 1, -2):
        if path[index] == path[sp]:
            matches += 1
            if index >= root_sp or matches >= 2:
                return 2
    return matches


@njit(cache=False)
def strict_repetition_count(
    path: np.ndarray, sp: int, reversible: int, barrier: int,
) -> int:
    """Count actual earlier occurrences without the search-cycle shortcut above."""
    matches = 0
    for index in range(sp - 2, max(barrier, sp - reversible) - 1, -2):
        if path[index] == path[sp]:
            matches += 1
            if matches >= 2:
                return matches
    return matches


@njit(cache=False)
def has_legal(
    boards: np.ndarray, meta: np.ndarray, moves: np.ndarray, count: int,
    undo: np.ndarray,
) -> bool:
    white = meta[0] == 0
    for index in range(count):
        c.apply_move(boards, meta, moves[index], undo)
        legal = not c.square_attacked(boards, c.king_square(boards, white), not white)
        c.unmake_move(boards, meta, undo)
        if legal:
            return True
    return False


@njit(cache=False)
def has_claimable_threefold(
    boards: np.ndarray, meta: np.ndarray, hashes: np.ndarray,
    moves: np.ndarray, count: int, undo: np.ndarray,
    path: np.ndarray, sp: int, barrier: int,
) -> bool:
    """Match the referee: a legal move creating a third occurrence is already claimable."""
    white = meta[0] == 0
    for index in range(count):
        c.apply_move_hashed(boards, meta, moves[index], undo, hashes)
        legal = not c.square_attacked(boards, c.king_square(boards, white), not white)
        claimable = False
        if legal:
            path[sp + 1] = hashes[0]
            claimable = strict_repetition_count(path, sp + 1, meta[3], barrier) >= 2
        c.unmake_move_hashed(boards, meta, undo, hashes)
        if claimable:
            return True
    return False


@njit(cache=False)
def score_moves(
    moves: np.ndarray, scores: np.ndarray, count: int, preferred: int,
    killers: np.ndarray, history: np.ndarray, side: int, ply: int,
) -> None:
    for index in range(count):
        move = moves[index]
        frm, to, piece, captured, promo, _ep, _dbl, _ck, _cq = c.unpack_move(move)
        score = int(history[side, frm, to])
        if captured != c.NO_PIECE:
            score = 1_000_000 + 16 * VALUES[captured % 6] - VALUES[piece % 6]
        if promo:
            score += 900_000 + 100 * promo
        if move == killers[ply, 0]:
            score = max(score, 800_000)
        elif move == killers[ply, 1]:
            score = max(score, 700_000)
        if move == preferred:
            score = 10_000_000
        scores[index] = score


@njit(cache=False)
def pick(moves: np.ndarray, scores: np.ndarray, index: int, count: int) -> int:
    best = index
    for other in range(index + 1, count):
        if scores[other] > scores[best]:
            best = other
    moves[index], moves[best] = moves[best], moves[index]
    scores[index], scores[best] = scores[best], scores[index]
    return int(moves[index])


@njit(cache=False)
def qsearch(
    boards: np.ndarray, meta: np.ndarray, hashes: np.ndarray,
    alpha: int, beta: int, ply: int, nodes: np.ndarray, deadline: float,
    aborted: np.ndarray, move_stack: np.ndarray, score_stack: np.ndarray,
    undo_stack: np.ndarray, path: np.ndarray, sp: int, barrier: int, root_sp: int,
) -> int:
    if check_time(nodes, aborted, deadline):
        return 0
    path[sp] = hashes[0]
    if repetition_count(path, sp, meta[3], barrier, root_sp) >= 2 or insufficient(boards):
        return 0
    white = meta[0] == 0
    in_check = c.square_attacked(boards, c.king_square(boards, white), not white)
    moves, scores, undo = move_stack[ply], score_stack[ply], undo_stack[ply]
    count = c.gen_pseudo_moves(boards, meta, moves)
    # Verify at least one move before stand-pat: a static cutoff must not hide stalemate.
    if not has_legal(boards, meta, moves, count, undo):
        return -MATE + ply if in_check else 0
    if (ply and meta[3] >= 7 and sp >= 5
            and has_claimable_threefold(
                boards, meta, hashes, moves, count, undo, path, sp, barrier
            )):
        return 0
    if meta[3] >= 100:
        return 0
    stand = c.eval_from_arrays(boards, meta)
    if ply >= MAX_PLY - 1:
        return stand
    best = -MATE if in_check else stand
    if not in_check:
        if stand >= beta:
            return stand
        alpha = max(alpha, stand)
        kept = 0
        for index in range(count):
            move = moves[index]
            frm, to, piece, captured, promo, ep, _dbl, _ck, _cq = c.unpack_move(move)
            if captured == c.NO_PIECE and promo == 0:
                continue
            exchange = 0
            if promo == 0 and not ep:
                exchange = c.see_capture(boards, frm, to, piece, captured)
            moves[kept] = move
            scores[kept] = exchange + (10000 if promo else 0)
            kept += 1
        count = kept
    else:
        for index in range(count):
            _f, _t, piece, cap, promo, _ep, _db, _ck, _cq = c.unpack_move(moves[index])
            scores[index] = (16 * VALUES[cap % 6] - VALUES[piece % 6]
                             if cap != c.NO_PIECE else 0) + 1000 * promo
    for index in range(count):
        move = pick(moves, scores, index, count)
        c.apply_move_hashed(boards, meta, move, undo, hashes)
        legal = not c.square_attacked(boards, c.king_square(boards, white), not white)
        # Only a losing SEE needs the checking-move exception to pruning.
        if not legal or (not in_check and scores[index] < 0
                         and not c.square_attacked(
                             boards, c.king_square(boards, not white), white)):
            c.unmake_move_hashed(boards, meta, undo, hashes)
            continue
        value = -qsearch(
            boards, meta, hashes, -beta, -alpha, ply + 1, nodes, deadline, aborted,
            move_stack, score_stack, undo_stack, path, sp + 1, barrier, root_sp,
        )
        c.unmake_move_hashed(boards, meta, undo, hashes)
        if aborted[0]:
            return 0
        best = max(best, value)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    return best


@njit(cache=False)
def negamax(
    boards: np.ndarray, meta: np.ndarray, hashes: np.ndarray, depth: int,
    alpha: int, beta: int, ply: int, nodes: np.ndarray, deadline: float,
    aborted: np.ndarray, keys: np.ndarray, depths: np.ndarray, values: np.ndarray,
    flags: np.ndarray, tt_moves: np.ndarray, killers: np.ndarray, history: np.ndarray,
    move_stack: np.ndarray, score_stack: np.ndarray, undo_stack: np.ndarray,
    path: np.ndarray, sp: int, barrier: int, root_sp: int, allow_null: bool,
    root_best: np.ndarray,
) -> int:
    if depth <= 0:
        return qsearch(
            boards, meta, hashes, alpha, beta, ply, nodes, deadline, aborted,
            move_stack, score_stack, undo_stack, path, sp, barrier, root_sp,
        )
    if check_time(nodes, aborted, deadline):
        return 0
    path[sp] = hashes[0]
    repetitions = repetition_count(path, sp, meta[3], barrier, root_sp)
    if ply and (repetitions >= 2 or insufficient(boards)):
        return 0
    # A reusable TT bound already came from a nonterminal position. Probe before
    # move generation and the legality scan, which would duplicate that work.
    # Keep draw/max-ply/mate-window guards: those returns take precedence below.
    bounded_alpha = max(alpha, -MATE + ply)
    bounded_beta = min(beta, MATE - ply - 1)
    is_pv = bounded_beta - bounded_alpha > 1
    claim_check_needed = ply > 0 and meta[3] >= 7 and sp >= 5
    # Include rule-50 state in the TT, but leave it out of repetition position identity.
    key = hashes[0] ^ (np.uint64(meta[3]) * CLOCK_SALT)
    slot = int(key & TT_MASK)
    preferred = -1
    tt_cutoff = False
    tt_value = 0
    if keys[slot] == key and depths[slot] >= 0:
        preferred = tt_moves[slot]
        if (not is_pv and 0 < ply < MAX_PLY - 1 and repetitions == 0
                and meta[3] < 100 and bounded_alpha < bounded_beta and depths[slot] >= depth):
            value = mate_from_table(values[slot], ply)
            if (flags[slot] == EXACT or (flags[slot] == LOWER and value >= bounded_beta)
                    or (flags[slot] == UPPER and value <= bounded_alpha)):
                if not claim_check_needed:
                    return value
                tt_cutoff, tt_value = True, value
    white = meta[0] == 0
    in_check = c.square_attacked(boards, c.king_square(boards, white), not white)
    moves, scores, undo = move_stack[ply], score_stack[ply], undo_stack[ply]
    # Reused per-ply arrays and legality at the point of search avoid making every move twice.
    count = c.gen_pseudo_moves(boards, meta, moves)
    if not has_legal(boards, meta, moves, count, undo):
        return -MATE + ply if in_check else 0
    if (claim_check_needed
            and has_claimable_threefold(
                boards, meta, hashes, moves, count, undo, path, sp, barrier
            )):
        return 0
    if meta[3] >= 100:
        return 0
    if ply >= MAX_PLY - 1:
        return c.eval_from_arrays(boards, meta)
    alpha, beta = bounded_alpha, bounded_beta
    if alpha >= beta:
        return alpha
    if tt_cutoff:
        return tt_value
    original_alpha = alpha
    if ply == 0 and root_best[0] >= 0:
        preferred = root_best[0]
    static = c.eval_from_arrays(boards, meta) if not in_check else -MATE
    if (not is_pv and not in_check and abs(beta) < MATE_BOUND
            and depth <= 3 and static - 150 * depth >= beta):
        return static
    non_pawns = (boards[c.WN] | boards[c.WB] | boards[c.WR] | boards[c.WQ]
                 if white else boards[c.BN] | boards[c.BB] | boards[c.BR] | boards[c.BQ])
    if (allow_null and not is_pv and not in_check and depth >= 4 and non_pawns
            and static >= beta and abs(beta) < MATE_BOUND):
        old_ep, old_hash = meta[2], hashes[0]
        if old_ep >= 0:
            hashes[0] ^= c.ZOBRIST_EP_FILE[old_ep % 8]
        hashes[0] ^= c.ZOBRIST_SIDE
        meta[0], meta[2] = 1 - meta[0], -1
        value = -negamax(
            boards, meta, hashes, depth - 3 - depth // 6, -beta, -beta + 1,
            ply + 1, nodes, deadline, aborted, keys, depths, values, flags, tt_moves,
            killers, history, move_stack, score_stack, undo_stack,
            path, sp + 1, sp + 1, root_sp, False, root_best,
        )
        meta[0], meta[2], hashes[0] = 1 - meta[0], old_ep, old_hash
        if aborted[0]:
            return 0
        if value >= beta:
            return beta if value >= MATE_BOUND else value
    score_moves(moves, scores, count, preferred, killers, history, meta[0], ply)
    best, best_move, searched = -MATE, -1, 0
    side = meta[0]
    for index in range(count):
        move = pick(moves, scores, index, count)
        frm, to, _piece, captured, promo, _ep, _db, _ck, _cq = c.unpack_move(move)
        quiet = captured == c.NO_PIECE and promo == 0
        c.apply_move_hashed(boards, meta, move, undo, hashes)
        if c.square_attacked(boards, c.king_square(boards, white), not white):
            c.unmake_move_hashed(boards, meta, undo, hashes)
            continue
        gives_check = c.square_attacked(boards, c.king_square(boards, not white), white)
        # Always search one legal move. Evasions and PV nodes must not be futility-pruned.
        if (searched > 0 and not is_pv and not in_check and quiet and not gives_check
                and depth == 1 and static + 140 <= alpha and abs(alpha) < MATE_BOUND):
            c.unmake_move_hashed(boards, meta, undo, hashes)
            continue
        child_depth = depth - 1
        reduction = 0
        if (depth >= 3 and searched >= 3 and quiet and not in_check and not gives_check
                and move != killers[ply, 0] and move != killers[ply, 1]):
            reduction = 1 + int(depth >= 6 and searched >= 8 and not is_pv)
        if searched == 0:
            value = -negamax(
                boards, meta, hashes, child_depth, -beta, -alpha, ply + 1,
                nodes, deadline, aborted, keys, depths, values, flags, tt_moves,
                killers, history, move_stack, score_stack, undo_stack,
                path, sp + 1, barrier, root_sp, True, root_best,
            )
        else:
            value = -negamax(
                boards, meta, hashes, child_depth - reduction, -alpha - 1, -alpha, ply + 1,
                nodes, deadline, aborted, keys, depths, values, flags, tt_moves,
                killers, history, move_stack, score_stack, undo_stack,
                path, sp + 1, barrier, root_sp, True, root_best,
            )
            if value > alpha and (reduction > 0 or value < beta):
                value = -negamax(
                    boards, meta, hashes, child_depth, -beta, -alpha, ply + 1,
                    nodes, deadline, aborted, keys, depths, values, flags, tt_moves,
                    killers, history, move_stack, score_stack, undo_stack,
                    path, sp + 1, barrier, root_sp, True, root_best,
                )
        c.unmake_move_hashed(boards, meta, undo, hashes)
        if aborted[0]:
            return 0
        searched += 1
        if value > best:
            best, best_move = value, move
            if ply == 0:
                root_best[0] = move
        alpha = max(alpha, value)
        if alpha >= beta:
            if quiet:
                if move != killers[ply, 0]:
                    killers[ply, 1], killers[ply, 0] = killers[ply, 0], move
                bonus = min(2000, depth * depth * 16)
                current = history[side, frm, to]
                history[side, frm, to] = current + bonus - current * bonus // HISTORY_MAX
            break
    if repetitions == 0:
        keys[slot], depths[slot] = key, depth
        values[slot], tt_moves[slot] = mate_to_table(best, ply), best_move
        flags[slot] = UPPER if best <= original_alpha else LOWER if best >= beta else EXACT
    return best


class Engine:
    def __init__(self) -> None:
        self.keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.depths = np.full(TT_SIZE, -1, dtype=np.int16)
        self.values = np.zeros(TT_SIZE, dtype=np.int32)
        self.flags = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_moves = np.full(TT_SIZE, -1, dtype=np.int64)
        self.killers = np.full((MAX_PLY, 2), -1, dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int64)
        self.move_stack = np.empty((MAX_PLY, c.MAX_MOVES), dtype=np.int64)
        self.score_stack = np.empty_like(self.move_stack)
        self.undo_stack = np.empty((MAX_PLY, 6), dtype=np.int64)
        self.path = np.zeros(1024, dtype=np.uint64)

    def search(
        self, boards: np.ndarray, meta: np.ndarray, time_budget_s: float,
        max_depth: int = 64, past: list[int] | None = None, soft_budget_s: float | None = None,
    ) -> tuple[int | None, int, int, int]:
        start = time.perf_counter()
        hashes = np.array([c.compute_hash(boards, meta)], dtype=np.uint64)
        recent = (past or [int(hashes[0])])[-101:]
        if recent[-1] != int(hashes[0]):
            recent = [int(hashes[0])]
        self.path[:len(recent)] = recent
        root_sp = len(recent) - 1
        nodes = np.zeros(1, dtype=np.int64)
        aborted = np.zeros(1, dtype=np.bool_)
        legal = np.empty(c.MAX_MOVES, dtype=np.int64)
        count = c.gen_legal_moves(boards, meta, legal)
        if count == 0:
            white = meta[0] == 0
            checked = c.square_attacked(boards, c.king_square(boards, white), not white)
            return None, -MATE if checked else 0, 0, 0
        best_move, best_score, completed = int(legal[0]), 0, 0
        root_best = np.array([best_move], dtype=np.int64)
        soft = time_budget_s if soft_budget_s is None else soft_budget_s
        self.history //= 2
        stable = 0
        for depth in range(1, min(max_depth, MAX_PLY - 2) + 1):
            if time.perf_counter() >= start + time_budget_s:
                break
            width = 35 if depth >= 4 else MATE * 2
            alpha, beta = max(-MATE, best_score - width), min(MATE, best_score + width)
            while True:
                score = negamax(
                    boards, meta, hashes, depth, alpha, beta, 0, nodes,
                    start + time_budget_s, aborted, self.keys, self.depths, self.values,
                    self.flags, self.tt_moves, self.killers, self.history,
                    self.move_stack, self.score_stack, self.undo_stack,
                    self.path, root_sp, 0, root_sp, True, root_best,
                )
                if aborted[0] or alpha < score < beta or width >= MATE * 2:
                    break
                width *= 2
                alpha, beta = max(-MATE, score - width), min(MATE, score + width)
            if aborted[0]:
                break
            stable = stable + 1 if int(root_best[0]) == best_move else 0
            best_move, best_score, completed = int(root_best[0]), int(score), depth
            if abs(best_score) >= MATE_BOUND:
                break
            elapsed = time.perf_counter() - start
            if elapsed >= soft * (0.75 if stable >= 3 else 1.0):
                break
        return best_move, best_score, completed, int(nodes[0])


def budgets(time_left_ms: int, fullmove: int) -> tuple[float, float]:
    remaining = max(0.0, time_left_ms / 1000.0)
    usable = max(0.0, remaining - max(0.025, min(0.25, remaining * 0.05)))
    # No fixed reserve cliff and no minimum larger than the clock. Increment is not an input.
    soft = min(3.5, usable / max(22, 48 - fullmove) + 0.02)
    hard = min(6.0, soft * 2.5, usable * 0.4)
    return min(soft, hard), hard


def conversion_budgets(
    soft: float, hard: float, time_left_ms: int, static: int,
    pieces: int, check_streak: int,
) -> tuple[float, float]:
    """Spend more reserve when a low-material edge needs conversion or defence."""
    if abs(static) < CONVERSION_SCORE or pieces > CONVERSION_PIECES:
        return soft, hard
    remaining = max(0.0, time_left_ms / 1000.0)
    usable = max(0.0, remaining - max(0.025, min(0.25, remaining * 0.05)))
    factor = 10.0 if check_streak >= 2 else 8.0
    extended_soft = min(6.0, usable * 0.70, soft * factor)
    extended_hard = min(6.0, usable * 0.85, max(hard, extended_soft * 1.5))
    return min(extended_soft, extended_hard), extended_hard


def defensive_budgets(
    soft: float, hard: float, time_left_ms: int, static: int,
    previous: int | None, pieces: int,
) -> tuple[float, float]:
    """Deepen when static and the prior search agree on a narrow defensive alert."""
    if (pieces <= CONVERSION_PIECES or previous is None
            or not (-CONVERSION_SCORE <= static <= -DEFENSE_ALERT_SCORE)
            or not (-CONVERSION_SCORE <= previous <= -DEFENSE_ALERT_SCORE)):
        return soft, hard
    remaining = max(0.0, time_left_ms / 1000.0)
    usable = max(0.0, remaining - max(0.025, min(0.25, remaining * 0.05)))
    extended_soft = min(6.0, usable * 0.32, soft * 6.5)
    new_soft = max(soft, extended_soft)
    extended_hard = min(6.0, usable * 0.48, max(hard, new_soft * 1.5))
    new_hard = max(hard, extended_hard)
    return min(new_soft, new_hard), new_hard


def critical_score(static: int, previous: int | None) -> int:
    """Keep one-turn tactical urgency when static evaluation hides the last search score."""
    if previous is not None and abs(previous) > abs(static):
        return previous
    return static


def board_key(board: chess.Board) -> int:
    boards, meta = c.boards_from_fen(board.fen())
    return int(c.compute_hash(boards, meta))


_ENGINE = Engine()
_PAST: list[int] = []
_AFTER_OUR_MOVE: chess.Board | None = None
_CHECK_STREAK = 0
_LAST_SCORE: int | None = None
_DEFENSE_COOLDOWN = 0


def observe(board: chess.Board) -> None:
    global _AFTER_OUR_MOVE, _CHECK_STREAK, _LAST_SCORE, _DEFENSE_COOLDOWN
    key = board_key(board)
    if _AFTER_OUR_MOVE is not None:
        for reply in list(_AFTER_OUR_MOVE.legal_moves):
            _AFTER_OUR_MOVE.push(reply)
            matches = _AFTER_OUR_MOVE.fen() == board.fen()
            _AFTER_OUR_MOVE.pop()
            if matches:
                _PAST.append(key)
                del _PAST[:-101]
                return
        _ENGINE.depths.fill(-1)
    _PAST[:] = [key]
    _AFTER_OUR_MOVE = None
    _CHECK_STREAK = 0
    _LAST_SCORE = None
    _DEFENSE_COOLDOWN = 0


def get_move(fen: str, time_left_ms: int) -> str:
    global _AFTER_OUR_MOVE, _CHECK_STREAK, _LAST_SCORE, _DEFENSE_COOLDOWN
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    started = time.perf_counter()
    observe(board)
    boards, meta = c.boards_from_fen(fen)
    soft, hard = budgets(time_left_ms - int((time.perf_counter() - started) * 1000),
                         board.fullmove_number)
    _CHECK_STREAK = _CHECK_STREAK + 1 if board.is_check() else 0
    static = c.eval_from_arrays(boards, meta)
    pieces = sum(c.popcount64(boards[code]) for code in range(12))
    urgency = critical_score(static, _LAST_SCORE)
    soft, hard = conversion_budgets(
        soft, hard, time_left_ms, urgency, pieces, _CHECK_STREAK,
    )
    if _DEFENSE_COOLDOWN > 0:
        _DEFENSE_COOLDOWN -= 1
    elif len(legal) > 1:
        original_soft, original_hard = soft, hard
        soft, hard = defensive_budgets(
            soft, hard, time_left_ms, static, _LAST_SCORE, pieces
        )
        if soft > original_soft or hard > original_hard:
            _DEFENSE_COOLDOWN = 2
    chosen = legal[0]
    searched_score: int | None = None
    if len(legal) > 1 and hard >= 0.003:
        move, score, depth, nodes = _ENGINE.search(boards, meta, hard, past=_PAST,
                                                  soft_budget_s=soft)
        searched_score = score
        if move is not None:
            candidate = chess.Move.from_uci(c.move_to_uci(move))
            if candidate in legal:
                chosen = candidate
        print(f"depth={depth} nodes={nodes} score={score} move={chosen.uci()}", flush=True)
    _LAST_SCORE = searched_score
    board.push(chosen)
    _PAST.append(board_key(board))
    _AFTER_OUR_MOVE = board
    return chosen.uci()


# Compile exactly the signatures used during play, then discard warm-up search state.
_warm_boards, _warm_meta = c.boards_from_fen(chess.STARTING_FEN)
_ENGINE.search(_warm_boards, _warm_meta, 0.3, max_depth=4)
_warm_result = _ENGINE.search(_warm_boards, _warm_meta, 5.0, max_depth=4)
if _warm_result[0] is not None:
    c.move_to_uci(_warm_result[0])
board_key(chess.Board())
_ENGINE.depths.fill(-1)
_ENGINE.history.fill(0)
_ENGINE.killers.fill(-1)
