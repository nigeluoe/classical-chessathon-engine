"""V44: classical PVS chess engine.

Only the perft-tested board representation and legal move substrate live in
``core.py``.  Move selection here is a new implementation: iterative deepening,
principal-variation search, a transposition table, conservative selective
pruning, and explicit clock management.  The engine does not ponder; the
competition process is suspended while the opponent thinks.
"""

from __future__ import annotations

import time

import chess
import core as c
import numpy as np
from numba import njit, objmode

INF = 32_000
MATE = 30_000
MATE_TT = MATE - 128
MAX_PLY = 96
TT_BITS = 21
TT_SIZE = 1 << TT_BITS
TT_MASK = np.uint64(TT_SIZE - 1)
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
NO_MOVE = -1
HISTORY_LIMIT = 16_384
CLOCK_KEY = np.uint64(0x9E3779B97F4A7C15)
PIECE_VALUE = np.array([100, 320, 330, 500, 900, 20_000], dtype=np.int64)
WHITE_KNIGHT_HOME = np.uint64((1 << 1) | (1 << 6))
BLACK_KNIGHT_HOME = np.uint64((1 << 57) | (1 << 62))
WHITE_BISHOP_HOME = np.uint64((1 << 2) | (1 << 5))
BLACK_BISHOP_HOME = np.uint64((1 << 58) | (1 << 61))


@njit(cache=False)
def wall_time() -> float:
    value = 0.0
    with objmode(value="float64"):
        value = time.perf_counter()
    return value


@njit(cache=False)
def stop_requested(nodes: np.ndarray, stopped: np.ndarray, deadline: float) -> bool:
    nodes[0] += 1
    if (nodes[0] & 1023) == 0 and wall_time() >= deadline:
        stopped[0] = True
    return bool(stopped[0])


@njit(cache=False)
def encode_mate(score: int, ply: int) -> int:
    if score >= MATE_TT:
        return score + ply
    if score <= -MATE_TT:
        return score - ply
    return score


@njit(cache=False)
def decode_mate(score: int, ply: int) -> int:
    if score >= MATE_TT:
        return score - ply
    if score <= -MATE_TT:
        return score + ply
    return score


@njit(cache=False)
def drawn_material(boards: np.ndarray) -> bool:
    heavy = boards[c.WP] | boards[c.BP] | boards[c.WR] | boards[c.BR]
    heavy |= boards[c.WQ] | boards[c.BQ]
    if heavy:
        return False
    minors = boards[c.WN] | boards[c.BN] | boards[c.WB] | boards[c.BB]
    if c.popcount64(minors) <= 1:
        return True
    if boards[c.WN] | boards[c.BN]:
        return False
    dark = np.uint64(0xAA55AA55AA55AA55)
    return not bool(minors & dark) or not bool(minors & ~dark)


@njit(cache=False)
def repeated(
    path: np.ndarray, path_index: int, halfmove: int, root_index: int, barrier: int,
) -> bool:
    matches = 0
    lower = max(barrier, path_index - halfmove)
    for index in range(path_index - 2, lower - 1, -2):
        if path[index] == path[path_index]:
            matches += 1
            # A cycle created inside this search can be repeated at will.  Before
            # the root, two previous occurrences are required for threefold.
            if index >= root_index or matches >= 2:
                return True
    return False


@njit(cache=False)
def static_eval(boards: np.ndarray, meta: np.ndarray) -> int:
    base = c.eval_from_arrays(boards, meta)
    white_adjustment = 0

    # In queenful positions, putting a minor back on its original square is a
    # real loss of time that PSTs alone understate.  The term is symmetric and
    # naturally disappears as the relevant pieces leave their home squares.
    if boards[c.WQ] and boards[c.BQ]:
        white_adjustment -= 14 * c.popcount64(boards[c.WN] & WHITE_KNIGHT_HOME)
        white_adjustment += 14 * c.popcount64(boards[c.BN] & BLACK_KNIGHT_HOME)
        white_adjustment -= 7 * c.popcount64(boards[c.WB] & WHITE_BISHOP_HOME)
        white_adjustment += 7 * c.popcount64(boards[c.BB] & BLACK_BISHOP_HOME)

    # The substrate's shield term counts a pawn anywhere in front of the king.
    # Correct that approximation when a king has committed to a wing: every
    # rank the closest shelter pawn advances becomes progressively more costly.
    for side in range(2):
        king = c.king_square(boards, side == c.WHITE)
        king_file, king_rank = king % 8, king // 8
        if (king_file <= 2 or king_file >= 5) and (king_rank <= 1 or king_rank >= 6):
            pawns = boards[c.WP if side == c.WHITE else c.BP]
            penalty = 0
            for file_index in range(max(0, king_file - 1), min(8, king_file + 2)):
                on_file = pawns & c.FILE_MASKS[file_index]
                if not on_file:
                    continue
                closest_advance = 7
                while on_file:
                    square, on_file = c.pop_lsb(on_file)
                    advance = square // 8 - 1 if side == c.WHITE else 6 - square // 8
                    if 0 <= advance < closest_advance:
                        closest_advance = advance
                penalty += 4 * closest_advance * closest_advance
            white_adjustment += -penalty if side == c.WHITE else penalty

    relative = white_adjustment if meta[0] == c.WHITE else -white_adjustment
    return base + relative + 8  # modest side-to-move initiative


@njit(cache=False)
def move_order(
    moves: np.ndarray,
    scores: np.ndarray,
    count: int,
    tt_move: int,
    killers: np.ndarray,
    history: np.ndarray,
    side: int,
    ply: int,
) -> None:
    for index in range(count):
        move = int(moves[index])
        frm, to, piece, captured, promo, _ep, _dbl, _ck, _cq = c.unpack_move(move)
        if move == tt_move:
            score = 20_000_000
        elif captured != c.NO_PIECE:
            # MVV-LVA is deliberately cheap here; SEE is used only where it can
            # actually prune in quiescence.
            score = 4_000_000 + 32 * PIECE_VALUE[captured % 6] - PIECE_VALUE[piece % 6]
        elif promo:
            score = 3_000_000 + promo * 10_000
        elif move == killers[ply, 0]:
            score = 2_000_000
        elif move == killers[ply, 1]:
            score = 1_900_000
        else:
            score = int(history[side, frm, to])
        scores[index] = score


@njit(cache=False)
def select_move(moves: np.ndarray, scores: np.ndarray, start: int, count: int) -> int:
    best = start
    for index in range(start + 1, count):
        if scores[index] > scores[best]:
            best = index
    moves[start], moves[best] = moves[best], moves[start]
    scores[start], scores[best] = scores[best], scores[start]
    return int(moves[start])


@njit(cache=False)
def qsearch(
    boards: np.ndarray,
    meta: np.ndarray,
    hash_value: np.ndarray,
    alpha: int,
    beta: int,
    ply: int,
    nodes: np.ndarray,
    stopped: np.ndarray,
    deadline: float,
    move_stack: np.ndarray,
    score_stack: np.ndarray,
    undo_stack: np.ndarray,
    path: np.ndarray,
    path_index: int,
    root_index: int,
    barrier: int,
) -> int:
    if stop_requested(nodes, stopped, deadline):
        return 0
    path[path_index] = hash_value[0]
    if repeated(path, path_index, meta[3], root_index, barrier) or meta[3] >= 100:
        return 0
    if drawn_material(boards):
        return 0

    white = meta[0] == c.WHITE
    in_check = c.square_attacked(boards, c.king_square(boards, white), not white)
    stand = static_eval(boards, meta)
    if ply >= MAX_PLY - 1:
        return stand
    if not in_check:
        if stand >= beta:
            return stand
        if stand > alpha:
            alpha = stand

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = c.gen_pseudo_moves(boards, meta, moves)
    kept = 0
    if in_check:
        for index in range(count):
            move = int(moves[index])
            _frm, _to, piece, captured, promo, _ep, _dbl, _ck, _cq = c.unpack_move(move)
            scores[index] = (32 * PIECE_VALUE[captured % 6] - PIECE_VALUE[piece % 6]
                             if captured != c.NO_PIECE else 0) + promo * 10_000
    else:
        for index in range(count):
            move = int(moves[index])
            frm, to, piece, captured, promo, ep, _dbl, _ck, _cq = c.unpack_move(move)
            if captured == c.NO_PIECE and promo == 0:
                continue
            see = 0 if ep or promo else c.see_capture(boards, frm, to, piece, captured)
            moves[kept] = move
            scores[kept] = 100_000 * promo + see
            kept += 1
        count = kept

    legal_count = 0
    best = -MATE + ply if in_check else stand
    for index in range(count):
        move = select_move(moves, scores, index, count)
        frm, to, _piece, captured, promo, _ep, _dbl, _ck, _cq = c.unpack_move(move)
        c.apply_move_hashed(boards, meta, move, undo, hash_value)
        if c.square_attacked(boards, c.king_square(boards, white), not white):
            c.unmake_move_hashed(boards, meta, undo, hash_value)
            continue
        legal_count += 1
        gives_check = c.square_attacked(boards, c.king_square(boards, not white), white)
        if not in_check and not gives_check:
            gain = PIECE_VALUE[captured % 6] if captured != c.NO_PIECE else 0
            if promo:
                gain += PIECE_VALUE[c.promo_piece_code(promo, white) % 6] - 100
            if stand + gain + 120 < alpha or (not ep and not promo and scores[index] < 0):
                c.unmake_move_hashed(boards, meta, undo, hash_value)
                continue
        score = -qsearch(
            boards, meta, hash_value, -beta, -alpha, ply + 1, nodes, stopped,
            deadline, move_stack, score_stack, undo_stack, path, path_index + 1,
            root_index, barrier,
        )
        c.unmake_move_hashed(boards, meta, undo, hash_value)
        if stopped[0]:
            return 0
        if score > best:
            best = score
        if score > alpha:
            alpha = score
            if alpha >= beta:
                return alpha
    if in_check and legal_count == 0:
        return -MATE + ply
    return best


@njit(cache=False)
def search_node(
    boards: np.ndarray,
    meta: np.ndarray,
    hash_value: np.ndarray,
    depth: int,
    alpha: int,
    beta: int,
    ply: int,
    pv_node: bool,
    allow_null: bool,
    nodes: np.ndarray,
    stopped: np.ndarray,
    deadline: float,
    tt_keys: np.ndarray,
    tt_depths: np.ndarray,
    tt_scores: np.ndarray,
    tt_bounds: np.ndarray,
    tt_moves: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    move_stack: np.ndarray,
    score_stack: np.ndarray,
    undo_stack: np.ndarray,
    path: np.ndarray,
    path_index: int,
    root_index: int,
    barrier: int,
    root_move: np.ndarray,
) -> int:
    if depth <= 0:
        return qsearch(
            boards, meta, hash_value, alpha, beta, ply, nodes, stopped, deadline,
            move_stack, score_stack, undo_stack, path, path_index, root_index, barrier,
        )
    if stop_requested(nodes, stopped, deadline):
        return 0
    path[path_index] = hash_value[0]
    if ply and (repeated(path, path_index, meta[3], root_index, barrier)
                or meta[3] >= 100 or drawn_material(boards)):
        return 0
    if ply >= MAX_PLY - 1:
        return static_eval(boards, meta)

    alpha = max(alpha, -MATE + ply)
    beta = min(beta, MATE - ply - 1)
    if alpha >= beta:
        return alpha
    original_alpha = alpha

    key = hash_value[0] ^ (np.uint64(meta[3]) * CLOCK_KEY)
    slot = int(key & TT_MASK)
    tt_move = NO_MOVE
    if tt_keys[slot] == key and tt_depths[slot] >= 0:
        tt_move = int(tt_moves[slot])
        if ply and not pv_node and tt_depths[slot] >= depth:
            cached = decode_mate(int(tt_scores[slot]), ply)
            bound = int(tt_bounds[slot])
            if bound == TT_EXACT:
                return cached
            if bound == TT_LOWER and cached >= beta:
                return cached
            if bound == TT_UPPER and cached <= alpha:
                return cached

    white = meta[0] == c.WHITE
    in_check = c.square_attacked(boards, c.king_square(boards, white), not white)
    if in_check:
        depth += 1
    static = -INF if in_check else static_eval(boards, meta)

    # Forward pruning is restricted to non-PV positions far from mate scores.
    if not pv_node and not in_check and abs(beta) < MATE_TT:
        if depth <= 6 and static - (80 + 85 * depth) >= beta:
            return static
        if depth <= 2 and static + 220 * depth < alpha:
            razor = qsearch(
                boards, meta, hash_value, alpha, beta, ply, nodes, stopped, deadline,
                move_stack, score_stack, undo_stack, path, path_index, root_index, barrier,
            )
            if razor <= alpha:
                return razor

    non_pawns = (boards[c.WN] | boards[c.WB] | boards[c.WR] | boards[c.WQ]
                 if white else boards[c.BN] | boards[c.BB] | boards[c.BR] | boards[c.BQ])
    if (allow_null and not pv_node and not in_check and depth >= 3 and non_pawns
            and static >= beta and abs(beta) < MATE_TT):
        old_side, old_ep, old_hash = meta[0], meta[2], hash_value[0]
        if old_ep >= 0:
            hash_value[0] ^= c.ZOBRIST_EP_FILE[old_ep % 8]
        hash_value[0] ^= c.ZOBRIST_SIDE
        meta[0], meta[2] = 1 - old_side, -1
        reduction = 2 + depth // 4
        score = -search_node(
            boards, meta, hash_value, depth - 1 - reduction, -beta, -beta + 1,
            ply + 1, False, False, nodes, stopped, deadline, tt_keys, tt_depths,
            tt_scores, tt_bounds, tt_moves, killers, history, move_stack, score_stack,
            undo_stack, path, path_index + 1, root_index, path_index + 1, root_move,
        )
        meta[0], meta[2], hash_value[0] = old_side, old_ep, old_hash
        if stopped[0]:
            return 0
        if score >= beta:
            return score

    moves = move_stack[ply]
    scores = score_stack[ply]
    undo = undo_stack[ply]
    count = c.gen_pseudo_moves(boards, meta, moves)
    if ply == 0 and root_move[0] != NO_MOVE:
        tt_move = int(root_move[0])
    move_order(moves, scores, count, tt_move, killers, history, meta[0], ply)

    best_score = -INF
    best_move = NO_MOVE
    legal_count = 0
    quiet_count = 0
    side = int(meta[0])
    for index in range(count):
        move = select_move(moves, scores, index, count)
        frm, to, _piece, captured, promo, _ep, _dbl, _ck, _cq = c.unpack_move(move)
        quiet = captured == c.NO_PIECE and promo == 0
        c.apply_move_hashed(boards, meta, move, undo, hash_value)
        if c.square_attacked(boards, c.king_square(boards, white), not white):
            c.unmake_move_hashed(boards, meta, undo, hash_value)
            continue
        gives_check = c.square_attacked(boards, c.king_square(boards, not white), white)
        legal_count += 1
        if quiet:
            quiet_count += 1

        # Late quiets at shallow non-PV nodes have little chance of improving alpha.
        prune = (
            legal_count > 1
            and not pv_node
            and not in_check
            and quiet
            and not gives_check
            and (
                (depth <= 3 and quiet_count > 3 + depth * depth)
                or (depth <= 2 and static + 100 + 120 * depth <= alpha)
            )
        )
        if prune:
            c.unmake_move_hashed(boards, meta, undo, hash_value)
            continue

        child_depth = depth - 1
        reduction = 0
        if (depth >= 3 and legal_count >= 4 and quiet and not in_check and not gives_check
                and move != killers[ply, 0] and move != killers[ply, 1]):
            reduction = 1
            if depth >= 6 and legal_count >= 8:
                reduction += 1
            if not pv_node and depth >= 9 and legal_count >= 12:
                reduction += 1
            reduction = min(reduction, child_depth - 1)

        if legal_count == 1:
            score = -search_node(
                boards, meta, hash_value, child_depth, -beta, -alpha, ply + 1,
                pv_node, True, nodes, stopped, deadline, tt_keys, tt_depths, tt_scores,
                tt_bounds, tt_moves, killers, history, move_stack, score_stack,
                undo_stack, path, path_index + 1, root_index, barrier, root_move,
            )
        else:
            score = -search_node(
                boards, meta, hash_value, child_depth - reduction, -alpha - 1, -alpha,
                ply + 1, False, True, nodes, stopped, deadline, tt_keys, tt_depths,
                tt_scores, tt_bounds, tt_moves, killers, history, move_stack, score_stack,
                undo_stack, path, path_index + 1, root_index, barrier, root_move,
            )
            if not stopped[0] and score > alpha and reduction:
                score = -search_node(
                    boards, meta, hash_value, child_depth, -alpha - 1, -alpha, ply + 1,
                    False, True, nodes, stopped, deadline, tt_keys, tt_depths, tt_scores,
                    tt_bounds, tt_moves, killers, history, move_stack, score_stack,
                    undo_stack, path, path_index + 1, root_index, barrier, root_move,
                )
            if not stopped[0] and score > alpha and score < beta:
                score = -search_node(
                    boards, meta, hash_value, child_depth, -beta, -alpha, ply + 1,
                    pv_node, True, nodes, stopped, deadline, tt_keys, tt_depths,
                    tt_scores, tt_bounds, tt_moves, killers, history, move_stack,
                    score_stack, undo_stack, path, path_index + 1, root_index,
                    barrier, root_move,
                )
        c.unmake_move_hashed(boards, meta, undo, hash_value)
        if stopped[0]:
            return 0
        if score > best_score:
            best_score, best_move = score, move
            if ply == 0:
                root_move[0] = move
        if score > alpha:
            alpha = score
            if alpha >= beta:
                if quiet:
                    if move != killers[ply, 0]:
                        killers[ply, 1] = killers[ply, 0]
                        killers[ply, 0] = move
                    bonus = min(2_000, depth * depth * 24)
                    old = int(history[side, frm, to])
                    history[side, frm, to] = old + bonus - old * bonus // HISTORY_LIMIT
                break

    if legal_count == 0:
        return -MATE + ply if in_check else 0

    replace = tt_keys[slot] != key or depth >= tt_depths[slot] or best_score >= beta
    if replace:
        tt_keys[slot] = key
        tt_depths[slot] = depth
        tt_scores[slot] = encode_mate(best_score, ply)
        tt_moves[slot] = best_move
        tt_bounds[slot] = (TT_UPPER if best_score <= original_alpha
                           else TT_LOWER if best_score >= beta else TT_EXACT)
    return best_score


class Engine:
    def __init__(self) -> None:
        self.tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_depths = np.full(TT_SIZE, -1, dtype=np.int16)
        self.tt_scores = np.zeros(TT_SIZE, dtype=np.int32)
        self.tt_bounds = np.zeros(TT_SIZE, dtype=np.int8)
        self.tt_moves = np.full(TT_SIZE, NO_MOVE, dtype=np.int64)
        self.killers = np.full((MAX_PLY, 2), NO_MOVE, dtype=np.int64)
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.move_stack = np.empty((MAX_PLY, c.MAX_MOVES), dtype=np.int64)
        self.score_stack = np.empty_like(self.move_stack)
        self.undo_stack = np.empty((MAX_PLY, 6), dtype=np.int64)
        self.path = np.zeros(1024, dtype=np.uint64)

    def clear(self) -> None:
        self.tt_depths.fill(-1)
        self.killers.fill(NO_MOVE)
        self.history.fill(0)

    def search(
        self,
        boards: np.ndarray,
        meta: np.ndarray,
        soft_seconds: float,
        hard_seconds: float,
        past: list[int] | None = None,
        max_depth: int = 64,
    ) -> tuple[int | None, int, int, int]:
        started = time.perf_counter()
        hard_deadline = started + hard_seconds
        hash_value = np.array([c.compute_hash(boards, meta)], dtype=np.uint64)
        recent = (past or [int(hash_value[0])])[-101:]
        if not recent or recent[-1] != int(hash_value[0]):
            recent = [int(hash_value[0])]
        root_index = len(recent) - 1
        self.path[:len(recent)] = recent

        legal = np.empty(c.MAX_MOVES, dtype=np.int64)
        count = c.gen_legal_moves(boards, meta, legal)
        if count == 0:
            return None, 0, 0, 0
        best_move = int(legal[0])
        best_score = 0
        completed_depth = 0
        nodes = np.zeros(1, dtype=np.int64)
        stopped = np.zeros(1, dtype=np.bool_)
        root_move = np.array([best_move], dtype=np.int64)
        self.history //= 2
        stable = 0

        for depth in range(1, min(max_depth, MAX_PLY - 3) + 1):
            if time.perf_counter() >= hard_deadline:
                break
            root_move[0] = best_move
            width = 28 if depth >= 5 else INF
            alpha = max(-INF, best_score - width)
            beta = min(INF, best_score + width)
            while True:
                score = search_node(
                    boards, meta, hash_value, depth, alpha, beta, 0, True, True,
                    nodes, stopped, hard_deadline, self.tt_keys, self.tt_depths,
                    self.tt_scores, self.tt_bounds, self.tt_moves, self.killers,
                    self.history, self.move_stack, self.score_stack, self.undo_stack,
                    self.path, root_index, root_index, 0, root_move,
                )
                if stopped[0] or alpha < score < beta or width >= INF:
                    break
                width = min(INF, width * 2)
                alpha = max(-INF, score - width)
                beta = min(INF, score + width)
            if stopped[0]:
                break
            candidate = int(root_move[0])
            stable = stable + 1 if candidate == best_move else 0
            best_move, best_score, completed_depth = candidate, int(score), depth
            if abs(best_score) >= MATE_TT:
                break
            elapsed = time.perf_counter() - started
            stability_factor = 0.72 if stable >= 3 else 0.86 if stable >= 2 else 1.0
            if elapsed >= soft_seconds * stability_factor:
                break
        return best_move, best_score, completed_depth, int(nodes[0])


def time_budget(time_left_ms: int, fullmove: int) -> tuple[float, float]:
    remaining = max(0.0, time_left_ms / 1000.0)
    reserve = max(0.03, min(0.30, remaining * 0.04))
    usable = max(0.0, remaining - reserve)
    moves_left = max(18, 38 - min(fullmove, 20))
    soft = min(4.0, usable / moves_left + 0.025)
    hard = min(7.0, usable * 0.35, soft * 2.25)
    return min(soft, hard), hard


def board_hash(board: chess.Board) -> int:
    boards, meta = c.boards_from_fen(board.fen())
    return int(c.compute_hash(boards, meta))


_ENGINE = Engine()
_HISTORY: list[int] = []
_AFTER_MOVE: chess.Board | None = None


def observe_position(board: chess.Board) -> None:
    global _AFTER_MOVE
    current = board_hash(board)
    if _AFTER_MOVE is not None:
        for reply in list(_AFTER_MOVE.legal_moves):
            _AFTER_MOVE.push(reply)
            same = _AFTER_MOVE.fen() == board.fen()
            _AFTER_MOVE.pop()
            if same:
                _HISTORY.append(current)
                del _HISTORY[:-101]
                return
        _ENGINE.clear()
    _HISTORY[:] = [current]
    _AFTER_MOVE = None


def get_move(fen: str, time_left_ms: int) -> str:
    global _AFTER_MOVE
    started = time.perf_counter()
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    observe_position(board)
    boards, meta = c.boards_from_fen(fen)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    soft, hard = time_budget(max(0, time_left_ms - elapsed_ms), board.fullmove_number)
    chosen = legal[0]
    score = depth = nodes = 0
    if len(legal) > 1 and hard >= 0.004:
        move, score, depth, nodes = _ENGINE.search(
            boards, meta, soft, hard, past=_HISTORY,
        )
        if move is not None:
            candidate = chess.Move.from_uci(c.move_to_uci(move))
            if candidate in legal:
                chosen = candidate
    print(f"v44 depth={depth} nodes={nodes} score={score} move={chosen.uci()}", flush=True)
    board.push(chosen)
    _HISTORY.append(board_hash(board))
    del _HISTORY[:-101]
    _AFTER_MOVE = board
    return chosen.uci()


# Compile every hot signature during the now-confirmed 90 second init window.
_warm_boards, _warm_meta = c.boards_from_fen(chess.STARTING_FEN)
_warm_result = _ENGINE.search(_warm_boards, _warm_meta, 0.2, 80.0, max_depth=2)
if _warm_result[0] is not None:
    c.move_to_uci(_warm_result[0])
_ENGINE.clear()
