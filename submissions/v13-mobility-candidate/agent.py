"""A small, readable chess engine for the AI Chessathon submission API.

This is V11 (V6's search plus a tapered, PeSTO-style midgame/endgame evaluator) with one
addition: a mobility term. The search already explores moves, but nothing in the static
evaluator previously scored raw mobility, so two positions with identical material and
piece placement but very different numbers of available legal moves for each side (e.g. a
piece boxed in behind its own pawns) evaluated identically. `evaluate()` is the hottest
function in the engine, called at every leaf node, so this candidate exists to answer one
question empirically: does the accuracy mobility buys back outweigh the extra move
generation it costs (roughly one extra `legal_moves` generation per evaluation), or does the
resulting shallower search lose more than it gains? Measure, don't assume.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import chess

# Scores are centipawns: 100 points is roughly one pawn. Used for move ordering (MVV-LVA)
# only; the static evaluation below uses its own tapered midgame/endgame material values.
PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
MATE_SCORE = 100_000
MAX_QUIESCENCE_DEPTH = 8
HISTORY_MAX = 16_384
NEXT_DEPTH_GROWTH = 2.5
TIME_CHECK_INTERVAL = 64
ASPIRATION_WINDOW = 45
NULL_MOVE_MIN_DEPTH = 4
FUTILITY_MARGIN = 110
MOBILITY_WEIGHT = 3


@dataclass(slots=True)
class TTEntry:
    """One transposition-table result, stored from the side to move's view."""

    depth: int
    score: int
    flag: str  # "exact", "lower", or "upper"
    best_move: chess.Move | None


class SearchTimeout(Exception):
    """Used to leave deeply nested search immediately when the move budget expires."""


# State lasts for a whole game because the platform keeps the Python process alive.
TRANSPOSITION_TABLE: dict[object, TTEntry] = {}
# [colour][from square][to square]. A fixed table is faster than hashing Move objects.
HISTORY = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
KILLERS: dict[int, tuple[chess.Move | None, chess.Move | None]] = {}
DEADLINE = 0.0
NODES = 0

# --- Tapered, PeSTO-style evaluation -----------------------------------------------------
#
# Piece-square tables below are transcribed in the usual "printed board" order: the first
# row is rank 8 (a8..h8) and the last row is rank 1 (a1..h1), each row left to right (a..h).
# This is the standard public-domain layout used by PeSTO and many derivative engines
# (see chessprogramming.org/PeSTO%27s_Evaluation_Function). `_to_square_order` converts that
# printed layout to python-chess's own square numbering (a1=0 ... h8=63) once, at import time,
# so the hot evaluation loop below only ever does plain list indexing.

MG_VALUE = [82, 337, 365, 477, 1025, 0]  # pawn, knight, bishop, rook, queen, king
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
    """Convert a printed-board-order table (rank 8 first row) to python-chess square order
    (a1=0 ... h8=63) by reversing the eight rank rows; each row's own a..h order is unchanged.
    """
    return [
        value
        for rank_index in range(7, -1, -1)
        for value in printed[rank_index * 8 : rank_index * 8 + 8]
    ]


# Index 0 is an unused placeholder so these can be indexed directly by chess.PieceType
# (PAWN=1 .. KING=6), matching PIECE_VALUE's keys.
_EMPTY_TABLE = [0] * 64
MG_PST = [
    _EMPTY_TABLE,
    _to_square_order(_MG_PAWN_PRINTED),
    _to_square_order(_MG_KNIGHT_PRINTED),
    _to_square_order(_MG_BISHOP_PRINTED),
    _to_square_order(_MG_ROOK_PRINTED),
    _to_square_order(_MG_QUEEN_PRINTED),
    _to_square_order(_MG_KING_PRINTED),
]
EG_PST = [
    _EMPTY_TABLE,
    _to_square_order(_EG_PAWN_PRINTED),
    _to_square_order(_EG_KNIGHT_PRINTED),
    _to_square_order(_EG_BISHOP_PRINTED),
    _to_square_order(_EG_ROOK_PRINTED),
    _to_square_order(_EG_QUEEN_PRINTED),
    _to_square_order(_EG_KING_PRINTED),
]

# Standard tapering weights: how much each piece counts toward "still in the middlegame".
PHASE_WEIGHT = [0, 0, 1, 1, 2, 4, 0]  # indexed like MG_PST/EG_PST above
MAX_PHASE = 24


def evaluate(board: chess.Board) -> int:
    """Evaluate from the side to move's perspective, as negamax requires.

    Each piece contributes a midgame score and an endgame score (material plus placement);
    the two totals are blended by a material-based phase estimate. A flat mobility term is
    then added: the difference between the side to move's legal-move count and its
    opponent's, which the search itself never scores directly.
    """
    mg_white = mg_black = 0
    eg_white = eg_black = 0
    phase = 0
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        mg = MG_VALUE[piece_type - 1] + MG_PST[piece_type][index]
        eg = EG_VALUE[piece_type - 1] + EG_PST[piece_type][index]
        if piece.color == chess.WHITE:
            mg_white += mg
            eg_white += eg
        else:
            mg_black += mg
            eg_black += eg
        phase += PHASE_WEIGHT[piece_type]

    phase = min(phase, MAX_PHASE)
    mg_score = mg_white - mg_black
    eg_score = eg_white - eg_black
    score_from_white = (mg_score * phase + eg_score * (MAX_PHASE - phase)) // MAX_PHASE

    # A bishop pair is a small but reliably useful advantage in open positions.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score_from_white += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score_from_white -= 30

    result = score_from_white if board.turn == chess.WHITE else -score_from_white

    own_mobility = len(list(board.legal_moves))
    board.push(chess.Move.null())
    try:
        opponent_mobility = len(list(board.legal_moves))
    finally:
        board.pop()
    result += MOBILITY_WEIGHT * (own_mobility - opponent_mobility)

    return result


def position_key(board: chess.Board) -> object:
    """Move clocks do not affect a chess position, so omit them from the table key."""
    return board._transposition_key()


def history_score(colour: chess.Color, move: chess.Move) -> int:
    """Return how often this quiet move has previously produced a beta cutoff."""
    return HISTORY[int(colour)][move.from_square][move.to_square]


def update_history(colour: chess.Color, move: chess.Move, bonus: int) -> None:
    """Reward or penalise a quiet move while keeping its score bounded."""
    current = history_score(colour, move)
    HISTORY[int(colour)][move.from_square][move.to_square] = (
        current + bonus - current * abs(bonus) // HISTORY_MAX
    )


def move_order_score(
    board: chess.Board, move: chess.Move, ply: int, tt_move: chess.Move | None
) -> int:
    """Put likely best moves first, which makes alpha-beta pruning effective."""
    if move == tt_move:
        return 10_000_000
    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        victim_value = PIECE_VALUE[victim.piece_type] if victim else PIECE_VALUE[chess.PAWN]
        attacker = board.piece_at(move.from_square)
        attacker_value = PIECE_VALUE[attacker.piece_type] if attacker else 0
        return 1_000_000 + 10 * victim_value - attacker_value
    if move.promotion:
        return 900_000 + PIECE_VALUE[move.promotion]
    first_killer, second_killer = KILLERS.get(ply, (None, None))
    if move == first_killer:
        return 800_000
    if move == second_killer:
        return 700_000
    return history_score(board.turn, move)


def ordered_moves(board: chess.Board, ply: int, tt_move: chess.Move | None) -> list[chess.Move]:
    """Generate once, then sort using tactical, table, killer, and history signals."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: move_order_score(board, move, ply, tt_move), reverse=True)
    return moves


def check_time() -> None:
    """Check infrequently enough to be cheap, but often enough to protect the clock."""
    global NODES
    NODES += 1
    if NODES % TIME_CHECK_INTERVAL == 0 and perf_counter() >= DEADLINE:
        raise SearchTimeout


def quiescence(board: chess.Board, alpha: int, beta: int, ply: int, depth: int) -> int:
    """Search forcing captures so we do not evaluate halfway through an exchange."""
    check_time()
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_insufficient_material() or depth >= MAX_QUIESCENCE_DEPTH:
        return evaluate(board)

    stand_pat = evaluate(board)
    if not board.is_check():
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        moves = [move for move in board.legal_moves if board.is_capture(move) or move.promotion]
    else:
        # In check, every legal reply matters; capture-only search would be wrong here.
        moves = list(board.legal_moves)

    moves.sort(key=lambda move: move_order_score(board, move, ply, None), reverse=True)
    for move in moves:
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1, depth + 1)
        finally:
            # A timeout must restore every pushed board before it reaches get_move().
            board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def has_non_pawn_material(board: chess.Board, colour: chess.Color) -> bool:
    """Avoid null-move pruning in pawn-only endings, where zugzwang is common."""
    return bool(
        board.pieces(chess.KNIGHT, colour)
        or board.pieces(chess.BISHOP, colour)
        or board.pieces(chess.ROOK, colour)
        or board.pieces(chess.QUEEN, colour)
    )


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int) -> int:
    """Alpha-beta negamax: one routine works for both white and black."""
    check_time()
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    if depth == 0:
        return quiescence(board, alpha, beta, ply, 0)

    key = position_key(board)
    entry = TRANSPOSITION_TABLE.get(key)
    original_alpha = alpha
    original_beta = beta
    if entry and entry.depth >= depth:
        if entry.flag == "exact":
            return entry.score
        if entry.flag == "lower":
            alpha = max(alpha, entry.score)
        else:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not board.is_check()
        and beta < MATE_SCORE - 100
        and has_non_pawn_material(board, board.turn)
    ):
        board.push(chess.Move.null())
        try:
            null_score = -negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
        finally:
            board.pop()
        if null_score >= beta:
            return beta

    best_score = -MATE_SCORE
    best_move: chess.Move | None = None
    quiet_moves: list[chess.Move] = []
    colour = board.turn
    static_eval = evaluate(board) if depth == 1 else -MATE_SCORE
    moves = ordered_moves(board, ply, entry.best_move if entry else None)
    for move_index, move in enumerate(moves):
        is_quiet = not board.is_capture(move) and not move.promotion
        gives_check = board.gives_check(move)
        if (
            depth == 1
            and is_quiet
            and not gives_check
            and static_eval + FUTILITY_MARGIN <= alpha
        ):
            continue
        if is_quiet:
            quiet_moves.append(move)
        board.push(move)
        try:
            if move_index == 0:
                score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Principal-variation search: most later moves cannot beat alpha.
                reduction = 0
                if depth >= 3 and move_index >= 3 and is_quiet and not gives_check:
                    reduction = 1 + int(depth >= 6 and move_index >= 8)
                score = -negamax(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1)
                if reduction and score > alpha:
                    score = -negamax(board, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            if is_quiet:
                first, _ = KILLERS.get(ply, (None, None))
                KILLERS[ply] = (move, first)
                update_history(colour, move, depth * depth)
                for previous_move in quiet_moves[:-1]:
                    update_history(colour, previous_move, -(depth * depth // 2))
            break

    flag = "exact"
    if best_score <= original_alpha:
        flag = "upper"
    elif best_score >= original_beta:
        flag = "lower"
    TRANSPOSITION_TABLE[key] = TTEntry(depth, best_score, flag, best_move)
    return best_score


def search_root(
    board: chess.Board, depth: int, preferred: chess.Move, alpha: int, beta: int
) -> tuple[chess.Move, int]:
    """Search one complete depth and return its principal-variation first move."""
    best_move = preferred
    best_score = -MATE_SCORE
    entry = TRANSPOSITION_TABLE.get(position_key(board))
    moves = ordered_moves(board, 0, entry.best_move if entry else preferred)
    for move_index, move in enumerate(moves):
        board.push(move)
        try:
            if move_index == 0:
                score = -negamax(board, depth - 1, -beta, -alpha, 1)
            else:
                score = -negamax(board, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -negamax(board, depth - 1, -beta, -alpha, 1)
        finally:
            board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
    return best_move, best_score


def move_budget_seconds(board: chess.Board, time_left_ms: int) -> float:
    """Spend more in the opening while retaining a clock reserve for the full game."""
    seconds_left = max(0.0, time_left_ms / 1000)
    expected_moves_left = max(25, 48 - board.fullmove_number)
    return max(0.05, min(3.0, seconds_left / expected_moves_left + 0.10))


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal UCI move before the platform clock expires."""
    global DEADLINE, NODES
    board = chess.Board(fen)
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        # The referee ends checkmate/stalemate games before asking, but this is defensive.
        return "0000"

    if len(TRANSPOSITION_TABLE) > 150_000:
        TRANSPOSITION_TABLE.clear()
        for colour_history in HISTORY:
            for from_history in colour_history:
                from_history[:] = [0] * 64
        KILLERS.clear()

    preferred = legal_moves[0]
    DEADLINE = perf_counter() + move_budget_seconds(board, time_left_ms)
    NODES = 0
    completed_depth = 0
    previous_score = 0
    try:
        for depth in range(1, 64):
            iteration_started = perf_counter()
            if depth == 1:
                preferred, score = search_root(
                    board, depth, preferred, -MATE_SCORE, MATE_SCORE
                )
            else:
                window = ASPIRATION_WINDOW
                alpha = max(-MATE_SCORE, previous_score - window)
                beta = min(MATE_SCORE, previous_score + window)
                while True:
                    preferred, score = search_root(board, depth, preferred, alpha, beta)
                    if score <= alpha:
                        alpha = max(-MATE_SCORE, alpha - window)
                    elif score >= beta:
                        beta = min(MATE_SCORE, beta + window)
                    else:
                        break
                    window *= 2
            completed_depth = depth
            previous_score = score
            if abs(score) >= MATE_SCORE - 100:
                break
            iteration_seconds = perf_counter() - iteration_started
            time_remaining = DEADLINE - perf_counter()
            if iteration_seconds * NEXT_DEPTH_GROWTH > time_remaining:
                break
    except SearchTimeout:
        pass

    print(f"depth={completed_depth} nodes={NODES} move={preferred.uci()}")
    return preferred.uci()
