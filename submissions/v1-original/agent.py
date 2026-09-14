"""A small, readable chess engine for the AI Chessathon submission API.

This deliberately classical first version is a dependable baseline for later
improvements such as a learned evaluator, an opening book, or faster move generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import chess

# Scores are centipawns: 100 points is roughly one pawn.
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
HISTORY: dict[chess.Move, int] = {}
KILLERS: dict[int, tuple[chess.Move | None, chess.Move | None]] = {}
DEADLINE = 0.0
NODES = 0


def piece_square_bonus(piece_type: int, square: chess.Square, colour: chess.Color) -> int:
    """Reward basic piece placement without an opaque 64-number lookup table."""
    file_distance = abs(chess.square_file(square) - 3.5)
    rank = chess.square_rank(square)
    forward_rank = rank if colour == chess.WHITE else 7 - rank
    centrality = int(7 - 2 * file_distance - 2 * abs(forward_rank - 3.5))

    if piece_type == chess.PAWN:
        return forward_rank * 9 - int(file_distance * 2)
    if piece_type == chess.KNIGHT:
        return centrality * 4
    if piece_type == chess.BISHOP:
        return centrality * 3
    if piece_type == chess.ROOK:
        return forward_rank * 2
    if piece_type == chess.QUEEN:
        return centrality
    return 0


def evaluate(board: chess.Board) -> int:
    """Evaluate from the side to move's perspective, as negamax requires."""
    white_score = 0
    black_score = 0
    for square, piece in board.piece_map().items():
        score = PIECE_VALUE[piece.piece_type] + piece_square_bonus(
            piece.piece_type, square, piece.color
        )
        if piece.color == chess.WHITE:
            white_score += score
        else:
            black_score += score

    # A bishop pair is a small but reliably useful advantage in open positions.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        white_score += 30
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        black_score += 30

    score_from_white = white_score - black_score
    return score_from_white if board.turn == chess.WHITE else -score_from_white


def position_key(board: chess.Board) -> object:
    """Move clocks do not affect a chess position, so omit them from the table key."""
    return board._transposition_key()


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
    return HISTORY.get(move, 0)


def ordered_moves(board: chess.Board, ply: int, tt_move: chess.Move | None) -> list[chess.Move]:
    """Generate once, then sort using tactical, table, killer, and history signals."""
    moves = list(board.legal_moves)
    moves.sort(key=lambda move: move_order_score(board, move, ply, tt_move), reverse=True)
    return moves


def check_time() -> None:
    """Check infrequently enough to be cheap, but often enough to protect the clock."""
    global NODES
    NODES += 1
    if NODES & 255 == 0 and perf_counter() >= DEADLINE:
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
        score = -quiescence(board, -beta, -alpha, ply + 1, depth + 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


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
    if entry and entry.depth >= depth:
        if entry.flag == "exact":
            return entry.score
        if entry.flag == "lower":
            alpha = max(alpha, entry.score)
        else:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    best_score = -MATE_SCORE
    best_move: chess.Move | None = None
    for move in ordered_moves(board, ply, entry.best_move if entry else None):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, score)
        if alpha >= beta:
            if not board.is_capture(move):
                first, _ = KILLERS.get(ply, (None, None))
                KILLERS[ply] = (move, first)
                HISTORY[move] = HISTORY.get(move, 0) + depth * depth
            break

    flag = "exact"
    if best_score <= original_alpha:
        flag = "upper"
    elif best_score >= beta:
        flag = "lower"
    TRANSPOSITION_TABLE[key] = TTEntry(depth, best_score, flag, best_move)
    return best_score


def search_root(board: chess.Board, depth: int, preferred: chess.Move) -> tuple[chess.Move, int]:
    """Search one complete depth and return its principal-variation first move."""
    alpha = -MATE_SCORE
    beta = MATE_SCORE
    best_move = preferred
    best_score = -MATE_SCORE
    entry = TRANSPOSITION_TABLE.get(position_key(board))
    for move in ordered_moves(board, 0, entry.best_move if entry else preferred):
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, 1)
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
        HISTORY.clear()
        KILLERS.clear()

    preferred = legal_moves[0]
    DEADLINE = perf_counter() + move_budget_seconds(board, time_left_ms)
    NODES = 0
    completed_depth = 0
    try:
        for depth in range(1, 64):
            preferred, score = search_root(board, depth, preferred)
            completed_depth = depth
            if abs(score) >= MATE_SCORE - 100:
                break
    except SearchTimeout:
        pass

    print(f"depth={completed_depth} nodes={NODES} move={preferred.uci()}")
    return preferred.uci()
