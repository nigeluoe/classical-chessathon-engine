"""A small, readable chess engine for the AI Chessathon submission API.

This deliberately classical first version is a dependable baseline for later
improvements such as a learned evaluator, an opening book, or faster move generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import chess
import numba
import numpy as np

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
HISTORY_MAX = 16_384
NEXT_DEPTH_GROWTH = 2.5
TIME_CHECK_INTERVAL = 64
NNUE_FEATURE_COUNT = 20_480
NNUE_RESIDUAL_WEIGHT = 0.50


@dataclass(slots=True)
class TTEntry:
    """One transposition-table result, stored from the side to move's view."""

    depth: int
    score: int
    flag: str  # "exact", "lower", or "upper"
    best_move: chess.Move | None


@dataclass(slots=True)
class EvalState:
    """Two incremental king-relative accumulators, always from White's view."""

    white: np.ndarray
    black: np.ndarray


class SearchTimeout(Exception):
    """Used to leave deeply nested search immediately when the move budget expires."""


# State lasts for a whole game because the platform keeps the Python process alive.
TRANSPOSITION_TABLE: dict[object, TTEntry] = {}
# [colour][from square][to square]. A fixed table is faster than hashing Move objects.
HISTORY = [[[0 for _ in range(64)] for _ in range(64)] for _ in range(2)]
KILLERS: dict[int, tuple[chess.Move | None, chess.Move | None]] = {}
DEADLINE = 0.0
NODES = 0


def load_nnue_weights() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load team-trained weights once, inside the platform's import-time budget."""
    with np.load(Path(__file__).with_name("incremental_nnue.npz")) as weights:
        if int(weights["feature_count"][0]) != NNUE_FEATURE_COUNT:
            raise ValueError("Unexpected NNUE feature count")
        return (
            weights["embedding"].astype(np.float32),
            weights["hidden_weight"].astype(np.float32),
            weights["hidden_bias"].astype(np.float32),
            weights["output_weight"].astype(np.float32),
            weights["output_bias"].astype(np.float32),
        )


NNUE_EMBEDDING, NNUE_HIDDEN_WEIGHT, NNUE_HIDDEN_BIAS, NNUE_OUTPUT_WEIGHT, NNUE_OUTPUT_BIAS = (
    load_nnue_weights()
)


@numba.njit(cache=False)
def nnue_output(
    white_accumulator: np.ndarray,
    black_accumulator: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
    output_weight: np.ndarray,
    output_bias: np.ndarray,
) -> float:
    """Run the tiny dense tail without Python or temporary NumPy arrays."""
    score = output_bias[0]
    accumulator_size = len(white_accumulator)
    for hidden_index in range(len(hidden_bias)):
        hidden_value = hidden_bias[hidden_index]
        for accumulator_index in range(accumulator_size):
            white_value = white_accumulator[accumulator_index]
            black_value = black_accumulator[accumulator_index]
            if white_value > 0.0:
                hidden_value += hidden_weight[hidden_index, accumulator_index] * white_value
            if black_value > 0.0:
                hidden_value += (
                    hidden_weight[hidden_index, accumulator_size + accumulator_index] * black_value
                )
        if hidden_value > 0.0:
            score += output_weight[0, hidden_index] * hidden_value
    return score


# Compile before the game clock starts; the platform grants a separate import budget.
nnue_output(
    np.zeros(NNUE_EMBEDDING.shape[1], dtype=np.float32),
    np.zeros(NNUE_EMBEDDING.shape[1], dtype=np.float32),
    NNUE_HIDDEN_WEIGHT,
    NNUE_HIDDEN_BIAS,
    NNUE_OUTPUT_WEIGHT,
    NNUE_OUTPUT_BIAS,
)


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


def evaluate_v2(board: chess.Board) -> int:
    """Return the original V2 static score from the side to move's perspective."""
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


def nnue_square(square: chess.Square, perspective: chess.Color, mirror_files: bool) -> int:
    """Orient a square exactly as the offline residual model was trained."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    if perspective == chess.BLACK:
        rank_index = 7 - rank_index
    if mirror_files:
        file_index = 7 - file_index
    return chess.square(file_index, rank_index)


def nnue_feature(
    board: chess.Board, piece: chess.Piece, square: chess.Square, perspective: chess.Color
) -> int:
    """Encode one non-king piece from a side's king-relative perspective."""
    king_square = board.king(perspective)
    if king_square is None:
        raise ValueError("Cannot evaluate a board without both kings")
    oriented_king = nnue_square(king_square, perspective, False)
    mirror_files = chess.square_file(oriented_king) < 4
    canonical_king = nnue_square(king_square, perspective, mirror_files)
    king_bucket = chess.square_rank(canonical_king) * 4 + chess.square_file(canonical_king) - 4
    category = piece.piece_type - 1
    if piece.color != perspective:
        category += 5
    return (king_bucket * 10 + category) * 64 + nnue_square(square, perspective, mirror_files)


def build_accumulator(board: chess.Board, perspective: chess.Color) -> np.ndarray:
    """Build one accumulator; used only at a root or after that king moves."""
    accumulator = np.zeros(NNUE_EMBEDDING.shape[1], dtype=np.float32)
    for square, piece in board.piece_map().items():
        if piece.piece_type != chess.KING:
            accumulator += NNUE_EMBEDDING[nnue_feature(board, piece, square, perspective)]
    return accumulator


def build_eval_state(board: chess.Board) -> EvalState:
    """Build both sides' state once at the start of each iterative-deepening pass."""
    return EvalState(build_accumulator(board, chess.WHITE), build_accumulator(board, chess.BLACK))


def adjust_piece(
    board: chess.Board,
    state: EvalState,
    piece: chess.Piece | None,
    square: chess.Square,
    direction: float,
) -> None:
    """Add or remove a non-king piece in both unchanged-king accumulators."""
    if piece is None or piece.piece_type == chess.KING:
        return
    state.white += direction * NNUE_EMBEDDING[nnue_feature(board, piece, square, chess.WHITE)]
    state.black += direction * NNUE_EMBEDDING[nnue_feature(board, piece, square, chess.BLACK)]


def push_with_eval(board: chess.Board, move: chess.Move, state: EvalState) -> EvalState:
    """Push a move and update only the features that the move can affect."""
    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError("Legal move has no piece on its source square")
    captured_square = move.to_square
    if board.is_en_passant(move):
        captured_square += -8 if board.turn == chess.WHITE else 8
    captured_piece = board.piece_at(captured_square)
    is_castling = board.is_castling(move)
    rook_from = rook_to = None
    if is_castling:
        rank = chess.square_rank(move.from_square)
        rook_from = chess.square(7 if chess.square_file(move.to_square) > 4 else 0, rank)
        rook_to = chess.square(5 if chess.square_file(move.to_square) > 4 else 3, rank)
    next_state = EvalState(state.white.copy(), state.black.copy())
    board.push(move)

    king_moved = moving_piece.piece_type == chess.KING
    if not king_moved:
        adjust_piece(board, next_state, moving_piece, move.from_square, -1.0)
        adjust_piece(board, next_state, board.piece_at(move.to_square), move.to_square, 1.0)
    adjust_piece(board, next_state, captured_piece, captured_square, -1.0)
    if rook_from is not None and rook_to is not None:
        rook = chess.Piece(chess.ROOK, moving_piece.color)
        adjust_piece(board, next_state, rook, rook_from, -1.0)
        adjust_piece(board, next_state, rook, rook_to, 1.0)
    if king_moved:
        if moving_piece.color == chess.WHITE:
            next_state.white = build_accumulator(board, chess.WHITE)
        else:
            next_state.black = build_accumulator(board, chess.BLACK)
    return next_state


def evaluate(board: chess.Board, state: EvalState) -> int:
    """Blend V2 with a cheap residual evaluated from incremental accumulators."""
    residual = nnue_output(
        state.white,
        state.black,
        NNUE_HIDDEN_WEIGHT,
        NNUE_HIDDEN_BIAS,
        NNUE_OUTPUT_WEIGHT,
        NNUE_OUTPUT_BIAS,
    )
    correction = round(residual * 100 * NNUE_RESIDUAL_WEIGHT)
    if board.turn == chess.BLACK:
        correction = -correction
    return evaluate_v2(board) + correction


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


def quiescence(
    board: chess.Board, state: EvalState, alpha: int, beta: int, ply: int, depth: int
) -> int:
    """Search forcing captures so we do not evaluate halfway through an exchange."""
    check_time()
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_insufficient_material() or depth >= MAX_QUIESCENCE_DEPTH:
        return evaluate(board, state)

    stand_pat = evaluate(board, state)
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
        child_state = push_with_eval(board, move, state)
        score = -quiescence(board, child_state, -beta, -alpha, ply + 1, depth + 1)
        board.pop()
        if score >= beta:
            return beta
        alpha = max(alpha, score)
    return alpha


def negamax(
    board: chess.Board, state: EvalState, depth: int, alpha: int, beta: int, ply: int
) -> int:
    """Alpha-beta negamax: one routine works for both white and black."""
    check_time()
    if board.is_checkmate():
        return -MATE_SCORE + ply
    if board.is_stalemate():
        return 0
    if board.is_insufficient_material():
        return 0
    if depth == 0:
        return quiescence(board, state, alpha, beta, ply, 0)

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

    best_score = -MATE_SCORE
    best_move: chess.Move | None = None
    quiet_moves: list[chess.Move] = []
    colour = board.turn
    moves = ordered_moves(board, ply, entry.best_move if entry else None)
    for move_index, move in enumerate(moves):
        is_quiet = not board.is_capture(move) and not move.promotion
        if is_quiet:
            quiet_moves.append(move)
        child_state = push_with_eval(board, move, state)
        try:
            if move_index == 0:
                score = -negamax(board, child_state, depth - 1, -beta, -alpha, ply + 1)
            else:
                # Principal-variation search: most later moves cannot beat alpha.
                score = -negamax(board, child_state, depth - 1, -alpha - 1, -alpha, ply + 1)
                if alpha < score < beta:
                    score = -negamax(board, child_state, depth - 1, -beta, -alpha, ply + 1)
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
    board: chess.Board, state: EvalState, depth: int, preferred: chess.Move
) -> tuple[chess.Move, int]:
    """Search one complete depth and return its principal-variation first move."""
    alpha = -MATE_SCORE
    beta = MATE_SCORE
    best_move = preferred
    best_score = -MATE_SCORE
    entry = TRANSPOSITION_TABLE.get(position_key(board))
    moves = ordered_moves(board, 0, entry.best_move if entry else preferred)
    for move_index, move in enumerate(moves):
        child_state = push_with_eval(board, move, state)
        try:
            if move_index == 0:
                score = -negamax(board, child_state, depth - 1, -beta, -alpha, 1)
            else:
                score = -negamax(board, child_state, depth - 1, -alpha - 1, -alpha, 1)
                if alpha < score < beta:
                    score = -negamax(board, child_state, depth - 1, -beta, -alpha, 1)
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
    try:
        for depth in range(1, 64):
            iteration_started = perf_counter()
            state = build_eval_state(board)
            preferred, score = search_root(board, state, depth, preferred)
            completed_depth = depth
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
