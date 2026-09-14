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
HISTORY_MAX = 16_384
NEXT_DEPTH_GROWTH = 2.5
TIME_CHECK_INTERVAL = 64
ASPIRATION_WINDOW = 45
NULL_MOVE_MIN_DEPTH = 4
FUTILITY_MARGIN = 110

ADJACENT_FILE_MASKS = tuple(
    (chess.BB_FILES[file_index - 1] if file_index > 0 else 0)
    | (chess.BB_FILES[file_index + 1] if file_index < 7 else 0)
    for file_index in range(8)
)


def make_passed_pawn_masks(colour: chess.Color) -> tuple[int, ...]:
    """Precompute opposing-pawn squares that would block each passed-pawn candidate."""
    masks: list[int] = []
    for square in chess.SQUARES:
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        files = chess.BB_FILES[file_index] | ADJACENT_FILE_MASKS[file_index]
        ranks = range(rank_index + 1, 8) if colour == chess.WHITE else range(rank_index - 1, -1, -1)
        mask = 0
        for rank in ranks:
            mask |= files & chess.BB_RANKS[rank]
        masks.append(mask)
    return tuple(masks)


PASSED_PAWN_MASKS = {
    chess.WHITE: make_passed_pawn_masks(chess.WHITE),
    chess.BLACK: make_passed_pawn_masks(chess.BLACK),
}


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


def pawn_structure_score(board: chess.Board, colour: chess.Color) -> int:
    """Score passed, isolated, doubled pawns and the immediate king pawn shield."""
    pawns = int(board.pieces(chess.PAWN, colour))
    enemy_pawns = int(board.pieces(chess.PAWN, not colour))
    score = 0
    for file_mask in chess.BB_FILES:
        count = (pawns & file_mask).bit_count()
        if count > 1:
            score -= 12 * (count - 1)
    for square in chess.scan_reversed(pawns):
        file_index = chess.square_file(square)
        rank_index = chess.square_rank(square)
        forward_rank = rank_index if colour == chess.WHITE else 7 - rank_index
        if not pawns & ADJACENT_FILE_MASKS[file_index]:
            score -= 10
        if not enemy_pawns & PASSED_PAWN_MASKS[colour][square]:
            score += 12 + forward_rank * 9

    king_square = board.king(colour)
    if king_square is not None:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        shield_rank = king_rank + 1 if colour == chess.WHITE else king_rank - 1
        if 0 <= shield_rank < 8:
            shield_mask = chess.BB_RANKS[shield_rank] & (
                chess.BB_FILES[king_file] | ADJACENT_FILE_MASKS[king_file]
            )
            score += 11 * (pawns & shield_mask).bit_count()
    return score


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

    white_score += pawn_structure_score(board, chess.WHITE)
    black_score += pawn_structure_score(board, chess.BLACK)

    score_from_white = white_score - black_score
    return score_from_white if board.turn == chess.WHITE else -score_from_white


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
        score = -quiescence(board, -beta, -alpha, ply + 1, depth + 1)
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

    if (
        depth >= NULL_MOVE_MIN_DEPTH
        and not board.is_check()
        and beta < MATE_SCORE - 100
        and has_non_pawn_material(board, board.turn)
    ):
        # A side that can pass and still fail high is normally already winning enough.
        board.push(chess.Move.null())
        try:
            null_score = -negamax(board, depth - 3, -beta, -beta + 1, ply + 1)
        finally:
            board.pop()
        if null_score >= beta:
            return beta

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
    static_eval = evaluate(board)
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
