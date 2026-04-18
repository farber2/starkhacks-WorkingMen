"""Helpers for extracting detailed metadata for a UCI move."""

from __future__ import annotations

from typing import Any

import chess

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


def get_move_info(board: chess.Board, uci_move: str) -> dict[str, Any]:
    """Return rich metadata for a move in the context of the given board."""
    move = chess.Move.from_uci(uci_move)

    if move not in board.legal_moves:
        raise ValueError(f"Illegal move for current position: {uci_move}")

    moving_piece = board.piece_at(move.from_square)
    if moving_piece is None:
        raise ValueError(f"No piece found on source square for move: {uci_move}")

    is_castling = board.is_castling(move)
    is_en_passant = board.is_en_passant(move)
    is_capture = board.is_capture(move)
    is_promotion = move.promotion is not None

    captured_piece = None
    if is_capture:
        capture_square = move.to_square
        if is_en_passant:
            step = -8 if board.turn == chess.WHITE else 8
            capture_square = move.to_square + step
        captured = board.piece_at(capture_square)
        captured_piece = PIECE_NAMES[captured.piece_type] if captured else None

    return {
        "from_square": chess.square_name(move.from_square),
        "to_square": chess.square_name(move.to_square),
        "piece": PIECE_NAMES[moving_piece.piece_type],
        "is_capture": is_capture,
        "is_castling": is_castling,
        "is_en_passant": is_en_passant,
        "is_promotion": is_promotion,
        "promotion_piece": PIECE_NAMES[move.promotion] if move.promotion else None,
        "captured_piece": captured_piece,
    }
